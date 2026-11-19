import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from typing import List, Tuple
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import anndata
import umap

def _save_figure(save_path: str, dpis: tuple = (300, 450, 600)):
    """Saves multi-resolution PNGs (300, 450, 600 DPI) and a lossless vector PDF."""
    if save_path:
        base, _ = os.path.splitext(save_path)
        for dpi in dpis:
            plt.savefig(f"{base}_{dpi}dpi.png", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{base}.pdf", bbox_inches='tight')

def plot_training_loss(losses: List[float], save_path: str = None):
    """Plots training loss trajectory across epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SOSM Training Loss Convergence')
    plt.grid(True, alpha=0.3)
    _save_figure(save_path)
    plt.close()

def plot_km_curves(km_fits: List[KaplanMeierFitter], title: str = "Kaplan-Meier Survival Curves", save_path: str = None):
    """Plots stratified Kaplan-Meier survival curves with confidence intervals."""
    plt.figure(figsize=(8, 6))
    for km in km_fits:
        km.plot_survival_function()
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.title(title)
    plt.legend(title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    _save_figure(save_path)
    plt.close()

def plot_stratified_km(
    risk_scores: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    cohort_label: str,
    cindex_label: str,
    cindex_val: float,
    percentile: float = 75.0,
    save_path: str = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Stratifies a cohort by a risk percentile threshold, runs a two-sample log-rank test,
    and renders uniform Kaplan-Meier survival curves.
    """
    risk_threshold = np.percentile(risk_scores, percentile)
    clusters = (risk_scores > risk_threshold).astype(int)
    mask_low, mask_high = clusters == 0, clusters == 1

    if mask_low.sum() > 0 and mask_high.sum() > 0:
        lr_res = logrank_test(
            durations[mask_low], durations[mask_high],
            event_observed_A=events[mask_low], event_observed_B=events[mask_high]
        )
        p_str = "p < 0.001" if lr_res.p_value < 0.001 else f"p = {lr_res.p_value:.4f}"
        km_title = f"{cohort_label} ({int(percentile)}th Percentile Risk Split) | {cindex_label}: {cindex_val:.3f} | {p_str}"
    else:
        p_str = "N/A"
        km_title = f"{cohort_label} ({int(percentile)}th Percentile Risk Split) | {cindex_label}: {cindex_val:.3f}"

    km_fits = []
    for c, label_name in [(0, "Low Risk"), (1, "High Risk")]:
        km = KaplanMeierFitter()
        mask = clusters == c
        if mask.sum() > 0:
            km.fit(durations[mask], events[mask], label=f"{label_name} (n={mask.sum()})")
            km_fits.append(km)

    plot_km_curves(km_fits, title=km_title, save_path=save_path)
    return mask_low, mask_high, p_str

def plot_manifold(
    Z: torch.Tensor,
    t: np.ndarray,
    cluster_labels: np.ndarray,
    adata: anndata.AnnData,
    save_dir: str = "plots/manifold",
    cluster_names: dict = None,
):
    """Generates UMAP projections of the latent Riemannian manifold colored by survival risk and pseudotime."""
    os.makedirs(save_dir, exist_ok=True)
    Z_np = Z.cpu().numpy()
    
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        embedding = reducer.fit_transform(Z_np)
        
        plt.figure(figsize=(8, 6))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        unique_clusters = np.unique(cluster_labels)
        
        for idx, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            label_name = (
                cluster_names.get(cluster_id, f'Cluster {cluster_id}')
                if cluster_names is not None
                else f'Cluster {cluster_id}'
            )
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                color=colors[idx % len(colors)],
                label=label_name,
                s=30,
                alpha=0.7
            )
            
        plt.legend(frameon=True, loc='best')
        plt.title('Latent Manifold (UMAP) — Stratified Risk')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        _save_figure(os.path.join(save_dir, 'umap_clusters.png'))
        plt.close()
        
        if t is not None:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=t, cmap='viridis', s=30, alpha=0.7)
            plt.colorbar(scatter, label='ssGSEA Pseudotime')
            plt.title('Latent Manifold (UMAP) — Geodesic Flow (Pseudotime)')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            _save_figure(os.path.join(save_dir, 'umap_t_colored.png'))
            plt.close()
    except ImportError:
        print("UMAP not installed. Skipping manifold visualization.")

def plot_correlation_diagnostic(score: np.ndarray, time: np.ndarray, save_path: str = None):
    """Plots correlation diagnostic between inferred pseudotime and true clinical survival time."""
    from scipy.stats import pearsonr
    corr, _ = pearsonr(score, time)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(time, score, alpha=0.6, s=30)
    plt.xlabel("Survival Time (days)")
    plt.ylabel("ssGSEA Combined Score")
    plt.title(f"Pseudotime vs Survival Time (corr = {corr:.3f})")
    plt.grid(True, alpha=0.3)
    
    z = np.polyfit(time, score, 1)
    p = np.poly1d(z)
    plt.plot(time, p(time), "r--", alpha=0.8, linewidth=2)
    
    _save_figure(save_path)
    plt.close()