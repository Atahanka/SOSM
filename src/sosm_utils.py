import numpy as np
from lifelines.utils import concordance_index as lifelines_concordance_index
from sosm_logging import log_training_start, log_epoch, log_training_end
import pandas as pd
import torch

def train(model, loader, optimizer, device, epochs: int = 500, logger=None):
    """
    Executes the self-organizing manifold optimization loop, minimizing 
    combined contrastive alignment and curvature energy.
    """
    if logger:
        log_training_start(logger, model, loader, epochs, device)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        for x_rna, x_methyl, x_mirna, t in loader:
            x_rna, x_methyl, x_mirna = x_rna.to(device), x_methyl.to(device), x_mirna.to(device)
            t = t.to(device)

            optimizer.zero_grad()
            loss, align, curv, Z = model(x_rna, x_methyl, x_mirna, t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            if logger:
                log_epoch(logger, epoch, avg_loss, align.item(), curv.item(), lr)
            else:
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Align: {align:.4f} | Curv: {curv:.4f}")

    if logger:
        log_training_end(logger, losses)
    
    return losses

def calculate_cindex(time: np.ndarray, labels: np.ndarray, event: np.ndarray) -> float:
    """Calculates Harrell's Concordance Index (C-index) using lifelines."""
    return float(lifelines_concordance_index(time, labels, event))

def compute_cross_modality_coactivation(
    model,
    top_rna_idx: np.ndarray,
    top_methyl_idx: np.ndarray,
    top_mirna_idx: np.ndarray,
    adata,
    percentile: float = 95.0
) -> pd.DataFrame:
    """
    Computes linearized input-to-latent sensitivity vectors across encoder layers
    and extracts the top cross-modality co-activation pairs in the shared manifold.
    """
    def _effective_latent_vectors(enc_module, top_idx):
        idx = np.ascontiguousarray(top_idx)
        w1 = enc_module.encoder[0].weight.data[:, idx]
        w2 = enc_module.encoder[3].weight.data
        w3 = enc_module.proj[0].weight.data
        eff = w3 @ w2 @ w1
        return torch.nn.functional.normalize(eff, p=2, dim=0).cpu().numpy()

    v_rna = _effective_latent_vectors(model.encoder.rna_enc, top_rna_idx)
    v_met = _effective_latent_vectors(model.encoder.methyl_enc, top_methyl_idx)
    v_mir = _effective_latent_vectors(model.encoder.mirna_enc, top_mirna_idx)

    V_all = np.concatenate([v_rna, v_met, v_mir], axis=1)
    sim_matrix = np.abs(V_all.T @ V_all)

    # Mask intra-modality blocks so only cross-modality interactions remain
    for start in (0, 20, 40):
        sim_matrix[start:start+20, start:start+20] = 0.0

    triu_i, triu_j = np.triu_indices(60, k=1)
    cross_mask = (triu_i // 20) != (triu_j // 20)
    pairs_i, pairs_j = triu_i[cross_mask], triu_j[cross_mask]
    scores = sim_matrix[pairs_i, pairs_j]

    top_edges = scores >= np.percentile(scores, percentile)
    all_names = np.concatenate([
        adata.uns['rna_names'][top_rna_idx],
        adata.uns['methyl_names'][top_methyl_idx],
        adata.uns['mirna_names'][top_mirna_idx]
    ])

    return pd.DataFrame({
        "Source_Idx": pairs_i[top_edges] + 1,
        "Source_Feature": all_names[pairs_i[top_edges]],
        "Target_Idx": pairs_j[top_edges] + 1,
        "Target_Feature": all_names[pairs_j[top_edges]],
        "Latent_CoActivation": scores[top_edges]
    }).sort_values("Latent_CoActivation", ascending=False)