# SOSM: Self-Organizing Survival Manifolds

**Prognosis Without Prediction: The Intrinsic Geometry of Survival**

SOSM is a deterministic, unsupervised deep learning framework designed to model time-to-event trajectories in complex systems plagued by the High-Dimension, Low-Sample-Size (HDLSS) bottleneck. Validated on a matched 184-patient multi-omics clear cell renal cell carcinoma cohort (TCGA-KIRC), SOSM projects disparate molecular profiles (RNA-seq, DNA methylation, and miRNA-seq) onto an $L_2$-normalized encoders ($\mathbb{S}^{15}$) into a unified 16-dimensional latent manifold.

By pairing cross-modality contrastive alignment (NT-Xent) with a pseudotime-weighted, curvature-minimizing graph Laplacian ($\mathcal{L}_{curv} = \Vert{}LZ\Vert{}_F^2$), the architecture decouples representation learning from explicit survival endpoints—organizing patient states along low-curvature geodesic trajectories where systemic failure manifests as a geometric phase transition.

## 📄 Paper

Read the full paper: **A. Karagöz, *"Prognosis Without Prediction: The Intrinsic Geometry of Survival,"* [arXiv:2611.24799](https://arxiv.org/abs/2611.24799), 2026.**

- **Theoretical Foundation:** A. Karagöz, *"Self-Organizing Survival Manifolds: A Theory for Unsupervised Discovery of Prognostic Structures in Biological Systems,"* [arXiv:2508.06539](https://arxiv.org/abs/2508.06539), 2025.

## 📂 Repository Structure

```
SOSM/
├── src/                                            # Core source code
│   ├── main_cv.py                                  # End-to-end deterministic 5-fold CV execution & evaluation
│   ├── sosm_data.py                                # Broad Firehose TCGA-KIRC ingestion, feature selection & ssGSEA proxy
│   ├── sosm_model.py                               # Multi-branch MLP encoders, NT-Xent loss & Laplacian curvature loss
│   ├── sosm_utils.py                               # Training loop, Harrell's C-index & linearized co-activation sensitivity
│   ├── sosm_plots.py                               # Multi-DPI (300/450/600) PNG & vector PDF figure rendering
│   └── sosm_logging.py                             # Timestamped console and file logging utilities
├── logs/                                           # Generated artifacts (logs, figures, and CSV tables)
│   ├── manifold_optimal/
│   │   ├── umap_clusters.{pdf,png}                 # UMAP latent projection colored by K-Means clusters
│   │   └── umap_t_colored.{pdf,png}                # UMAP latent projection colored by ssGSEA pseudotime
│   ├── correlation_diagnostic_optimal.{pdf,png}    # ssGSEA proxy vs. clinical survival time correlation check
│   ├── km_curves_global_oof.{pdf,png}              # Global out-of-fold Kaplan-Meier survival curves (N=184)
│   ├── km_curves_test_optimal.{pdf,png}            # Held-out optimal fold Kaplan-Meier survival curves (n=37)
│   ├── training_loss_optimal.{pdf,png}             # Optimal fold training loss trajectory over 500 epochs
│   ├── sosm_top_biomarkers_mapped.csv              # Top 20 primary encoder attribution weights per modality
│   ├── sosm_cross_modality_coactivation_top5pct.csv # Top 5% cross-modality linearized sensitivity pairs
│   └── sosm_*.log                                  # Timestamped execution logs
├── CODEOWNERS
├── requirements.txt
├── README.md
└── LICENSE
```

## 🔧 Setup

```
# Clone and create an isolated virtual environment
git clone https://github.com/Atahanka/SOSM.git
cd SOSM
python3.10 -m venv .venv_sosm
source .venv_sosm/bin/activate

# Install pinned dependencies (lifelines installed with --no-deps to preserve exact pandas/numpy ABI)
pip install -r requirements.txt
pip install --no-deps lifelines==0.30.0
```

## 🧪 Usage

The entire pipeline—automated Broad Firehose TCGA-KIRC download and parsing, leak-free per-fold standardization, ssGSEA prognostic pseudotime initialization, 5-fold cross-validation (`seed = 3`, 500 epochs per fold), downstream $L_2$-penalized Cox evaluation, and figure/table generation—is executed via a single entry point:

```
python src/main_cv.py
```

### Pipeline Execution Overview (`src/main_cv.py`)

1. **Multi-Omics Data Ingestion (`sosm_data.py`):** Downloads and aligns matched clinical, RNA-seq (5,000 features), DNA Methylation450k (5,000 CpGs via chunked variance filtering), and miRNA-seq (2,000 miRNAs) profiles for $N = 184$ TCGA-KIRC patients.

2. **Leak-Free Fold Preparation (`sosm_data.py` & `main_cv.py`):** Computes the label-free proliferation-minus-immune ssGSEA pseudotime proxy ($+1.5 \times \text{proliferation} - 1.2 \times \text{immune}$), then fits top-variance feature selection (5,000 RNAs, 5,000 CpGs, 2,000 miRNAs), `StandardScaler` normalization, and global one-bit polarity sign calibration strictly on each 80% training split before transforming held-out test folds.

3. **Manifold Optimization (`sosm_model.py` & `sosm_utils.py`):** Trains three modality-specific MLPs (`[512, 128] -> 16`, $L_2$-normalized onto $\mathbb{S}^{15}$) using Adam (`lr=1e-3`, `weight_decay=1e-4`, `batch_size=128`) under $\mathcal{L}_{total} = \mathcal{L}_{align} + \lambda_{curv}\mathcal{L}_{curv}$ ($\tau = 0.1$, $\sigma = 1.0$, $\lambda_{curv} = 0.1$).

4. **Evaluation & Biomarker Extraction (`main_cv.py`, `sosm_utils.py`, `sosm_plots.py`):** Evaluates frozen embeddings via a downstream $L_2$-penalized `CoxPHFitter(penalizer=1.0)` and discrete `KMeans(n_clusters=2)`, quantifies test-set Laplacian dissipation $\delta_i = \Vert{}(L_{\text{test}}Z_{\text{test}})_i\Vert{}_2$, extracts top primary weights and linearized cross-modality sensitivity pairs ($W_{\text{proj}}W_{h2}W_{h1}$), and exports all figures and CSVs to `logs/`.

## 📊 Key Results (TCGA-KIRC, $N = 184$)

- **Out-of-Fold Prognostic Discrimination:** Mean C-index of $0.768 \pm 0.068$ across 5-fold cross-validation (reaching $0.848$ on the optimal fold), substantially outperforming discrete K-Means partitioning ($0.507$).

- **Survival Stratification:** Significant separation between 75th-percentile High-Risk ($n = 46$) and Low-Risk ($n = 138$) cohorts (Mantel-Cox log-rank $p < 0.001$ globally; $p = 0.0243$ on the held-out optimal fold).

- **Geometric Phase Transition:** High-risk patients exhibit a $14.7\%$ increase in local graph Laplacian dissipation ($\delta_i = \Vert{}(L_{\text{test}}Z_{\text{test}})_i\Vert{}_2$, $0.5887$ vs. $0.5131$).

- **Label-Free Biomarker Discovery:** Linearized encoder sensitivity autonomously recovers canonical ccRCC metabolic and stromal-immune drivers (*HMGCS2*, *PTGS2*, *GREM1*, *LCN2*, *CCL18*) and cross-modality miRNA hubs (`hsa-mir-621`, `hsa-mir-492`).

## 📜 Dual-License Model

Because this repository contains novel geometric deep learning architectures, discrete graph Laplacian curvature penalties, and multi-omics pipelines with commercial and clinical applications, it is distributed under a **Dual-License** model.

**1. Academic / Open-Source (GNU AGPL v3.0)**

By default, the source code in this repository is available under the GNU Affero General Public License v3.0.

⚠️ **Important Copyleft Provision:** If you distribute modified versions of this code, or modify it and allow external users to interact with it over a network (AGPL v3 Section 13), you are legally required to **open-source your entire derivative application** under the GNU AGPL v3.

**2. Commercial / Proprietary License**

If you intend to distribute, host, or integrate this architecture, its loss formulations, or derivative models in a commercial product, closed-source SaaS/API, client deliverable, or proprietary workflow (e.g., biotech/pharma R&D, clinical diagnostics, algorithmic trading, or foundation model training) without complying with AGPL v3 source-disclosure obligations, you **MUST** obtain a separate written Commercial License.

- **Pre-Computed Artifacts (`logs/`):** Author-generated logs, figures, and biomarker tables remain the exclusive intellectual property of the Author and may not be redistributed or used for competitive benchmarking without permission.
- **Commercial Licensing Inquiries:** See Section 4 of the [LICENSE](./LICENSE) file for instructions on requesting a Commercial License.

**🔒 For complete terms, conditions, and the full GNU AGPL v3.0 text**, please consult the [LICENSE](./LICENSE) file.

> **© 2026 Atahan Karagöz.**  
> Dual-Licensed under the **GNU AGPL v3.0** or a separate **Commercial License**. Unauthorized proprietary use is strictly prohibited.