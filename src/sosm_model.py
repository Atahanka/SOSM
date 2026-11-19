import torch
import torch.nn as nn
import torch.nn.functional as F

class OmicsEncoder(nn.Module):
    """
    Multilayer Perceptron (MLP) encoder for an individual omics modality (e.g., RNA, Methylation, miRNA).
    Applies batch normalization and ReLU activations, mapping high-dimensional molecular profiles
    to a unit-normalized latent space via L2 normalization.
    """
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU()])
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(prev_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.proj(h)
        return F.normalize(z, p=2, dim=1)

class MultiOmicsEncoder(nn.Module):
    """
    Integrates multiple omics encoders (transcriptomic, epigenomic, post-transcriptional) into a unified architecture.
    """
    def __init__(self, rna_dim: int, methyl_dim: int, mirna_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        self.rna_enc = OmicsEncoder(rna_dim, hidden_dims, latent_dim)
        self.methyl_enc = OmicsEncoder(methyl_dim, hidden_dims, latent_dim)
        self.mirna_enc = OmicsEncoder(mirna_dim, hidden_dims, latent_dim)

    def forward(self, x_rna: torch.Tensor, x_methyl: torch.Tensor, x_mirna: torch.Tensor):
        return self.rna_enc(x_rna), self.methyl_enc(x_methyl), self.mirna_enc(x_mirna)

def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) loss for multi-modal contrastive alignment.
    Maximizes consistency between paired omics views of the same biological sample.
    """
    B = z_i.shape[0]
    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    pos_i_j = torch.diag(sim, B)
    pos_j_i = torch.diag(sim, -B)
    pos = torch.cat([pos_i_j, pos_j_i]).view(2 * B, 1)
    mask = (~torch.eye(2 * B, dtype=torch.bool, device=z.device)).float()
    neg = sim * mask
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)

def survival_kernel(t: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Computes pairwise survival affinity weights using a Gaussian RBF kernel over survival pseudotimes.
    """
    diff = (t.unsqueeze(0) - t.unsqueeze(1)).abs()
    return torch.exp(-diff ** 2 / (2 * sigma ** 2))

def sosm_curvature_loss(Z: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Computes the discrete graph Laplacian Dirichlet energy (geodesic curvature penalty).
    Measures local manifold bending and promotes alignment along low-curvature survival flows.
    """
    N = Z.shape[0]
    D = W.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(D + 1e-8, -0.5))
    L = torch.eye(N, device=Z.device) - D_inv_sqrt @ W @ D_inv_sqrt
    LZ = L @ Z
    return torch.norm(LZ, p='fro') ** 2

class SOSMModel(nn.Module):
    """
    Self-Organizing Survival Manifolds (SOSM) core model.
    Combines multi-modal contrastive alignment with curvature-constrained geometric optimization.
    """
    def __init__(self, rna_dim: int, methyl_dim: int, mirna_dim: int, hidden_dims: list, latent_dim: int,
                 temp: float = 0.5, sigma: float = 0.1, lambda_curv: float = 0.01):
        super().__init__()
        self.encoder = MultiOmicsEncoder(rna_dim, methyl_dim, mirna_dim, hidden_dims, latent_dim)
        self.temp = temp
        self.sigma = sigma
        self.lambda_curv = lambda_curv

    def forward(self, x_rna: torch.Tensor, x_methyl: torch.Tensor, x_mirna: torch.Tensor, t: torch.Tensor):
        z_rna, z_methyl, z_mirna = self.encoder(x_rna, x_methyl, x_mirna)
        align_loss = nt_xent_loss(z_rna, z_methyl, self.temp) + nt_xent_loss(z_rna, z_mirna, self.temp)

        # Fuse modalities into a shared Riemannian latent coordinate space
        Z = (z_rna + z_methyl + z_mirna) / 3.0
        Z = F.normalize(Z, p=2, dim=1)

        W = survival_kernel(t, self.sigma)
        curv_loss = sosm_curvature_loss(Z, W)

        total_loss = align_loss + self.lambda_curv * curv_loss
        return total_loss, align_loss, curv_loss, Z