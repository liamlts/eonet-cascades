"""Full-covariance bivariate Mixture Density Network spatial head."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as nnf
from torch import nn


class MDNHead(nn.Module):
    """Mixture of N_mix bivariate Gaussians with full covariance via Cholesky.

    Output dims per component: 6 = 2 (mean) + 3 (Cholesky L: l00, l10, l11) + 1 (mixture logit).
    l00 and l11 are pushed positive via softplus; l10 is unconstrained.
    """

    def __init__(self, input_dim: int, n_components: int = 8) -> None:
        super().__init__()
        self.n_components = n_components
        self.head = nn.Linear(input_dim, n_components * 6)
        nn.init.xavier_uniform_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)
        self._log_2pi = math.log(2.0 * math.pi)

    def _unpack(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (means [b, k, 2], cholesky [b, k, 2, 2], log_weights [b, k])."""
        batch = h.shape[0]
        n_mix = self.n_components
        out = self.head(h)  # (batch, n_mix*6)
        out = out.view(batch, n_mix, 6)
        means = out[..., 0:2]
        l00 = nnf.softplus(out[..., 2]) + 1e-3
        l10 = out[..., 3]
        l11 = nnf.softplus(out[..., 4]) + 1e-3
        chol = torch.zeros(batch, n_mix, 2, 2, device=h.device, dtype=h.dtype)
        chol[..., 0, 0] = l00
        chol[..., 1, 0] = l10
        chol[..., 1, 1] = l11
        log_w = nnf.log_softmax(out[..., 5], dim=-1)
        return means, chol, log_w

    def log_prob(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x | h). Shapes: h (batch, input_dim), x (batch, 2). Returns (batch,)."""
        means, chol, log_w = self._unpack(h)  # (b,k,2), (b,k,2,2), (b,k)
        # Per-component log Gaussian density.
        # log N(x; mu, L L^T) = -log(2pi) - log|L| - 0.5 * ||L^-1 (x - mu)||^2
        diff = x.unsqueeze(1) - means  # (b, k, 2)
        # Solve L * z = diff for z: z = L^{-1} diff.
        chol_inv_diff = torch.linalg.solve_triangular(
            chol, diff.unsqueeze(-1), upper=False
        ).squeeze(-1)  # (b, k, 2)
        quad = (chol_inv_diff * chol_inv_diff).sum(dim=-1)  # (b, k)
        log_det = torch.log(chol[..., 0, 0]) + torch.log(chol[..., 1, 1])  # (b, k)
        comp_log = -self._log_2pi - log_det - 0.5 * quad  # (b, k)
        return torch.logsumexp(log_w + comp_log, dim=-1)  # (b,)

    def sample(self, h: torch.Tensor) -> torch.Tensor:
        """Sample one point per row from the mixture. Returns (batch, 2)."""
        means, chol, log_w = self._unpack(h)
        # Sample component index via Gumbel-max trick.
        gumbel = -torch.log(-torch.log(torch.rand_like(log_w) + 1e-12) + 1e-12)
        idx = (log_w + gumbel).argmax(dim=-1)  # (batch,)
        b_idx = torch.arange(h.shape[0], device=h.device)
        mu = means[b_idx, idx]  # (batch, 2)
        chol_sel = chol[b_idx, idx]  # (batch, 2, 2)
        eps = torch.randn(h.shape[0], 2, device=h.device)
        return mu + (chol_sel @ eps.unsqueeze(-1)).squeeze(-1)
