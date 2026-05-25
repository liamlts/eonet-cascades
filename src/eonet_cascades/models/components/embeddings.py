"""Mark and spatial embeddings for Tier 1 Neural Hawkes."""

from __future__ import annotations

import torch
from torch import nn


class MarkEmbedding(nn.Module):
    """Learned per-mark embedding."""

    def __init__(self, n_marks: int, dim: int = 16) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_marks, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, mark_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(mark_idx)


class SpatialEmbedding(nn.Module):
    """Small MLP R^2 -> R^dim for (lon, lat) -> spatial embedding."""

    def __init__(self, dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
