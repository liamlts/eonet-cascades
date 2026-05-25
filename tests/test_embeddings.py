"""Mark + spatial embedding shape tests."""

from __future__ import annotations

import torch

from eonet_cascades.models.components.embeddings import MarkEmbedding, SpatialEmbedding


def test_mark_embedding_shape():
    n_marks, dim = 12, 16
    emb = MarkEmbedding(n_marks=n_marks, dim=dim)
    idx = torch.tensor([0, 3, 7, 11])
    out = emb(idx)
    assert out.shape == (4, dim)


def test_mark_embedding_distinct_marks_distinct_vectors():
    emb = MarkEmbedding(n_marks=5, dim=8)
    out = emb(torch.arange(5))
    # All 5 vectors should be distinguishable (unique rows).
    norms = torch.cdist(out, out)
    off_diag = norms[~torch.eye(5, dtype=bool)]
    assert (off_diag > 1e-6).all(), "embeddings should not collapse"


def test_spatial_embedding_shape():
    emb = SpatialEmbedding(dim=16)
    x = torch.tensor([[-100.0, 35.0], [-95.0, 40.0], [-110.0, 25.0]])
    out = emb(x)
    assert out.shape == (3, 16)


def test_spatial_embedding_deterministic():
    emb = SpatialEmbedding(dim=8)
    x = torch.tensor([[-100.0, 35.0]])
    a = emb(x)
    b = emb(x)
    assert torch.allclose(a, b)
