"""Forward-sim transition matrix shape / sanity tests."""

from __future__ import annotations

import pytest
import torch

from eonet_cascades.interpret.forward_sim_matrix import compute_transition_matrix
from eonet_cascades.models.neural_hawkes import NeuralHawkes


@pytest.mark.slow
def test_transition_matrix_shape_and_rowsums():
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2
    )
    t_matrix = compute_transition_matrix(
        model, n_marks=3, bbox=(-10.0, -10.0, 10.0, 10.0),
        n_trajectories=20, window_days=5.0,
    )
    assert t_matrix.shape == (3, 3)
    # Each row should sum to 0 (no events) or 1 (events occurred).
    for r in range(3):
        s = float(t_matrix[r].sum().item())
        assert s == pytest.approx(1.0, abs=1e-6) or s == pytest.approx(0.0, abs=1e-6)
