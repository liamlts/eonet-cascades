"""Tier 1 cascade recovery gate.

Per spec §6.3: train on synthetic Hawkes data with known alpha. The
aggregated attribution matrix should qualitatively match the true alpha
sparsity pattern.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.optim import AdamW

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.interpret.attribution import compute_attribution_matrix
from eonet_cascades.models.hawkes import HawkesParams
from eonet_cascades.models.neural_hawkes import NeuralHawkes
from eonet_cascades.training.neural_loop import TrainChunk, train_one_epoch


@pytest.mark.slow
def test_neural_cascade_recovery():
    torch.manual_seed(0)
    np.random.seed(0)
    n_marks = 3
    bbox = (-10.0, -10.0, 10.0, 10.0)
    alpha_true = np.array(
        [
            [0.30, 0.10, 0.00],
            [0.00, 0.40, 0.15],
            [0.05, 0.00, 0.20],
        ]
    )
    truth = HawkesParams(
        mu=np.array([0.5, 0.3, 0.2]),
        alpha=alpha_true,
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    rng = np.random.default_rng(0)
    events = simulate_hawkes(truth, bbox=bbox, t_end=200.0, rng=rng)
    chunk = TrainChunk(
        times=torch.tensor(events["time"], dtype=torch.float32),
        lons=torch.tensor(events["lon"], dtype=torch.float32),
        lats=torch.tensor(events["lat"], dtype=torch.float32),
        marks=torch.tensor(events["mark"], dtype=torch.long),
        window=(0.0, 200.0),
    )
    model = NeuralHawkes(n_marks=n_marks, hidden_dim=32, mark_emb_dim=8, spatial_emb_dim=8, n_mix=4)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    for _ in range(50):
        train_one_epoch(model, [chunk], optimizer)

    a_matrix = compute_attribution_matrix(
        model, chunk.times, chunk.lons, chunk.lats, chunk.marks, n_marks=n_marks
    )
    # Use tolist() -> np.array to avoid torch/numpy 1.x vs 2.x ABI clash.
    a_np = np.array(a_matrix.tolist())
    print("True alpha:\n", alpha_true)
    print("Attribution A:\n", a_np)

    quart = np.quantile(a_np, 0.75)
    nonzero_mask = alpha_true > 1e-3
    zero_mask = alpha_true < 1e-3
    nonzero_in_top = (a_np[nonzero_mask] >= quart).mean()
    zero_in_top = (a_np[zero_mask] >= quart).mean()
    print(f"non-zero entries in top quartile: {nonzero_in_top:.2f}")
    print(f"true-zero entries in top quartile: {zero_in_top:.2f}")
    assert nonzero_in_top >= 0.7, (
        f"only {nonzero_in_top:.2f} of true non-zero alpha entries in top quartile"
    )
    assert zero_in_top <= 0.25, (
        f"{zero_in_top:.2f} of true-zero entries in top quartile (should be <=0.25)"
    )
