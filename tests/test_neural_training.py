"""Verify training loop decreases NLL on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.optim import AdamW

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.models.hawkes import HawkesParams
from eonet_cascades.models.neural_hawkes import NeuralHawkes
from eonet_cascades.training.neural_loop import TrainChunk, train_one_epoch


@pytest.mark.slow
def test_neural_training_reduces_nll():
    torch.manual_seed(0)
    np.random.seed(0)
    n_marks = 3
    bbox = (-10.0, -10.0, 10.0, 10.0)
    truth = HawkesParams(
        mu=np.array([0.5, 0.3, 0.2]),
        alpha=np.array([[0.30, 0.10, 0.00], [0.00, 0.40, 0.15], [0.05, 0.00, 0.20]]),
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    rng = np.random.default_rng(0)
    events = simulate_hawkes(truth, bbox=bbox, t_end=80.0, rng=rng)
    chunk = TrainChunk(
        times=torch.tensor(events["time"], dtype=torch.float32),
        lons=torch.tensor(events["lon"], dtype=torch.float32),
        lats=torch.tensor(events["lat"], dtype=torch.float32),
        marks=torch.tensor(events["mark"], dtype=torch.long),
        window=(0.0, 80.0),
    )
    model = NeuralHawkes(n_marks=n_marks, hidden_dim=16, mark_emb_dim=8, spatial_emb_dim=8, n_mix=4)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(5):
        info = train_one_epoch(model, [chunk], optimizer, device="cpu")
        losses.append(info["nll_per_event"])
    print(f"NLL per event over 5 epochs: {losses}")
    assert losses[-1] < losses[0] - 0.05, f"NLL did not decrease enough: {losses}"
