"""NeuralHawkes end-to-end forward + intensity tests."""

from __future__ import annotations

import torch

from eonet_cascades.models.neural_hawkes import NeuralHawkes


def test_forward_shapes():
    model = NeuralHawkes(n_marks=4, hidden_dim=16, mark_emb_dim=8, spatial_emb_dim=8, n_mix=4)
    # Simulate 7 events
    times = torch.tensor([0.1, 0.5, 1.2, 2.0, 3.3, 4.5, 6.0])
    lons = torch.linspace(-10.0, 10.0, 7)
    lats = torch.linspace(0.0, 5.0, 7)
    marks = torch.tensor([0, 1, 2, 0, 3, 1, 2])
    out = model.forward(times, lons, lats, marks)
    # Output should contain per-event log lambda_t, mark log-prob at observed mark,
    # and spatial log prob at the event location (each shape (n_events,)).
    assert out["log_lambda_t"].shape == (7,)
    assert out["log_p_mark"].shape == (7,)
    assert out["log_p_x"].shape == (7,)


def test_forward_history_grows_with_index():
    """Hidden state at event i should depend on events 0..i-1 only (causality)."""
    torch.manual_seed(0)
    model = NeuralHawkes(n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2)
    times = torch.tensor([0.1, 0.5, 1.2, 2.0])
    lons = torch.zeros(4)
    lats = torch.zeros(4)
    marks = torch.tensor([0, 1, 2, 0])
    out_full = model.forward(times, lons, lats, marks)
    # If we trim to first 2 events, the first 2 outputs should match (causality).
    out_trim = model.forward(times[:2], lons[:2], lats[:2], marks[:2])
    assert torch.allclose(out_full["log_lambda_t"][:2], out_trim["log_lambda_t"], atol=1e-5)
    assert torch.allclose(out_full["log_p_mark"][:2], out_trim["log_p_mark"], atol=1e-5)
