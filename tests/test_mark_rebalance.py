"""Tests for the Tier 1.5 class-rebalance plumbing.

Two surfaces:
  * `mark_rebalance_weights` helper -- weight shapes, normalization, mode dispatch.
  * `NeuralHawkes.log_likelihood(..., mark_weights=...)` -- unweighted-default
    matches the pre-rebalance behavior (bit-exact); uniform weights match too.
"""

from __future__ import annotations

import numpy as np
import torch

from eonet_cascades.models.neural_hawkes import NeuralHawkes
from eonet_cascades.training.neural_loop import mark_rebalance_weights


def test_inverse_sqrt_weights_normalize_to_mean_one():
    marks = np.array([0, 0, 0, 0, 1, 1, 2])
    w = mark_rebalance_weights(marks, n_marks=3, mode="inverse-sqrt")
    assert w.shape == (3,)
    # Mean weight should be exactly 1 (within float roundoff).
    assert abs(float(w.mean().item()) - 1.0) < 1e-6
    # Rare marks should have larger weight than common marks.
    assert float(w[2].item()) > float(w[1].item()) > float(w[0].item())


def test_inverse_frequency_weights_are_more_aggressive_than_sqrt():
    marks = np.array([0] * 100 + [1] * 1)
    w_sqrt = mark_rebalance_weights(marks, n_marks=2, mode="inverse-sqrt")
    w_inv = mark_rebalance_weights(marks, n_marks=2, mode="inverse-frequency")
    # Both put more weight on mark 1; inverse-frequency should put MORE.
    assert float(w_inv[1].item()) > float(w_sqrt[1].item())


def test_none_mode_returns_unit_weights():
    marks = np.array([0, 1, 2, 0, 1])
    w = mark_rebalance_weights(marks, n_marks=3, mode="none")
    assert torch.allclose(w, torch.ones(3))


def test_zero_count_mark_does_not_explode():
    marks = np.array([0, 0, 0])  # mark 1 is absent
    w = mark_rebalance_weights(marks, n_marks=2, mode="inverse-sqrt")
    assert torch.isfinite(w).all()
    # Absent mark gets the floored count of 1, so its weight is finite.


def test_log_likelihood_unweighted_default_matches_no_mark_weights_kwarg():
    """Passing mark_weights=None must be a no-op (backwards-compat)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_marks = 3
    n_events = 20
    model = NeuralHawkes(n_marks=n_marks, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2)
    model.eval()

    times = torch.tensor(np.sort(rng.uniform(0.0, 20.0, size=n_events)), dtype=torch.float32)
    lons = torch.tensor(rng.uniform(-10.0, 10.0, size=n_events), dtype=torch.float32)
    lats = torch.tensor(rng.uniform(-5.0, 5.0, size=n_events), dtype=torch.float32)
    marks = torch.tensor(rng.integers(0, n_marks, size=n_events), dtype=torch.long)
    window = (0.0, 20.0)

    ll_none = model.log_likelihood(times, lons, lats, marks, window)
    ll_unit = model.log_likelihood(
        times, lons, lats, marks, window, mark_weights=torch.ones(n_marks)
    )
    # Bit-exact: same forward + same multiply-by-1 per event.
    assert torch.allclose(ll_none, ll_unit, atol=1e-7, rtol=0.0)


def test_log_likelihood_with_skewed_weights_changes_value():
    """Sanity: non-uniform weights actually change the log-likelihood."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_marks = 3
    n_events = 30
    model = NeuralHawkes(n_marks=n_marks, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2)
    model.eval()

    times = torch.tensor(np.sort(rng.uniform(0.0, 20.0, size=n_events)), dtype=torch.float32)
    lons = torch.tensor(rng.uniform(-10.0, 10.0, size=n_events), dtype=torch.float32)
    lats = torch.tensor(rng.uniform(-5.0, 5.0, size=n_events), dtype=torch.float32)
    marks = torch.tensor(rng.integers(0, n_marks, size=n_events), dtype=torch.long)
    window = (0.0, 20.0)

    ll_none = model.log_likelihood(times, lons, lats, marks, window)
    skewed = torch.tensor([0.5, 1.0, 3.0])
    ll_skewed = model.log_likelihood(times, lons, lats, marks, window, mark_weights=skewed)
    assert not torch.allclose(ll_none, ll_skewed), "skewed weights should change log-lik"
