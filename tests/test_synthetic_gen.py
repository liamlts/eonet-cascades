"""Synthetic Hawkes generator tests."""

from __future__ import annotations

import numpy as np

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.models.hawkes import HawkesParams


def test_pure_poisson_when_alpha_zero():
    n_marks = 2
    p = HawkesParams(
        mu=np.array([0.5, 1.0]),
        alpha=np.zeros((n_marks, n_marks)),
        beta=np.ones((n_marks, n_marks)),
        sigma=np.ones((n_marks, n_marks)),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    rng = np.random.default_rng(42)
    events = simulate_hawkes(p, bbox=bbox, t_end=100.0, rng=rng)
    # Expected total count = (0.5 + 1.0) * t_end = 150
    n_total = events["time"].shape[0]
    assert 100 < n_total < 200, f"got {n_total} events, expected ~150"


def test_branching_increases_count():
    """Higher alpha -> more events overall."""
    bbox = (-10.0, -10.0, 10.0, 10.0)
    p_low = HawkesParams(
        mu=np.array([0.5]),
        alpha=np.array([[0.0]]),
        beta=np.array([[1.0]]),
        sigma=np.array([[1.0]]),
    )
    p_high = HawkesParams(
        mu=np.array([0.5]),
        alpha=np.array([[0.5]]),
        beta=np.array([[1.0]]),
        sigma=np.array([[1.0]]),
    )
    n_low = simulate_hawkes(p_low, bbox=bbox, t_end=100.0, rng=np.random.default_rng(0))[
        "time"
    ].shape[0]
    n_high = simulate_hawkes(p_high, bbox=bbox, t_end=100.0, rng=np.random.default_rng(0))[
        "time"
    ].shape[0]
    # With branching ratio 0.5, total events expected = N_immigrants / (1 - 0.5) = 2x.
    assert n_high > 1.5 * n_low


def test_offspring_within_sigma_of_parent():
    """Spatially-close offspring on a sharply-peaked kernel."""
    p = HawkesParams(
        mu=np.array([0.1]),
        alpha=np.array([[0.9]]),
        beta=np.array([[2.0]]),
        sigma=np.array([[0.1]]),  # very tight spatial kernel
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    rng = np.random.default_rng(0)
    events = simulate_hawkes(p, bbox=bbox, t_end=20.0, rng=rng)
    # If there are >= 5 events the cluster should be spatially concentrated.
    if events["time"].shape[0] >= 5:
        lons = events["lon"]
        lats = events["lat"]
        assert lons.std() < 5.0
        assert lats.std() < 5.0
