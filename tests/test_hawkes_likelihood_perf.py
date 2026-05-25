"""Vectorized vs loop performance regression test."""

from __future__ import annotations

import time

import numpy as np
import pytest

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.models.hawkes import (
    HawkesParams,
    hawkes_log_likelihood,
    hawkes_log_likelihood_vectorized,
)


def _uniform_pi(k, x, b):
    min_lon, min_lat, max_lon, max_lat = b
    return np.full(x.shape[0], 1.0 / ((max_lon - min_lon) * (max_lat - min_lat)))


@pytest.mark.slow
def test_vectorized_at_least_20x_faster_than_loop():
    rng = np.random.default_rng(3)
    n_marks = 3
    p = HawkesParams(
        mu=np.array([0.5, 0.3, 0.2]),
        alpha=np.array([[0.30, 0.10, 0.00], [0.00, 0.40, 0.15], [0.05, 0.00, 0.20]]),
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    events = simulate_hawkes(p, bbox=bbox, t_end=200.0, rng=rng)
    n = events["time"].shape[0]
    print(f"benchmark on {n} events")

    t0 = time.perf_counter()
    ll_loop = hawkes_log_likelihood(p, events, (0.0, 200.0), _uniform_pi, bbox)
    dt_loop = time.perf_counter() - t0

    t1 = time.perf_counter()
    ll_vec = hawkes_log_likelihood_vectorized(p, events, (0.0, 200.0), _uniform_pi, bbox)
    dt_vec = time.perf_counter() - t1

    print(f"  loop:       {dt_loop:.3f} s   (ll={ll_loop:.4f})")
    print(f"  vectorized: {dt_vec:.3f} s   (ll={ll_vec:.4f})")
    print(f"  speedup:    {dt_loop / dt_vec:.1f}x")
    assert abs(ll_loop - ll_vec) < 1e-6
    assert dt_loop / dt_vec >= 20.0, f"only {dt_loop / dt_vec:.1f}x speedup, expected >=20x"
