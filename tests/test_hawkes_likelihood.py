"""Hawkes log-likelihood tests."""

from __future__ import annotations

import math

import numpy as np

from eonet_cascades.models.hawkes import HawkesParams, hawkes_log_likelihood


def _trivial_pi(k, x, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    area = (max_lon - min_lon) * (max_lat - min_lat)
    return np.full(x.shape[0], 1.0 / area)


def test_likelihood_no_events_is_minus_mu_t():
    n_marks = 2
    p = HawkesParams(
        mu=np.array([1.0, 2.0]),
        alpha=np.zeros((n_marks, n_marks)),
        beta=np.ones((n_marks, n_marks)),
        sigma=np.ones((n_marks, n_marks)),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    events = {
        "time": np.array([], dtype=np.float64),
        "lon": np.array([], dtype=np.float64),
        "lat": np.array([], dtype=np.float64),
        "mark": np.array([], dtype=np.int64),
    }
    t_end = 10.0
    ll = hawkes_log_likelihood(p, events, (0.0, t_end), _trivial_pi, bbox)
    # log L = -sum_k mu_k * t_end  (no events, no triggering, integral has only baseline part)
    expected = -((1.0 + 2.0) * t_end)
    assert math.isclose(ll, expected, rel_tol=1e-9)


def test_likelihood_single_event_no_triggering():
    p = HawkesParams(
        mu=np.array([1.0]),
        alpha=np.zeros((1, 1)),
        beta=np.ones((1, 1)),
        sigma=np.ones((1, 1)),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    area = 400.0
    events = {
        "time": np.array([2.0]),
        "lon": np.array([0.0]),
        "lat": np.array([0.0]),
        "mark": np.array([0], dtype=np.int64),
    }
    t_end = 10.0
    ll = hawkes_log_likelihood(p, events, (0.0, t_end), _trivial_pi, bbox)
    # log lambda at event = log(mu * 1/area) = log(1/area)
    # integral = mu * t_end
    expected = math.log(1.0 / area) - 1.0 * t_end
    assert math.isclose(ll, expected, rel_tol=1e-6)


def test_likelihood_monotone_in_event_count_baseline():
    p = HawkesParams(
        mu=np.array([0.1]),
        alpha=np.zeros((1, 1)),
        beta=np.ones((1, 1)),
        sigma=np.ones((1, 1)),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    t_end = 100.0

    def make_events(n):
        return {
            "time": np.linspace(1.0, t_end - 1.0, n),
            "lon": np.zeros(n),
            "lat": np.zeros(n),
            "mark": np.zeros(n, dtype=np.int64),
        }

    # 10 events that match the baseline rate (~10 expected in t_end=100 with mu=0.1).
    ll10 = hawkes_log_likelihood(p, make_events(10), (0.0, t_end), _trivial_pi, bbox)
    # 100 events: far above baseline -- log-likelihood should be lower (model is wrong).
    ll100 = hawkes_log_likelihood(p, make_events(100), (0.0, t_end), _trivial_pi, bbox)
    # With alpha=0 and high event count, the per-event log terms are log(small density)
    # which are very negative -- the more events, the lower the total LL at fixed mu.
    assert ll10 > ll100


def test_vectorized_matches_loop_on_small_synthetic():
    """Both formulations must agree to ~1e-7 on identical inputs."""
    import numpy as np

    from eonet_cascades.eval.synthetic import simulate_hawkes
    from eonet_cascades.models.hawkes import (
        HawkesParams,
        hawkes_log_likelihood,
        hawkes_log_likelihood_vectorized,
    )

    rng = np.random.default_rng(7)
    n_marks = 3
    p = HawkesParams(
        mu=np.array([0.4, 0.3, 0.2]),
        alpha=np.array([[0.25, 0.05, 0.0], [0.0, 0.30, 0.10], [0.05, 0.0, 0.20]]),
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    events = simulate_hawkes(p, bbox=bbox, t_end=80.0, rng=rng)

    def _uniform_pi(k, x, b):
        ml, mlat, max_lon, max_lat = b
        return np.full(x.shape[0], 1.0 / ((max_lon - ml) * (max_lat - mlat)))

    ll_loop = hawkes_log_likelihood(p, events, (0.0, 80.0), _uniform_pi, bbox)
    ll_vec = hawkes_log_likelihood_vectorized(p, events, (0.0, 80.0), _uniform_pi, bbox)
    assert abs(ll_loop - ll_vec) < 1e-7, f"loop {ll_loop} vs vectorized {ll_vec}"
