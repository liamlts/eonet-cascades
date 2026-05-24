"""Hawkes MLE fit tests."""

from __future__ import annotations

import numpy as np

from eonet_cascades.models.hawkes import ParametricHawkes


def _trivial_pi(k, x, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    area = (max_lon - min_lon) * (max_lat - min_lat)
    return np.full(x.shape[0], 1.0 / area)


def test_fit_recovers_homogeneous_poisson_mu():
    """With alpha=0 (no triggering), MLE for mu_k should match (count_k / T) closely."""
    K = 2  # noqa: N806
    rng = np.random.default_rng(0)
    T = 200.0  # noqa: N806
    bbox = (-10.0, -10.0, 10.0, 10.0)
    # Generate by hand: rates 0.5 and 1.0, uniform over bbox.
    rates = [0.5, 1.0]
    times: list[float] = []
    lons: list[float] = []
    lats: list[float] = []
    marks: list[int] = []
    for k in range(K):
        n = rng.poisson(rates[k] * T)
        ts = np.sort(rng.uniform(0, T, n))
        times.extend(ts.tolist())
        lons.extend(rng.uniform(-10, 10, n).tolist())
        lats.extend(rng.uniform(-10, 10, n).tolist())
        marks.extend([k] * n)
    order = np.argsort(times)
    events = {
        "time": np.array(times)[order],
        "lon": np.array(lons)[order],
        "lat": np.array(lats)[order],
        "mark": np.array(marks, dtype=np.int64)[order],
    }
    model = ParametricHawkes(K=K, bbox=bbox, pi_k=_trivial_pi)
    result = model.fit(events, (0.0, T), fix_alpha_zero=True)
    # μ recovered within 20% (Poisson statistical error at these sample sizes).
    assert abs(model.params.mu[0] - 0.5) / 0.5 < 0.2
    assert abs(model.params.mu[1] - 1.0) / 1.0 < 0.2
    assert "nll_final" in result
    assert result["status"] in {"success", "converged"}


def test_fit_stable_on_short_window():
    """Smoke test -- fit completes without error on a small mixed dataset."""
    K = 2  # noqa: N806
    rng = np.random.default_rng(1)
    T = 50.0  # noqa: N806
    bbox = (-5.0, -5.0, 5.0, 5.0)
    n = 30
    events = {
        "time": np.sort(rng.uniform(0, T, n)),
        "lon": rng.uniform(-5, 5, n),
        "lat": rng.uniform(-5, 5, n),
        "mark": rng.integers(0, K, n).astype(np.int64),
    }
    model = ParametricHawkes(K=K, bbox=bbox, pi_k=_trivial_pi)
    result = model.fit(events, (0.0, T))
    assert "nll_final" in result
