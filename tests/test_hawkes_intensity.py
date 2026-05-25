"""Hawkes parameter container + intensity arithmetic tests."""

from __future__ import annotations

import math

import numpy as np

from eonet_cascades.models.hawkes import HawkesParams, conditional_intensity


def _trivial_pi(k: int, x: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    # Uniform density over bbox: 1 / (lon_range * lat_range)
    min_lon, min_lat, max_lon, max_lat = bbox
    area = (max_lon - min_lon) * (max_lat - min_lat)
    return np.full(x.shape[0], 1.0 / area)


def test_params_default_shapes():
    K = 3  # noqa: N806
    p = HawkesParams.zeros(K)
    assert p.mu.shape == (K,)
    assert p.alpha.shape == (K, K)
    assert p.beta.shape == (K, K)
    assert p.sigma.shape == (K, K)


def test_intensity_with_no_history_equals_baseline_only():
    K = 2  # noqa: N806
    p = HawkesParams(
        mu=np.array([0.5, 0.3]),
        alpha=np.zeros((K, K)),
        beta=np.ones((K, K)),
        sigma=np.ones((K, K)),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    area = 400.0
    # No history at all.
    history = {
        "time": np.array([], dtype=np.float64),
        "lon": np.array([], dtype=np.float64),
        "lat": np.array([], dtype=np.float64),
        "mark": np.array([], dtype=np.int64),
    }
    t = 1.0
    x = np.array([[0.0, 0.0]])
    lam = conditional_intensity(p, t, x, history, _trivial_pi, bbox)
    # baseline only: mu_k * pi_k(x) for each k -> mu_k / area
    assert lam.shape == (K,)
    assert math.isclose(lam[0], 0.5 / area, rel_tol=1e-9)
    assert math.isclose(lam[1], 0.3 / area, rel_tol=1e-9)


def test_intensity_with_single_past_event_self_excites():
    K = 2  # noqa: N806
    p = HawkesParams(
        mu=np.array([0.0, 0.0]),  # baseline off so we measure trigger only
        alpha=np.array([[0.5, 0.1], [0.0, 0.0]]),  # mark-0 triggers self and a bit of mark-1
        beta=np.full((K, K), 1.0),
        sigma=np.full((K, K), 1.0),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    history = {
        "time": np.array([0.0]),
        "lon": np.array([0.0]),
        "lat": np.array([0.0]),
        "mark": np.array([0], dtype=np.int64),
    }
    # Evaluate at the same location, time = 0.5 (so exp(-beta*0.5) = exp(-0.5))
    t = 0.5
    x = np.array([[0.0, 0.0]])
    lam = conditional_intensity(p, t, x, history, _trivial_pi, bbox)
    # lam_0 = alpha_{0->0} * beta_{0->0} * exp(-beta*dt) * g_x(0; sigma=1)
    expected_temporal = 0.5 * 1.0 * math.exp(-0.5)
    expected_spatial = 1.0 / (2 * math.pi * 1.0 * 1.0)  # Gaussian at center, sigma=1
    expected_lam0 = expected_temporal * expected_spatial
    assert math.isclose(lam[0], expected_lam0, rel_tol=1e-6)
    # lam_1 = alpha_{0->1} * beta_{0->1} * exp(-beta*dt) * g_x
    expected_lam1 = 0.1 * 1.0 * math.exp(-0.5) * expected_spatial
    assert math.isclose(lam[1], expected_lam1, rel_tol=1e-6)


def test_intensity_at_future_event_only_uses_past():
    K = 1  # noqa: N806, F841
    p = HawkesParams(
        mu=np.array([0.0]),
        alpha=np.array([[1.0]]),
        beta=np.array([[1.0]]),
        sigma=np.array([[1.0]]),
    )
    bbox = (-10.0, -10.0, 10.0, 10.0)
    history = {
        "time": np.array([1.0, 5.0]),
        "lon": np.array([0.0, 0.0]),
        "lat": np.array([0.0, 0.0]),
        "mark": np.array([0, 0], dtype=np.int64),
    }
    # Evaluate at t=3 -- only the event at t=1 should contribute.
    t = 3.0
    x = np.array([[0.0, 0.0]])
    lam = conditional_intensity(p, t, x, history, _trivial_pi, bbox)
    expected = 1.0 * 1.0 * math.exp(-1.0 * 2.0) / (2 * math.pi * 1.0)
    assert math.isclose(lam[0], expected, rel_tol=1e-6)


def test_kde_baseline_integrates_to_one_per_mark():
    from eonet_cascades.models.hawkes import KDESpatialBaseline

    bbox = (-10.0, -10.0, 10.0, 10.0)
    rng = np.random.default_rng(0)
    # Mark 0 events cluster around (-5, -5); mark 1 events around (5, 5).
    n = 500
    events_df = {
        "time_start": np.array([np.datetime64("2024-01-01")] * (2 * n)),
        "longitude": np.concatenate([rng.normal(-5, 1, n), rng.normal(5, 1, n)]),
        "latitude": np.concatenate([rng.normal(-5, 1, n), rng.normal(5, 1, n)]),
        "mark": np.array(["a"] * n + ["b"] * n),
    }
    import polars as pl

    df = pl.DataFrame(events_df)
    baseline = KDESpatialBaseline.from_events(df, mark_names=["a", "b"], bbox=bbox, grid_step=1.0)
    # Integral check: sum over a fine grid times cell area should be ~1.
    fine_lon = np.linspace(-10, 10, 41)
    fine_lat = np.linspace(-10, 10, 41)
    LL, AA = np.meshgrid(fine_lon, fine_lat)  # noqa: N806
    pts = np.column_stack([LL.ravel(), AA.ravel()])
    for k in (0, 1):
        vals = baseline(k, pts, bbox)
        # cell width 0.5 deg, so cell area = 0.25 deg^2
        integral = float(vals.sum() * 0.25)
        assert 0.7 < integral < 1.3, f"mark {k} integral {integral} not near 1"
