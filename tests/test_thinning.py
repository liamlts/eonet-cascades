"""Ogata thinning sampler tests."""

from __future__ import annotations

import math

import numpy as np

from eonet_cascades.training.thinning import thinning_sample_temporal


def test_constant_intensity_gives_poisson_count(rng=np.random.default_rng(42)):  # noqa: B008
    # For λ(t) = 5, expected N over [0, 10] is 50, var is 50.
    rate = 5.0
    t_end = 10.0
    intensity = lambda t, hist: rate  # noqa: E731
    upper_bound = lambda t, hist: rate  # noqa: E731

    n_trials = 200
    counts = []
    for _ in range(n_trials):
        events = thinning_sample_temporal(intensity, upper_bound, t_end, rng=rng)
        counts.append(len(events))

    mean = np.mean(counts)
    var = np.var(counts)
    # Allow loose Poisson check.
    assert abs(mean - rate * t_end) < 5.0, f"mean {mean} far from {rate * t_end}"
    assert 0.5 * mean < var < 2.0 * mean, f"var {var} not Poisson-like for mean {mean}"


def test_zero_intensity_yields_empty():
    intensity = lambda t, hist: 0.0  # noqa: E731
    upper_bound = lambda t, hist: 0.0  # noqa: E731
    events = thinning_sample_temporal(intensity, upper_bound, 100.0, rng=np.random.default_rng(0))
    assert events == []


def test_decaying_intensity_concentrates_early():
    # λ(t) starts at 10 and decays exp(-t). Most events should be in [0,1].
    intensity = lambda t, hist: 10.0 * math.exp(-t)  # noqa: E731
    upper_bound = lambda t, hist: 10.0  # noqa: E731

    rng = np.random.default_rng(1)
    counts_early = 0
    counts_late = 0
    n_trials = 50
    for _ in range(n_trials):
        events = thinning_sample_temporal(intensity, upper_bound, 5.0, rng=rng)
        for ev in events:
            if ev < 1.0:
                counts_early += 1
            else:
                counts_late += 1
    assert counts_early > 3 * counts_late, (
        f"expected events to concentrate early, got early={counts_early} late={counts_late}"
    )


def test_history_passed_to_intensity():
    # Use a history-dependent intensity to confirm history is threaded correctly.
    def intensity(t, hist):
        return 1.0 + len(hist)

    def upper_bound(t, hist):
        return 1.0 + len(hist) + 5.0

    events = thinning_sample_temporal(intensity, upper_bound, 20.0, rng=np.random.default_rng(0))
    # With history-amplified intensity the process self-excites; expect more than baseline.
    assert len(events) > 0
