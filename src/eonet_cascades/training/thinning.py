"""Ogata thinning algorithm for point process simulation.

Reference: Ogata (1981), "On Lewis' Simulation Method for Point Processes",
IEEE Trans. Information Theory.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def thinning_sample_temporal(
    intensity_fn: Callable[[float, list[float]], float],
    upper_bound_fn: Callable[[float, list[float]], float],
    t_end: float,
    t0: float = 0.0,
    rng: np.random.Generator | None = None,
    max_events: int = 1_000_000,
) -> list[float]:
    """Sample event times from a temporal point process on [t0, t_end] via thinning.

    Parameters
    ----------
    intensity_fn : (t, history) -> λ(t | history)
        The true conditional intensity. Must be <= upper_bound_fn at all points.
    upper_bound_fn : (t, history) -> λ_bar
        A computable upper bound on λ over the next inter-arrival.
    t_end : float
        Upper time bound (exclusive).
    t0 : float
        Lower time bound (inclusive). Default 0.
    rng : np.random.Generator
        Numpy RNG. If None, uses np.random.default_rng() with a fresh seed.
    max_events : int
        Safety cap to prevent runaway sampling on unstable processes.

    Returns
    -------
    list[float]
        Sorted event times in [t0, t_end).
    """
    if rng is None:
        rng = np.random.default_rng()

    events: list[float] = []
    t = t0
    while t < t_end:
        lam_bar = upper_bound_fn(t, events)
        if lam_bar <= 0:
            # No more events possible.
            break
        tau = rng.exponential(scale=1.0 / lam_bar)
        t = t + tau
        if t >= t_end:
            break
        lam = intensity_fn(t, events)
        if lam > lam_bar + 1e-12:
            raise ValueError(
                f"upper bound {lam_bar} smaller than true intensity {lam} at t={t}"
            )
        u = rng.uniform()
        if u * lam_bar <= lam:
            events.append(t)
            if len(events) >= max_events:
                # Safety cap: return partial list rather than crashing.
                # Caller should detect len(events) == max_events and warn.
                break
    return events
