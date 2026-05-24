"""Tier 0 — Parametric Multivariate Hawkes Process.

Implements the intensity, log-likelihood, sampling, and MLE-based fitting for
the spatio-temporal marked Hawkes model defined in
docs/superpowers/specs/2026-05-24-eonet-cascade-benchmark-design.md §4.2.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class HawkesParams:
    """Parametric multivariate Hawkes parameters.

    Shapes:
      mu:    (K,)       -- per-mark immigration rate (events / day, integrated over bbox)
      alpha: (K, K)     -- alpha[i, j] = branching ratio from mark i parent to mark j child
      beta:  (K, K)     -- beta[i, j] = exponential decay rate of i->j trigger (1/day)
      sigma: (K, K)     -- sigma[i, j] = isotropic Gaussian bandwidth of i->j trigger (degrees)
    """

    mu: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    sigma: np.ndarray

    @classmethod
    def zeros(cls, n_marks: int) -> HawkesParams:
        return cls(
            mu=np.zeros(n_marks),
            alpha=np.zeros((n_marks, n_marks)),
            beta=np.ones((n_marks, n_marks)),
            sigma=np.ones((n_marks, n_marks)),
        )

    @property
    def n_marks(self) -> int:
        return self.mu.shape[0]

    # K is conventional physics notation; expose as alias.
    @property
    def K(self) -> int:  # noqa: N802
        return self.n_marks

    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.alpha))))


# A spatial-density callable: (mark_index, points (N, 2), bbox) -> density values (N,)
SpatialDensityFn = Callable[[int, np.ndarray, tuple[float, float, float, float]], np.ndarray]


def conditional_intensity(
    params: HawkesParams,
    t: float,
    x: np.ndarray,
    history: dict[str, np.ndarray],
    pi_k: SpatialDensityFn,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Compute lambda_k(t, x | H_t) for all marks k.

    Parameters
    ----------
    params : HawkesParams
    t : float
        Evaluation time. Only events with history["time"] < t contribute.
    x : np.ndarray of shape (1, 2)
        Single evaluation point in (lon, lat).
    history : dict with keys "time", "lon", "lat", "mark" (each 1-D np arrays of equal length)
    pi_k : SpatialDensityFn
        Empirical spatial density per mark; values at the eval point.
    bbox : (min_lon, min_lat, max_lon, max_lat)

    Returns
    -------
    np.ndarray of shape (K,)
        Intensity per mark at (t, x).
    """
    n_marks = params.n_marks
    # Baseline
    pi_vals = np.array([pi_k(k, x, bbox)[0] for k in range(n_marks)])
    lam = params.mu * pi_vals  # shape (n_marks,)

    # Triggering -- only past events contribute.
    t_hist = history["time"]
    past_mask = t_hist < t
    if not np.any(past_mask):
        return lam

    t_past = t_hist[past_mask]
    lon_past = history["lon"][past_mask]
    lat_past = history["lat"][past_mask]
    k_past = history["mark"][past_mask].astype(np.int64)

    dt = t - t_past  # (M,)
    # Spatial distance squared from each past event to x.
    dlon = x[0, 0] - lon_past
    dlat = x[0, 1] - lat_past
    d2 = dlon * dlon + dlat * dlat  # (M,)

    # For each past event j with mark k_j, and each child mark k,
    # contribution = alpha[k_j, k] * beta[k_j, k] * exp(-beta*dt) * Gauss2D(d2; sigma[k_j, k])
    for k in range(n_marks):
        a_col = params.alpha[k_past, k]       # (M,)
        b_col = params.beta[k_past, k]        # (M,)
        s_col = params.sigma[k_past, k]       # (M,)
        temporal = a_col * b_col * np.exp(-b_col * dt)
        spatial = np.exp(-d2 / (2.0 * s_col * s_col)) / (2.0 * math.pi * s_col * s_col)
        lam[k] += float(np.sum(temporal * spatial))
    return lam
