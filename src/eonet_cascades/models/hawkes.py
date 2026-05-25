"""Tier 0 — Parametric Multivariate Hawkes Process.

Implements the intensity, log-likelihood, sampling, and MLE-based fitting for
the spatio-temporal marked Hawkes model defined in
docs/superpowers/specs/2026-05-24-eonet-cascade-benchmark-design.md §4.2.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize


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


def hawkes_log_likelihood(
    params: HawkesParams,
    events: dict[str, np.ndarray],
    window: tuple[float, float],
    pi_k: SpatialDensityFn,
    bbox: tuple[float, float, float, float],
    spatial_mass_approx_one: bool = True,
) -> float:
    """Compute the log-likelihood of `events` under `params`.

    log L = sum_i log lambda_{k_i}(t_i, x_i | H_{t_i})
          - sum_k mu_k (t_end - t0)
          - sum_j sum_k alpha[k_j, k] * (1 - exp(-beta[k_j, k] (t_end - t_j))) * G_x(bbox | x_j, sigma)

    `spatial_mass_approx_one=True` substitutes G_x approx 1 (Assumption A1) — see plan header.
    """
    t0, t_end = window
    t_arr = events["time"]
    lon_arr = events["lon"]
    lat_arr = events["lat"]
    k_arr = events["mark"].astype(np.int64)
    n = t_arr.shape[0]

    # Sum log-intensity at each event (using only strictly earlier events as history).
    sum_log = 0.0
    # Pre-sort if not already.
    order = np.argsort(t_arr, kind="stable")
    t_s = t_arr[order]
    lon_s = lon_arr[order]
    lat_s = lat_arr[order]
    k_s = k_arr[order]

    for i in range(n):
        t_i = t_s[i]
        x_i = np.array([[lon_s[i], lat_s[i]]])
        hist = {
            "time": t_s[:i],
            "lon": lon_s[:i],
            "lat": lat_s[:i],
            "mark": k_s[:i],
        }
        lam_vec = conditional_intensity(params, t_i, x_i, hist, pi_k, bbox)
        lam_i = lam_vec[k_s[i]]
        if lam_i <= 0:
            return -np.inf
        sum_log += math.log(lam_i)

    # Integrated intensity.
    # Baseline part: sum_k mu_k * (t_end - t0) — pi_k integrates to 1.
    integral_baseline = float(np.sum(params.mu) * (t_end - t0))

    # Triggering part: for each event j, contribution to total integrated intensity is
    # sum_k alpha[k_j, k] * (1 - exp(-beta[k_j, k] * (t_end - t_j))) * G_x.
    if n == 0:
        integral_trigger = 0.0
    else:
        decay = np.exp(-params.beta[k_s, :] * (t_end - t_s)[:, None])  # (n, n_marks)
        per_event = params.alpha[k_s, :] * (1.0 - decay)  # (n, n_marks)
        if not spatial_mass_approx_one:
            raise NotImplementedError("Exact spatial mass not implemented in v1")
        integral_trigger = float(np.sum(per_event))

    return sum_log - integral_baseline - integral_trigger


@dataclass
class ParametricHawkes:
    """Tier 0 — multivariate marked Hawkes model with exponential temporal kernel and
    isotropic Gaussian spatial kernel. Conforms to `PointProcessModel` protocol."""

    K: int  # K is standard notation for number of marks
    bbox: tuple[float, float, float, float]
    pi_k: SpatialDensityFn
    params: HawkesParams = field(init=False)
    name: str = field(default="hawkes_tier0")

    def __post_init__(self) -> None:
        # Sensible initial values.
        self.params = HawkesParams(
            mu=np.full(self.K, 0.1),
            alpha=np.full((self.K, self.K), 0.05),
            beta=np.full((self.K, self.K), 1.0),
            sigma=np.full((self.K, self.K), 1.0),
        )

    def log_likelihood(
        self,
        events: dict[str, np.ndarray] | pl.DataFrame,
        window: tuple[float, float],
    ) -> float:
        if isinstance(events, pl.DataFrame):
            events = _df_to_event_dict(events)
        return hawkes_log_likelihood(self.params, events, window, self.pi_k, self.bbox)

    def sample(self, history, window):  # pragma: no cover — placeholder
        raise NotImplementedError("Sampling lands in a later task")

    def fit(
        self,
        events: dict[str, np.ndarray] | pl.DataFrame,
        window: tuple[float, float],
        *,
        fix_alpha_zero: bool = False,
        max_iter: int = 200,
    ) -> dict[str, Any]:
        """MLE fit of (mu, alpha, beta, sigma) via L-BFGS-B with positive bounds.

        `fix_alpha_zero=True` clamps alpha=0 (homogeneous Poisson baseline-only fit) --
        useful for validating the mu recovery path independently of the triggering kernels.
        """
        if isinstance(events, pl.DataFrame):
            events = _df_to_event_dict(events)

        K = self.K  # noqa: N806
        n_mu = K
        n_pair = K * K
        # Flat parameter vector: [mu (K), alpha (K^2), beta (K^2), sigma (K^2)]

        def unpack(theta: np.ndarray) -> HawkesParams:
            mu = theta[:n_mu]
            alpha = theta[n_mu : n_mu + n_pair].reshape(K, K)
            beta = theta[n_mu + n_pair : n_mu + 2 * n_pair].reshape(K, K)
            sigma = theta[n_mu + 2 * n_pair :].reshape(K, K)
            if fix_alpha_zero:
                alpha = np.zeros_like(alpha)
            return HawkesParams(mu=mu, alpha=alpha, beta=beta, sigma=sigma)

        def nll(theta: np.ndarray) -> float:
            params = unpack(theta)
            ll = hawkes_log_likelihood(params, events, window, self.pi_k, self.bbox)
            if not np.isfinite(ll):
                return 1e20
            return -ll

        theta0 = np.concatenate(
            [self.params.mu, self.params.alpha.ravel(), self.params.beta.ravel(), self.params.sigma.ravel()]
        )
        # Bounds: keep alpha bounded *strictly* above 0; scipy's finite-difference
        # gradient steps will otherwise occasionally land just below 0 and raise.
        lower = np.concatenate(
            [np.full(n_mu, 1e-6), np.full(n_pair, 1e-6), np.full(n_pair, 1e-3), np.full(n_pair, 1e-3)]
        )
        upper = np.concatenate(
            [np.full(n_mu, 100.0), np.full(n_pair, 0.95), np.full(n_pair, 100.0), np.full(n_pair, 100.0)]
        )
        # Clamp theta0 inside bounds to avoid scipy's strict bound check rejecting it.
        theta0 = np.clip(theta0, lower + 1e-9, upper - 1e-9)
        bounds = list(zip(lower.tolist(), upper.tolist(), strict=True))

        res = minimize(
            nll,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-9},
        )
        self.params = unpack(res.x)

        return {
            "nll_final": float(res.fun),
            "n_iter": int(res.nit),
            "status": "success" if res.success else "failed",
            "message": res.message if isinstance(res.message, str) else res.message.decode("utf-8", "ignore"),
            "spectral_radius": self.params.spectral_radius(),
        }


def _df_to_event_dict(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Convert a polars events DataFrame (with mark as string) to the numpy dict form.

    Mark strings are mapped to integer indices in alphabetical order.
    """
    times = df["time_start"].to_numpy().astype("datetime64[us]")
    t0_ref = times.min()
    t_days = (times - t0_ref).astype("timedelta64[us]").astype(np.float64) / (86_400 * 1e6)
    marks_sorted = sorted(df["mark"].unique().to_list())
    mark_to_idx = {m: i for i, m in enumerate(marks_sorted)}
    mark_idx = np.array([mark_to_idx[m] for m in df["mark"].to_list()], dtype=np.int64)
    return {
        "time": t_days,
        "lon": df["longitude"].to_numpy().astype(np.float64),
        "lat": df["latitude"].to_numpy().astype(np.float64),
        "mark": mark_idx,
    }


from scipy.ndimage import gaussian_filter  # noqa: E402


@dataclass
class KDESpatialBaseline:
    """Per-mark spatial baseline density estimated from an empirical event distribution.

    Stores a (K, n_lat, n_lon) grid of normalized densities. Calling the instance
    with (mark_index, points (N, 2), bbox) returns density values at those points
    via nearest-grid lookup.
    """

    densities: np.ndarray   # shape (n_marks, n_lat, n_lon)
    bbox: tuple[float, float, float, float]
    grid_step: float
    mark_names: list[str]

    @classmethod
    def from_events(
        cls,
        events_df,
        mark_names: list[str],
        bbox: tuple[float, float, float, float],
        grid_step: float = 1.0,
        smooth_sigma: float = 1.5,
    ) -> KDESpatialBaseline:
        min_lon, min_lat, max_lon, max_lat = bbox
        n_lon = round((max_lon - min_lon) / grid_step)
        n_lat = round((max_lat - min_lat) / grid_step)
        n_marks = len(mark_names)
        densities = np.zeros((n_marks, n_lat, n_lon), dtype=np.float64)
        # Accept either polars or dict input.
        if isinstance(events_df, pl.DataFrame):
            lon = events_df["longitude"].to_numpy().astype(np.float64)
            lat = events_df["latitude"].to_numpy().astype(np.float64)
            marks = events_df["mark"].to_list()
        else:
            lon = np.asarray(events_df["longitude"], dtype=np.float64)
            lat = np.asarray(events_df["latitude"], dtype=np.float64)
            marks = list(events_df["mark"])
        for i, name in enumerate(mark_names):
            mask = np.array([m == name for m in marks])
            if not mask.any():
                # Uniform fallback so density is non-zero everywhere.
                densities[i] = 1.0
            else:
                lons_k = lon[mask]
                lats_k = lat[mask]
                hist, _, _ = np.histogram2d(
                    lats_k, lons_k,
                    bins=[n_lat, n_lon],
                    range=[[min_lat, max_lat], [min_lon, max_lon]],
                )
                densities[i] = gaussian_filter(hist, sigma=smooth_sigma) + 1e-6  # floor for log
            # Normalize so cell-area integral = 1.
            cell_area = grid_step * grid_step
            densities[i] /= densities[i].sum() * cell_area
        return cls(densities=densities, bbox=bbox, grid_step=grid_step, mark_names=mark_names)

    def __call__(
        self,
        k: int,
        x: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        min_lon, _min_lat, _max_lon, _max_lat = self.bbox
        min_lat = self.bbox[1]
        n_lat, n_lon = self.densities.shape[1], self.densities.shape[2]
        lon_idx = np.clip(((x[:, 0] - min_lon) / self.grid_step).astype(int), 0, n_lon - 1)
        lat_idx = np.clip(((x[:, 1] - min_lat) / self.grid_step).astype(int), 0, n_lat - 1)
        return self.densities[k, lat_idx, lon_idx]
