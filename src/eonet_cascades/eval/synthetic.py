"""Synthetic Hawkes data generator.

Uses Ogata's branching-process method: simulate immigrants from a Poisson(mu_k)
baseline over [0, t_end] x bbox, then recursively spawn offspring per the
(alpha, beta, sigma) triggering kernels.
"""

from __future__ import annotations

import numpy as np

from eonet_cascades.models.hawkes import HawkesParams


def simulate_hawkes(
    params: HawkesParams,
    bbox: tuple[float, float, float, float],
    t_end: float,
    t0: float = 0.0,
    rng: np.random.Generator | None = None,
    max_events: int = 200_000,
) -> dict[str, np.ndarray]:
    """Forward-simulate a multivariate marked Hawkes process on [t0, t_end] x bbox.

    Spatial baseline is uniform over the bbox. (Plan 3 adds nonuniform baselines.)

    Returns
    -------
    dict with keys "time", "lon", "lat", "mark" -- each a 1-D np.ndarray, sorted by time.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_marks = params.K
    min_lon, min_lat, max_lon, max_lat = bbox

    times: list[float] = []
    lons: list[float] = []
    lats: list[float] = []
    marks: list[int] = []

    # Generation 0 -- immigrants from Poisson(mu_k * t_end) uniformly over the bbox.
    for k in range(n_marks):
        n_imm = rng.poisson(params.mu[k] * (t_end - t0))
        for _ in range(n_imm):
            times.append(rng.uniform(t0, t_end))
            lons.append(rng.uniform(min_lon, max_lon))
            lats.append(rng.uniform(min_lat, max_lat))
            marks.append(k)

    # Generations 1..inf -- BFS. Each event j of mark kj spawns Poisson(alpha[kj, k])
    # offspring of mark k, temporal offsets ~ Exp(beta[kj, k]), spatial offsets
    # isotropic Gaussian sigma[kj, k].
    pending = list(zip(times, lons, lats, marks, strict=True))
    while pending:
        next_pending = []
        for tj, xj, yj, kj in pending:
            for k in range(n_marks):
                a = params.alpha[kj, k]
                if a <= 0:
                    continue
                n_off = rng.poisson(a)
                for _ in range(n_off):
                    dt = rng.exponential(1.0 / params.beta[kj, k])
                    tc = tj + dt
                    if tc >= t_end:
                        continue
                    dx = rng.normal(0.0, params.sigma[kj, k])
                    dy = rng.normal(0.0, params.sigma[kj, k])
                    xc = xj + dx
                    yc = yj + dy
                    if not (min_lon <= xc <= max_lon and min_lat <= yc <= max_lat):
                        continue
                    times.append(tc)
                    lons.append(xc)
                    lats.append(yc)
                    marks.append(k)
                    next_pending.append((tc, xc, yc, k))
                    if len(times) > max_events:
                        raise RuntimeError(
                            f"simulate_hawkes exceeded max_events={max_events} -- "
                            "likely unstable (alpha spectral radius > 1)"
                        )
        pending = next_pending

    order = np.argsort(times)
    return {
        "time": np.asarray(times)[order],
        "lon": np.asarray(lons)[order],
        "lat": np.asarray(lats)[order],
        "mark": np.asarray(marks, dtype=np.int64)[order],
    }
