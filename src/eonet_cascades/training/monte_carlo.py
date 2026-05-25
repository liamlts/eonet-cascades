"""Monte Carlo helpers for point-process likelihood integrals."""

from __future__ import annotations

from collections.abc import Callable

import torch


def mc_integrate_lambda_t(
    lambda_fn: Callable[[torch.Tensor], torch.Tensor],
    t_start: float,
    t_end: float,
    n_samples: int = 20,
    seed: int | None = None,
) -> float:
    """Estimate the integral of a 1-D temporal intensity over [t_start, t_end].

    Uses uniform Monte Carlo: average of n_samples evaluations of lambda_fn at
    uniform sample points, multiplied by the interval length.
    """
    if t_end <= t_start:
        return 0.0
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    samples = torch.rand(n_samples, generator=gen) * (t_end - t_start) + t_start
    vals = lambda_fn(samples)
    return float(vals.mean() * (t_end - t_start))
