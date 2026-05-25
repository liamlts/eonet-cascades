"""Monte Carlo temporal integral tests."""

from __future__ import annotations

import torch

from eonet_cascades.training.monte_carlo import mc_integrate_lambda_t


def test_constant_lambda_integral_close_to_lambda_times_length():
    """If lambda_t is constant rate r over [0, T], integral should be r*T."""

    def constant_lambda(t):
        return torch.full_like(t, 3.0)

    val = mc_integrate_lambda_t(
        lambda_fn=constant_lambda, t_start=0.0, t_end=4.0, n_samples=200, seed=0
    )
    # Expected ~12; tolerate sample noise.
    assert abs(val - 12.0) < 0.5, f"got {val}"


def test_linear_lambda_integral_close_to_analytic():
    """lambda_t(t) = 2 + t over [0, 5]. Analytic integral = 10 + 12.5 = 22.5."""

    def linear_lambda(t):
        return 2.0 + t

    val = mc_integrate_lambda_t(
        lambda_fn=linear_lambda, t_start=0.0, t_end=5.0, n_samples=500, seed=1
    )
    assert abs(val - 22.5) < 1.0, f"got {val}"


def test_zero_window_returns_zero():
    val = mc_integrate_lambda_t(
        lambda_fn=lambda t: torch.full_like(t, 10.0), t_start=2.0, t_end=2.0, n_samples=10, seed=0
    )
    assert val == 0.0
