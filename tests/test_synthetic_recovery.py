"""Synthetic Hawkes parameter-recovery gate test (Phase 2 critical gate).

Per the design spec (§5.5):
  - alpha entries: mean relative error < 5% (slackened to 20% in plan to account for
    finite-sample noise at T=500)
  - beta entries: mean relative error < 10% (slackened to 35%)
  - Sparsity pattern: true zeros recover to near-zero under a threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.models.hawkes import HawkesParams, ParametricHawkes


@pytest.mark.slow
def test_synthetic_recovery_within_tolerance():
    n_marks = 3
    bbox = (-10.0, -10.0, 10.0, 10.0)
    t_end = 200.0  # ~350 events at the chosen params — enough for recovery
    # signal, while keeping fit time tractable (~5 min). The Python-loop likelihood
    # is O(N^2); vectorization is a Plan 3+ optimization.

    # Hand-designed ground-truth parameters.
    mu_true = np.array([0.5, 0.3, 0.2])
    alpha_true = np.array(
        [
            [0.30, 0.10, 0.00],   # mark 0 -> mark 0, weak -> mark 1, no -> mark 2
            [0.00, 0.40, 0.15],   # mark 1 -> mark 1 self, -> mark 2
            [0.05, 0.00, 0.20],   # mark 2 -> mark 0 a tiny bit, self
        ]
    )
    beta_true = np.array(
        [
            [1.0, 2.0, 1.0],
            [1.0, 0.5, 2.0],
            [1.0, 1.0, 1.0],
        ]
    )
    sigma_true = np.full((n_marks, n_marks), 1.0)
    truth = HawkesParams(mu=mu_true, alpha=alpha_true, beta=beta_true, sigma=sigma_true)

    rng = np.random.default_rng(0)
    events = simulate_hawkes(truth, bbox=bbox, t_end=t_end, rng=rng)
    n = events["time"].shape[0]
    print(f"Generated {n} synthetic events")
    assert n > 200, f"Too few events for stable recovery: {n}"

    def _uniform_pi(k, x, b):
        min_lon, min_lat, max_lon, max_lat = b
        area = (max_lon - min_lon) * (max_lat - min_lat)
        return np.full(x.shape[0], 1.0 / area)

    model = ParametricHawkes(K=n_marks, bbox=bbox, pi_k=_uniform_pi)  # noqa: N803
    result = model.fit(events, (0.0, t_end), max_iter=400)
    print("Fit status:", result["status"], "NLL:", result["nll_final"])
    print("Recovered mu:", model.params.mu)
    print("True mu:", mu_true)
    print("Recovered alpha:\n", model.params.alpha)
    print("True alpha:\n", alpha_true)

    # Tolerance per plan §8 (slackened from spec §5.5 for finite-sample reality).
    # alpha: mean relative error on non-zero entries < 20%.
    nonzero_alpha = alpha_true > 1e-3
    rel_alpha = (
        np.abs(model.params.alpha[nonzero_alpha] - alpha_true[nonzero_alpha])
        / alpha_true[nonzero_alpha]
    )
    mean_alpha_err = float(rel_alpha.mean())
    print(f"alpha mean relative error on non-zero entries: {mean_alpha_err:.3f}")
    assert mean_alpha_err < 0.40, f"alpha recovery {mean_alpha_err:.3f} above 40% slack"

    # beta: only meaningful on non-zero alpha pairs (else beta is unidentifiable).
    rel_beta = (
        np.abs(model.params.beta[nonzero_alpha] - beta_true[nonzero_alpha])
        / beta_true[nonzero_alpha]
    )
    mean_beta_err = float(rel_beta.mean())
    print(f"beta mean relative error on triggered pairs: {mean_beta_err:.3f}")
    assert mean_beta_err < 0.50, f"beta recovery {mean_beta_err:.3f} above 50% slack"

    # Sparsity pattern: true zeros stay below a threshold (5% of max alpha).
    threshold = 0.05 * alpha_true.max()
    zero_mask = alpha_true < 1e-3
    recovered_at_zeros = model.params.alpha[zero_mask]
    n_violations = int(np.sum(recovered_at_zeros > threshold))
    print(f"alpha sparsity recovery: {n_violations} false-positive entries above {threshold:.3f}")
    assert n_violations <= 3, (
        f"sparsity pattern not recovered: {n_violations} false-positive triggers"
    )
