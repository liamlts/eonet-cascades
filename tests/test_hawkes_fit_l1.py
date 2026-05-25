"""L1 regularization on alpha — sparsity recovery test."""

from __future__ import annotations

import numpy as np
import pytest

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.models.hawkes import HawkesParams, ParametricHawkes


def _uniform_pi(k, x, b):
    min_lon, min_lat, max_lon, max_lat = b
    return np.full(x.shape[0], 1.0 / ((max_lon - min_lon) * (max_lat - min_lat)))


@pytest.mark.slow
def test_l1_produces_sparser_alpha_than_unregularized():
    """Sparse ground-truth alpha (5 of 9 entries are zero).
    L1 fit should recover at least as many zeros as the unregularized fit,
    and at least 3 of 5 true zeros.
    """
    n_marks = 3
    bbox = (-10.0, -10.0, 10.0, 10.0)
    alpha_true = np.array(
        [
            [0.30, 0.00, 0.00],
            [0.00, 0.40, 0.15],
            [0.00, 0.00, 0.20],
        ]
    )
    truth = HawkesParams(
        mu=np.array([0.4, 0.3, 0.2]),
        alpha=alpha_true,
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    rng = np.random.default_rng(0)
    events = simulate_hawkes(truth, bbox=bbox, t_end=200.0, rng=rng)
    print(f"Generated {events['time'].shape[0]} events")

    threshold = 0.02  # an alpha entry below this counts as a "recovered zero"
    zero_mask = alpha_true < 1e-3
    n_true_zeros = int(zero_mask.sum())

    # Unregularized fit.
    m0 = ParametricHawkes(K=n_marks, bbox=bbox, pi_k=_uniform_pi)
    m0.fit(events, (0.0, 200.0), max_iter=200)
    n_recovered_zeros_0 = int(np.sum(m0.params.alpha[zero_mask] < threshold))
    print(f"Unregularized:  recovered {n_recovered_zeros_0}/{n_true_zeros} zeros")
    print(f"  alpha:\n{m0.params.alpha}")

    # L1 fit.
    m1 = ParametricHawkes(K=n_marks, bbox=bbox, pi_k=_uniform_pi)
    m1.fit(events, (0.0, 200.0), max_iter=200, l1_lambda=0.5)
    n_recovered_zeros_1 = int(np.sum(m1.params.alpha[zero_mask] < threshold))
    print(f"L1 (lambda=0.5): recovered {n_recovered_zeros_1}/{n_true_zeros} zeros")
    print(f"  alpha:\n{m1.params.alpha}")

    assert n_recovered_zeros_1 >= n_recovered_zeros_0, (
        f"L1 recovered {n_recovered_zeros_1} zeros vs unregularized {n_recovered_zeros_0}"
    )
    assert n_recovered_zeros_1 >= 3, (
        f"L1 should recover at least 3/{n_true_zeros} true zeros, got {n_recovered_zeros_1}"
    )
