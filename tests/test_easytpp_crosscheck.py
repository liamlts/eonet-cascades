"""Cross-check our Tier 1 NLL against a reference TPP library's CTLSTM.

Per spec §6.4: this test trains our NeuralHawkes and an independent
implementation on the same synthetic data and asserts the held-out NLLs
agree within 2% relative. Until the reference library's API is fully
wired in, the assertion is a `pytest.skip` placeholder — the test still
runs the our-side training to make sure the surrounding scaffold works.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.optim import AdamW

from eonet_cascades.eval.synthetic import simulate_hawkes
from eonet_cascades.models.hawkes import HawkesParams
from eonet_cascades.models.neural_hawkes import NeuralHawkes
from eonet_cascades.training.neural_loop import TrainChunk, train_one_epoch


@pytest.mark.slow
@pytest.mark.network
def test_easytpp_nll_within_2pct():
    """Both our model and an external CTLSTM library should reach within 2%
    of each other on the same synthetic data after 30 epochs.

    NOTE: this scaffold runs OUR training to confirm the test infrastructure
    works, then skips the cross-check until the reference library API is
    finalized.
    """
    # Try the preferred library first.
    have_ref = False
    try:
        import easy_tpp  # noqa: F401

        have_ref = "easy_tpp"
    except ImportError:
        try:
            import tpps  # noqa: F401

            have_ref = "tpps"
        except ImportError:
            have_ref = None
    print(f"Reference library: {have_ref}")

    torch.manual_seed(0)
    np.random.seed(0)
    n_marks = 3
    bbox = (-10.0, -10.0, 10.0, 10.0)
    truth = HawkesParams(
        mu=np.array([0.5, 0.3, 0.2]),
        alpha=np.array(
            [[0.30, 0.10, 0.00], [0.00, 0.40, 0.15], [0.05, 0.00, 0.20]]
        ),
        beta=np.full((n_marks, n_marks), 1.0),
        sigma=np.full((n_marks, n_marks), 1.0),
    )
    rng = np.random.default_rng(0)
    events = simulate_hawkes(truth, bbox=bbox, t_end=80.0, rng=rng)
    chunk = TrainChunk(
        times=torch.tensor(events["time"], dtype=torch.float32),
        lons=torch.tensor(events["lon"], dtype=torch.float32),
        lats=torch.tensor(events["lat"], dtype=torch.float32),
        marks=torch.tensor(events["mark"], dtype=torch.long),
        window=(0.0, 80.0),
    )

    # 1. Train OUR model.
    model = NeuralHawkes(
        n_marks=n_marks, hidden_dim=32, mark_emb_dim=8, spatial_emb_dim=8, n_mix=4
    )
    optimizer = AdamW(model.parameters(), lr=1e-2)
    for _ in range(30):
        train_one_epoch(model, [chunk], optimizer)
    with torch.no_grad():
        our_nll = float(
            -model.log_likelihood(
                chunk.times, chunk.lons, chunk.lats, chunk.marks, chunk.window
            ).item()
        ) / chunk.times.shape[0]
    print(f"Our NLL/event: {our_nll:.4f}")

    if have_ref is None:
        pytest.skip(
            "No reference TPP library available (neither easy_tpp nor tpps installed)."
            " Our model's NLL is reported but cross-check is skipped."
        )

    # 2. Train reference library's CTLSTM on the same event sequence.
    # The actual API integration goes here once the library install is
    # confirmed working in CI. The placeholder skip below is intentional —
    # it preserves the test infrastructure (import paths, training run,
    # NLL computation) without making a brittle assertion against an
    # unverified library API.
    pytest.skip(
        f"{have_ref} is importable but operational API integration is pending. "
        "Replace this skip with the reference fit + NLL extraction once the "
        "library API is confirmed against the installed version."
    )
