"""Tests for the mark_head constructor argument added in commit 40cf6d3 spec.

Surfaces tested:
  * NeuralHawkes(mark_head="linear") is bit-identical to the current default.
  * NeuralHawkes(mark_head="mlp") builds an MLP head with expected shape.
  * log_likelihood returns finite scalars under both modes.
  * Unknown mark_head values raise ValueError at construction time.
  * mark_head round-trips through state_dict + config dict save/load.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eonet_cascades.models.neural_hawkes import NeuralHawkes


def _small_inputs(n_marks: int = 3, n_events: int = 20, seed: int = 0):
    """Reusable synthetic event sequence for forward-pass tests."""
    rng = np.random.default_rng(seed)
    times = torch.tensor(np.sort(rng.uniform(0.0, 20.0, size=n_events)), dtype=torch.float32)
    lons = torch.tensor(rng.uniform(-10.0, 10.0, size=n_events), dtype=torch.float32)
    lats = torch.tensor(rng.uniform(-5.0, 5.0, size=n_events), dtype=torch.float32)
    marks = torch.tensor(rng.integers(0, n_marks, size=n_events), dtype=torch.long)
    return times, lons, lats, marks


def test_linear_mark_head_default_is_bit_identical():
    """NeuralHawkes() and NeuralHawkes(mark_head='linear') must be bit-equal.

    Same seed before construction → same param initialization → same forward
    output on the same input. Guards against accidentally shifting RNG state.
    """
    torch.manual_seed(0)
    m_default = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2
    )

    torch.manual_seed(0)
    m_explicit = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="linear",
    )

    # All parameters must match exactly.
    for (n1, p1), (n2, p2) in zip(
        m_default.named_parameters(), m_explicit.named_parameters(), strict=True
    ):
        assert n1 == n2, f"parameter name mismatch: {n1} vs {n2}"
        assert torch.equal(p1, p2), f"parameter {n1} differs between default and explicit"

    times, lons, lats, marks = _small_inputs(n_marks=3, n_events=20, seed=0)
    out_default = m_default(times, lons, lats, marks)
    out_explicit = m_explicit(times, lons, lats, marks)
    assert torch.equal(
        out_default["log_lambda_k_at_event"], out_explicit["log_lambda_k_at_event"]
    )


def test_mlp_mark_head_constructs_with_expected_shape():
    """NeuralHawkes(mark_head='mlp') builds an nn.Sequential head with the
    correct (hidden_dim → hidden_dim // 2 → n_marks) shape."""
    model = NeuralHawkes(
        n_marks=8, hidden_dim=64, mark_emb_dim=8, spatial_emb_dim=8, n_mix=2,
        mark_head="mlp",
    )

    head = model.W_lambda_k
    assert isinstance(head, nn.Sequential), f"expected Sequential, got {type(head)}"
    assert len(head) == 3, f"expected 3 sub-modules (Linear, ReLU, Linear), got {len(head)}"
    assert isinstance(head[0], nn.Linear)
    assert head[0].in_features == 64
    assert head[0].out_features == 32  # hidden_dim // 2
    assert isinstance(head[1], nn.ReLU)
    assert isinstance(head[2], nn.Linear)
    assert head[2].in_features == 32
    assert head[2].out_features == 8

    assert model.mark_head == "mlp"


def test_mlp_mark_head_log_likelihood_is_finite():
    """log_likelihood under the MLP head returns a finite scalar tensor."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=3, n_events=20, seed=0)
    ll = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0))
    assert ll.dim() == 0, f"expected scalar, got shape {tuple(ll.shape)}"
    assert torch.isfinite(ll), f"non-finite log_likelihood: {ll.item()}"


def test_invalid_mark_head_raises_value_error():
    """Unknown mark_head value raises a clear ValueError, not a silent fallback."""
    import pytest
    with pytest.raises(ValueError, match="unknown mark_head"):
        NeuralHawkes(
            n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
            mark_head="transformer",
        )


def test_mlp_mark_head_state_dict_round_trip(tmp_path):
    """Save an MLP-head model + config dict, reload it, verify forward output
    matches bit-exactly. Mirrors the CLI's checkpoint save/load pattern."""
    torch.manual_seed(0)
    m_src = NeuralHawkes(
        n_marks=3, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    m_src.eval()

    ckpt_path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "state_dict": m_src.state_dict(),
            "mark_names": ["a", "b", "c"],
            "config": {
                "hidden_dim": 8,
                "mark_emb_dim": 4,
                "spatial_emb_dim": 4,
                "n_mix": 2,
                "n_marks": 3,
                "mark_head": "mlp",
            },
        },
        ckpt_path,
    )

    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["config"]
    assert cfg["mark_head"] == "mlp"

    m_dst = NeuralHawkes(
        n_marks=cfg["n_marks"],
        hidden_dim=cfg["hidden_dim"],
        mark_emb_dim=cfg["mark_emb_dim"],
        spatial_emb_dim=cfg["spatial_emb_dim"],
        n_mix=cfg["n_mix"],
        mark_head=cfg["mark_head"],
    )
    m_dst.load_state_dict(ckpt["state_dict"])
    m_dst.eval()

    times, lons, lats, marks = _small_inputs(n_marks=3, n_events=15, seed=1)
    out_src = m_src(times, lons, lats, marks)
    out_dst = m_dst(times, lons, lats, marks)
    assert torch.equal(out_src["log_lambda_k_at_event"], out_dst["log_lambda_k_at_event"])
