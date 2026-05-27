"""Tests for the auxiliary mark-classification loss (H4 experiment).

Surfaces tested:
  * NeuralHawkes.forward() returns a 'z_at_events' key with raw mark-head
    logits (shape (N, n_marks)), pre-softplus.
  * NeuralHawkes.log_likelihood(..., aux_lambda=0.0) is bit-identical to
    the current default (backwards-compat guard).
  * Non-zero aux_lambda changes the log_likelihood value (sanity).
  * aux_lambda=1.0 produces a finite log_likelihood on a tiny synthetic.
  * The aux gradient flows to W_lambda_k parameters after backward().
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eonet_cascades.models.neural_hawkes import NeuralHawkes


def _small_inputs(n_marks: int = 3, n_events: int = 20, seed: int = 0):
    """Reusable synthetic event sequence."""
    rng = np.random.default_rng(seed)
    times = torch.tensor(np.sort(rng.uniform(0.0, 20.0, size=n_events)), dtype=torch.float32)
    lons = torch.tensor(rng.uniform(-10.0, 10.0, size=n_events), dtype=torch.float32)
    lats = torch.tensor(rng.uniform(-5.0, 5.0, size=n_events), dtype=torch.float32)
    marks = torch.tensor(rng.integers(0, n_marks, size=n_events), dtype=torch.long)
    return times, lons, lats, marks


def test_forward_returns_z_at_events_with_correct_shape():
    """forward() output dict must include 'z_at_events' with shape (N, K)."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=5, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=5, n_events=12, seed=0)
    out = model(times, lons, lats, marks)
    assert "z_at_events" in out, f"missing 'z_at_events' key; got {sorted(out.keys())}"
    z = out["z_at_events"]
    assert z.shape == (12, 5), f"expected (12, 5), got {tuple(z.shape)}"
    # z is pre-softplus logits and can be negative; sanity-check it's finite.
    assert torch.isfinite(z).all()


def test_aux_lambda_zero_is_bit_identical_to_no_aux():
    """log_likelihood(..., aux_lambda=0.0) and log_likelihood(...) with no
    aux_lambda kwarg must return the EXACT same tensor. Backwards-compat."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=4, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=4, n_events=15, seed=1)
    ll_default = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0))
    ll_zero = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=0.0)
    assert torch.equal(ll_default, ll_zero), "aux_lambda=0.0 should be no-op"


def test_nonzero_aux_lambda_changes_log_likelihood():
    """Sanity: aux_lambda > 0 actually changes the loss."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=4, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=4, n_events=15, seed=1)
    ll_zero = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=0.0)
    ll_one = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=1.0)
    assert not torch.equal(ll_zero, ll_one), "aux_lambda=1.0 should change the loss"
    # The aux term is + aux_lambda * sum log P(k_obs | h). Since log probabilities
    # are <= 0, this strictly decreases log_likelihood relative to aux_lambda=0.
    assert ll_one.item() < ll_zero.item(), (
        f"aux loss should reduce log_likelihood; got ll_one={ll_one.item():.4f}, "
        f"ll_zero={ll_zero.item():.4f}"
    )


def test_aux_lambda_one_log_likelihood_finite():
    """log_likelihood(..., aux_lambda=1.0) returns a finite scalar."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=4, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    model.eval()

    times, lons, lats, marks = _small_inputs(n_marks=4, n_events=15, seed=2)
    ll = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=1.0)
    assert ll.dim() == 0, f"expected scalar, got shape {tuple(ll.shape)}"
    assert torch.isfinite(ll), f"non-finite log_likelihood: {ll.item()}"


def test_aux_loss_gradient_flows_to_mark_head():
    """When aux_lambda > 0, backward() on -log_likelihood populates .grad on
    the W_lambda_k mark-head parameters. Guards against the aux loss being
    accidentally detached from the mark head."""
    torch.manual_seed(0)
    model = NeuralHawkes(
        n_marks=4, hidden_dim=8, mark_emb_dim=4, spatial_emb_dim=4, n_mix=2,
        mark_head="mlp",
    )
    # Zero out all .grad before computing.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    times, lons, lats, marks = _small_inputs(n_marks=4, n_events=15, seed=3)
    ll = model.log_likelihood(times, lons, lats, marks, window=(0.0, 20.0), aux_lambda=1.0)
    loss = -ll
    loss.backward()

    # Verify the mark head's linear layers got gradient.
    head = model.W_lambda_k
    assert isinstance(head, nn.Sequential), "expected MLP head for this test"
    for i, sub in enumerate(head):
        if isinstance(sub, nn.Linear):
            g = sub.weight.grad
            assert g is not None, f"head[{i}].weight has no .grad"
            assert torch.isfinite(g).all(), f"head[{i}].weight.grad has non-finite values"
            assert g.abs().sum().item() > 0.0, f"head[{i}].weight.grad is all zeros"
