"""Regression test for the vectorized gradient-attribution implementation.

The production `compute_attribution_matrix` in src/.../interpret/attribution.py
collapses the original O(n^2) inner loop of per-(i, j) `autograd.grad` calls
into one call per child with `inputs=h_list[:i]`. The two formulations must
give bit-identical results — this test pins that guarantee on a tiny
synthetic sequence so any future regression surfaces immediately.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as nnf

from eonet_cascades.interpret.attribution import compute_attribution_matrix
from eonet_cascades.models.neural_hawkes import NeuralHawkes


def _reference_attribution_per_pair(
    model: NeuralHawkes,
    times: torch.Tensor,
    lons: torch.Tensor,
    lats: torch.Tensor,
    marks: torch.Tensor,
    n_marks: int,
    tau_days: float = 7.0,
) -> torch.Tensor:
    """Per-(i, j) `autograd.grad` reference. Slow but maximally explicit.

    Mirrors the pre-vectorization implementation: one backward pass per
    (child i, parent j) pair, accumulating into the K x K matrix one
    pair at a time. Used only to pin the vectorized fast path.
    """
    model.eval()
    a_matrix = torch.zeros(n_marks, n_marks, dtype=torch.float64)
    n = times.shape[0]
    if n < 2:
        return a_matrix

    hidden_dim = model.hidden_dim
    device = times.device
    c_post = torch.zeros(1, hidden_dim, device=device)
    c_bar = torch.zeros(1, hidden_dim, device=device)
    delta = torch.ones(1, hidden_dim, device=device)
    o = torch.zeros(1, hidden_dim, device=device)
    t_last = torch.zeros(1, device=device)

    h_list: list[torch.Tensor] = []
    per_event_list: list[torch.Tensor] = []

    for i in range(n):
        t_i = times[i : i + 1]
        dt = (t_i - t_last).clamp(min=0.0).unsqueeze(-1)
        h_at_t, _ = model.cell.evolve(c_post, c_bar, delta, o, dt)
        if h_at_t.requires_grad:
            h_at_t.retain_grad()
        h_list.append(h_at_t)

        lam_k = nnf.softplus(model.W_lambda_k(h_at_t)).clamp_min(1e-12)
        log_lam_at_obs = torch.log(lam_k[0, marks[i]])
        mark_e = model.mark_emb(marks[i : i + 1])
        mdn_input = torch.cat([h_at_t, mark_e], dim=-1)
        x_i = torch.stack([lons[i : i + 1], lats[i : i + 1]], dim=-1)
        log_p_x_i = model.mdn.log_prob(mdn_input, x_i)
        per_event_list.append(log_lam_at_obs + log_p_x_i.squeeze())

        ev_inp = model._event_input(lons[i : i + 1], lats[i : i + 1], marks[i : i + 1])
        _, c_post, c_bar, delta, o = model.cell.update(ev_inp, h_at_t, c_post, c_bar)
        t_last = t_i

    for i in range(1, n):
        child = int(marks[i])
        for j in range(i):
            if not h_list[j].requires_grad:
                continue
            g = torch.autograd.grad(
                per_event_list[i], h_list[j], retain_graph=True, allow_unused=True
            )[0]
            if g is None:
                continue
            grad_norm = float(g.abs().sum().item())
            dt_pair = float((times[i] - times[j]).clamp(min=0.0).item())
            decay = float(torch.exp(torch.tensor(-dt_pair / tau_days)).item())
            parent = int(marks[j])
            a_matrix[parent, child] += grad_norm * decay

    return a_matrix


def test_vectorized_matches_per_pair_reference():
    """Vectorized fast path must equal the per-(i, j) reference to within
    float roundoff. Tiny n=25 synthetic with K=3 keeps the reference fast.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_marks = 3
    n_events = 25

    model = NeuralHawkes(
        n_marks=n_marks,
        hidden_dim=8,
        mark_emb_dim=4,
        spatial_emb_dim=4,
        n_mix=2,
    )
    model.eval()

    times = torch.tensor(np.sort(rng.uniform(0.0, 30.0, size=n_events)), dtype=torch.float32)
    lons = torch.tensor(rng.uniform(-10.0, 10.0, size=n_events), dtype=torch.float32)
    lats = torch.tensor(rng.uniform(-5.0, 5.0, size=n_events), dtype=torch.float32)
    marks = torch.tensor(rng.integers(0, n_marks, size=n_events), dtype=torch.long)

    a_vec = compute_attribution_matrix(model, times, lons, lats, marks, n_marks=n_marks)
    a_ref = _reference_attribution_per_pair(model, times, lons, lats, marks, n_marks=n_marks)

    assert a_vec.shape == a_ref.shape == (n_marks, n_marks)
    # Both implementations route the same autograd.grad call shape through
    # the same scalar accumulation pattern, so the residual is float
    # roundoff in the .item() / .tolist() conversions. atol=1e-9 is tight
    # but not paranoid.
    assert torch.allclose(a_vec, a_ref, atol=1e-9, rtol=0.0), (
        f"vectorized vs per-(i,j) reference disagree:\n"
        f"  vectorized:\n{a_vec.numpy()}\n"
        f"  reference :\n{a_ref.numpy()}\n"
        f"  max abs diff: {(a_vec - a_ref).abs().max().item():.3e}"
    )
    # Sanity: the matrix is non-trivial (otherwise the test is vacuous).
    assert float(a_ref.abs().sum().item()) > 0.0
