"""Per-event gradient attribution -> K x K neural excitation matrix.

Per spec §5.A:
    A[k_parent, k_child] += || grad_{h_parent} log lambda_{k_child}(t_child, x_child | h) ||_1
                            * exp( -(t_child - t_parent) / tau )

with tau = 7 days.

Implementation note: we replicate the NeuralHawkes forward loop here so that each
h_at_t is an explicit intermediate tensor we can differentiate against individually.
The stacked h_at_events tensor returned by model.forward() is created after the loop
and does not lie on the computation path of the per-event log-likelihood scalars, so
torch.autograd.grad(per_event[i], stacked_h_events) always returns None.
"""

from __future__ import annotations

import torch
import torch.nn.functional as nnf

from eonet_cascades.models.neural_hawkes import NeuralHawkes


def compute_attribution_matrix(
    model: NeuralHawkes,
    times: torch.Tensor,
    lons: torch.Tensor,
    lats: torch.Tensor,
    marks: torch.Tensor,
    n_marks: int,
    tau_days: float = 7.0,
) -> torch.Tensor:
    """Aggregate per-event gradient attributions into a K x K matrix.

    Result: rows = parent_mark, columns = child_mark.
    Same convention as Tier 0's alpha matrix.
    """
    model.eval()
    a_matrix = torch.zeros(n_marks, n_marks, dtype=torch.float64)
    n = times.shape[0]
    if n < 2:
        return a_matrix

    # Replicate NeuralHawkes.forward() keeping each h_at_t as an explicit node.
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

        lam_k = nnf.softplus(model.W_lambda_k(h_at_t)).clamp_min(1e-12)  # (1, n_marks)
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
            grads = torch.autograd.grad(
                per_event_list[i], h_list[j], retain_graph=True, allow_unused=True
            )[0]
            if grads is None:
                continue
            grad_norm = float(grads.abs().sum().item())
            dt = float((times[i] - times[j]).clamp(min=0.0).item())
            decay = float(torch.exp(torch.tensor(-dt / tau_days)).item())
            parent = int(marks[j])
            a_matrix[parent, child] += grad_norm * decay

    return a_matrix
