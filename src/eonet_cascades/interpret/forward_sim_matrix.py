"""Forward-simulation transition matrix for Tier 1 cascade interpretation.

For each parent mark, seed N_traj chains from a single event of that mark
at the bbox center, simulate forward via Ogata thinning on sum_k lambda_k
with mark drawn proportional to per-mark intensities, count which child
mark appears first within a fixed time window. Produces (K, K) transition
probability matrix.
"""

from __future__ import annotations

import torch

from eonet_cascades.models.neural_hawkes import NeuralHawkes


def compute_transition_matrix(
    model: NeuralHawkes,
    n_marks: int,
    bbox: tuple[float, float, float, float],
    n_trajectories: int = 1000,
    window_days: float = 14.0,
) -> torch.Tensor:
    """Estimate empirical P(first child mark | parent mark) by forward simulation.

    Returns (K, K) tensor T where T[p, c] = empirical count of child=c given
    parent=p, normalized so rows sum to 1 (or 0 if no events were sampled).
    """
    model.eval()
    transitions = torch.zeros(n_marks, n_marks, dtype=torch.float64)
    min_lon, min_lat, max_lon, max_lat = bbox
    cx = 0.5 * (min_lon + max_lon)
    cy = 0.5 * (min_lat + max_lat)
    device = next(model.parameters()).device

    for parent in range(n_marks):
        for _ in range(n_trajectories):
            times = torch.tensor([0.0], dtype=torch.float32, device=device)
            lons = torch.tensor([cx], dtype=torch.float32, device=device)
            lats = torch.tensor([cy], dtype=torch.float32, device=device)
            marks = torch.tensor([parent], dtype=torch.long, device=device)
            with torch.no_grad():
                child = _sample_first_child_mark(
                    model, times, lons, lats, marks, window_days
                )
            if child is not None:
                transitions[parent, child] += 1
    row_sums = transitions.sum(dim=1, keepdim=True).clamp(min=1.0)
    return transitions / row_sums


def _sample_first_child_mark(
    model: NeuralHawkes,
    history_times: torch.Tensor,
    history_lons: torch.Tensor,
    history_lats: torch.Tensor,
    history_marks: torch.Tensor,
    window_days: float,
    lambda_upper: float = 10.0,
) -> int | None:
    """Ogata thinning to draw the first child event time and mark.

    Uses sum_k lambda_k as the total intensity. When an event is accepted,
    samples its mark from p(k | h) = lambda_k / sum_k lambda_k.
    Returns the child mark, or None if no event occurs within the window.
    """
    t = float(history_times[-1].item())
    end = t + window_days
    while t < end:
        tau = float(torch.distributions.Exponential(rate=lambda_upper).sample().item())
        t = t + tau
        if t >= end:
            return None
        # Advance hidden state through history + up to time t, get per-mark intensities.
        lam_k = _lambda_k_at(model, history_times, history_lons, history_lats, history_marks, t)
        lam_total = float(lam_k.sum().item())
        if lam_total > lambda_upper + 1e-9:
            # Re-do with a larger upper bound; safer to skip this iteration.
            lambda_upper = lam_total * 1.5
            continue
        u = float(torch.rand(1).item())
        if u * lambda_upper <= lam_total:
            # Accept. Sample mark proportional to lam_k.
            probs = lam_k / lam_total
            return int(torch.multinomial(probs, 1).item())
    return None


def _lambda_k_at(
    model: NeuralHawkes,
    event_times: torch.Tensor,
    event_lons: torch.Tensor,
    event_lats: torch.Tensor,
    event_marks: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Evaluate lambda_k (per-mark intensity vector, shape (K,)) at time t given history."""
    device = event_times.device
    hidden_dim = model.hidden_dim
    c_post = torch.zeros(1, hidden_dim, device=device)
    c_bar = torch.zeros(1, hidden_dim, device=device)
    delta = torch.ones(1, hidden_dim, device=device)
    o = torch.zeros(1, hidden_dim, device=device)
    t_last = torch.zeros(1, device=device)
    for i in range(event_times.shape[0]):
        dt = (event_times[i:i + 1] - t_last).clamp(min=0.0).unsqueeze(-1)
        h_at_t, _ = model.cell.evolve(c_post, c_bar, delta, o, dt)
        ev_inp = model._event_input(
            event_lons[i:i + 1], event_lats[i:i + 1], event_marks[i:i + 1]
        )
        _, c_post, c_bar, delta, o = model.cell.update(ev_inp, h_at_t, c_post, c_bar)
        t_last = event_times[i:i + 1]
    dt = torch.tensor([[t]], device=device) - t_last.unsqueeze(-1)
    h_at_q, _ = model.cell.evolve(c_post, c_bar, delta, o, dt.clamp(min=0.0))
    lam_k = model._lambda_k(h_at_q)  # (1, K)
    return lam_k.squeeze(0)
