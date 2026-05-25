"""Continuous-time LSTM cell (Mei & Eisner 2017)."""

from __future__ import annotations

import torch
from torch import nn


class CTLSTMCell(nn.Module):
    """Continuous-time LSTM cell.

    Differs from standard LSTM by maintaining a SECOND cell-state target
    (c_bar = asymptote as t -> infinity) and a learned decay rate (delta).
    Between events, c(t) decays exponentially from c_post toward c_bar with
    rate delta. See Mei & Eisner 2017 NeurIPS, eqs. 5-9.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 7 gate sets: i, f, g, o, i_bar, f_bar, delta. Each is a linear from
        # concat(input, h_prev) to hidden_dim.
        self.W = nn.Linear(input_dim + hidden_dim, 7 * hidden_dim)
        nn.init.xavier_uniform_(self.W.weight, gain=0.5)
        nn.init.zeros_(self.W.bias)

    def update(
        self,
        x: torch.Tensor,  # (B, input_dim) — event embedding
        h_prev: torch.Tensor,  # (B, hidden_dim)
        c_prev: torch.Tensor,  # (B, hidden_dim)
        c_bar_prev: torch.Tensor,  # (B, hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one event step. Returns (h, c_post, c_bar, delta, o)."""
        u = torch.cat([x, h_prev], dim=-1)  # (B, in + hid)
        gates = self.W(u)  # (B, 7 * hid)
        i, f, g, o, i_bar, f_bar, d = gates.chunk(7, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        i_bar = torch.sigmoid(i_bar)
        f_bar = torch.sigmoid(f_bar)
        delta = torch.nn.functional.softplus(d) + 1e-6

        c_post = f * c_prev + i * g
        c_bar = f_bar * c_bar_prev + i_bar * g
        h = o * torch.tanh(c_post)
        return h, c_post, c_bar, delta, o

    def evolve(
        self,
        c_post: torch.Tensor,  # (B, hidden_dim) — cell just after last event
        c_bar: torch.Tensor,  # (B, hidden_dim) — asymptote
        delta: torch.Tensor,  # (B, hidden_dim) — decay rate per dim
        o: torch.Tensor,  # (B, hidden_dim) — output gate at last event
        dt: torch.Tensor,  # (B, 1) or (B,) — elapsed time since last event
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (h(t), c(t)) at time t = t_last_event + dt via closed-form decay."""
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)
        c_t = c_bar + (c_post - c_bar) * torch.exp(-delta * dt)
        h_t = o * torch.tanh(c_t)
        return h_t, c_t
