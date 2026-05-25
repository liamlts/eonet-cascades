"""Tier 1 — Neural Hawkes (CTLSTM + MDN) model.

Architecture: continuous-time LSTM hidden state h(t), per-mark temporal
intensity heads lambda_k(t|h), shared full-covariance bivariate MDN spatial
head conditioned on (h, mark_emb).

Per-mark intensity decomposition (multivariate Hawkes form):
    lambda_k(t, x | h(t)) = softplus(W_lambda_k . h(t)) * p(x | h, k)

So the per-event log-likelihood contribution at (t_i, x_i, k_i) is:
    log lambda_{k_i}(t_i | h(t_i)) + log p(x_i | h(t_i), k_i)

The total intensity integrated over time (for the NLL integral) is:
    integral over [t_start, t_end] of sum_k lambda_k(t | h(t)) dt
"""

from __future__ import annotations

import torch
import torch.nn.functional as nnf
from torch import nn

from eonet_cascades.models.components.ctlstm import CTLSTMCell
from eonet_cascades.models.components.embeddings import MarkEmbedding, SpatialEmbedding
from eonet_cascades.models.components.mdn_head import MDNHead


class NeuralHawkes(nn.Module):
    name = "neural_hawkes_tier1"

    def __init__(
        self,
        n_marks: int,
        hidden_dim: int = 64,
        mark_emb_dim: int = 16,
        spatial_emb_dim: int = 16,
        n_mix: int = 8,
    ) -> None:
        super().__init__()
        self.n_marks = n_marks
        self.hidden_dim = hidden_dim
        self.mark_emb_dim = mark_emb_dim
        self.mark_emb = MarkEmbedding(n_marks=n_marks, dim=mark_emb_dim)
        self.spatial_emb = SpatialEmbedding(dim=spatial_emb_dim)
        input_dim = mark_emb_dim + spatial_emb_dim
        self.cell = CTLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)
        # Per-mark temporal intensity head — replaces the old W_lambda_t (scalar)
        # and W_mark (softmax) pair.
        self.W_lambda_k = nn.Linear(hidden_dim, n_marks)
        self.mdn = MDNHead(input_dim=hidden_dim + mark_emb_dim, n_components=n_mix)

    def _event_input(
        self, lon: torch.Tensor, lat: torch.Tensor, mark: torch.Tensor
    ) -> torch.Tensor:
        x = torch.stack([lon, lat], dim=-1)
        return torch.cat([self.mark_emb(mark), self.spatial_emb(x)], dim=-1)

    def _lambda_k(self, h: torch.Tensor) -> torch.Tensor:
        """Per-mark intensities at hidden state h. Returns shape (..., n_marks)."""
        return nnf.softplus(self.W_lambda_k(h)).clamp_min(1e-12)

    def forward(
        self,
        times: torch.Tensor,
        lons: torch.Tensor,
        lats: torch.Tensor,
        marks: torch.Tensor,
        init_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process a 1-D event sequence. Returns per-event intensity components.

        Returns dict with:
            log_lambda_k_at_event : (N,) log lambda_{k_i}(t_i | h(t_i))
            log_p_x               : (N,) log p(x_i | h, k_i)
            h_at_events           : (N, hidden_dim) hidden state at event times
        """
        n = times.shape[0]
        device = times.device
        hidden_dim = self.hidden_dim
        if init_state is None:
            c_post_i = torch.zeros(1, hidden_dim, device=device)
            c_bar_i = torch.zeros(1, hidden_dim, device=device)
            delta_i = torch.ones(1, hidden_dim, device=device)
            o_i = torch.zeros(1, hidden_dim, device=device)
            t_last_i = torch.zeros(1, device=device)
        else:
            c_post_i, c_bar_i, delta_i, o_i, t_last_i = init_state

        log_lambda_k_list: list[torch.Tensor] = []
        log_p_x_list: list[torch.Tensor] = []
        h_event_list: list[torch.Tensor] = []

        for i in range(n):
            t_i = times[i : i + 1]
            dt = (t_i - t_last_i).clamp(min=0.0).unsqueeze(-1)
            h_at_t, _ = self.cell.evolve(c_post_i, c_bar_i, delta_i, o_i, dt)

            lam_k = self._lambda_k(h_at_t)  # (1, n_marks)
            log_lam_at_obs = torch.log(lam_k[0, marks[i]])  # scalar

            mark_e = self.mark_emb(marks[i : i + 1])
            mdn_input = torch.cat([h_at_t, mark_e], dim=-1)
            x_i = torch.stack([lons[i : i + 1], lats[i : i + 1]], dim=-1)
            log_p_x_i = self.mdn.log_prob(mdn_input, x_i)

            log_lambda_k_list.append(log_lam_at_obs)
            log_p_x_list.append(log_p_x_i.squeeze())
            h_event_list.append(h_at_t.squeeze(0))

            ev_inp = self._event_input(lons[i : i + 1], lats[i : i + 1], marks[i : i + 1])
            _, c_post_i, c_bar_i, delta_i, o_i = self.cell.update(ev_inp, h_at_t, c_post_i, c_bar_i)
            t_last_i = t_i

        return {
            "log_lambda_k_at_event": torch.stack(log_lambda_k_list),
            "log_p_x": torch.stack(log_p_x_list),
            "h_at_events": torch.stack(h_event_list),
        }

    def log_likelihood(
        self,
        times: torch.Tensor,
        lons: torch.Tensor,
        lats: torch.Tensor,
        marks: torch.Tensor,
        window: tuple[float, float],
        n_mc_samples: int = 20,
    ) -> torch.Tensor:
        """Compute log L over a single event sequence in window.

        log L = sum_i (log lambda_{k_i}(t_i | h) + log p(x_i | h, k_i))
              - integral over [t_start, t_end] of sum_k lambda_k(t | h(t)) dt
        """
        t_start, t_end = window
        out = self.forward(times, lons, lats, marks)
        per_event = out["log_lambda_k_at_event"] + out["log_p_x"]
        sum_per_event = per_event.sum()

        device = times.device
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        sample_times, _ = torch.sort(
            torch.rand(n_mc_samples, generator=gen) * (t_end - t_start) + t_start
        )
        sample_times = sample_times.to(device)
        lam_total_at_samples = self._lambda_total_at(times, lons, lats, marks, sample_times)
        integral = lam_total_at_samples.mean() * (t_end - t_start)
        return sum_per_event - integral

    def _lambda_total_at(
        self,
        event_times: torch.Tensor,
        event_lons: torch.Tensor,
        event_lats: torch.Tensor,
        event_marks: torch.Tensor,
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate sum_k lambda_k at arbitrary query_times given event history."""
        device = event_times.device
        hidden_dim = self.hidden_dim
        c_post = torch.zeros(1, hidden_dim, device=device)
        c_bar = torch.zeros(1, hidden_dim, device=device)
        delta = torch.ones(1, hidden_dim, device=device)
        o = torch.zeros(1, hidden_dim, device=device)
        t_last = torch.zeros(1, device=device)
        qt_sorted, _ = torch.sort(query_times)
        out_vals: list[torch.Tensor] = []
        ei = 0
        n_events = event_times.shape[0]
        for q in qt_sorted:
            while ei < n_events and event_times[ei] <= q:
                dt = (event_times[ei : ei + 1] - t_last).clamp(min=0.0).unsqueeze(-1)
                h_at_t, _ = self.cell.evolve(c_post, c_bar, delta, o, dt)
                ev_inp = self._event_input(
                    event_lons[ei : ei + 1], event_lats[ei : ei + 1], event_marks[ei : ei + 1]
                )
                _, c_post, c_bar, delta, o = self.cell.update(ev_inp, h_at_t, c_post, c_bar)
                t_last = event_times[ei : ei + 1]
                ei += 1
            dt = (q.unsqueeze(0) - t_last).clamp(min=0.0).unsqueeze(-1)
            h_at_q, _ = self.cell.evolve(c_post, c_bar, delta, o, dt)
            lam_k = self._lambda_k(h_at_q)  # (1, n_marks)
            out_vals.append(lam_k.sum(dim=-1).squeeze())
        return torch.stack(out_vals)
