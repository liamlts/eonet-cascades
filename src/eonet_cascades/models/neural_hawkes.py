"""Tier 1 — Neural Hawkes (CTLSTM + MDN) model.

Per spec docs/superpowers/specs/2026-05-25-tier-1-neural-hawkes-design.md §3.
"""

from __future__ import annotations

import torch
import torch.nn.functional as nnf
from torch import nn

from eonet_cascades.models.components.ctlstm import CTLSTMCell
from eonet_cascades.models.components.embeddings import MarkEmbedding, SpatialEmbedding
from eonet_cascades.models.components.mdn_head import MDNHead


class NeuralHawkes(nn.Module):
    """Spatio-temporal marked Neural Hawkes.

    Intensity decomposition (spec §3.2):
        lambda(t, x, k | h(t)) = lambda_t(t | h) * p(k | h, t) * p(x | h, t, k)
    """

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
        self.W_lambda_t = nn.Linear(hidden_dim, 1)
        self.W_mark = nn.Linear(hidden_dim, n_marks)
        self.mdn = MDNHead(input_dim=hidden_dim + mark_emb_dim, n_components=n_mix)

    def _event_input(self, lon: torch.Tensor, lat: torch.Tensor, mark: torch.Tensor) -> torch.Tensor:
        x = torch.stack([lon, lat], dim=-1)
        return torch.cat([self.mark_emb(mark), self.spatial_emb(x)], dim=-1)

    def forward(
        self,
        times: torch.Tensor,
        lons: torch.Tensor,
        lats: torch.Tensor,
        marks: torch.Tensor,
        init_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process a 1-D event sequence. Returns per-event intensity components.

        Returns dict with:
            log_lambda_t : (N,) log temporal intensity at event times
            log_p_mark   : (N,) log p(k_i | h(t_i))
            log_p_x      : (N,) log p(x_i | h(t_i), k_i)
            h_at_events  : (N, hidden_dim) hidden state queried at event times
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

        log_lambda_t_list: list[torch.Tensor] = []
        log_p_mark_list: list[torch.Tensor] = []
        log_p_x_list: list[torch.Tensor] = []
        h_event_list: list[torch.Tensor] = []

        for i in range(n):
            t_i = times[i:i + 1]
            dt = (t_i - t_last_i).clamp(min=0.0).unsqueeze(-1)  # (1, 1)
            # Hidden state at the event time (just BEFORE the event update).
            h_at_t, _ = self.cell.evolve(c_post_i, c_bar_i, delta_i, o_i, dt)
            # Intensity components computed from h(t_i) before the event update.
            log_lambda_t = nnf.softplus(self.W_lambda_t(h_at_t)).clamp_min(1e-12).log()
            mark_logits = nnf.log_softmax(self.W_mark(h_at_t), dim=-1)
            log_p_mark_i = mark_logits[0, marks[i]]
            # MDN spatial: conditioned on (h(t_i), mark_emb[k_i]).
            mark_e = self.mark_emb(marks[i:i + 1])
            mdn_input = torch.cat([h_at_t, mark_e], dim=-1)
            x_i = torch.stack([lons[i:i + 1], lats[i:i + 1]], dim=-1)
            log_p_x_i = self.mdn.log_prob(mdn_input, x_i)

            log_lambda_t_list.append(log_lambda_t.squeeze())
            log_p_mark_list.append(log_p_mark_i)
            log_p_x_list.append(log_p_x_i.squeeze())
            h_event_list.append(h_at_t.squeeze(0))

            # Event update: incorporate (mark, location) into the CTLSTM state.
            ev_inp = self._event_input(lons[i:i + 1], lats[i:i + 1], marks[i:i + 1])
            _, c_post_i, c_bar_i, delta_i, o_i = self.cell.update(
                ev_inp, h_at_t, c_post_i, c_bar_i
            )
            t_last_i = t_i

        return {
            "log_lambda_t": torch.stack(log_lambda_t_list),
            "log_p_mark": torch.stack(log_p_mark_list),
            "log_p_x": torch.stack(log_p_x_list),
            "h_at_events": torch.stack(h_event_list),
        }
