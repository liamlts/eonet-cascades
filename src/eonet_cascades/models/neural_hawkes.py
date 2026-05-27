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
        mark_head: str = "linear",
    ) -> None:
        super().__init__()
        self.n_marks = n_marks
        self.hidden_dim = hidden_dim
        self.mark_emb_dim = mark_emb_dim
        self.mark_emb = MarkEmbedding(n_marks=n_marks, dim=mark_emb_dim)
        self.spatial_emb = SpatialEmbedding(dim=spatial_emb_dim)
        input_dim = mark_emb_dim + spatial_emb_dim
        self.cell = CTLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)
        # Per-mark temporal intensity head. The "linear" branch is the
        # original Tier 1 architecture (single nn.Linear). The "mlp" branch
        # (added 2026-05-26) is a 2-layer ReLU MLP that tests whether
        # non-linear capacity breaks the rank-1 mark-head collapse documented
        # in docs/notes/tier1_5-result.md.
        if mark_head == "linear":
            self.W_lambda_k = nn.Linear(hidden_dim, n_marks)
        elif mark_head == "mlp":
            self.W_lambda_k = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_marks),
            )
        else:
            raise ValueError(
                f"unknown mark_head: {mark_head!r} (expected 'linear' or 'mlp')"
            )
        self.mark_head = mark_head
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
            z_at_events           : (N, n_marks) raw mark-head logits pre-softplus
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
        z_event_list: list[torch.Tensor] = []

        for i in range(n):
            t_i = times[i : i + 1]
            dt = (t_i - t_last_i).clamp(min=0.0).unsqueeze(-1)
            h_at_t, _ = self.cell.evolve(c_post_i, c_bar_i, delta_i, o_i, dt)

            z_at_t = self.W_lambda_k(h_at_t)  # (1, n_marks) raw logits pre-softplus
            lam_k = nnf.softplus(z_at_t).clamp_min(1e-12)
            log_lam_at_obs = torch.log(lam_k[0, marks[i]])  # scalar

            mark_e = self.mark_emb(marks[i : i + 1])
            mdn_input = torch.cat([h_at_t, mark_e], dim=-1)
            x_i = torch.stack([lons[i : i + 1], lats[i : i + 1]], dim=-1)
            log_p_x_i = self.mdn.log_prob(mdn_input, x_i)

            log_lambda_k_list.append(log_lam_at_obs)
            log_p_x_list.append(log_p_x_i.squeeze())
            h_event_list.append(h_at_t.squeeze(0))
            z_event_list.append(z_at_t.squeeze(0))

            ev_inp = self._event_input(lons[i : i + 1], lats[i : i + 1], marks[i : i + 1])
            _, c_post_i, c_bar_i, delta_i, o_i = self.cell.update(ev_inp, h_at_t, c_post_i, c_bar_i)
            t_last_i = t_i

        return {
            "log_lambda_k_at_event": torch.stack(log_lambda_k_list),
            "log_p_x": torch.stack(log_p_x_list),
            "h_at_events": torch.stack(h_event_list),
            "z_at_events": torch.stack(z_event_list),
        }

    def log_likelihood(
        self,
        times: torch.Tensor,
        lons: torch.Tensor,
        lats: torch.Tensor,
        marks: torch.Tensor,
        window: tuple[float, float],
        n_mc_samples: int = 20,
        mark_weights: torch.Tensor | None = None,
        aux_lambda: float = 0.0,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute log L over a single event sequence in window.

        log L = sum_i (log lambda_{k_i}(t_i | h) + log p(x_i | h, k_i))
              - integral over [t_start, t_end] of sum_k lambda_k(t | h(t)) dt

        Optional `mark_weights`: a (K,) tensor weighting the per-event log-lik
        contributions by `w[marks[i]]`. Used for class-rebalanced TRAINING when
        the mark distribution is heavily imbalanced (per the Tier 1.5 retrain
        addressing the mark-head class-collapse diagnosed in commit 420d5a3).
        Pass None (default) for evaluation / true-NLL reporting — weighted
        log_likelihood is NOT the true Hawkes likelihood.

        Optional `aux_lambda`: coefficient for the H4 auxiliary mark-classification
        loss. When > 0, adds `aux_lambda * sum_i log softmax(z_i)_{k_i_observed}`
        to the log-likelihood, where z = W_lambda_k(h) are the raw mark-head
        logits. Provides explicit gradient on softmax(z) (mark composition)
        which is decoupled from the rate gradient via softplus(z). Pass 0.0
        (default) for evaluation — val NLL reported elsewhere should always be
        pure Hawkes NLL.

        Optional `return_components`: when True, returns a dict
        {"total", "hawkes", "aux"} of scalar tensors where total == hawkes + aux.
        "hawkes" is the pure Hawkes log-likelihood (unaffected by aux_lambda);
        "aux" is the aux_lambda * sum log P(k_obs|h) contribution (0 when
        aux_lambda == 0). Lets the training loop record a semantically-
        consistent pure-Hawkes NLL alongside the blended training objective.
        """
        t_start, t_end = window
        out = self.forward(times, lons, lats, marks)
        per_event = out["log_lambda_k_at_event"] + out["log_p_x"]
        if mark_weights is not None:
            w = mark_weights.to(per_event.device).index_select(0, marks)
            per_event = per_event * w
        sum_per_event = per_event.sum()

        # H4 auxiliary mark-classification loss (cross-entropy on softmax(z)).
        # Provides explicit gradient on RELATIVE z magnitudes (mark composition).
        # Softmax is shift-invariant in z, so this does not affect the rate
        # gradient which flows through softplus(z). Eval should always pass
        # aux_lambda=0.0 to keep val NLL comparable to the original Tier 1.
        aux_contribution: torch.Tensor | None = None
        if aux_lambda > 0.0:
            z = out["z_at_events"]  # (N, K) raw logits
            log_p_mark = nnf.log_softmax(z, dim=-1)
            log_p_obs = log_p_mark.gather(1, marks.unsqueeze(1)).squeeze(1)  # (N,)
            aux_term = log_p_obs.sum()  # SUM of log P(observed mark | h)
            # Added to per-event sum: log_p_obs entries are <= 0, so adding
            # aux_lambda * aux_term (with aux_lambda > 0) strictly decreases
            # log_likelihood. The training loop negates to get a minimizable
            # loss; this becomes -aux_lambda * sum log P(k_obs|h) = aux_lambda
            # times the standard categorical cross-entropy.
            aux_contribution = aux_lambda * aux_term
            sum_per_event = sum_per_event + aux_contribution

        device = times.device
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        sample_times, _ = torch.sort(
            torch.rand(n_mc_samples, generator=gen) * (t_end - t_start) + t_start
        )
        sample_times = sample_times.to(device)
        lam_total_at_samples = self._lambda_total_at(times, lons, lats, marks, sample_times)
        integral = lam_total_at_samples.mean() * (t_end - t_start)
        total = sum_per_event - integral
        if return_components:
            if aux_contribution is None:
                aux_contribution = total.new_zeros(())
                hawkes = total
            else:
                hawkes = total - aux_contribution
            return {"total": total, "hawkes": hawkes, "aux": aux_contribution}
        return total

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
