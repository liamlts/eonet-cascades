"""Training driver for Tier 1 NeuralHawkes.

Per spec §4.2: chunk events into 7-day windows, truncated BPTT with hidden
state carryover between chunks, AdamW + cosine schedule + grad-clipping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from eonet_cascades.models.neural_hawkes import NeuralHawkes


@dataclass
class TrainChunk:
    """One 7-day chunk of events, ready to feed the model."""

    times: torch.Tensor  # (N,) in days since window start
    lons: torch.Tensor  # (N,)
    lats: torch.Tensor  # (N,)
    marks: torch.Tensor  # (N,) int64
    window: tuple[float, float]  # chunk start / end in same units as times


def mark_rebalance_weights(
    marks_idx: np.ndarray, n_marks: int, mode: str = "inverse-sqrt"
) -> torch.Tensor:
    """Compute per-mark loss weights for class-rebalanced training.

    Returns a (K,) tensor normalized so mean weight = 1. Counts of 0 are
    floored to 1 to avoid division-by-zero (the mark gets the floor weight;
    it has no training signal anyway).

    Modes:
        "inverse-sqrt"      w[k] proportional to 1 / sqrt(count[k])
        "inverse-frequency" w[k] proportional to 1 / count[k]   (more aggressive)
        "none"              w[k] = 1 for all k (equivalent to passing None)
    """
    counts = np.zeros(n_marks, dtype=np.float64)
    unique, c = np.unique(marks_idx, return_counts=True)
    counts[unique] = c
    counts = np.maximum(counts, 1.0)
    if mode == "inverse-sqrt":
        raw = 1.0 / np.sqrt(counts)
    elif mode == "inverse-frequency":
        raw = 1.0 / counts
    elif mode == "none":
        raw = np.ones(n_marks, dtype=np.float64)
    else:
        raise ValueError(f"unknown rebalance mode {mode!r}")
    weights = raw / raw.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: NeuralHawkes,
    chunks: Iterable[TrainChunk],
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | None = None,
    grad_clip: float = 1.0,
    device: str = "cpu",
    mark_weights: torch.Tensor | None = None,
    aux_lambda: float = 0.0,
) -> dict[str, float]:
    """Run one epoch of training over the chunk iterator.

    If `mark_weights` is provided, applies a class-rebalanced training
    objective (see NeuralHawkes.log_likelihood). If `aux_lambda > 0`,
    adds the H4 auxiliary mark-classification cross-entropy loss. The
    eval/NLL reporting path should always use both at default (None / 0.0)
    so val numbers stay comparable across runs.

    Return dict keys:
      - loss_sum: accumulated -log_likelihood (blended objective when aux>0)
      - n_events: total event count across chunks
      - nll_per_event: loss_sum / n_events (blended when aux_lambda > 0)
      - nll_hawkes_per_event: pure Hawkes NLL per event, independent of
        aux_lambda. This is the semantically-consistent column to write to
        train_curves.csv for cross-run comparison.
      - aux_per_event: aux contribution per event (0 when aux_lambda == 0),
        equal to nll_per_event - nll_hawkes_per_event up to fp roundoff.
    """
    model.train()
    total_loss = 0.0
    total_hawkes_loss = 0.0
    total_aux_loss = 0.0
    total_events = 0
    for chunk in chunks:
        optimizer.zero_grad()
        times = chunk.times.to(device)
        lons = chunk.lons.to(device)
        lats = chunk.lats.to(device)
        marks = chunk.marks.to(device)
        if times.numel() == 0:
            continue
        components = model.log_likelihood(
            times,
            lons,
            lats,
            marks,
            chunk.window,
            mark_weights=mark_weights,
            aux_lambda=aux_lambda,
            return_components=True,
        )
        ll = components["total"]
        loss = -ll
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        total_hawkes_loss += float(-components["hawkes"].item())
        total_aux_loss += float(-components["aux"].item())
        total_events += int(times.shape[0])
    n = max(1, total_events)
    return {
        "loss_sum": total_loss,
        "n_events": total_events,
        "nll_per_event": total_loss / n,
        "nll_hawkes_per_event": total_hawkes_loss / n,
        "aux_per_event": total_aux_loss / n,
    }
