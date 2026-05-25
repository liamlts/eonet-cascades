"""Training driver for Tier 1 NeuralHawkes.

Per spec §4.2: chunk events into 7-day windows, truncated BPTT with hidden
state carryover between chunks, AdamW + cosine schedule + grad-clipping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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


def train_one_epoch(
    model: NeuralHawkes,
    chunks: Iterable[TrainChunk],
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | None = None,
    grad_clip: float = 1.0,
    device: str = "cpu",
) -> dict[str, float]:
    """Run one epoch of training over the chunk iterator.

    Returns a dict with mean train loss and number of events seen.
    """
    model.train()
    total_loss = 0.0
    total_events = 0
    for chunk in chunks:
        optimizer.zero_grad()
        times = chunk.times.to(device)
        lons = chunk.lons.to(device)
        lats = chunk.lats.to(device)
        marks = chunk.marks.to(device)
        if times.numel() == 0:
            continue
        ll = model.log_likelihood(times, lons, lats, marks, chunk.window)
        loss = -ll
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        total_events += int(times.shape[0])
    return {
        "loss_sum": total_loss,
        "n_events": total_events,
        "nll_per_event": total_loss / max(1, total_events),
    }
