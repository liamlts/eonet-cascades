"""Common interface for point process models (Tiers 0-3)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class PointProcessModel(Protocol):
    """All Hawkes / Neural Hawkes / Transformer Hawkes tiers conform to this.

    `events` is a polars DataFrame with at minimum columns
    (time_start: datetime, longitude: float, latitude: float, mark: str).
    """

    name: str

    def log_likelihood(
        self,
        events: pl.DataFrame,
        window: tuple[float, float],
    ) -> float:
        """Sum log-intensity at each event minus the integrated intensity."""

    def sample(
        self,
        history: pl.DataFrame,
        window: tuple[float, float],
    ) -> pl.DataFrame:
        """Forward-simulate new events given conditioning history (Ogata thinning)."""

    def fit(
        self,
        events: pl.DataFrame,
        window: tuple[float, float],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Fit model parameters. Returns a summary dict (final NLL, status, ...)."""
