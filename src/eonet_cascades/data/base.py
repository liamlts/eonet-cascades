"""Common interface for catalog fetchers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from eonet_cascades.data.schema import Event, RawEvent


@runtime_checkable
class CatalogFetcher(Protocol):
    """A catalog fetcher knows how to pull raw records and harmonize them."""

    name: str

    def fetch(self, since: datetime, until: datetime) -> Iterable[RawEvent]:
        """Yield raw events covering [since, until)."""

    def harmonize(self, raw: RawEvent) -> Event | None:
        """Convert a raw event to the unified Event schema.

        Returning None signals the raw record is outside the unified mark
        vocabulary (or otherwise should be skipped without erroring).
        """
