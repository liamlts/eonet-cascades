"""NASA EONET v3 fetcher."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from eonet_cascades.data.http import RateLimitedClient
from eonet_cascades.data.marks import harmonize_mark
from eonet_cascades.data.schema import Event, RawEvent

EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"


class EONETFetcher:
    """Fetches events from NASA EONET v3."""

    name = "eonet"

    def __init__(self, rate_per_sec: float = 1.0) -> None:
        self._client = RateLimitedClient(rate_per_sec=rate_per_sec)

    def fetch(self, since: datetime, until: datetime) -> Iterable[RawEvent]:
        params = {
            "start": since.date().isoformat(),
            "end": until.date().isoformat(),
            "status": "all",
        }
        r = self._client.get(EONET_URL, params=params)
        payload = r.json()
        yield from self._iter_raw_from_payload(payload)

    def _iter_raw_from_payload(self, payload: dict[str, Any]) -> Iterable[RawEvent]:
        for ev in payload.get("events", []):
            yield RawEvent(source_catalog="eonet", source_id=ev["id"], payload=ev)

    def harmonize(self, raw: RawEvent) -> Event | None:
        p = raw.payload
        categories = p.get("categories", [])
        if not categories:
            return None
        cat_id = categories[0].get("id", "")
        mark = harmonize_mark("eonet", cat_id)
        if mark is None:
            return None

        geometry = p.get("geometry", [])
        if not geometry:
            return None
        first = geometry[0]
        if first.get("type") != "Point":
            return None
        coords = first.get("coordinates")
        if not coords or len(coords) < 2:
            return None
        lon, lat = float(coords[0]), float(coords[1])

        time_start = _parse_iso8601(first["date"])
        time_end = None
        if len(geometry) > 1:
            time_end = _parse_iso8601(geometry[-1]["date"])

        return Event(
            event_id=f"eonet:{raw.source_id}",
            source_catalog="eonet",
            time_start=time_start,
            time_end=time_end,
            longitude=lon,
            latitude=lat,
            mark=mark,
            magnitude=None,
            metadata={"title": p.get("title"), "sources": p.get("sources", [])},
            ingested_at=datetime.now(UTC),
            dedup_group_id=None,
        )


def _parse_iso8601(ts: str) -> datetime:
    """EONET emits 'Z'-suffixed UTC timestamps. fromisoformat needs explicit handling."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
