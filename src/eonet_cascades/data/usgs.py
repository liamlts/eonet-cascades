"""USGS ComCat (FDSN web service) earthquake fetcher."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from eonet_cascades.data.http import RateLimitedClient
from eonet_cascades.data.schema import Event, Mark, RawEvent

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


class USGSFetcher:
    """Fetches earthquakes from the USGS ComCat FDSN web service."""

    name = "usgs"

    def __init__(self, rate_per_sec: float = 2.0, min_magnitude: float = 2.5) -> None:
        self._client = RateLimitedClient(rate_per_sec=rate_per_sec)
        self._min_magnitude = min_magnitude

    def fetch(
        self,
        since: datetime,
        until: datetime,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> Iterable[RawEvent]:
        params: dict[str, str] = {
            "format": "geojson",
            "starttime": since.isoformat(),
            "endtime": until.isoformat(),
            "minmagnitude": str(self._min_magnitude),
            "orderby": "time-asc",
        }
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            params["minlongitude"] = str(min_lon)
            params["maxlongitude"] = str(max_lon)
            params["minlatitude"] = str(min_lat)
            params["maxlatitude"] = str(max_lat)
        r = self._client.get(USGS_URL, params=params)
        payload = r.json()
        yield from self._iter_raw_from_payload(payload)

    def _iter_raw_from_payload(self, payload: dict[str, Any]) -> Iterable[RawEvent]:
        for feat in payload.get("features", []):
            yield RawEvent(source_catalog="usgs", source_id=feat["id"], payload=feat)

    def harmonize(self, raw: RawEvent) -> Event | None:
        p = raw.payload
        props = p.get("properties", {})
        geom = p.get("geometry", {})
        coords = geom.get("coordinates") or []
        if len(coords) < 2 or geom.get("type") != "Point":
            return None
        lon, lat = float(coords[0]), float(coords[1])
        ts_ms = props.get("time")
        if ts_ms is None:
            return None
        time_start = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
        return Event(
            event_id=f"usgs:{raw.source_id}",
            source_catalog="usgs",
            time_start=time_start,
            time_end=None,
            longitude=lon,
            latitude=lat,
            mark=Mark.EARTHQUAKE,
            magnitude=props.get("mag"),
            metadata={
                "place": props.get("place"),
                "depth_km": coords[2] if len(coords) >= 3 else None,
            },
            ingested_at=datetime.now(UTC),
            dedup_group_id=None,
        )
