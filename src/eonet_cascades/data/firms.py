"""NASA FIRMS (VIIRS / MODIS active fire) fetcher.

API docs: https://firms.modaps.eosdis.nasa.gov/api/area/
Endpoint shape (bbox variant):
  /api/area/csv/<MAP_KEY>/<SOURCE>/<W>,<S>,<E>,<N>/<DAY_RANGE>/<DATE>
"""

from __future__ import annotations

import io
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import polars as pl

from eonet_cascades.data.http import RateLimitedClient
from eonet_cascades.data.schema import Event, Mark, RawEvent

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# Confidence levels: VIIRS uses 'l', 'n', 'h' (low/nominal/high); MODIS uses 0-100.
_CONF_ORDER = {"l": 0, "n": 1, "h": 2}


class FIRMSFetcher:
    name = "firms"

    def __init__(
        self,
        api_key: str | None,
        rate_per_sec: float = 0.5,
        source: str = "VIIRS_SNPP_SP",
        min_confidence: str = "n",
    ) -> None:
        self._api_key = api_key
        self._client = RateLimitedClient(rate_per_sec=rate_per_sec)
        self._source = source
        self._min_conf = min_confidence

    def fetch(
        self,
        since: datetime,
        until: datetime,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> Iterable[RawEvent]:
        if not self._api_key:
            raise RuntimeError(
                "FIRMS API key required; set EONET_FIRMS_API_KEY or configs/data/conus.yaml"
            )
        if bbox is None:
            bbox = (-180.0, -90.0, 180.0, 90.0)
        w, s, e, n = bbox
        # FIRMS API caps day_range at 5 — window accordingly.
        # FIRMS uses 400 Bad Request as its rate-limit response (not 429).
        # The client retries 5xx only, so we catch 4xx here, back off, retry once.
        cursor = since
        skipped_windows = 0
        while cursor < until:
            window_end = min(cursor + timedelta(days=5), until)
            day_range = (window_end - cursor).days
            if day_range == 0:
                day_range = 1
            date_str = cursor.date().isoformat()
            url = f"{BASE}/{self._api_key}/{self._source}/{w},{s},{e},{n}/{day_range}/{date_str}"
            try:
                r = self._client.get(url)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    # Likely rate-limited. Back off 60s and try once more.
                    print(
                        f"[firms] 400 on window {date_str}/+{day_range}d — "
                        "rate-limit suspected, sleeping 60s then retrying",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(60)
                    try:
                        r = self._client.get(url)
                    except httpx.HTTPStatusError:
                        skipped_windows += 1
                        print(
                            f"[firms] still failing on {date_str}/+{day_range}d "
                            "after backoff — skipping window",
                            file=sys.stderr,
                            flush=True,
                        )
                        cursor = window_end
                        continue
                else:
                    raise
            yield from self._iter_raw_from_csv(r.text)
            cursor = window_end
        if skipped_windows:
            print(
                f"[firms] total windows skipped after retries: {skipped_windows}",
                file=sys.stderr,
                flush=True,
            )

    def _iter_raw_from_csv(self, text: str) -> Iterable[RawEvent]:
        df = pl.read_csv(io.StringIO(text), infer_schema_length=1000, ignore_errors=True)
        if df.height == 0:
            return
        for i, row in enumerate(df.iter_rows(named=True)):
            sid = f"{row.get('acq_date','')}_{row.get('acq_time','')}_{row.get('latitude','')}_{row.get('longitude','')}_{i}"
            yield RawEvent(
                source_catalog="firms",
                source_id=sid,
                payload=row,
            )

    def harmonize(self, raw: RawEvent) -> Event | None:
        p = raw.payload
        lat = _safe_float(p.get("latitude"))
        lon = _safe_float(p.get("longitude"))
        if lat is None or lon is None:
            return None

        conf = str(p.get("confidence", "")).lower()
        if conf in _CONF_ORDER:
            if _CONF_ORDER[conf] < _CONF_ORDER.get(self._min_conf, 1):
                return None

        time_start = _parse_firms_datetime(p.get("acq_date"), p.get("acq_time"))
        if time_start is None:
            return None

        frp = _safe_float(p.get("frp"))

        return Event(
            event_id=f"firms:{raw.source_id}",
            source_catalog="firms",
            time_start=time_start,
            time_end=None,
            longitude=lon,
            latitude=lat,
            mark=Mark.WILDFIRE,
            magnitude=frp,
            metadata={
                "satellite": p.get("satellite"),
                "instrument": p.get("instrument"),
                "confidence": conf,
                "bright_ti4": p.get("bright_ti4"),
                "daynight": p.get("daynight"),
            },
            ingested_at=datetime.now(UTC),
            dedup_group_id=None,
        )


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_firms_datetime(date_s: Any, time_s: Any) -> datetime | None:
    if date_s is None or time_s is None:
        return None
    try:
        t = int(time_s)
        hh, mm = divmod(t, 100)
        d = datetime.strptime(str(date_s), "%Y-%m-%d")
        return d.replace(hour=hh, minute=mm, tzinfo=UTC)
    except (TypeError, ValueError):
        return None
