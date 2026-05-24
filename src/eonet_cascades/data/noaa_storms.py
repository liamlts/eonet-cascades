"""NOAA Storm Events Database fetcher.

Source: annual bulk CSV.gz files at
https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

This fetcher downloads the per-year details file, parses it, and yields raw rows.
"""

from __future__ import annotations

import gzip
import io
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from eonet_cascades.data.http import RateLimitedClient
from eonet_cascades.data.marks import harmonize_mark
from eonet_cascades.data.schema import Event, RawEvent

INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
# Filename pattern: StormEvents_details-ftp_v1.0_d2024_cYYYYMMDD.csv.gz
_FILENAME_RE = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz")

# Columns we care about; many more exist in the source.
_COLUMNS = [
    "EVENT_ID",
    "STATE",
    "EVENT_TYPE",
    "BEGIN_DATE_TIME",
    "END_DATE_TIME",
    "BEGIN_LAT",
    "BEGIN_LON",
    "MAGNITUDE",
    "MAGNITUDE_TYPE",
    "TOR_F_SCALE",
]


class NOAAStormsFetcher:
    name = "noaa"

    def __init__(self, rate_per_sec: float = 0.5, cache_dir: Path | None = None) -> None:
        self._client = RateLimitedClient(rate_per_sec=rate_per_sec)
        self._cache_dir = cache_dir

    def fetch(self, since: datetime, until: datetime) -> Iterable[RawEvent]:
        years = range(since.year, until.year + 1)
        for year in years:
            csv_bytes = self._download_year(year)
            if not csv_bytes:
                continue
            with io.BytesIO(csv_bytes) as fh, gzip.GzipFile(fileobj=fh) as gz:
                yield from self._iter_raw_from_bytes(gz.read())

    def _download_year(self, year: int) -> bytes:
        index_html = self._client.get(INDEX_URL).text
        full_names = [
            m.group(0)
            for m in _FILENAME_RE.finditer(index_html)
            if int(m.group(1)) == year
        ]
        if not full_names:
            return b""
        full_names.sort()
        url = INDEX_URL + full_names[-1]
        r = self._client.get(url)
        return r.content

    def _iter_raw_from_bytes(self, csv_bytes: bytes) -> Iterable[RawEvent]:
        df = pl.read_csv(io.BytesIO(csv_bytes), infer_schema_length=10000, ignore_errors=True)
        cols = [c for c in _COLUMNS if c in df.columns]
        for row in df.select(cols).iter_rows(named=True):
            yield RawEvent(
                source_catalog="noaa",
                source_id=str(row["EVENT_ID"]),
                payload=row,
            )

    def _iter_raw_from_csv(self, path: Path) -> Iterable[RawEvent]:
        """Test helper: parse a plain (uncompressed) CSV fixture."""
        df = pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)
        cols = [c for c in _COLUMNS if c in df.columns]
        for row in df.select(cols).iter_rows(named=True):
            yield RawEvent(
                source_catalog="noaa",
                source_id=str(row["EVENT_ID"]),
                payload=row,
            )

    def harmonize(self, raw: RawEvent) -> Event | None:
        p = raw.payload
        event_type = (p.get("EVENT_TYPE") or "").strip()
        mark = harmonize_mark("noaa", event_type)
        if mark is None:
            return None

        lat = _safe_float(p.get("BEGIN_LAT"))
        lon = _safe_float(p.get("BEGIN_LON"))
        if lat is None or lon is None:
            return None

        t_start = _parse_noaa_datetime(p.get("BEGIN_DATE_TIME"))
        t_end = _parse_noaa_datetime(p.get("END_DATE_TIME"))
        if t_start is None:
            return None

        mag = _safe_float(p.get("MAGNITUDE"))

        return Event(
            event_id=f"noaa:{raw.source_id}",
            source_catalog="noaa",
            time_start=t_start,
            time_end=t_end,
            longitude=lon,
            latitude=lat,
            mark=mark,
            magnitude=mag,
            metadata={
                "state": p.get("STATE"),
                "event_type": event_type,
                "magnitude_type": p.get("MAGNITUDE_TYPE"),
                "tor_f_scale": p.get("TOR_F_SCALE"),
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


def _parse_noaa_datetime(s: Any) -> datetime | None:
    """NOAA format: '15-JAN-24 14:30:00' (UTC by convention in the source)."""
    if not s:
        return None
    try:
        dt = datetime.strptime(str(s), "%d-%b-%y %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC)
