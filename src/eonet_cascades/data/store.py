"""DuckDB-backed event store."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from eonet_cascades.data.schema import Event

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id        VARCHAR PRIMARY KEY,
    source_catalog  VARCHAR NOT NULL,
    time_start      TIMESTAMP WITH TIME ZONE NOT NULL,
    time_end        TIMESTAMP WITH TIME ZONE,
    longitude       DOUBLE NOT NULL,
    latitude        DOUBLE NOT NULL,
    mark            VARCHAR NOT NULL,
    magnitude       DOUBLE,
    metadata_json   VARCHAR NOT NULL,
    ingested_at     TIMESTAMP WITH TIME ZONE NOT NULL,
    dedup_group_id  VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(time_start);
CREATE INDEX IF NOT EXISTS idx_events_mark ON events(mark);
CREATE INDEX IF NOT EXISTS idx_events_catalog ON events(source_catalog);
"""


class EventStore:
    """Thin wrapper over DuckDB with the unified Event schema."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path), read_only=read_only)

    def init_schema(self) -> None:
        if self.read_only:
            return  # nothing to do — DDL is a write
        self._conn.execute(_DDL)

    def write_events(self, events: Iterable[Event]) -> int:
        rows = [_event_to_row(e) for e in events]
        if not rows:
            return 0
        df = _rows_to_polars(rows)
        # Idempotent insert: ON CONFLICT DO NOTHING by primary key.
        self._conn.register("incoming", df.to_arrow())
        try:
            self._conn.execute(
                "INSERT INTO events SELECT * FROM incoming ON CONFLICT(event_id) DO NOTHING"
            )
        except duckdb.ProgrammingError:
            # Fallback for older DuckDB that may not support ON CONFLICT syntax.
            self._conn.execute(
                "INSERT INTO events SELECT * FROM incoming "
                "WHERE event_id NOT IN (SELECT event_id FROM events)"
            )
        self._conn.unregister("incoming")
        return len(rows)

    def count_events(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def query_events(
        self,
        *,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        marks: list[str] | None = None,
        source_catalogs: list[str] | None = None,
    ) -> pl.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if time_start is not None:
            clauses.append("time_start >= ?")
            params.append(time_start)
        if time_end is not None:
            clauses.append("time_start <= ?")
            params.append(time_end)
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            clauses.append("longitude BETWEEN ? AND ?")
            params.extend([min_lon, max_lon])
            clauses.append("latitude BETWEEN ? AND ?")
            params.extend([min_lat, max_lat])
        if marks:
            clauses.append(f"mark IN ({','.join(['?'] * len(marks))})")
            params.extend(marks)
        if source_catalogs:
            clauses.append(f"source_catalog IN ({','.join(['?'] * len(source_catalogs))})")
            params.extend(source_catalogs)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM events {where} ORDER BY time_start"
        return self._conn.execute(sql, params).pl()

    def query_sql(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyRelation:
        return self._conn.execute(sql, params or [])

    def close(self) -> None:
        self._conn.close()


_EXPLICIT_SCHEMA: dict[str, pl.DataType] = {
    "event_id": pl.Utf8,
    "source_catalog": pl.Utf8,
    "time_start": pl.Datetime("us", "UTC"),
    "time_end": pl.Datetime("us", "UTC"),
    "longitude": pl.Float64,
    "latitude": pl.Float64,
    "mark": pl.Utf8,
    "magnitude": pl.Float64,
    "metadata_json": pl.Utf8,
    "ingested_at": pl.Datetime("us", "UTC"),
    "dedup_group_id": pl.Utf8,
}


def _rows_to_polars(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a Polars DataFrame with an explicit schema.

    Inferring from data is unsafe — when early rows have None magnitudes,
    Polars infers `pl.Null` and later rows with floats trigger a ComputeError.
    The explicit schema also keeps datetime columns timezone-aware so DuckDB
    accepts them into TIMESTAMP WITH TIME ZONE columns.
    """
    return pl.DataFrame(rows, schema=_EXPLICIT_SCHEMA, strict=False)


def _event_to_row(e: Event) -> dict[str, Any]:
    return {
        "event_id": e.event_id,
        "source_catalog": e.source_catalog,
        "time_start": e.time_start,
        "time_end": e.time_end,
        "longitude": e.longitude,
        "latitude": e.latitude,
        "mark": e.mark.value,
        "magnitude": e.magnitude,
        "metadata_json": json.dumps(e.metadata, default=str),
        "ingested_at": e.ingested_at,
        "dedup_group_id": e.dedup_group_id,
    }
