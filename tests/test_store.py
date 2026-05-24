"""DuckDB store roundtrip tests."""

from datetime import UTC, datetime, timedelta

import pytest

from eonet_cascades.data.schema import Event, Mark
from eonet_cascades.data.store import EventStore


def _event(i: int) -> Event:
    return Event(
        event_id=f"eonet:E{i}",
        source_catalog="eonet",
        time_start=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
        time_end=None,
        longitude=-100.0 + i * 0.01,
        latitude=35.0 + i * 0.01,
        mark=Mark.WILDFIRE,
        magnitude=float(i),
        metadata={"i": i},
        ingested_at=datetime(2024, 6, 1, tzinfo=UTC),
        dedup_group_id=None,
    )


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "events.duckdb")
    s.init_schema()
    yield s
    s.close()


def test_init_schema_creates_events_table(store):
    rows = store.query_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
    assert ("events",) in rows


def test_write_and_count(store):
    events = [_event(i) for i in range(5)]
    store.write_events(events)
    assert store.count_events() == 5


def test_write_is_idempotent_on_event_id(store):
    events = [_event(i) for i in range(3)]
    store.write_events(events)
    store.write_events(events)  # second write must not duplicate
    assert store.count_events() == 3


def test_query_by_time_range(store):
    store.write_events([_event(i) for i in range(10)])
    df = store.query_events(
        time_start=datetime(2024, 1, 3, tzinfo=UTC),
        time_end=datetime(2024, 1, 6, tzinfo=UTC),
    )
    assert len(df) == 4  # days 2..5 inclusive


def test_query_by_bbox(store):
    store.write_events([_event(i) for i in range(10)])
    df = store.query_events(bbox=(-100.05, 35.0, -100.0, 35.05))
    assert 0 < len(df) <= 10


def test_query_by_mark(store):
    store.write_events([_event(i) for i in range(3)])
    df_wildfire = store.query_events(marks=["wildfire"])
    df_quake = store.query_events(marks=["earthquake"])
    assert len(df_wildfire) == 3
    assert len(df_quake) == 0
