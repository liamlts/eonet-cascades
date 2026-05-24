"""Cross-catalog deduplication tests."""

from datetime import UTC, datetime, timedelta

from eonet_cascades.data.dedup import DEFAULT_THRESHOLDS, assign_dedup_groups
from eonet_cascades.data.schema import Event, Mark


def _ev(eid: str, catalog: str, mark: Mark, t: datetime, lon: float, lat: float) -> Event:
    return Event(
        event_id=f"{catalog}:{eid}",
        source_catalog=catalog,
        time_start=t,
        time_end=None,
        longitude=lon,
        latitude=lat,
        mark=mark,
        magnitude=None,
        metadata={},
        ingested_at=datetime(2024, 6, 1, tzinfo=UTC),
        dedup_group_id=None,
    )


def test_same_event_in_two_catalogs_collapses():
    t = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)
    events = [
        _ev("E1", "eonet", Mark.TROPICAL_CYCLONE, t, -75.0, 25.0),
        _ev("S1", "noaa", Mark.TROPICAL_CYCLONE, t + timedelta(hours=2), -75.5, 25.2),
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id is not None
    assert out[0].dedup_group_id == out[1].dedup_group_id


def test_distant_events_get_distinct_groups():
    t = datetime(2024, 9, 1, tzinfo=UTC)
    events = [
        _ev("E1", "eonet", Mark.WILDFIRE, t, -120.0, 35.0),
        _ev("F1", "firms", Mark.WILDFIRE, t, -80.0, 35.0),  # 4000+ km away
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id != out[1].dedup_group_id


def test_different_marks_never_collapse():
    t = datetime(2024, 9, 1, tzinfo=UTC)
    events = [
        _ev("E1", "eonet", Mark.WILDFIRE, t, -120.0, 35.0),
        _ev("E2", "eonet", Mark.EARTHQUAKE, t, -120.0, 35.0),  # same place + time, different mark
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id != out[1].dedup_group_id


def test_earthquake_threshold_is_tight():
    t = datetime(2024, 9, 1, tzinfo=UTC)
    events = [
        # 10 km apart — outside the 5 km earthquake threshold
        _ev("U1", "usgs", Mark.EARTHQUAKE, t, -120.0, 35.0),
        _ev("U2", "usgs", Mark.EARTHQUAKE, t + timedelta(minutes=30), -120.0, 35.09),
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id != out[1].dedup_group_id


def test_drought_threshold_is_loose():
    t = datetime(2024, 9, 1, tzinfo=UTC)
    events = [
        # 80 km / 3 days apart — within the loose drought threshold
        _ev("E1", "eonet", Mark.DROUGHT, t, -100.0, 35.0),
        _ev("E2", "eonet", Mark.DROUGHT, t + timedelta(days=3), -100.0, 35.8),
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id == out[1].dedup_group_id


def test_default_thresholds_cover_all_marks():
    missing = set(Mark) - set(DEFAULT_THRESHOLDS.keys())
    assert not missing, f"thresholds missing for: {missing}"


def test_empty_input_returns_empty():
    assert assign_dedup_groups([]) == []
