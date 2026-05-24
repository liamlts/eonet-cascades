"""Schema invariant tests."""

from datetime import UTC, datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from eonet_cascades.data.schema import Event, Mark, RawEvent

UNIFIED_MARKS = {m.value for m in Mark}


def _make_event(**overrides):
    base = dict(
        event_id="eonet:EONET_12345",
        source_catalog="eonet",
        time_start=datetime(2024, 1, 1, tzinfo=UTC),
        time_end=None,
        longitude=-100.0,
        latitude=35.0,
        mark="wildfire",
        magnitude=None,
        metadata={"foo": "bar"},
        ingested_at=datetime(2024, 6, 1, tzinfo=UTC),
        dedup_group_id=None,
    )
    base.update(overrides)
    return Event(**base)


def test_event_minimal_fields_ok():
    ev = _make_event()
    assert ev.mark == Mark.WILDFIRE


def test_mark_must_be_in_vocab():
    with pytest.raises(ValidationError):
        _make_event(mark="not_a_real_mark")


def test_timestamps_must_be_utc():
    with pytest.raises(ValidationError, match="timezone"):
        _make_event(time_start=datetime(2024, 1, 1))  # naive


def test_longitude_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _make_event(longitude=200.0)


def test_latitude_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _make_event(latitude=-95.0)


def test_time_end_before_time_start_rejected():
    with pytest.raises(ValidationError, match="time_end"):
        _make_event(
            time_start=datetime(2024, 1, 10, tzinfo=UTC),
            time_end=datetime(2024, 1, 1, tzinfo=UTC),
        )


def test_event_id_must_be_namespaced():
    with pytest.raises(ValidationError, match="event_id"):
        _make_event(event_id="not-namespaced")


@given(
    lon=st.floats(min_value=-180, max_value=180, allow_nan=False),
    lat=st.floats(min_value=-90, max_value=90, allow_nan=False),
)
def test_event_accepts_any_valid_coords(lon, lat):
    ev = _make_event(longitude=lon, latitude=lat)
    assert -180 <= ev.longitude <= 180
    assert -90 <= ev.latitude <= 90


@given(mark=st.sampled_from(sorted(UNIFIED_MARKS)))
def test_event_accepts_all_vocab_marks(mark):
    ev = _make_event(mark=mark)
    assert ev.mark.value == mark


def test_raw_event_is_freeform_dict():
    raw = RawEvent(source_catalog="eonet", source_id="x", payload={"any": "thing"})
    assert raw.payload["any"] == "thing"
