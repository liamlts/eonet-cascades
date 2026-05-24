"""NOAA Storm Events fetcher + harmonization tests."""

from datetime import UTC, datetime
from pathlib import Path

from eonet_cascades.data.noaa_storms import NOAAStormsFetcher
from eonet_cascades.data.schema import Mark, RawEvent

FIXTURE = Path(__file__).parent / "fixtures" / "noaa_sample.csv"


def test_harmonize_tornado():
    fetcher = NOAAStormsFetcher()
    raws = list(fetcher._iter_raw_from_csv(FIXTURE))
    by_id = {r.source_id: r for r in raws}
    ev = fetcher.harmonize(by_id["1000001"])
    assert ev is not None
    assert ev.mark == Mark.TORNADO
    assert ev.longitude == -97.2
    assert ev.latitude == 30.5
    assert ev.time_start == datetime(2024, 1, 15, 14, 30, tzinfo=UTC)


def test_harmonize_hurricane_maps_to_tropical_cyclone():
    fetcher = NOAAStormsFetcher()
    raws = list(fetcher._iter_raw_from_csv(FIXTURE))
    by_id = {r.source_id: r for r in raws}
    ev = fetcher.harmonize(by_id["1000002"])
    assert ev is not None
    assert ev.mark == Mark.TROPICAL_CYCLONE
    assert ev.magnitude == 95.0


def test_harmonize_flash_flood_maps_to_flood():
    fetcher = NOAAStormsFetcher()
    raws = list(fetcher._iter_raw_from_csv(FIXTURE))
    by_id = {r.source_id: r for r in raws}
    ev = fetcher.harmonize(by_id["1000003"])
    assert ev is not None
    assert ev.mark == Mark.FLOOD


def test_harmonize_skips_missing_coords():
    fetcher = NOAAStormsFetcher()
    raws = list(fetcher._iter_raw_from_csv(FIXTURE))
    by_id = {r.source_id: r for r in raws}
    ev = fetcher.harmonize(by_id["1000006"])  # has no BEGIN_LAT/LON
    assert ev is None


def test_harmonize_unknown_type_returns_none():
    fetcher = NOAAStormsFetcher()
    raw = RawEvent(
        source_catalog="noaa",
        source_id="999999",
        payload={
            "EVENT_ID": "999999",
            "EVENT_TYPE": "Astronaut Re-entry",
            "BEGIN_DATE_TIME": "01-JAN-24 00:00:00",
            "END_DATE_TIME": "01-JAN-24 01:00:00",
            "BEGIN_LAT": "30.0",
            "BEGIN_LON": "-90.0",
            "MAGNITUDE": "",
        },
    )
    assert fetcher.harmonize(raw) is None
