"""NASA FIRMS fetcher + harmonization tests."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import respx

from eonet_cascades.data.firms import FIRMSFetcher
from eonet_cascades.data.schema import Mark

FIXTURE = Path(__file__).parent / "fixtures" / "firms_sample.csv"


def test_harmonize_fixture_rows():
    fetcher = FIRMSFetcher(api_key="dummy", rate_per_sec=100.0)
    raws = list(fetcher._iter_raw_from_csv(FIXTURE.read_text()))
    assert len(raws) == 3
    events = [fetcher.harmonize(r) for r in raws]
    assert all(e is not None for e in events)
    assert all(e.mark == Mark.WILDFIRE for e in events)
    assert events[0].longitude == -118.234
    assert events[0].latitude == 34.123
    assert events[0].time_start == datetime(2024, 6, 15, 18, 42, tzinfo=UTC)
    assert events[0].magnitude == 12.5  # FRP


def test_harmonize_low_confidence_dropped():
    fetcher = FIRMSFetcher(api_key="dummy", rate_per_sec=100.0, min_confidence="n")
    raws = list(fetcher._iter_raw_from_csv(FIXTURE.read_text()))
    # All sample rows are 'n' or 'h'; with min_confidence='n' all pass.
    events = [fetcher.harmonize(r) for r in raws]
    assert sum(1 for e in events if e is not None) == 3

    fetcher_strict = FIRMSFetcher(api_key="dummy", rate_per_sec=100.0, min_confidence="h")
    events_strict = [fetcher_strict.harmonize(r) for r in raws]
    assert sum(1 for e in events_strict if e is not None) == 1


@respx.mock
def test_fetch_passes_api_key_and_bbox():
    route = respx.get(url__regex=r"https://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/.*").mock(
        return_value=httpx.Response(200, text="latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n")
    )
    fetcher = FIRMSFetcher(api_key="MAP_KEY_123", rate_per_sec=100.0)
    list(
        fetcher.fetch(
            since=datetime(2024, 6, 15, tzinfo=UTC),
            until=datetime(2024, 6, 16, tzinfo=UTC),
            bbox=(-130.0, 14.0, -65.0, 50.0),
        )
    )
    assert route.called
    url = str(route.calls[0].request.url)
    assert "MAP_KEY_123" in url
