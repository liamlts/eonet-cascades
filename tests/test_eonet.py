"""EONET fetcher + harmonization tests."""

from datetime import UTC, datetime

import httpx
import respx

from eonet_cascades.data.eonet import EONETFetcher
from eonet_cascades.data.schema import Mark


def test_harmonize_skips_unknown_category(eonet_payload):
    fetcher = EONETFetcher(rate_per_sec=100.0)
    raw_events = list(fetcher._iter_raw_from_payload(eonet_payload))
    assert len(raw_events) > 0
    # Every harmonized result is either a valid Event or None (unknown category).
    for r in raw_events:
        result = fetcher.harmonize(r)
        if result is not None:
            assert result.mark in set(Mark)
            assert result.source_catalog == "eonet"
            assert result.event_id.startswith("eonet:")


def test_harmonize_handles_point_geometry():
    fetcher = EONETFetcher(rate_per_sec=100.0)
    raw_payload = {
        "events": [
            {
                "id": "EONET_TEST_1",
                "title": "Test Wildfire",
                "categories": [{"id": "wildfires", "title": "Wildfires"}],
                "geometry": [
                    {
                        "date": "2024-06-15T12:00:00Z",
                        "type": "Point",
                        "coordinates": [-120.5, 38.2],
                    }
                ],
                "sources": [],
            }
        ]
    }
    raws = list(fetcher._iter_raw_from_payload(raw_payload))
    assert len(raws) == 1
    ev = fetcher.harmonize(raws[0])
    assert ev is not None
    assert ev.mark == Mark.WILDFIRE
    assert ev.longitude == -120.5
    assert ev.latitude == 38.2
    assert ev.time_start == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)


def test_harmonize_uses_first_geometry_point_if_multiple():
    fetcher = EONETFetcher(rate_per_sec=100.0)
    raw_payload = {
        "events": [
            {
                "id": "EONET_TEST_2",
                "title": "Hurricane Track",
                "categories": [{"id": "severeStorms", "title": "Severe Storms"}],
                "geometry": [
                    {
                        "date": "2024-09-01T00:00:00Z",
                        "type": "Point",
                        "coordinates": [-75.0, 25.0],
                    },
                    {
                        "date": "2024-09-02T00:00:00Z",
                        "type": "Point",
                        "coordinates": [-76.0, 26.0],
                    },
                ],
                "sources": [],
            }
        ]
    }
    raws = list(fetcher._iter_raw_from_payload(raw_payload))
    ev = fetcher.harmonize(raws[0])
    assert ev is not None
    assert ev.mark == Mark.SEVERE_STORM
    assert (ev.longitude, ev.latitude) == (-75.0, 25.0)
    assert ev.time_start == datetime(2024, 9, 1, tzinfo=UTC)
    assert ev.time_end == datetime(2024, 9, 2, tzinfo=UTC)


@respx.mock
def test_fetch_hits_eonet_endpoint():
    respx.get("https://eonet.gsfc.nasa.gov/api/v3/events").mock(
        return_value=httpx.Response(200, json={"events": []})
    )
    fetcher = EONETFetcher(rate_per_sec=100.0)
    result = list(
        fetcher.fetch(
            since=datetime(2024, 1, 1, tzinfo=UTC),
            until=datetime(2024, 2, 1, tzinfo=UTC),
        )
    )
    assert result == []
