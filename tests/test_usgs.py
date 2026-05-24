"""USGS ComCat fetcher + harmonization tests."""

from datetime import UTC, datetime

import httpx
import respx

from eonet_cascades.data.schema import Mark
from eonet_cascades.data.usgs import USGSFetcher


def test_harmonize_real_fixture(usgs_payload):
    fetcher = USGSFetcher(rate_per_sec=100.0)
    raws = list(fetcher._iter_raw_from_payload(usgs_payload))
    assert len(raws) > 0
    for r in raws:
        ev = fetcher.harmonize(r)
        assert ev is not None
        assert ev.mark == Mark.EARTHQUAKE
        assert ev.event_id.startswith("usgs:")
        assert ev.magnitude is not None


def test_harmonize_minimal_feature():
    fetcher = USGSFetcher(rate_per_sec=100.0)
    payload = {
        "features": [
            {
                "id": "us6000abcd",
                "type": "Feature",
                "properties": {
                    "mag": 5.2,
                    "place": "10km W of Test",
                    "time": 1704067200000,  # 2024-01-01T00:00:00Z in ms
                    "type": "earthquake",
                },
                "geometry": {"type": "Point", "coordinates": [-120.0, 35.0, 10.0]},
            }
        ]
    }
    raws = list(fetcher._iter_raw_from_payload(payload))
    ev = fetcher.harmonize(raws[0])
    assert ev is not None
    assert ev.event_id == "usgs:us6000abcd"
    assert ev.magnitude == 5.2
    assert (ev.longitude, ev.latitude) == (-120.0, 35.0)
    assert ev.time_start == datetime(2024, 1, 1, tzinfo=UTC)


@respx.mock
def test_fetch_passes_bbox():
    route = respx.get("https://earthquake.usgs.gov/fdsnws/event/1/query").mock(
        return_value=httpx.Response(200, json={"features": []})
    )
    fetcher = USGSFetcher(rate_per_sec=100.0)
    list(
        fetcher.fetch(
            since=datetime(2024, 1, 1, tzinfo=UTC),
            until=datetime(2024, 1, 8, tzinfo=UTC),
            bbox=(-130.0, 14.0, -65.0, 50.0),
        )
    )
    assert route.called
    qs = dict(route.calls[0].request.url.params)
    assert qs["minlongitude"] == "-130.0"
    assert qs["maxlongitude"] == "-65.0"
