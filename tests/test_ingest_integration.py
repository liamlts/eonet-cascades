"""End-to-end ingest pipeline integration test."""

import gzip
import json
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from eonet_cascades.cli import app
from eonet_cascades.data.store import EventStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.slow
@respx.mock
def test_end_to_end_ingest(tmp_path, monkeypatch):
    # 1. Point config at tmp_path
    monkeypatch.setenv("EONET_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("EONET_FIRMS_API_KEY", "dummy")

    # 2. Mock all four upstream endpoints
    eonet_payload = json.loads((FIXTURE_DIR / "eonet_sample.json").read_text())
    respx.get("https://eonet.gsfc.nasa.gov/api/v3/events").mock(
        return_value=httpx.Response(200, json=eonet_payload)
    )

    usgs_payload = json.loads((FIXTURE_DIR / "usgs_sample.geojson").read_text())
    respx.get("https://earthquake.usgs.gov/fdsnws/event/1/query").mock(
        return_value=httpx.Response(200, json=usgs_payload)
    )

    # NOAA: stub index page + per-year file
    respx.get("https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/").mock(
        return_value=httpx.Response(200, text="StormEvents_details-ftp_v1.0_d2024_c20250101.csv.gz")
    )
    csv_bytes = (FIXTURE_DIR / "noaa_sample.csv").read_bytes()
    respx.get(url__regex=r".*StormEvents_details-ftp_v1\.0_d2024_c\d+\.csv\.gz").mock(
        return_value=httpx.Response(200, content=gzip.compress(csv_bytes))
    )

    # FIRMS
    respx.get(url__regex=r"https://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/.*").mock(
        return_value=httpx.Response(200, text=(FIXTURE_DIR / "firms_sample.csv").read_text())
    )

    # 3. Run the CLI
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ingest",
            "--catalogs", "eonet,usgs,noaa,firms",
            "--since", "2024-01-01",
            "--until", "2024-02-15",
        ],
    )
    if result.exit_code != 0:
        print(result.stdout)
        print(result.exception)
    assert result.exit_code == 0

    # 4. Verify each catalog wrote at least one event
    store = EventStore(tmp_path / "events.duckdb")
    store.init_schema()
    for cat in ["eonet", "usgs", "noaa", "firms"]:
        df = store.query_events(source_catalogs=[cat])
        assert df.height > 0, f"no events written for {cat}"

    # 5. Verify dedup column is populated (every event gets a group id under new semantics)
    df = store.query_events()
    assert "dedup_group_id" in df.columns
    store.close()
