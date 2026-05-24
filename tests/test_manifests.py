"""Manifest state I/O tests."""

from datetime import UTC, datetime

from eonet_cascades.data.manifests import ManifestStore


def test_read_missing_manifest_returns_none(tmp_path):
    store = ManifestStore(tmp_path)
    assert store.last_fetched("eonet") is None


def test_write_and_read_roundtrip(tmp_path):
    store = ManifestStore(tmp_path)
    ts = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
    store.set_last_fetched("eonet", ts)
    assert store.last_fetched("eonet") == ts


def test_independent_catalogs(tmp_path):
    store = ManifestStore(tmp_path)
    ts1 = datetime(2024, 1, 1, tzinfo=UTC)
    ts2 = datetime(2024, 2, 1, tzinfo=UTC)
    store.set_last_fetched("eonet", ts1)
    store.set_last_fetched("usgs", ts2)
    assert store.last_fetched("eonet") == ts1
    assert store.last_fetched("usgs") == ts2
