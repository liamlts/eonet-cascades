"""Mark harmonization registry tests."""

from eonet_cascades.data.marks import harmonize_mark
from eonet_cascades.data.schema import Mark


def test_eonet_wildfire_maps():
    assert harmonize_mark("eonet", "wildfires") == Mark.WILDFIRE
    assert harmonize_mark("eonet", "Wildfires") == Mark.WILDFIRE  # case-insensitive


def test_eonet_severe_storms_maps():
    assert harmonize_mark("eonet", "severeStorms") == Mark.SEVERE_STORM


def test_usgs_earthquake_maps():
    assert harmonize_mark("usgs", "earthquake") == Mark.EARTHQUAKE


def test_noaa_tornado_maps():
    assert harmonize_mark("noaa", "Tornado") == Mark.TORNADO


def test_noaa_hurricane_maps_to_tropical_cyclone():
    assert harmonize_mark("noaa", "Hurricane") == Mark.TROPICAL_CYCLONE
    assert harmonize_mark("noaa", "Tropical Storm") == Mark.TROPICAL_CYCLONE


def test_firms_active_fire_maps():
    assert harmonize_mark("firms", "active_fire") == Mark.WILDFIRE


def test_unknown_mark_returns_none():
    assert harmonize_mark("eonet", "unicorn_uprising") is None
    assert harmonize_mark("unknown_catalog", "wildfires") is None


def test_all_unified_marks_have_at_least_one_source_mapping():
    # Every Mark in the v1 vocab should be reachable from at least one catalog.
    from eonet_cascades.data.marks import _REGISTRY

    reached: set[Mark] = set()
    for mapping in _REGISTRY.values():
        reached.update(mapping.values())
    missing = set(Mark) - reached
    assert not missing, f"unreachable marks: {missing}"
