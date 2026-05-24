"""Per-catalog → unified mark mapping registry."""

from __future__ import annotations

from eonet_cascades.data.schema import Mark

# Each catalog defines its own native category strings; we map them to the
# unified Mark vocab. Keys are lowercased; the lookup normalizes input.
_REGISTRY: dict[str, dict[str, Mark]] = {
    "eonet": {
        "wildfires": Mark.WILDFIRE,
        "severestorms": Mark.SEVERE_STORM,
        "volcanoes": Mark.VOLCANIC_ERUPTION,
        "sealakeice": Mark.SEA_LAKE_ICE,
        "earthquakes": Mark.EARTHQUAKE,
        "floods": Mark.FLOOD,
        "landslides": Mark.LANDSLIDE,
        "drought": Mark.DROUGHT,
        "dusthaze": Mark.DUST_HAZE,
        "tempextremes": Mark.TEMPERATURE_EXTREME,
    },
    "usgs": {
        "earthquake": Mark.EARTHQUAKE,
    },
    "noaa": {
        # NOAA Storm Events EVENT_TYPE values (subset relevant to our vocab)
        "tornado": Mark.TORNADO,
        "hurricane": Mark.TROPICAL_CYCLONE,
        "hurricane (typhoon)": Mark.TROPICAL_CYCLONE,
        "tropical storm": Mark.TROPICAL_CYCLONE,
        "tropical depression": Mark.TROPICAL_CYCLONE,
        "flood": Mark.FLOOD,
        "flash flood": Mark.FLOOD,
        "coastal flood": Mark.FLOOD,
        "thunderstorm wind": Mark.SEVERE_STORM,
        "hail": Mark.SEVERE_STORM,
        "high wind": Mark.SEVERE_STORM,
        "winter storm": Mark.SEVERE_STORM,
        "blizzard": Mark.SEVERE_STORM,
        "drought": Mark.DROUGHT,
        "excessive heat": Mark.TEMPERATURE_EXTREME,
        "heat": Mark.TEMPERATURE_EXTREME,
        "cold/wind chill": Mark.TEMPERATURE_EXTREME,
        "extreme cold/wind chill": Mark.TEMPERATURE_EXTREME,
        "wildfire": Mark.WILDFIRE,
        "debris flow": Mark.LANDSLIDE,
        "dust storm": Mark.DUST_HAZE,
        "dust devil": Mark.DUST_HAZE,
        "lake-effect snow": Mark.SEA_LAKE_ICE,
    },
    "firms": {
        "active_fire": Mark.WILDFIRE,
    },
}


def harmonize_mark(catalog: str, native: str) -> Mark | None:
    """Return the unified Mark for a catalog-native category, or None if unknown."""
    mapping = _REGISTRY.get(catalog.lower())
    if mapping is None:
        return None
    return mapping.get(native.lower())
