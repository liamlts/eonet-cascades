"""Unified Event schema and raw fetch envelope."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Mark(StrEnum):
    """Unified mark vocabulary, fixed for v1."""

    WILDFIRE = "wildfire"
    SEVERE_STORM = "severe_storm"
    TROPICAL_CYCLONE = "tropical_cyclone"
    TORNADO = "tornado"
    FLOOD = "flood"
    EARTHQUAKE = "earthquake"
    VOLCANIC_ERUPTION = "volcanic_eruption"
    LANDSLIDE = "landslide"
    DROUGHT = "drought"
    DUST_HAZE = "dust_haze"
    TEMPERATURE_EXTREME = "temperature_extreme"
    SEA_LAKE_ICE = "sea_lake_ice"


class Event(BaseModel):
    """A harmonized event ready for modeling.

    All timestamps are timezone-aware UTC. Coordinates are in WGS84.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(description='"{catalog}:{native_id}"')
    source_catalog: str
    time_start: datetime
    time_end: datetime | None = None
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    mark: Mark
    magnitude: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime
    dedup_group_id: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> Event:
        if ":" not in self.event_id:
            raise ValueError(
                f"event_id must be '{{catalog}}:{{native_id}}', got: {self.event_id!r}"
            )
        if self.time_start.tzinfo is None:
            raise ValueError("time_start must be timezone-aware (UTC)")
        if self.time_end is not None:
            if self.time_end.tzinfo is None:
                raise ValueError("time_end must be timezone-aware (UTC)")
            if self.time_end < self.time_start:
                raise ValueError("time_end must be >= time_start")
        if self.ingested_at.tzinfo is None:
            raise ValueError("ingested_at must be timezone-aware (UTC)")
        return self


class RawEvent(BaseModel):
    """Freeform envelope for catalog responses prior to harmonization."""

    model_config = ConfigDict(extra="forbid")

    source_catalog: str
    source_id: str
    payload: dict[str, Any]
