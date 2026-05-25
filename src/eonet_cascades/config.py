"""Configuration models for the data layer."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    """Data-layer configuration.

    Resolved from environment variables prefixed `EONET_`, then YAML overlay,
    then explicit kwargs. Defaults target CONUS + Mexico on the external drive.
    """

    model_config = SettingsConfigDict(env_prefix="EONET_", extra="ignore")

    data_root: Path = Field(default=Path("/Volumes/Seagate_Ext/eonet-cascades-data"))
    catalogs: list[str] = Field(default_factory=lambda: ["eonet", "usgs", "noaa", "firms"])
    bbox: tuple[float, float, float, float] = Field(
        default=(-130.0, 14.0, -65.0, 50.0),
        description="(min_lon, min_lat, max_lon, max_lat) — CONUS + Mexico default.",
    )
    start_date: date = Field(default=date(2000, 1, 1))
    firms_api_key: str | None = Field(default=None)

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError(f"invalid longitude range: {min_lon}..{max_lon}")
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError(f"invalid latitude range: {min_lat}..{max_lat}")
        return v

    @property
    def duckdb_path(self) -> Path:
        return self.data_root / "events.duckdb"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def manifests_dir(self) -> Path:
        return self.data_root / "manifests"

    def ensure_exists(self) -> None:
        """Verify the data root exists (drive mounted). Raise if not."""
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"data_root does not exist: {self.data_root}. Is the external drive mounted?"
            )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)


def load_data_config(yaml_path: Path) -> DataConfig:
    """Load `DataConfig` from a YAML file, applying env-var overlay on top."""
    with yaml_path.open() as f:
        overlay = yaml.safe_load(f) or {}
    return DataConfig(**overlay)
