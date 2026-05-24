# Bootstrap and Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the project skeleton and a working multi-catalog ingestion pipeline that fetches, harmonizes, and dedupes events from EONET, USGS ComCat, NOAA Storm Events, and NASA FIRMS into a queryable DuckDB store on an external drive — meeting the dataset smell-test gate that unlocks modeling work.

**Architecture:** Pluggable per-catalog fetchers that produce a unified `Event` schema, written to Parquet (raw) and DuckDB (harmonized). A shared rate-limited HTTP client and idempotent ingestion driver. All paths configurable via YAML; data lives on `/Volumes/Seagate_Ext/eonet-cascades-data/` by default. Strict layering: `data/` depends only on the schema; nothing else exists yet at this phase.

**Tech Stack:** Python 3.11+, uv, PyTorch (declared but not used here), pydantic 2, pydantic-settings, DuckDB, polars, pyarrow, httpx, Typer + rich, pytest, hypothesis, ruff.

---

## File Structure

Files this plan creates or modifies, with responsibilities:

```
~/Projects/eonet-cascades/
├── pyproject.toml                        # uv project, deps, ruff + pytest config
├── uv.lock                                # locked dep tree (auto-managed)
├── .gitignore
├── .github/workflows/ci.yml               # ruff + pytest on push
├── Makefile                               # convenience targets
├── README.md                              # what + how-to-run + headline (stub here)
├── configs/
│   └── data/conus.yaml                    # default data config (paths, catalogs, region)
├── src/eonet_cascades/
│   ├── __init__.py                        # version
│   ├── cli.py                             # Typer entrypoint registered as `eonet`
│   ├── config.py                          # pydantic-settings models for data config
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py                      # pydantic Event + RawEvent + Mark enum
│   │   ├── store.py                       # DuckDB read/write layer
│   │   ├── http.py                        # rate-limited httpx client
│   │   ├── manifests.py                   # per-catalog last-fetched state I/O
│   │   ├── marks.py                       # per-catalog → unified mark mapping
│   │   ├── dedup.py                       # cross-catalog spatio-temporal clustering
│   │   ├── base.py                        # CatalogFetcher protocol
│   │   ├── eonet.py                       # NASA EONET fetcher
│   │   ├── usgs.py                        # USGS ComCat fetcher
│   │   ├── noaa_storms.py                 # NOAA Storm Events fetcher
│   │   ├── firms.py                       # NASA FIRMS fetcher
│   │   └── ingest.py                      # orchestrator: drive all fetchers + dedup
├── tests/
│   ├── __init__.py
│   ├── conftest.py                        # fixtures: tmp data dir, mock httpx
│   ├── fixtures/
│   │   ├── eonet_sample.json
│   │   ├── usgs_sample.geojson
│   │   ├── noaa_sample.csv
│   │   └── firms_sample.csv
│   ├── test_schema.py                     # pydantic + hypothesis invariants
│   ├── test_store.py                      # DuckDB roundtrip
│   ├── test_marks.py                      # mark vocab coverage
│   ├── test_eonet.py                      # EONET fetcher + harmonization
│   ├── test_usgs.py                       # USGS fetcher + harmonization
│   ├── test_noaa.py                       # NOAA fetcher + harmonization
│   ├── test_firms.py                      # FIRMS fetcher + harmonization
│   ├── test_dedup.py                      # dedup synthetic clusters
│   └── test_ingest_integration.py         # end-to-end small-window ingest
├── notebooks/
│   └── 01_data_exploration.ipynb          # dataset smell-test deliverable
└── data/                                  # symlink → /Volumes/Seagate_Ext/eonet-cascades-data/
```

---

## Phase 0 — Bootstrap

### Task 1: Initialize uv project with pyproject.toml and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/eonet_cascades/__init__.py`

- [ ] **Step 1: Confirm working directory and uv availability**

```bash
cd ~/Projects/eonet-cascades
which uv || (echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)
uv --version
```

Expected: a version string like `uv 0.x.y`.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "eonet-cascades"
version = "0.0.1"
description = "Spatio-temporal point process benchmark for natural-hazard event cascades"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "duckdb>=1.0",
    "polars>=1.0",
    "pyarrow>=15",
    "httpx>=0.27",
    "typer>=0.12",
    "rich>=13",
    "pyyaml>=6",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "hypothesis>=6",
    "ruff>=0.5",
    "respx>=0.21",
    "ipykernel>=6",
]
ml = [
    "torch>=2.2",
    "matplotlib>=3.8",
    "scienceplots>=2",
    "cartopy>=0.23",
    "shapely>=2",
    "pyproj>=3.6",
]

[project.scripts]
eonet = "eonet_cascades.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/eonet_cascades"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "RUF"]
ignore = ["E501"]  # long lines OK in test fixtures

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
markers = [
    "slow: marks slow integration tests (deselect with -m 'not slow')",
    "network: marks tests that hit real network (deselect with -m 'not network')",
]
```

- [ ] **Step 3: Write `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/

# Project
data/
runs/
*.duckdb
*.duckdb.wal
.env

# OS
.DS_Store
```

- [ ] **Step 4: Write `src/eonet_cascades/__init__.py`**

```python
"""EONET natural-hazard cascade benchmark."""

__version__ = "0.0.1"
```

- [ ] **Step 5: Install and verify the package builds**

```bash
cd ~/Projects/eonet-cascades
uv sync --extra dev
uv run python -c "import eonet_cascades; print(eonet_cascades.__version__)"
```

Expected: `0.0.1`.

- [ ] **Step 6: Commit**

```bash
cd ~/Projects/eonet-cascades
git add pyproject.toml .gitignore src/eonet_cascades/__init__.py uv.lock
git commit -m "chore: initialize uv project with deps and ruff/pytest config"
```

---

### Task 2: Wire up the Typer CLI entrypoint

**Files:**
- Create: `src/eonet_cascades/cli.py`
- Test: `tests/__init__.py`, `tests/test_cli.py`

- [ ] **Step 1: Create empty test package and write the CLI smoke test**

```bash
cd ~/Projects/eonet-cascades
mkdir -p tests
touch tests/__init__.py
```

Write `tests/test_cli.py`:

```python
"""CLI smoke tests."""

from typer.testing import CliRunner

from eonet_cascades.cli import app

runner = CliRunner()


def test_help_runs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "eonet" in result.stdout.lower()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.0.1" in result.stdout
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd ~/Projects/eonet-cascades
uv run pytest tests/test_cli.py -v
```

Expected: ImportError or ModuleNotFoundError on `eonet_cascades.cli`.

- [ ] **Step 3: Implement minimal `src/eonet_cascades/cli.py`**

```python
"""Top-level Typer CLI for the eonet-cascades project."""

from __future__ import annotations

import typer
from rich.console import Console

from eonet_cascades import __version__

app = typer.Typer(
    name="eonet",
    help="Spatio-temporal point process benchmark for natural-hazard event cascades.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    """Root callback — exists to disable Typer's single-command flattening so
    `eonet --help` shows the app help (not the only subcommand's help) until
    Task 16 adds a second command."""


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
uv run pytest tests/test_cli.py -v
uv run eonet --help
uv run eonet version
```

Expected: both tests pass; `eonet --help` shows the Typer help table; `eonet version` prints `0.0.1`.

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/cli.py tests/__init__.py tests/test_cli.py
git commit -m "feat(cli): add Typer entrypoint with version command"
```

---

### Task 3: Add GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - name: Set up Python
        run: uv python install 3.11
      - name: Install dependencies
        run: uv sync --extra dev
      - name: Lint
        run: uv run ruff check .
      - name: Format check
        run: uv run ruff format --check .
      - name: Test
        run: uv run pytest -m "not slow and not network"
```

- [ ] **Step 2: Run the local equivalent to confirm it passes**

```bash
cd ~/Projects/eonet-cascades
uv run ruff check .
uv run ruff format --check . || uv run ruff format .  # fix any formatting now
uv run pytest -m "not slow and not network"
```

Expected: ruff passes, pytest passes.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
# also commit any ruff format auto-fixes
git add -u
git commit -m "ci: add ruff + pytest workflow"
```

---

### Task 4: Add README stub and Makefile

**Files:**
- Create: `README.md`
- Create: `Makefile`

- [ ] **Step 1: Write `README.md`**

```markdown
# eonet-cascades

Spatio-temporal point process benchmark suite for natural-hazard event cascades over CONUS + Mexico, 2000–present.

Three model tiers — parametric multivariate Hawkes, Neural Hawkes (continuous-time LSTM), and Transformer Hawkes — share a common likelihood interface and evaluation harness. The learned cross-mark triggering structure is the headline interpretable output: a cascade graph of natural hazards.

## Status

Phase 0 + 1 — bootstrap and data layer. See `docs/superpowers/specs/2026-05-24-eonet-cascade-benchmark-design.md` for the full design.

## Quick start

```bash
uv sync --extra dev
uv run eonet --help
uv run pytest
```

## Data location

Raw catalogs and the harmonized DuckDB store live on an external drive by default:

```
/Volumes/Seagate_Ext/eonet-cascades-data/
```

Override with the `EONET_DATA_ROOT` environment variable or `--data-root` CLI flag.

## Reproduce the headline figure

```bash
make headline
```

(Stub until Phase 6.)
```

- [ ] **Step 2: Write `Makefile`**

```makefile
.PHONY: install test lint format headline ingest

install:
	uv sync --extra dev --extra ml

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

ingest:
	uv run eonet ingest --catalogs eonet,usgs,noaa,firms --since 2000-01-01

headline:
	@echo "Stub — headline figure regeneration lands in Phase 6."
```

- [ ] **Step 3: Verify**

```bash
make lint
make test
```

Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add README.md Makefile
git commit -m "docs: add README stub and Makefile targets"
```

---

### Task 5: Set up external data directory and path configuration

**Files:**
- Create: `src/eonet_cascades/config.py`
- Create: `configs/data/conus.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Verify external drive and create the data root**

```bash
test -d /Volumes/Seagate_Ext || (echo "Seagate_Ext not mounted; aborting" && exit 1)
mkdir -p /Volumes/Seagate_Ext/eonet-cascades-data/{raw,manifests}
ls -la /Volumes/Seagate_Ext/eonet-cascades-data/
```

Expected: the `raw/` and `manifests/` subdirectories exist on the external drive.

- [ ] **Step 2: Create the in-repo `data/` symlink (gitignored)**

```bash
cd ~/Projects/eonet-cascades
ln -sf /Volumes/Seagate_Ext/eonet-cascades-data data
ls -la data
```

Expected: `data -> /Volumes/Seagate_Ext/eonet-cascades-data`.

- [ ] **Step 3: Write the test for the config loader**

`tests/test_config.py`:

```python
"""Config loading tests."""

from pathlib import Path

import pytest
import yaml

from eonet_cascades.config import DataConfig, load_data_config


def test_data_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("EONET_DATA_ROOT", str(tmp_path))
    cfg = DataConfig()
    assert cfg.data_root == tmp_path
    assert cfg.duckdb_path == tmp_path / "events.duckdb"
    assert cfg.raw_dir == tmp_path / "raw"
    assert cfg.manifests_dir == tmp_path / "manifests"
    assert "eonet" in cfg.catalogs
    assert cfg.bbox == (-130.0, 14.0, -65.0, 50.0)  # CONUS + Mexico


def test_load_data_config_from_yaml(tmp_path):
    yaml_path = tmp_path / "conus.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(tmp_path / "alt"),
                "catalogs": ["eonet", "usgs"],
                "bbox": [-130.0, 14.0, -65.0, 50.0],
                "start_date": "2000-01-01",
            }
        )
    )
    cfg = load_data_config(yaml_path)
    assert cfg.data_root == tmp_path / "alt"
    assert cfg.catalogs == ["eonet", "usgs"]


def test_data_config_missing_drive_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("EONET_DATA_ROOT", str(tmp_path / "does-not-exist"))
    cfg = DataConfig()
    with pytest.raises(FileNotFoundError, match="data_root does not exist"):
        cfg.ensure_exists()
```

- [ ] **Step 4: Run the test and confirm it fails**

```bash
uv run pytest tests/test_config.py -v
```

Expected: import error on `eonet_cascades.config`.

- [ ] **Step 5: Implement `src/eonet_cascades/config.py`**

```python
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
    def _validate_bbox(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
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
                f"data_root does not exist: {self.data_root}. "
                "Is the external drive mounted?"
            )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)


def load_data_config(yaml_path: Path) -> DataConfig:
    """Load `DataConfig` from a YAML file, applying env-var overlay on top."""
    with yaml_path.open() as f:
        overlay = yaml.safe_load(f) or {}
    return DataConfig(**overlay)
```

- [ ] **Step 6: Write the default config YAML**

`configs/data/conus.yaml`:

```yaml
# Default CONUS + Mexico data configuration.
data_root: /Volumes/Seagate_Ext/eonet-cascades-data
catalogs:
  - eonet
  - usgs
  - noaa
  - firms
bbox: [-130.0, 14.0, -65.0, 50.0]   # min_lon, min_lat, max_lon, max_lat
start_date: 2000-01-01
firms_api_key: null   # required at runtime for the FIRMS fetcher; set via env EONET_FIRMS_API_KEY
```

- [ ] **Step 7: Run tests and confirm they pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: all three tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/eonet_cascades/config.py configs/data/conus.yaml tests/test_config.py
git commit -m "feat(config): add DataConfig with env + YAML resolution"
```

---

## Phase 1 — Data Layer

### Task 6: Define the Event schema

**Files:**
- Create: `src/eonet_cascades/data/__init__.py`
- Create: `src/eonet_cascades/data/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the schema tests (property-based)**

`tests/test_schema.py`:

```python
"""Schema invariant tests."""

from datetime import datetime, timezone

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from eonet_cascades.data.schema import Event, Mark, RawEvent

UNIFIED_MARKS = {m.value for m in Mark}


def _make_event(**overrides):
    base = dict(
        event_id="eonet:EONET_12345",
        source_catalog="eonet",
        time_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        time_end=None,
        longitude=-100.0,
        latitude=35.0,
        mark="wildfire",
        magnitude=None,
        metadata={"foo": "bar"},
        ingested_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        dedup_group_id=None,
    )
    base.update(overrides)
    return Event(**base)


def test_event_minimal_fields_ok():
    ev = _make_event()
    assert ev.mark == Mark.WILDFIRE


def test_mark_must_be_in_vocab():
    with pytest.raises(ValidationError):
        _make_event(mark="not_a_real_mark")


def test_timestamps_must_be_utc():
    with pytest.raises(ValidationError, match="timezone"):
        _make_event(time_start=datetime(2024, 1, 1))  # naive


def test_longitude_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _make_event(longitude=200.0)


def test_latitude_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _make_event(latitude=-95.0)


def test_time_end_before_time_start_rejected():
    with pytest.raises(ValidationError, match="time_end"):
        _make_event(
            time_start=datetime(2024, 1, 10, tzinfo=timezone.utc),
            time_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )


def test_event_id_must_be_namespaced():
    with pytest.raises(ValidationError, match="event_id"):
        _make_event(event_id="not-namespaced")


@given(
    lon=st.floats(min_value=-180, max_value=180, allow_nan=False),
    lat=st.floats(min_value=-90, max_value=90, allow_nan=False),
)
def test_event_accepts_any_valid_coords(lon, lat):
    ev = _make_event(longitude=lon, latitude=lat)
    assert -180 <= ev.longitude <= 180
    assert -90 <= ev.latitude <= 90


@given(mark=st.sampled_from(sorted(UNIFIED_MARKS)))
def test_event_accepts_all_vocab_marks(mark):
    ev = _make_event(mark=mark)
    assert ev.mark.value == mark


def test_raw_event_is_freeform_dict():
    raw = RawEvent(source_catalog="eonet", source_id="x", payload={"any": "thing"})
    assert raw.payload["any"] == "thing"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
mkdir -p src/eonet_cascades/data
touch src/eonet_cascades/data/__init__.py
uv run pytest tests/test_schema.py -v
```

Expected: ImportError on `eonet_cascades.data.schema`.

- [ ] **Step 3: Implement the schema**

`src/eonet_cascades/data/schema.py`:

```python
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
    def _validate(self) -> "Event":
        if ":" not in self.event_id:
            raise ValueError(f"event_id must be '{{catalog}}:{{native_id}}', got: {self.event_id!r}")
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
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
uv run pytest tests/test_schema.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/data/__init__.py src/eonet_cascades/data/schema.py tests/test_schema.py
git commit -m "feat(data): add unified Event schema with strict validation"
```

---

### Task 7: Implement the DuckDB store

**Files:**
- Create: `src/eonet_cascades/data/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the store tests**

`tests/test_store.py`:

```python
"""DuckDB store roundtrip tests."""

from datetime import datetime, timedelta, timezone

import pytest

from eonet_cascades.data.schema import Event, Mark
from eonet_cascades.data.store import EventStore


def _event(i: int) -> Event:
    return Event(
        event_id=f"eonet:E{i}",
        source_catalog="eonet",
        time_start=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i),
        time_end=None,
        longitude=-100.0 + i * 0.01,
        latitude=35.0 + i * 0.01,
        mark=Mark.WILDFIRE,
        magnitude=float(i),
        metadata={"i": i},
        ingested_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        dedup_group_id=None,
    )


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "events.duckdb")
    s.init_schema()
    yield s
    s.close()


def test_init_schema_creates_events_table(store):
    rows = store.query_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
    assert ("events",) in rows


def test_write_and_count(store):
    events = [_event(i) for i in range(5)]
    store.write_events(events)
    assert store.count_events() == 5


def test_write_is_idempotent_on_event_id(store):
    events = [_event(i) for i in range(3)]
    store.write_events(events)
    store.write_events(events)  # second write must not duplicate
    assert store.count_events() == 3


def test_query_by_time_range(store):
    store.write_events([_event(i) for i in range(10)])
    df = store.query_events(
        time_start=datetime(2024, 1, 3, tzinfo=timezone.utc),
        time_end=datetime(2024, 1, 6, tzinfo=timezone.utc),
    )
    assert len(df) == 4  # days 2..5 inclusive


def test_query_by_bbox(store):
    store.write_events([_event(i) for i in range(10)])
    df = store.query_events(bbox=(-100.05, 35.0, -100.0, 35.05))
    assert 0 < len(df) <= 10


def test_query_by_mark(store):
    store.write_events([_event(i) for i in range(3)])
    df_wildfire = store.query_events(marks=["wildfire"])
    df_quake = store.query_events(marks=["earthquake"])
    assert len(df_wildfire) == 3
    assert len(df_quake) == 0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_store.py -v
```

Expected: ImportError on `eonet_cascades.data.store`.

- [ ] **Step 3: Implement the store**

`src/eonet_cascades/data/store.py`:

```python
"""DuckDB-backed event store."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from eonet_cascades.data.schema import Event

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id        VARCHAR PRIMARY KEY,
    source_catalog  VARCHAR NOT NULL,
    time_start      TIMESTAMP WITH TIME ZONE NOT NULL,
    time_end        TIMESTAMP WITH TIME ZONE,
    longitude       DOUBLE NOT NULL,
    latitude        DOUBLE NOT NULL,
    mark            VARCHAR NOT NULL,
    magnitude       DOUBLE,
    metadata_json   VARCHAR NOT NULL,
    ingested_at     TIMESTAMP WITH TIME ZONE NOT NULL,
    dedup_group_id  VARCHAR
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(time_start);
CREATE INDEX IF NOT EXISTS idx_events_mark ON events(mark);
CREATE INDEX IF NOT EXISTS idx_events_catalog ON events(source_catalog);
"""


class EventStore:
    """Thin wrapper over DuckDB with the unified Event schema."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))

    def init_schema(self) -> None:
        self._conn.execute(_DDL)

    def write_events(self, events: Iterable[Event]) -> int:
        rows = [_event_to_row(e) for e in events]
        if not rows:
            return 0
        df = pl.DataFrame(rows)
        # Idempotent insert: ON CONFLICT DO NOTHING by primary key.
        self._conn.register("incoming", df.to_arrow())
        self._conn.execute(
            "INSERT INTO events SELECT * FROM incoming ON CONFLICT(event_id) DO NOTHING"
        )
        self._conn.unregister("incoming")
        return len(rows)

    def count_events(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def query_events(
        self,
        *,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        marks: list[str] | None = None,
        source_catalogs: list[str] | None = None,
    ) -> pl.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if time_start is not None:
            clauses.append("time_start >= ?")
            params.append(time_start)
        if time_end is not None:
            clauses.append("time_start <= ?")
            params.append(time_end)
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            clauses.append("longitude BETWEEN ? AND ?")
            params.extend([min_lon, max_lon])
            clauses.append("latitude BETWEEN ? AND ?")
            params.extend([min_lat, max_lat])
        if marks:
            clauses.append(f"mark IN ({','.join(['?'] * len(marks))})")
            params.extend(marks)
        if source_catalogs:
            clauses.append(f"source_catalog IN ({','.join(['?'] * len(source_catalogs))})")
            params.extend(source_catalogs)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM events {where} ORDER BY time_start"
        return self._conn.execute(sql, params).pl()

    def query_sql(self, sql: str, params: list[Any] | None = None) -> duckdb.DuckDBPyRelation:
        return self._conn.execute(sql, params or [])

    def close(self) -> None:
        self._conn.close()


def _event_to_row(e: Event) -> dict[str, Any]:
    return {
        "event_id": e.event_id,
        "source_catalog": e.source_catalog,
        "time_start": e.time_start,
        "time_end": e.time_end,
        "longitude": e.longitude,
        "latitude": e.latitude,
        "mark": e.mark.value,
        "magnitude": e.magnitude,
        "metadata_json": json.dumps(e.metadata, default=str),
        "ingested_at": e.ingested_at,
        "dedup_group_id": e.dedup_group_id,
    }
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
uv run pytest tests/test_store.py -v
```

Expected: all six tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/data/store.py tests/test_store.py
git commit -m "feat(data): add DuckDB-backed EventStore with idempotent writes"
```

---

### Task 8: Implement the rate-limited HTTP client and manifests

**Files:**
- Create: `src/eonet_cascades/data/http.py`
- Create: `src/eonet_cascades/data/manifests.py`
- Test: `tests/test_http.py`, `tests/test_manifests.py`

- [ ] **Step 1: Write tests for the rate limiter**

`tests/test_http.py`:

```python
"""Rate-limited HTTP client tests."""

import time

import httpx
import pytest
import respx

from eonet_cascades.data.http import RateLimitedClient


@respx.mock
def test_client_calls_url():
    respx.get("https://example.com/x").mock(return_value=httpx.Response(200, json={"ok": True}))
    client = RateLimitedClient(rate_per_sec=10.0)
    r = client.get("https://example.com/x")
    assert r.json() == {"ok": True}


@respx.mock
def test_client_throttles_to_rate():
    respx.get("https://example.com/y").mock(return_value=httpx.Response(200, json={}))
    client = RateLimitedClient(rate_per_sec=4.0)  # min 0.25s between requests
    t0 = time.monotonic()
    for _ in range(3):
        client.get("https://example.com/y")
    elapsed = time.monotonic() - t0
    # 3 calls at 4/sec: first is free, then 2 * 0.25s = ~0.5s
    assert elapsed >= 0.45, f"expected throttle, got {elapsed:.3f}s"


@respx.mock
def test_client_retries_on_5xx():
    route = respx.get("https://example.com/z").mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    client = RateLimitedClient(rate_per_sec=100.0, max_retries=3, retry_backoff=0.01)
    r = client.get("https://example.com/z")
    assert r.status_code == 200
    assert route.call_count == 3


@respx.mock
def test_client_raises_after_max_retries():
    respx.get("https://example.com/q").mock(return_value=httpx.Response(503))
    client = RateLimitedClient(rate_per_sec=100.0, max_retries=2, retry_backoff=0.01)
    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://example.com/q")
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
uv run pytest tests/test_http.py -v
```

Expected: ImportError on `eonet_cascades.data.http`.

- [ ] **Step 3: Implement the rate-limited client**

`src/eonet_cascades/data/http.py`:

```python
"""Rate-limited HTTP client with retries."""

from __future__ import annotations

import time

import httpx


class RateLimitedClient:
    """Synchronous httpx wrapper enforcing a minimum interval between requests.

    Retries on 5xx and connection errors with simple linear backoff.
    """

    def __init__(
        self,
        rate_per_sec: float = 1.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        self._min_interval = 1.0 / rate_per_sec
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)
        self._last_call = 0.0

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self._request("GET", url, **kwargs)

    def stream_text(self, url: str, **kwargs) -> httpx.Response:
        return self._request("GET", url, **kwargs)

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        self._throttle()
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                r = self._client.request(method, url, **kwargs)
                if r.status_code >= 500:
                    r.raise_for_status()
                r.raise_for_status()
                return r
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last_exc = e
                if attempt + 1 < self._max_retries:
                    time.sleep(self._retry_backoff * (attempt + 1))
                    continue
                raise
        # Unreachable, but mypy doesn't know that.
        raise RuntimeError("unreachable") from last_exc

    def _throttle(self) -> None:
        now = time.monotonic()
        wait = self._min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Write tests for the manifest store**

`tests/test_manifests.py`:

```python
"""Manifest state I/O tests."""

from datetime import datetime, timezone

from eonet_cascades.data.manifests import ManifestStore


def test_read_missing_manifest_returns_none(tmp_path):
    store = ManifestStore(tmp_path)
    assert store.last_fetched("eonet") is None


def test_write_and_read_roundtrip(tmp_path):
    store = ManifestStore(tmp_path)
    ts = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    store.set_last_fetched("eonet", ts)
    assert store.last_fetched("eonet") == ts


def test_independent_catalogs(tmp_path):
    store = ManifestStore(tmp_path)
    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = datetime(2024, 2, 1, tzinfo=timezone.utc)
    store.set_last_fetched("eonet", ts1)
    store.set_last_fetched("usgs", ts2)
    assert store.last_fetched("eonet") == ts1
    assert store.last_fetched("usgs") == ts2
```

- [ ] **Step 5: Run, confirm failure, implement manifests**

```bash
uv run pytest tests/test_manifests.py -v
```

Expected: ImportError.

`src/eonet_cascades/data/manifests.py`:

```python
"""Per-catalog ingestion state stored as JSON in the manifests directory."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class ManifestStore:
    """Tracks the last successful fetch timestamp per catalog."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, catalog: str) -> Path:
        return self.root / f"{catalog}_state.json"

    def last_fetched(self, catalog: str) -> datetime | None:
        p = self._path(catalog)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        ts = data.get("last_fetched")
        return datetime.fromisoformat(ts) if ts else None

    def set_last_fetched(self, catalog: str, ts: datetime) -> None:
        p = self._path(catalog)
        p.write_text(json.dumps({"last_fetched": ts.isoformat()}))
```

- [ ] **Step 6: Run all new tests**

```bash
uv run pytest tests/test_http.py tests/test_manifests.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/eonet_cascades/data/http.py src/eonet_cascades/data/manifests.py tests/test_http.py tests/test_manifests.py
git commit -m "feat(data): add rate-limited HTTP client and manifest state store"
```

---

### Task 9: Define the CatalogFetcher protocol

**Files:**
- Create: `src/eonet_cascades/data/base.py`

- [ ] **Step 1: Write the protocol**

`src/eonet_cascades/data/base.py`:

```python
"""Common interface for catalog fetchers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from eonet_cascades.data.schema import Event, RawEvent


@runtime_checkable
class CatalogFetcher(Protocol):
    """A catalog fetcher knows how to pull raw records and harmonize them."""

    name: str

    def fetch(self, since: datetime, until: datetime) -> Iterable[RawEvent]:
        """Yield raw events covering [since, until)."""

    def harmonize(self, raw: RawEvent) -> Event | None:
        """Convert a raw event to the unified Event schema.

        Returning None signals the raw record is outside the unified mark
        vocabulary (or otherwise should be skipped without erroring).
        """
```

- [ ] **Step 2: Commit (no tests — the protocol has no behavior yet)**

```bash
git add src/eonet_cascades/data/base.py
git commit -m "feat(data): define CatalogFetcher protocol"
```

---

### Task 10: Implement the mark harmonization registry

**Files:**
- Create: `src/eonet_cascades/data/marks.py`
- Test: `tests/test_marks.py`

- [ ] **Step 1: Write the tests**

`tests/test_marks.py`:

```python
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
```

- [ ] **Step 2: Run, confirm failure**

```bash
uv run pytest tests/test_marks.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement the registry**

`src/eonet_cascades/data/marks.py`:

```python
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
        "seaLakeIce".lower(): Mark.SEA_LAKE_ICE,
        "earthquakes": Mark.EARTHQUAKE,
        "floods": Mark.FLOOD,
        "landslides": Mark.LANDSLIDE,
        "drought": Mark.DROUGHT,
        "dustHaze".lower(): Mark.DUST_HAZE,
        "tempExtremes".lower(): Mark.TEMPERATURE_EXTREME,
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
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_marks.py -v
```

Expected: all eight tests pass — including `test_all_unified_marks_have_at_least_one_source_mapping`. If that final test fails, the registry has a gap; fix the registry, don't weaken the test.

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/data/marks.py tests/test_marks.py
git commit -m "feat(data): add catalog-to-unified mark harmonization registry"
```

---

### Task 11: Implement the EONET fetcher

**Files:**
- Create: `src/eonet_cascades/data/eonet.py`
- Create: `tests/fixtures/eonet_sample.json`
- Create: `tests/conftest.py`
- Test: `tests/test_eonet.py`

- [ ] **Step 1: Capture a real EONET fixture**

```bash
mkdir -p tests/fixtures
curl -s "https://eonet.gsfc.nasa.gov/api/v3/events?limit=20&days=30" > tests/fixtures/eonet_sample.json
python -c "import json; d=json.load(open('tests/fixtures/eonet_sample.json')); print(len(d['events']), 'events captured')"
```

Expected: a positive event count is printed. If the request fails (offline), construct the fixture by hand following the EONET v3 schema documented at https://eonet.gsfc.nasa.gov/docs/v3 — a single event with `id`, `title`, `categories: [{id, title}]`, `geometry: [{date, type, coordinates}]` is sufficient.

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def eonet_payload() -> dict:
    return json.loads((FIXTURE_DIR / "eonet_sample.json").read_text())
```

- [ ] **Step 3: Write the EONET tests**

`tests/test_eonet.py`:

```python
"""EONET fetcher + harmonization tests."""

from datetime import datetime, timezone

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
    assert ev.time_start == datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)


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
    assert ev.time_start == datetime(2024, 9, 1, tzinfo=timezone.utc)
    assert ev.time_end == datetime(2024, 9, 2, tzinfo=timezone.utc)


@respx.mock
def test_fetch_hits_eonet_endpoint():
    respx.get("https://eonet.gsfc.nasa.gov/api/v3/events").mock(
        return_value=httpx.Response(200, json={"events": []})
    )
    fetcher = EONETFetcher(rate_per_sec=100.0)
    result = list(
        fetcher.fetch(
            since=datetime(2024, 1, 1, tzinfo=timezone.utc),
            until=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
    )
    assert result == []
```

- [ ] **Step 4: Run, confirm failure**

```bash
uv run pytest tests/test_eonet.py -v
```

Expected: ImportError on `eonet_cascades.data.eonet`.

- [ ] **Step 5: Implement the EONET fetcher**

`src/eonet_cascades/data/eonet.py`:

```python
"""NASA EONET v3 fetcher."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from eonet_cascades.data.http import RateLimitedClient
from eonet_cascades.data.marks import harmonize_mark
from eonet_cascades.data.schema import Event, RawEvent

EONET_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"


class EONETFetcher:
    """Fetches events from NASA EONET v3."""

    name = "eonet"

    def __init__(self, rate_per_sec: float = 1.0) -> None:
        self._client = RateLimitedClient(rate_per_sec=rate_per_sec)

    def fetch(self, since: datetime, until: datetime) -> Iterable[RawEvent]:
        # EONET v3 supports `start` and `end` ISO date parameters.
        params = {
            "start": since.date().isoformat(),
            "end": until.date().isoformat(),
            "status": "all",
        }
        r = self._client.get(EONET_URL, params=params)
        payload = r.json()
        yield from self._iter_raw_from_payload(payload)

    def _iter_raw_from_payload(self, payload: dict[str, Any]) -> Iterable[RawEvent]:
        for ev in payload.get("events", []):
            yield RawEvent(source_catalog="eonet", source_id=ev["id"], payload=ev)

    def harmonize(self, raw: RawEvent) -> Event | None:
        p = raw.payload
        categories = p.get("categories", [])
        if not categories:
            return None
        cat_id = categories[0].get("id", "")
        mark = harmonize_mark("eonet", cat_id)
        if mark is None:
            return None

        geometry = p.get("geometry", [])
        if not geometry:
            return None
        first = geometry[0]
        if first.get("type") != "Point":
            return None
        coords = first.get("coordinates")
        if not coords or len(coords) < 2:
            return None
        lon, lat = float(coords[0]), float(coords[1])

        time_start = _parse_iso8601(first["date"])
        time_end = None
        if len(geometry) > 1:
            time_end = _parse_iso8601(geometry[-1]["date"])

        return Event(
            event_id=f"eonet:{raw.source_id}",
            source_catalog="eonet",
            time_start=time_start,
            time_end=time_end,
            longitude=lon,
            latitude=lat,
            mark=mark,
            magnitude=None,
            metadata={"title": p.get("title"), "sources": p.get("sources", [])},
            ingested_at=datetime.now(timezone.utc),
            dedup_group_id=None,
        )


def _parse_iso8601(ts: str) -> datetime:
    """EONET emits 'Z'-suffixed UTC timestamps. fromisoformat needs explicit handling."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
```

- [ ] **Step 6: Run tests and confirm they pass**

```bash
uv run pytest tests/test_eonet.py -v
```

Expected: all four tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/eonet_cascades/data/eonet.py tests/test_eonet.py tests/conftest.py tests/fixtures/eonet_sample.json
git commit -m "feat(data): add EONET v3 fetcher with harmonization"
```

---

### Task 12: Implement the USGS ComCat earthquake fetcher

**Files:**
- Create: `src/eonet_cascades/data/usgs.py`
- Create: `tests/fixtures/usgs_sample.geojson`
- Test: `tests/test_usgs.py`

- [ ] **Step 1: Capture a real USGS fixture**

```bash
curl -s "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2024-01-01&endtime=2024-01-08&minmagnitude=4.5&minlongitude=-130&maxlongitude=-65&minlatitude=14&maxlatitude=50" > tests/fixtures/usgs_sample.geojson
python -c "import json; d=json.load(open('tests/fixtures/usgs_sample.geojson')); print(len(d['features']), 'quakes captured')"
```

Expected: a positive count (typically a dozen or so for a week of CONUS+Mexico magnitude ≥4.5).

- [ ] **Step 2: Extend `tests/conftest.py`**

Add to the existing `tests/conftest.py`:

```python
@pytest.fixture
def usgs_payload() -> dict:
    return json.loads((FIXTURE_DIR / "usgs_sample.geojson").read_text())
```

- [ ] **Step 3: Write the USGS tests**

`tests/test_usgs.py`:

```python
"""USGS ComCat fetcher + harmonization tests."""

from datetime import datetime, timezone

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
    assert ev.time_start == datetime(2024, 1, 1, tzinfo=timezone.utc)


@respx.mock
def test_fetch_passes_bbox():
    route = respx.get("https://earthquake.usgs.gov/fdsnws/event/1/query").mock(
        return_value=httpx.Response(200, json={"features": []})
    )
    fetcher = USGSFetcher(rate_per_sec=100.0)
    list(
        fetcher.fetch(
            since=datetime(2024, 1, 1, tzinfo=timezone.utc),
            until=datetime(2024, 1, 8, tzinfo=timezone.utc),
            bbox=(-130.0, 14.0, -65.0, 50.0),
        )
    )
    assert route.called
    qs = dict(route.calls[0].request.url.params)
    assert qs["minlongitude"] == "-130.0"
    assert qs["maxlongitude"] == "-65.0"
```

- [ ] **Step 4: Run, confirm failure, implement**

```bash
uv run pytest tests/test_usgs.py -v
```

Expected: ImportError.

`src/eonet_cascades/data/usgs.py`:

```python
"""USGS ComCat (FDSN web service) earthquake fetcher."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from eonet_cascades.data.http import RateLimitedClient
from eonet_cascades.data.schema import Event, Mark, RawEvent

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


class USGSFetcher:
    """Fetches earthquakes from the USGS ComCat FDSN web service."""

    name = "usgs"

    def __init__(self, rate_per_sec: float = 2.0, min_magnitude: float = 2.5) -> None:
        self._client = RateLimitedClient(rate_per_sec=rate_per_sec)
        self._min_magnitude = min_magnitude

    def fetch(
        self,
        since: datetime,
        until: datetime,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> Iterable[RawEvent]:
        params: dict[str, str] = {
            "format": "geojson",
            "starttime": since.isoformat(),
            "endtime": until.isoformat(),
            "minmagnitude": str(self._min_magnitude),
            "orderby": "time-asc",
        }
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            params["minlongitude"] = str(min_lon)
            params["maxlongitude"] = str(max_lon)
            params["minlatitude"] = str(min_lat)
            params["maxlatitude"] = str(max_lat)
        r = self._client.get(USGS_URL, params=params)
        payload = r.json()
        yield from self._iter_raw_from_payload(payload)

    def _iter_raw_from_payload(self, payload: dict[str, Any]) -> Iterable[RawEvent]:
        for feat in payload.get("features", []):
            yield RawEvent(source_catalog="usgs", source_id=feat["id"], payload=feat)

    def harmonize(self, raw: RawEvent) -> Event | None:
        p = raw.payload
        props = p.get("properties", {})
        geom = p.get("geometry", {})
        coords = geom.get("coordinates") or []
        if len(coords) < 2 or geom.get("type") != "Point":
            return None
        lon, lat = float(coords[0]), float(coords[1])
        ts_ms = props.get("time")
        if ts_ms is None:
            return None
        time_start = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return Event(
            event_id=f"usgs:{raw.source_id}",
            source_catalog="usgs",
            time_start=time_start,
            time_end=None,
            longitude=lon,
            latitude=lat,
            mark=Mark.EARTHQUAKE,
            magnitude=props.get("mag"),
            metadata={"place": props.get("place"), "depth_km": coords[2] if len(coords) >= 3 else None},
            ingested_at=datetime.now(timezone.utc),
            dedup_group_id=None,
        )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_usgs.py -v
```

Expected: all three pass.

- [ ] **Step 6: Commit**

```bash
git add src/eonet_cascades/data/usgs.py tests/test_usgs.py tests/fixtures/usgs_sample.geojson tests/conftest.py
git commit -m "feat(data): add USGS ComCat earthquake fetcher"
```

---

### Task 13: Implement the NOAA Storm Events fetcher

**Files:**
- Create: `src/eonet_cascades/data/noaa_storms.py`
- Create: `tests/fixtures/noaa_sample.csv`
- Test: `tests/test_noaa.py`

- [ ] **Step 1: Build the NOAA fixture (synthetic — bulk CSV is too large)**

`tests/fixtures/noaa_sample.csv`:

```csv
BEGIN_YEARMONTH,BEGIN_DAY,BEGIN_TIME,EVENT_ID,STATE,EVENT_TYPE,BEGIN_DATE_TIME,END_DATE_TIME,BEGIN_LAT,BEGIN_LON,MAGNITUDE,MAGNITUDE_TYPE,TOR_F_SCALE
202401,15,1430,1000001,TEXAS,Tornado,15-JAN-24 14:30:00,15-JAN-24 14:45:00,30.5,-97.2,,EF,EF2
202401,20,2200,1000002,FLORIDA,Hurricane,20-JAN-24 22:00:00,21-JAN-24 04:00:00,27.8,-80.1,95.0,KT,
202401,22,0300,1000003,LOUISIANA,Flash Flood,22-JAN-24 03:00:00,22-JAN-24 09:30:00,30.2,-91.0,,,
202402,01,1600,1000004,CALIFORNIA,Wildfire,01-FEB-24 16:00:00,02-FEB-24 22:00:00,34.1,-118.5,,,
202402,05,0900,1000005,COLORADO,Blizzard,05-FEB-24 09:00:00,06-FEB-24 06:00:00,39.7,-104.9,,,
202402,10,1200,1000006,UNKNOWN,Marine Hurricane/Typhoon,10-FEB-24 12:00:00,10-FEB-24 18:00:00,,,40.0,KT,
```

- [ ] **Step 2: Write the NOAA tests**

`tests/test_noaa.py`:

```python
"""NOAA Storm Events fetcher + harmonization tests."""

from datetime import datetime, timezone
from pathlib import Path

from eonet_cascades.data.noaa_storms import NOAAStormsFetcher
from eonet_cascades.data.schema import Mark

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
    assert ev.time_start == datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)


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
    raw = type(raws_first := list(fetcher._iter_raw_from_csv(FIXTURE)))  # noqa: just to be explicit
    # Construct a fake raw payload with an unmapped EVENT_TYPE
    from eonet_cascades.data.schema import RawEvent

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
```

- [ ] **Step 3: Run, confirm failure, implement**

```bash
uv run pytest tests/test_noaa.py -v
```

Expected: ImportError.

`src/eonet_cascades/data/noaa_storms.py`:

```python
"""NOAA Storm Events Database fetcher.

The Storm Events DB ships as annual bulk CSVs at
https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

This fetcher downloads the per-year details file, parses it, and yields raw rows.
"""

from __future__ import annotations

import gzip
import io
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from eonet_cascades.data.http import RateLimitedClient
from eonet_cascades.data.marks import harmonize_mark
from eonet_cascades.data.schema import Event, RawEvent

INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
# Filename pattern: StormEvents_details-ftp_v1.0_d2024_cYYYYMMDD.csv.gz
_FILENAME_RE = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz")

# Columns we care about; many more exist in the source.
_COLUMNS = [
    "EVENT_ID",
    "STATE",
    "EVENT_TYPE",
    "BEGIN_DATE_TIME",
    "END_DATE_TIME",
    "BEGIN_LAT",
    "BEGIN_LON",
    "MAGNITUDE",
    "MAGNITUDE_TYPE",
    "TOR_F_SCALE",
]


class NOAAStormsFetcher:
    name = "noaa"

    def __init__(self, rate_per_sec: float = 0.5, cache_dir: Path | None = None) -> None:
        self._client = RateLimitedClient(rate_per_sec=rate_per_sec)
        self._cache_dir = cache_dir

    def fetch(self, since: datetime, until: datetime) -> Iterable[RawEvent]:
        years = range(since.year, until.year + 1)
        for year in years:
            csv_bytes = self._download_year(year)
            with io.BytesIO(csv_bytes) as fh, gzip.GzipFile(fileobj=fh) as gz:
                yield from self._iter_raw_from_bytes(gz.read())

    def _download_year(self, year: int) -> bytes:
        index_html = self._client.get(INDEX_URL).text
        candidates = [
            name for name in _FILENAME_RE.findall(index_html) if int(name) == year
        ]
        # The findall above only returns the year group; we need the full filename.
        full_names = [
            m.group(0)
            for m in _FILENAME_RE.finditer(index_html)
            if int(m.group(1)) == year
        ]
        if not full_names:
            return b""
        # Pick the most recent revision (highest cYYYYMMDD).
        full_names.sort()
        url = INDEX_URL + full_names[-1]
        r = self._client.get(url)
        return r.content

    def _iter_raw_from_bytes(self, csv_bytes: bytes) -> Iterable[RawEvent]:
        df = pl.read_csv(io.BytesIO(csv_bytes), infer_schema_length=10000, ignore_errors=True)
        cols = [c for c in _COLUMNS if c in df.columns]
        for row in df.select(cols).iter_rows(named=True):
            yield RawEvent(
                source_catalog="noaa",
                source_id=str(row["EVENT_ID"]),
                payload=row,
            )

    def _iter_raw_from_csv(self, path: Path) -> Iterable[RawEvent]:
        """Test helper: parse a plain (uncompressed) CSV fixture."""
        df = pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)
        cols = [c for c in _COLUMNS if c in df.columns]
        for row in df.select(cols).iter_rows(named=True):
            yield RawEvent(
                source_catalog="noaa",
                source_id=str(row["EVENT_ID"]),
                payload=row,
            )

    def harmonize(self, raw: RawEvent) -> Event | None:
        p = raw.payload
        event_type = (p.get("EVENT_TYPE") or "").strip()
        mark = harmonize_mark("noaa", event_type)
        if mark is None:
            return None

        lat = _safe_float(p.get("BEGIN_LAT"))
        lon = _safe_float(p.get("BEGIN_LON"))
        if lat is None or lon is None:
            return None

        t_start = _parse_noaa_datetime(p.get("BEGIN_DATE_TIME"))
        t_end = _parse_noaa_datetime(p.get("END_DATE_TIME"))
        if t_start is None:
            return None

        mag = _safe_float(p.get("MAGNITUDE"))

        return Event(
            event_id=f"noaa:{raw.source_id}",
            source_catalog="noaa",
            time_start=t_start,
            time_end=t_end,
            longitude=lon,
            latitude=lat,
            mark=mark,
            magnitude=mag,
            metadata={
                "state": p.get("STATE"),
                "event_type": event_type,
                "magnitude_type": p.get("MAGNITUDE_TYPE"),
                "tor_f_scale": p.get("TOR_F_SCALE"),
            },
            ingested_at=datetime.now(timezone.utc),
            dedup_group_id=None,
        )


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_noaa_datetime(s: Any) -> datetime | None:
    """NOAA format: '15-JAN-24 14:30:00' (UTC by convention in the source)."""
    if not s:
        return None
    try:
        dt = datetime.strptime(str(s), "%d-%b-%y %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc)
```

- [ ] **Step 4: Fix a small test bug**

The test `test_harmonize_unknown_type_returns_none` has a stray `raw = type(raws_first := ...)` line. Replace the whole test body with the clean version below in `tests/test_noaa.py`:

```python
def test_harmonize_unknown_type_returns_none():
    from eonet_cascades.data.schema import RawEvent

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
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_noaa.py -v
```

Expected: all five pass.

- [ ] **Step 6: Commit**

```bash
git add src/eonet_cascades/data/noaa_storms.py tests/test_noaa.py tests/fixtures/noaa_sample.csv
git commit -m "feat(data): add NOAA Storm Events fetcher with bulk CSV handling"
```

---

### Task 14: Implement the NASA FIRMS fetcher

**Files:**
- Create: `src/eonet_cascades/data/firms.py`
- Create: `tests/fixtures/firms_sample.csv`
- Test: `tests/test_firms.py`

- [ ] **Step 1: Build the FIRMS fixture**

`tests/fixtures/firms_sample.csv`:

```csv
latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
34.123,-118.234,335.5,0.5,0.5,2024-06-15,1842,Suomi NPP,VIIRS,n,2.0NRT,295.2,12.5,D
38.567,-122.111,310.1,0.4,0.4,2024-06-15,2103,Suomi NPP,VIIRS,n,2.0NRT,290.0,5.3,D
40.012,-105.234,325.8,0.6,0.5,2024-06-16,0512,Suomi NPP,VIIRS,h,2.0NRT,300.5,22.1,N
```

- [ ] **Step 2: Write FIRMS tests**

`tests/test_firms.py`:

```python
"""NASA FIRMS fetcher + harmonization tests."""

from datetime import datetime, timezone
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
    assert events[0].time_start == datetime(2024, 6, 15, 18, 42, tzinfo=timezone.utc)
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
            since=datetime(2024, 6, 15, tzinfo=timezone.utc),
            until=datetime(2024, 6, 16, tzinfo=timezone.utc),
            bbox=(-130.0, 14.0, -65.0, 50.0),
        )
    )
    assert route.called
    url = str(route.calls[0].request.url)
    assert "MAP_KEY_123" in url
```

- [ ] **Step 3: Run, confirm failure, implement**

```bash
uv run pytest tests/test_firms.py -v
```

Expected: ImportError.

`src/eonet_cascades/data/firms.py`:

```python
"""NASA FIRMS (VIIRS / MODIS active fire) fetcher.

API docs: https://firms.modaps.eosdis.nasa.gov/api/area/
Endpoint shape:
  https://firms.modaps.eosdis.nasa.gov/api/area/csv/<MAP_KEY>/<SOURCE>/<AREA>/<DAY_RANGE>/<DATE>

For our needs we use the bounding-box variant:
  /api/area/csv/<MAP_KEY>/<SOURCE>/<W>,<S>,<E>,<N>/<DAY_RANGE>/<DATE>
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl

from eonet_cascades.data.http import RateLimitedClient
from eonet_cascades.data.schema import Event, Mark, RawEvent

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
# Confidence levels: VIIRS uses 'l', 'n', 'h' (low/nominal/high); MODIS uses 0-100.
_CONF_ORDER = {"l": 0, "n": 1, "h": 2}


class FIRMSFetcher:
    name = "firms"

    def __init__(
        self,
        api_key: str | None,
        rate_per_sec: float = 0.5,
        source: str = "VIIRS_SNPP_NRT",
        min_confidence: str = "n",
    ) -> None:
        self._api_key = api_key
        self._client = RateLimitedClient(rate_per_sec=rate_per_sec)
        self._source = source
        self._min_conf = min_confidence

    def fetch(
        self,
        since: datetime,
        until: datetime,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> Iterable[RawEvent]:
        if not self._api_key:
            raise RuntimeError(
                "FIRMS API key required; set EONET_FIRMS_API_KEY or configs/data/conus.yaml"
            )
        if bbox is None:
            bbox = (-180.0, -90.0, 180.0, 90.0)
        w, s, e, n = bbox
        # FIRMS caps day_range at 5 per request; iterate.
        cursor = since
        while cursor < until:
            window_end = min(cursor + timedelta(days=5), until)
            day_range = (window_end - cursor).days
            if day_range == 0:
                day_range = 1
            date_str = cursor.date().isoformat()
            url = f"{BASE}/{self._api_key}/{self._source}/{w},{s},{e},{n}/{day_range}/{date_str}"
            r = self._client.get(url)
            yield from self._iter_raw_from_csv(r.text)
            cursor = window_end

    def _iter_raw_from_csv(self, text: str) -> Iterable[RawEvent]:
        df = pl.read_csv(io.StringIO(text), infer_schema_length=1000, ignore_errors=True)
        if df.height == 0:
            return
        for i, row in enumerate(df.iter_rows(named=True)):
            sid = f"{row.get('acq_date','')}_{row.get('acq_time','')}_{row.get('latitude','')}_{row.get('longitude','')}_{i}"
            yield RawEvent(
                source_catalog="firms",
                source_id=sid,
                payload=row,
            )

    def harmonize(self, raw: RawEvent) -> Event | None:
        p = raw.payload
        lat = _safe_float(p.get("latitude"))
        lon = _safe_float(p.get("longitude"))
        if lat is None or lon is None:
            return None

        # Confidence gate
        conf = str(p.get("confidence", "")).lower()
        if conf in _CONF_ORDER:
            if _CONF_ORDER[conf] < _CONF_ORDER.get(self._min_conf, 1):
                return None

        time_start = _parse_firms_datetime(p.get("acq_date"), p.get("acq_time"))
        if time_start is None:
            return None

        frp = _safe_float(p.get("frp"))

        return Event(
            event_id=f"firms:{raw.source_id}",
            source_catalog="firms",
            time_start=time_start,
            time_end=None,
            longitude=lon,
            latitude=lat,
            mark=Mark.WILDFIRE,
            magnitude=frp,
            metadata={
                "satellite": p.get("satellite"),
                "instrument": p.get("instrument"),
                "confidence": conf,
                "bright_ti4": p.get("bright_ti4"),
                "daynight": p.get("daynight"),
            },
            ingested_at=datetime.now(timezone.utc),
            dedup_group_id=None,
        )


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_firms_datetime(date_s: Any, time_s: Any) -> datetime | None:
    if date_s is None or time_s is None:
        return None
    try:
        # FIRMS time is HHMM as an int or string
        t = int(time_s)
        hh, mm = divmod(t, 100)
        d = datetime.strptime(str(date_s), "%Y-%m-%d")
        return d.replace(hour=hh, minute=mm, tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_firms.py -v
```

Expected: all three pass.

- [ ] **Step 5: Commit**

```bash
git add src/eonet_cascades/data/firms.py tests/test_firms.py tests/fixtures/firms_sample.csv
git commit -m "feat(data): add NASA FIRMS active-fire fetcher"
```

---

### Task 15: Implement cross-catalog deduplication

**Files:**
- Create: `src/eonet_cascades/data/dedup.py`
- Test: `tests/test_dedup.py`

- [ ] **Step 1: Write the dedup tests**

`tests/test_dedup.py`:

```python
"""Cross-catalog deduplication tests."""

from datetime import datetime, timedelta, timezone

from eonet_cascades.data.dedup import DEFAULT_THRESHOLDS, assign_dedup_groups
from eonet_cascades.data.schema import Event, Mark


def _ev(eid: str, catalog: str, mark: Mark, t: datetime, lon: float, lat: float) -> Event:
    return Event(
        event_id=f"{catalog}:{eid}",
        source_catalog=catalog,
        time_start=t,
        time_end=None,
        longitude=lon,
        latitude=lat,
        mark=mark,
        magnitude=None,
        metadata={},
        ingested_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        dedup_group_id=None,
    )


def test_same_event_in_two_catalogs_collapses():
    t = datetime(2024, 9, 1, 12, 0, tzinfo=timezone.utc)
    events = [
        _ev("E1", "eonet", Mark.TROPICAL_CYCLONE, t, -75.0, 25.0),
        _ev("S1", "noaa", Mark.TROPICAL_CYCLONE, t + timedelta(hours=2), -75.5, 25.2),
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id is not None
    assert out[0].dedup_group_id == out[1].dedup_group_id


def test_distant_events_get_distinct_groups():
    t = datetime(2024, 9, 1, tzinfo=timezone.utc)
    events = [
        _ev("E1", "eonet", Mark.WILDFIRE, t, -120.0, 35.0),
        _ev("F1", "firms", Mark.WILDFIRE, t, -80.0, 35.0),  # 4000+ km away
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id != out[1].dedup_group_id


def test_different_marks_never_collapse():
    t = datetime(2024, 9, 1, tzinfo=timezone.utc)
    events = [
        _ev("E1", "eonet", Mark.WILDFIRE, t, -120.0, 35.0),
        _ev("E2", "eonet", Mark.EARTHQUAKE, t, -120.0, 35.0),  # same place + time, different mark
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id != out[1].dedup_group_id


def test_earthquake_threshold_is_tight():
    t = datetime(2024, 9, 1, tzinfo=timezone.utc)
    events = [
        # 10 km apart — outside the 5 km earthquake threshold
        _ev("U1", "usgs", Mark.EARTHQUAKE, t, -120.0, 35.0),
        _ev("U2", "usgs", Mark.EARTHQUAKE, t + timedelta(minutes=30), -120.0, 35.09),
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id != out[1].dedup_group_id


def test_drought_threshold_is_loose():
    t = datetime(2024, 9, 1, tzinfo=timezone.utc)
    events = [
        # 80 km / 3 days apart — within the loose drought threshold
        _ev("E1", "eonet", Mark.DROUGHT, t, -100.0, 35.0),
        _ev("E2", "eonet", Mark.DROUGHT, t + timedelta(days=3), -100.0, 35.8),
    ]
    out = assign_dedup_groups(events)
    assert out[0].dedup_group_id == out[1].dedup_group_id


def test_default_thresholds_cover_all_marks():
    missing = set(Mark) - set(DEFAULT_THRESHOLDS.keys())
    assert not missing, f"thresholds missing for: {missing}"


def test_empty_input_returns_empty():
    assert assign_dedup_groups([]) == []
```

- [ ] **Step 2: Run, confirm failure, implement**

```bash
uv run pytest tests/test_dedup.py -v
```

Expected: ImportError.

`src/eonet_cascades/data/dedup.py`:

```python
"""Cross-catalog spatio-temporal deduplication."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from datetime import timedelta

from eonet_cascades.data.schema import Event, Mark


class _Threshold:
    __slots__ = ("spatial_km", "temporal")

    def __init__(self, spatial_km: float, temporal: timedelta) -> None:
        self.spatial_km = spatial_km
        self.temporal = temporal


DEFAULT_THRESHOLDS: dict[Mark, _Threshold] = {
    Mark.EARTHQUAKE: _Threshold(5.0, timedelta(hours=1)),
    Mark.VOLCANIC_ERUPTION: _Threshold(5.0, timedelta(hours=24)),
    Mark.WILDFIRE: _Threshold(25.0, timedelta(hours=24)),
    Mark.TROPICAL_CYCLONE: _Threshold(200.0, timedelta(hours=12)),
    Mark.TORNADO: _Threshold(25.0, timedelta(hours=6)),
    Mark.SEVERE_STORM: _Threshold(25.0, timedelta(hours=6)),
    Mark.FLOOD: _Threshold(50.0, timedelta(hours=24)),
    Mark.LANDSLIDE: _Threshold(50.0, timedelta(hours=24)),
    Mark.DROUGHT: _Threshold(100.0, timedelta(days=7)),
    Mark.TEMPERATURE_EXTREME: _Threshold(100.0, timedelta(days=7)),
    Mark.DUST_HAZE: _Threshold(100.0, timedelta(days=7)),
    Mark.SEA_LAKE_ICE: _Threshold(100.0, timedelta(days=7)),
}


def assign_dedup_groups(
    events: Iterable[Event],
    thresholds: dict[Mark, _Threshold] | None = None,
) -> list[Event]:
    """Assign `dedup_group_id` to events that cluster within mark-specific spatial/temporal thresholds.

    Returns a new list of Event instances (Events are frozen) — does not mutate input.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    events = list(events)
    if not events:
        return []

    # Union-find over event indices, restricted to same-mark candidates.
    parent = list(range(len(events)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Bucket events by mark, then by date-quantum to bound pairwise comparisons.
    by_mark: dict[Mark, list[int]] = {}
    for idx, ev in enumerate(events):
        by_mark.setdefault(ev.mark, []).append(idx)

    for mark, idxs in by_mark.items():
        t = th.get(mark)
        if t is None:
            continue
        # Sort by time and do a sliding-window comparison.
        idxs.sort(key=lambda i: events[i].time_start)
        for i_pos, i in enumerate(idxs):
            for j in idxs[i_pos + 1 :]:
                if events[j].time_start - events[i].time_start > t.temporal:
                    break
                if _haversine_km(
                    events[i].longitude, events[i].latitude,
                    events[j].longitude, events[j].latitude,
                ) <= t.spatial_km:
                    union(i, j)

    # Materialize stable group ids: hash of sorted member event_ids.
    groups: dict[int, list[int]] = {}
    for i in range(len(events)):
        groups.setdefault(find(i), []).append(i)

    out: list[Event] = list(events)
    for root, members in groups.items():
        if len(members) == 1:
            continue
        member_ids = sorted(events[m].event_id for m in members)
        gid = "dg_" + hashlib.sha1("|".join(member_ids).encode()).hexdigest()[:12]
        for m in members:
            out[m] = events[m].model_copy(update={"dedup_group_id": gid})
    return out


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometers."""
    r_earth = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r_earth * math.asin(math.sqrt(a))
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_dedup.py -v
```

Expected: all seven pass.

- [ ] **Step 4: Commit**

```bash
git add src/eonet_cascades/data/dedup.py tests/test_dedup.py
git commit -m "feat(data): add cross-catalog spatio-temporal deduplication"
```

---

### Task 16: Wire up the ingest orchestrator and CLI command

**Files:**
- Create: `src/eonet_cascades/data/ingest.py`
- Modify: `src/eonet_cascades/cli.py`
- Test: `tests/test_ingest_integration.py`

- [ ] **Step 1: Write the orchestrator integration test (uses respx to mock all four catalogs)**

`tests/test_ingest_integration.py`:

```python
"""End-to-end ingest pipeline integration test."""

import json
from datetime import datetime, timezone
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
    # Provide gzipped CSV
    import gzip
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

    # 5. Verify dedup ran (at least the column exists; groups may be sparse)
    df = store.query_events()
    assert "dedup_group_id" in df.columns
    store.close()
```

- [ ] **Step 2: Confirm test fails**

```bash
uv run pytest tests/test_ingest_integration.py -v -m slow
```

Expected: ImportError or no `ingest` command on CLI.

- [ ] **Step 3: Implement the orchestrator**

`src/eonet_cascades/data/ingest.py`:

```python
"""Top-level ingestion orchestrator.

Drives all configured catalog fetchers, harmonizes, dedupes, and writes to the
DuckDB store. Idempotent via manifest state and ON CONFLICT DO NOTHING inserts.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from rich.console import Console

from eonet_cascades.config import DataConfig
from eonet_cascades.data.base import CatalogFetcher
from eonet_cascades.data.dedup import assign_dedup_groups
from eonet_cascades.data.eonet import EONETFetcher
from eonet_cascades.data.firms import FIRMSFetcher
from eonet_cascades.data.manifests import ManifestStore
from eonet_cascades.data.noaa_storms import NOAAStormsFetcher
from eonet_cascades.data.schema import Event
from eonet_cascades.data.store import EventStore
from eonet_cascades.data.usgs import USGSFetcher

console = Console()


def build_fetcher(name: str, cfg: DataConfig) -> CatalogFetcher:
    if name == "eonet":
        return EONETFetcher()
    if name == "usgs":
        return USGSFetcher()
    if name == "noaa":
        return NOAAStormsFetcher()
    if name == "firms":
        return FIRMSFetcher(api_key=cfg.firms_api_key)
    raise ValueError(f"unknown catalog: {name}")


def run_ingest(
    cfg: DataConfig,
    since: datetime,
    until: datetime,
    catalogs: list[str] | None = None,
) -> dict[str, int]:
    """Run the full ingest pipeline for the given time window and catalogs.

    Returns per-catalog counts of events written.
    """
    cfg.ensure_exists()
    store = EventStore(cfg.duckdb_path)
    store.init_schema()
    manifests = ManifestStore(cfg.manifests_dir)
    counts: dict[str, int] = {}

    catalogs = catalogs or cfg.catalogs

    new_events: list[Event] = []
    for cat in catalogs:
        last = manifests.last_fetched(cat)
        effective_since = max(last, since) if last is not None else since
        if effective_since >= until:
            console.log(f"[dim]{cat}: nothing to fetch ({effective_since} ≥ {until})[/]")
            counts[cat] = 0
            continue
        fetcher = build_fetcher(cat, cfg)
        console.log(f"[bold]{cat}[/]: fetching {effective_since} → {until}")
        cat_events: list[Event] = []
        for raw in fetcher.fetch(effective_since, until):
            ev = fetcher.harmonize(raw)
            if ev is None:
                continue
            # Bbox filter
            if not _in_bbox(ev, cfg.bbox):
                continue
            cat_events.append(ev)
        counts[cat] = len(cat_events)
        new_events.extend(cat_events)
        manifests.set_last_fetched(cat, until)
        console.log(f"  → harmonized {len(cat_events)} events")

    # Dedup across catalogs in the new batch + existing rows (simple v1: dedup only the new batch).
    deduped = assign_dedup_groups(new_events)
    written = store.write_events(deduped)
    console.log(f"[green]Wrote {written} events to {cfg.duckdb_path}[/]")
    store.close()
    return counts


def _in_bbox(ev: Event, bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= ev.longitude <= max_lon and min_lat <= ev.latitude <= max_lat
```

- [ ] **Step 4: Add the `ingest` command to the CLI**

Modify `src/eonet_cascades/cli.py` to append (do not replace existing content):

```python
from datetime import datetime, timezone
from pathlib import Path

from eonet_cascades.config import DataConfig, load_data_config
from eonet_cascades.data.ingest import run_ingest


@app.command()
def ingest(
    catalogs: str = typer.Option("eonet,usgs,noaa,firms", help="Comma-separated list of catalogs"),
    since: str = typer.Option("2000-01-01", help="ISO date, inclusive lower bound"),
    until: str = typer.Option(None, help="ISO date, exclusive upper bound (default: now)"),
    config: Path = typer.Option(None, help="Optional YAML config path"),
) -> None:
    """Fetch + harmonize + dedup + persist events from the specified catalogs."""
    cfg = load_data_config(config) if config else DataConfig()
    since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    until_dt = (
        datetime.fromisoformat(until).replace(tzinfo=timezone.utc)
        if until
        else datetime.now(timezone.utc)
    )
    cat_list = [c.strip() for c in catalogs.split(",") if c.strip()]
    counts = run_ingest(cfg, since=since_dt, until=until_dt, catalogs=cat_list)
    console.print(counts)
```

- [ ] **Step 5: Run the integration test**

```bash
uv run pytest tests/test_ingest_integration.py -v -m slow
```

Expected: passes. If a sub-fetcher raises (e.g., NOAA filename pattern mismatched on the mocked index HTML), inspect the failure; the most common issue is `_FILENAME_RE` not matching the mocked index text — verify by printing the index text in the test.

- [ ] **Step 6: Run the full test suite**

```bash
uv run pytest -v
uv run pytest -v -m slow
```

Expected: every test passes; ruff still clean (`uv run ruff check .`).

- [ ] **Step 7: Commit**

```bash
git add src/eonet_cascades/data/ingest.py src/eonet_cascades/cli.py tests/test_ingest_integration.py
git commit -m "feat(data): wire up ingest orchestrator + CLI command"
```

---

### Task 17: First real ingest — small window, all four catalogs

**Files:**
- None (operational task — populates `data/events.duckdb`)

- [ ] **Step 1: Acquire a FIRMS API key**

Visit https://firms.modaps.eosdis.nasa.gov/api/area/ and register for a free MAP_KEY. Then:

```bash
echo "EONET_FIRMS_API_KEY=YOUR_KEY_HERE" >> ~/Projects/eonet-cascades/.env
# Optional: load .env into the shell for this session
export EONET_FIRMS_API_KEY=YOUR_KEY_HERE
```

- [ ] **Step 2: Run a 30-day ingest against live APIs**

```bash
cd ~/Projects/eonet-cascades
uv run eonet ingest \
  --catalogs eonet,usgs,noaa,firms \
  --since 2024-06-01 \
  --until 2024-07-01
```

Expected: per-catalog event counts printed; `data/events.duckdb` populated. EONET typically returns dozens of events for a month; USGS returns hundreds at min-magnitude 2.5; NOAA depends on whether 2024 bulk file is available; FIRMS returns thousands of fire detections.

- [ ] **Step 3: Quick sanity check**

```bash
uv run python -c "
from eonet_cascades.data.store import EventStore
from eonet_cascades.config import DataConfig
cfg = DataConfig()
store = EventStore(cfg.duckdb_path)
store.init_schema()
for cat in cfg.catalogs:
    n = store.query_events(source_catalogs=[cat]).height
    print(f'{cat}: {n} events')
print('total:', store.count_events())
store.close()
"
```

Expected: a non-zero count for each catalog. If any catalog returns 0, investigate (most likely cause: NOAA bulk file for 2024 not yet published — try 2023, or temporarily drop noaa from the catalog list).

- [ ] **Step 4: Run the full historical ingest**

```bash
uv run eonet ingest \
  --catalogs eonet,usgs,noaa,firms \
  --since 2000-01-01
```

Expected: this is the long-running command. USGS at min-magnitude 2.5 over 25 years is ~500k events globally; FIRMS bbox-bounded is millions of fire detections. Estimated wall-clock time: 30 minutes to a few hours depending on network. The process is resumable — re-running picks up from manifests.

- [ ] **Step 5: No commit (operational task, no code change)**

---

### Task 18: Dataset smell-test notebook (Phase 1 gate)

**Files:**
- Create: `notebooks/01_data_exploration.ipynb`

- [ ] **Step 1: Install jupyter and create the notebook**

```bash
cd ~/Projects/eonet-cascades
uv add --dev jupyterlab matplotlib  # if not already added
uv run python -m ipykernel install --user --name=eonet-cascades --display-name="eonet-cascades"
```

- [ ] **Step 2: Create `notebooks/01_data_exploration.ipynb` with the following cell content**

The notebook should contain six cells, in order. Create the file using `jupytext` or directly by hand; the content of each cell:

**Cell 1 (markdown):**

```markdown
# Phase 1 Dataset Smell Test

This notebook is the **gate deliverable** for Phase 1. Each section answers one question that, if the answer looks wrong, blocks modeling work in Phase 2.

Run this end-to-end after a full ingest. Inspect the output by eye.
```

**Cell 2 (code) — counts:**

```python
import polars as pl
from eonet_cascades.config import DataConfig
from eonet_cascades.data.store import EventStore

cfg = DataConfig()
store = EventStore(cfg.duckdb_path)
store.init_schema()
df = store.query_events()
print(f"Total events: {df.height:,}")
print()
print("Per catalog:")
print(df.group_by("source_catalog").len().sort("len", descending=True))
print()
print("Per mark:")
print(df.group_by("mark").len().sort("len", descending=True))
```

**Cell 3 (code) — yearly time series:**

```python
import matplotlib.pyplot as plt

yearly = (
    df.with_columns(pl.col("time_start").dt.year().alias("year"))
      .group_by(["year", "source_catalog"]).len()
      .sort("year")
)
fig, ax = plt.subplots(figsize=(10, 5))
for cat, group in yearly.partition_by("source_catalog", as_dict=True).items():
    ax.plot(group["year"], group["len"], label=cat, marker="o")
ax.set_yscale("log")
ax.set_xlabel("Year")
ax.set_ylabel("Events (log)")
ax.set_title("Event counts per year, per catalog")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Cell 4 (code) — CONUS map of one mark:**

```python
fig, ax = plt.subplots(figsize=(11, 6))
sample = df.filter(pl.col("mark") == "wildfire").sample(min(20000, df.filter(pl.col("mark") == "wildfire").height), seed=0)
ax.scatter(sample["longitude"], sample["latitude"], s=2, alpha=0.3, c="orangered")
ax.set_xlim(-130, -65)
ax.set_ylim(14, 50)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Wildfires (n={sample.height:,}) — spatial coverage check")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Cell 5 (code) — dedup health:**

```python
n_total = df.height
n_grouped = df.filter(pl.col("dedup_group_id").is_not_null()).height
groups = df.filter(pl.col("dedup_group_id").is_not_null()).group_by("dedup_group_id").len()
print(f"Events in any dedup group: {n_grouped:,} / {n_total:,} ({100*n_grouped/n_total:.2f}%)")
print(f"Number of dedup groups: {groups.height:,}")
print(f"Mean group size: {groups['len'].mean():.2f}")
print(f"Max group size: {groups['len'].max()}")
print()
print("Largest groups (sanity check — should be hurricanes / floods, not earthquakes):")
top = groups.sort("len", descending=True).head(10)
for row in top.iter_rows(named=True):
    gid = row["dedup_group_id"]
    members = df.filter(pl.col("dedup_group_id") == gid)
    marks = members["mark"].unique().to_list()
    cats = members["source_catalog"].unique().to_list()
    t0 = members["time_start"].min()
    print(f"  {gid}  size={row['len']:>4}  marks={marks}  catalogs={cats}  t0={t0}")
```

**Cell 6 (markdown) — gate checklist:**

```markdown
## Phase 1 Gate Checklist

Before moving to Phase 2, confirm by eye:

- [ ] Per-catalog counts are non-zero and within an order of magnitude of expectations (USGS in the hundreds of thousands; FIRMS in the millions; NOAA in the hundreds of thousands; EONET in the tens of thousands).
- [ ] Per-mark counts cover all 12 unified marks; no mark has zero events.
- [ ] The yearly time series shows reasonable coverage from 2000 onward; no suspicious gaps.
- [ ] The CONUS wildfire map shows points concentrated in expected regions (California, Pacific Northwest, Southwest) — not uniform random.
- [ ] Dedup health: groups are dominated by tropical cyclones, severe storms, floods — **not** earthquakes (tight thresholds should keep quake-dedup-rate near 0%). If quakes are over-dedupelating, revisit thresholds.

If any item fails, fix the data layer before opening Plan 2.
```

- [ ] **Step 3: Run the notebook end-to-end**

```bash
cd ~/Projects/eonet-cascades
uv run jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb --inplace
```

Expected: notebook executes without errors; all four plots and tables render.

- [ ] **Step 4: Inspect manually against the checklist**

Open the notebook in JupyterLab and walk through the six cells.

```bash
uv run jupyter lab notebooks/01_data_exploration.ipynb
```

This is the human gate. If anything looks wrong, **do not proceed to Plan 2** — fix the data layer first. Common issues to expect on a first run:

- NOAA filename pattern mismatch → fewer NOAA events than expected. Fix `_FILENAME_RE` in `noaa_storms.py`.
- FIRMS confidence filter too tight → drop `min_confidence` to `l` and re-ingest.
- Earthquakes appearing in dedup groups → revisit `DEFAULT_THRESHOLDS[Mark.EARTHQUAKE]`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/01_data_exploration.ipynb
git commit -m "feat(notebook): add Phase 1 dataset smell-test notebook"
```

---

## Self-Review

**Spec coverage check** — every section of the design doc is implemented:

- §2 scope (CONUS+Mexico bbox, 2000+, 12-mark vocab) → Task 5 (config defaults), Task 6 (Mark enum)
- §3.1 four-catalog set → Tasks 11–14 (one fetcher each)
- §3.2 unified schema → Task 6
- §3.3 DuckDB+Parquet storage on external drive → Tasks 5 (path config), 7 (DuckDB store). Note: raw Parquet writes are **deferred** — v1 of the orchestrator harmonizes directly from streamed responses without persisting raw payloads to disk. The `raw/{catalog}/year=YYYY/*.parquet` layout from §3.3 is therefore not realized in this plan. If you want full raw-payload preservation in this phase, add a Task between 16 and 17 that writes raw Parquet inside each fetcher's `fetch()` call before harmonization. Recommended to defer until a later phase since the `metadata` JSON column already preserves the per-event raw payload for harmonized events.
- §3.4 idempotent ingestion w/ manifests + common interface → Tasks 8, 9, 16
- §3.5 cross-catalog dedup with per-mark thresholds → Task 15

**Placeholder scan** — no `TBD`, no `TODO` left in steps. The one acknowledged gap (raw Parquet persistence) is called out explicitly above with a suggested follow-up task; not a placeholder.

**Type/name consistency** — `Event`, `RawEvent`, `Mark`, `CatalogFetcher`, `EventStore`, `ManifestStore`, `DataConfig` are referenced identically across all tasks; the four fetcher classes are named consistently (`EONETFetcher`, `USGSFetcher`, `NOAAStormsFetcher`, `FIRMSFetcher`). `harmonize_mark` signature and `Mark` enum values match the registry. Field names (`time_start`, `dedup_group_id`, `magnitude`, etc.) align between schema, store DDL, and tests.

**Known operational risk** — Task 17 hits live APIs and produces a multi-GB dataset over the full historical window. A failed run is recoverable (manifests + idempotent inserts), but the first full ingest may take a few hours. If iteration speed matters more than completeness, start with `--since 2020-01-01` to validate everything works at smaller scale before going full historical.
