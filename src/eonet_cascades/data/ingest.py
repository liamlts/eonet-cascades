"""Top-level ingestion orchestrator.

Drives all configured catalog fetchers, harmonizes, dedupes, and writes to the
DuckDB store. Idempotent via manifest state and ON CONFLICT DO NOTHING inserts.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError
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

    # NOTE: v1 design wrote ALL catalogs in one batch and did cross-catalog
    # dedup before a single store.write_events. That lost data on any
    # mid-stream failure (e.g. a FIRMS rate-limit 400 after NOAA succeeded).
    # We now write per catalog so partial progress survives. Cross-catalog
    # dedup becomes a later in-store pass (TODO Plan 2+); for now each
    # catalog gets within-catalog dedup at write time.
    for cat in catalogs:
        last = manifests.last_fetched(cat)
        effective_since = max(last, since) if last is not None else since
        if effective_since >= until:
            console.log(f"[dim]{cat}: nothing to fetch ({effective_since} >= {until})[/]")
            counts[cat] = 0
            continue
        fetcher = build_fetcher(cat, cfg)
        console.log(f"[bold]{cat}[/]: fetching {effective_since} -> {until}")
        cat_events: list[Event] = []
        skipped_validation = 0
        try:
            fetch_iter = fetcher.fetch(effective_since, until, bbox=cfg.bbox)  # type: ignore[arg-type]
        except TypeError:
            fetch_iter = fetcher.fetch(effective_since, until)
        for raw in fetch_iter:
            try:
                ev = fetcher.harmonize(raw)
            except ValidationError:
                skipped_validation += 1
                continue
            if ev is None:
                continue
            if not _in_bbox(ev, cfg.bbox):
                continue
            cat_events.append(ev)
        counts[cat] = len(cat_events)
        skip_note = f" ({skipped_validation} skipped: invalid coords)" if skipped_validation else ""
        console.log(f"  -> harmonized {len(cat_events)} events{skip_note}")

        # Per-catalog dedup + write — preserves progress across catalog failures.
        deduped = assign_dedup_groups(cat_events)
        written = store.write_events(deduped)
        console.log(f"  -> wrote {written} {cat} events to store")
        manifests.set_last_fetched(cat, until)

    store.close()
    return counts


def _in_bbox(ev: Event, bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lon <= ev.longitude <= max_lon and min_lat <= ev.latitude <= max_lat
