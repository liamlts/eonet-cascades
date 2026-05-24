"""Top-level Typer CLI for the eonet-cascades project."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from eonet_cascades import __version__
from eonet_cascades.config import DataConfig, load_data_config
from eonet_cascades.data.ingest import run_ingest

app = typer.Typer(
    name="eonet",
    help="Spatio-temporal point process benchmark for natural-hazard event cascades.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    """Root callback — disables Typer's single-command flattening so
    `eonet --help` shows the app help (not the only subcommand's help) until
    Task 16 adds a second command."""


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


@app.command()
def ingest(
    catalogs: Annotated[
        str,
        typer.Option(help="Comma-separated list of catalogs"),
    ] = "eonet,usgs,noaa,firms",
    since: Annotated[
        str,
        typer.Option(help="ISO date, inclusive lower bound"),
    ] = "2000-01-01",
    until: Annotated[
        str | None,
        typer.Option(help="ISO date, exclusive upper bound (default: now)"),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(help="Optional YAML config path"),
    ] = None,
) -> None:
    """Fetch + harmonize + dedup + persist events from the specified catalogs."""
    cfg = load_data_config(config) if config else DataConfig()
    since_dt = datetime.fromisoformat(since).replace(tzinfo=UTC)
    until_dt = (
        datetime.fromisoformat(until).replace(tzinfo=UTC)
        if until
        else datetime.now(UTC)
    )
    cat_list = [c.strip() for c in catalogs.split(",") if c.strip()]
    counts = run_ingest(cfg, since=since_dt, until=until_dt, catalogs=cat_list)
    console.print(counts)
