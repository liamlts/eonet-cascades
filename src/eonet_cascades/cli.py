"""Top-level Typer CLI for the eonet-cascades project."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from eonet_cascades import __version__
from eonet_cascades.config import DataConfig, load_data_config
from eonet_cascades.data.ingest import run_ingest
from eonet_cascades.data.store import EventStore
from eonet_cascades.interpret.excitation import excitation_to_dataframe, plot_excitation_heatmap
from eonet_cascades.models.hawkes import KDESpatialBaseline, ParametricHawkes

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


# --- model subcommand group (Plan 2 Task 11) ---

model_app = typer.Typer(help="Fit and inspect point-process models.")
app.add_typer(model_app, name="model")


@model_app.command("train-hawkes")
def model_train_hawkes(
    since: Annotated[str, typer.Option(help="Train-window start (ISO date)")] = "2023-01-01",
    until: Annotated[str, typer.Option(help="Train-window end (ISO date)")] = "2024-01-01",
    sample: Annotated[int, typer.Option(help="Max events to fit on (random subsample)")] = 5000,
    config: Annotated[Path | None, typer.Option(help="Optional YAML data config")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Output dir; default runs/tier0/{timestamp}"),
    ] = None,
    max_iter: Annotated[int, typer.Option(help="L-BFGS-B max iterations")] = 300,
) -> None:
    """Fit Tier 0 parametric Hawkes on a windowed subsample of the event archive."""
    import numpy as np

    cfg = load_data_config(config) if config else DataConfig()
    since_dt = datetime.fromisoformat(since).replace(tzinfo=UTC)
    until_dt = datetime.fromisoformat(until).replace(tzinfo=UTC)
    # Snapshot the DB to /tmp so we coexist with anything that holds the
    # write lock (e.g. an interactive notebook kernel). DuckDB read-only
    # mode is not enough — it still conflicts with an outstanding RW lock.
    import shutil
    import tempfile
    snapshot_dir = Path(tempfile.mkdtemp(prefix="eonet_tier0_"))
    snapshot_path = snapshot_dir / "events.duckdb"
    console.print(f"Snapshotting DB to {snapshot_path}...")
    shutil.copy2(cfg.duckdb_path, snapshot_path)
    store = EventStore(snapshot_path, read_only=True)
    df = store.query_events(time_start=since_dt, time_end=until_dt)
    console.print(f"Loaded {df.height:,} events in window [{since}, {until})")
    if df.height > sample:
        df = df.sample(sample, seed=seed)
        console.print(f"Subsampled to {df.height:,}")
    mark_names = sorted(df["mark"].unique().to_list())
    n_marks = len(mark_names)
    console.print(f"K = {n_marks} marks: {mark_names}")

    bbox = cfg.bbox
    baseline = KDESpatialBaseline.from_events(df, mark_names, bbox, grid_step=1.0)
    model = ParametricHawkes(K=n_marks, bbox=bbox, pi_k=baseline)

    # Convert to numpy event dict using time-since-window-start in days.
    times = df["time_start"].to_numpy().astype("datetime64[us]")
    t0_np = np.datetime64(since_dt.replace(tzinfo=None))
    t_days = (times - t0_np).astype("timedelta64[us]").astype(np.float64) / (86_400 * 1e6)
    mark_to_idx = {m: i for i, m in enumerate(mark_names)}
    events_dict = {
        "time": t_days,
        "lon": df["longitude"].to_numpy().astype(np.float64),
        "lat": df["latitude"].to_numpy().astype(np.float64),
        "mark": np.array([mark_to_idx[m] for m in df["mark"].to_list()], dtype=np.int64),
    }
    t_end_days = (until_dt - since_dt).total_seconds() / 86_400.0
    result = model.fit(events_dict, (0.0, t_end_days), max_iter=max_iter)
    console.print(result)

    out = out_dir or (
        Path("runs") / "tier0" / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "params.pkl", "wb") as f:
        pickle.dump(
            {
                "params": model.params,
                "mark_names": mark_names,
                "bbox": bbox,
                "window": (since, until),
                "fit_result": result,
                "n_events_used": df.height,
            },
            f,
        )
    excitation_to_dataframe(model.params, mark_names).write_csv(out / "alpha.csv")
    fig = plot_excitation_heatmap(model.params, mark_names)
    fig.savefig(out / "alpha.png", dpi=150)
    console.print(f"Saved checkpoint + figures to {out}")
    store.close()
