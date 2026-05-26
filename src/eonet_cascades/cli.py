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
from eonet_cascades.models.neural_hawkes import NeuralHawkes
from eonet_cascades.training.neural_loop import (
    TrainChunk,
    mark_rebalance_weights,
    train_one_epoch,
)

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
    until_dt = datetime.fromisoformat(until).replace(tzinfo=UTC) if until else datetime.now(UTC)
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
    l1_lambda: Annotated[float, typer.Option(help="L1 regularization on alpha (0 = none)")] = 0.0,
) -> None:
    """Fit Tier 0 parametric Hawkes on a windowed subsample of the event archive."""
    import numpy as np

    cfg = load_data_config(config) if config else DataConfig()
    since_dt = datetime.fromisoformat(since).replace(tzinfo=UTC)
    until_dt = datetime.fromisoformat(until).replace(tzinfo=UTC)
    # Snapshot the DB to /tmp so we coexist with anything that holds the
    # write lock (e.g. an interactive notebook kernel). DuckDB read-only
    # mode is not enough — it still conflicts with an outstanding RW lock.
    import atexit
    import shutil
    import tempfile

    snapshot_dir = Path(tempfile.mkdtemp(prefix="eonet_tier0_"))
    atexit.register(shutil.rmtree, snapshot_dir, ignore_errors=True)
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
    result = model.fit(events_dict, (0.0, t_end_days), max_iter=max_iter, l1_lambda=l1_lambda)
    console.print(result)

    out = out_dir or (Path("runs") / "tier0" / datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
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


# --- Tier 1 train-neural-hawkes (Plan 4 Task 9) ---


@model_app.command("train-neural-hawkes")
def model_train_neural_hawkes(
    since: Annotated[str, typer.Option(help="Train start (ISO date)")] = "2022-01-01",
    until: Annotated[str, typer.Option(help="Train end (ISO date)")] = "2024-06-30",
    val_until: Annotated[str, typer.Option(help="Val end (ISO date)")] = "2024-12-31",
    sample: Annotated[
        int, typer.Option(help="Max events to fit on (random subsample of train)")
    ] = 200000,
    config: Annotated[Path | None, typer.Option(help="Optional YAML data config")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    hidden_dim: Annotated[int, typer.Option(help="CTLSTM hidden dim")] = 64,
    n_epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 10,
    lr: Annotated[float, typer.Option(help="AdamW learning rate")] = 1e-3,
    chunk_days: Annotated[float, typer.Option(help="BPTT chunk size in days")] = 7.0,
    device: Annotated[str, typer.Option(help="cpu / cuda / mps")] = "cpu",
    out_dir: Annotated[
        Path | None,
        typer.Option(help="Output dir; default runs/tier1/{timestamp}"),
    ] = None,
    mark_rebalance: Annotated[
        bool,
        typer.Option(
            "--mark-rebalance/--no-mark-rebalance",
            help=(
                "Apply class-rebalanced training to break the mark-head "
                "class-collapse documented in commit 420d5a3 (Tier 1.5)."
            ),
        ),
    ] = False,
    rebalance_mode: Annotated[
        str,
        typer.Option(
            help="When --mark-rebalance is set: 'inverse-sqrt' (default) or 'inverse-frequency'."
        ),
    ] = "inverse-sqrt",
    stratify_train: Annotated[
        bool,
        typer.Option(
            "--stratify-train/--no-stratify-train",
            help=(
                "Stratified subsample: force ALL events of marks below "
                "--stratify-threshold into the training subsample, then "
                "random-fill the rest. Prevents rare marks from being "
                "absent entirely after random subsampling."
            ),
        ),
    ] = False,
    stratify_threshold: Annotated[
        float,
        typer.Option(
            help=(
                "Mark-frequency fraction below which a mark is forced into the "
                "training subsample whole. Only applies when --stratify-train is set."
            )
        ),
    ] = 0.01,
    mark_head: Annotated[
        str,
        typer.Option(
            help=(
                "Mark-intensity head architecture. 'linear' is the original Tier 1 "
                "single nn.Linear head. 'mlp' is a 2-layer ReLU MLP "
                "(H -> H//2 -> n_marks) added 2026-05-26 to test whether non-linear "
                "capacity breaks the rank-1 collapse documented in tier1_5-result.md."
            )
        ),
    ] = "linear",
) -> None:
    """Train Tier 1 Neural Hawkes on a windowed sample of the event archive."""
    import atexit
    import shutil
    import tempfile
    import time

    import numpy as np
    import polars as pl
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = load_data_config(config) if config else DataConfig()
    since_dt = datetime.fromisoformat(since).replace(tzinfo=UTC)
    until_dt = datetime.fromisoformat(until).replace(tzinfo=UTC)
    val_until_dt = datetime.fromisoformat(val_until).replace(tzinfo=UTC)

    snapshot_dir = Path(tempfile.mkdtemp(prefix="eonet_tier1_"))
    atexit.register(shutil.rmtree, snapshot_dir, ignore_errors=True)
    snapshot_path = snapshot_dir / "events.duckdb"
    console.print(f"Snapshotting DB to {snapshot_path}...")
    shutil.copy2(cfg.duckdb_path, snapshot_path)
    store = EventStore(snapshot_path, read_only=True)
    df_train = store.query_events(time_start=since_dt, time_end=until_dt)
    df_val = store.query_events(time_start=until_dt, time_end=val_until_dt)
    console.print(f"Loaded {df_train.height:,} train events and {df_val.height:,} val events")
    if df_train.height > sample:
        if stratify_train:
            df_train = _stratified_subsample(
                df_train, n_target=sample, rare_threshold_frac=stratify_threshold, seed=seed
            )
            counts = df_train.group_by("mark").len().sort("len", descending=True)
            console.print(
                f"Stratified-subsampled train to {df_train.height:,} (rare threshold "
                f"= {stratify_threshold:.3f}). Mark counts:"
            )
            console.print(counts)
        else:
            df_train = df_train.sample(sample, seed=seed)
            console.print(f"Subsampled train to {df_train.height:,}")

    mark_names = sorted(
        set(df_train["mark"].unique().to_list()) | set(df_val["mark"].unique().to_list())
    )
    n_marks = len(mark_names)
    mark_to_idx = {m: i for i, m in enumerate(mark_names)}
    console.print(f"K = {n_marks} marks: {mark_names}")

    # Compute mark-rebalance weights from the train marks (post-stratification).
    if mark_rebalance:
        train_marks_idx = np.array(
            [mark_to_idx[m] for m in df_train["mark"].to_list()], dtype=np.int64
        )
        mark_weights = mark_rebalance_weights(train_marks_idx, n_marks=n_marks, mode=rebalance_mode)
        console.print(
            f"Mark rebalance ({rebalance_mode}): weights = "
            + ", ".join(f"{m}={mark_weights[i].item():.3f}" for i, m in enumerate(mark_names))
        )
    else:
        mark_weights = None

    def chunked(df: pl.DataFrame, t0_dt: datetime) -> list[TrainChunk]:
        df = df.sort("time_start")
        if df.height == 0:
            return []
        times_np = df["time_start"].to_numpy().astype("datetime64[us]")
        t_arr = (times_np - np.datetime64(t0_dt.replace(tzinfo=None))).astype(
            "timedelta64[us]"
        ).astype(np.float64) / (86_400 * 1e6)
        marks_idx = np.array([mark_to_idx[m] for m in df["mark"].to_list()], dtype=np.int64)
        chunks: list[TrainChunk] = []
        max_t = float(t_arr.max())
        c_start = 0.0
        while c_start < max_t + 1e-9:
            c_end = c_start + chunk_days
            mask = (t_arr >= c_start) & (t_arr < c_end)
            if mask.any():
                chunks.append(
                    TrainChunk(
                        times=torch.tensor(t_arr[mask], dtype=torch.float32),
                        lons=torch.tensor(df["longitude"].to_numpy()[mask], dtype=torch.float32),
                        lats=torch.tensor(df["latitude"].to_numpy()[mask], dtype=torch.float32),
                        marks=torch.tensor(marks_idx[mask], dtype=torch.long),
                        window=(c_start, c_end),
                    )
                )
            c_start = c_end
        return chunks

    train_chunks = chunked(df_train, since_dt)
    val_chunks = chunked(df_val, until_dt)
    console.print(f"Built {len(train_chunks)} train chunks, {len(val_chunks)} val chunks")

    model = NeuralHawkes(
        n_marks=n_marks, hidden_dim=hidden_dim, mark_head=mark_head
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs * max(1, len(train_chunks)))

    best_val_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []
    for epoch in range(n_epochs):
        t0_e = time.perf_counter()
        train_info = train_one_epoch(
            model,
            train_chunks,
            optimizer,
            scheduler,
            device=device,
            mark_weights=mark_weights,
        )
        val_info = _tier1_eval_loop(model, val_chunks, device=device)
        elapsed = time.perf_counter() - t0_e
        record = {
            "epoch": epoch,
            "train_nll": train_info["nll_per_event"],
            "val_nll": val_info["nll_per_event"],
            "elapsed_s": elapsed,
        }
        history.append(record)
        console.print(record)
        if val_info["nll_per_event"] < best_val_nll:
            best_val_nll = val_info["nll_per_event"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    out = out_dir or (Path("runs") / "tier1" / datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "mark_names": mark_names,
            "config": {
                "since": since,
                "until": until,
                "val_until": val_until,
                "hidden_dim": hidden_dim,
                "mark_head": mark_head,
                "n_marks": n_marks,
            },
        },
        out / "checkpoint_best.pt",
    )
    torch.save(
        {"state_dict": model.state_dict(), "mark_names": mark_names},
        out / "checkpoint_final.pt",
    )
    pl.DataFrame(history).write_csv(out / "train_curves.csv")
    console.print(f"Saved checkpoints + curves to {out}")
    store.close()


def _stratified_subsample(df, n_target: int, rare_threshold_frac: float, seed: int):
    """Force all events of rare marks into the subsample; random-fill the rest.

    A mark is "rare" if its share of the input is < rare_threshold_frac.
    Returns up to n_target rows, sorted by time_start. Untyped on df to keep
    the top-level cli.py free of a polars import.
    """
    import polars as pl_

    total = df.height
    if total <= n_target:
        return df.sort("time_start")
    counts = df.group_by("mark").len()
    rare_marks = [
        m for m, c in zip(counts["mark"].to_list(), counts["len"].to_list(), strict=True)
        if c / total < rare_threshold_frac
    ]
    if not rare_marks:
        return df.sample(n_target, seed=seed).sort("time_start")
    keep_df = df.filter(pl_.col("mark").is_in(rare_marks))
    rest_df = df.filter(~pl_.col("mark").is_in(rare_marks))
    n_rest = max(0, n_target - keep_df.height)
    rest_sample = (
        rest_df.sample(min(n_rest, rest_df.height), seed=seed)
        if n_rest > 0
        else rest_df.head(0)
    )
    return pl_.concat([keep_df, rest_sample], how="vertical").sort("time_start")


def _tier1_eval_loop(model: NeuralHawkes, chunks: list, device: str) -> dict[str, float]:
    """Evaluate val/test NLL with no_grad."""
    import torch as _torch

    model.eval()
    total_loss = 0.0
    total_events = 0
    with _torch.no_grad():
        for chunk in chunks:
            if chunk.times.numel() == 0:
                continue
            ll = model.log_likelihood(
                chunk.times.to(device),
                chunk.lons.to(device),
                chunk.lats.to(device),
                chunk.marks.to(device),
                chunk.window,
            )
            total_loss += float(-ll.item())
            total_events += int(chunk.times.shape[0])
    return {"nll_per_event": total_loss / max(1, total_events)}
