"""Task 13 v2: stratified n=5000 attribution + forward-sim regeneration.

Differences vs scripts/run_task13.py:
  * SAMPLE_N = 5000 (was 2000)
  * Stratified sampling: include ALL dust_haze and ALL landslide events from
    the Jul-2024 slice (the v1 random sample missed both, leaving those rows
    and columns of the attribution matrix structurally zero). Remaining
    slots are filled by uniform-random draw from the other marks.
  * Overwrites runs/tier1/<run>/attribution_matrix.{csv,png} and
    forward_sim_matrix.{csv,png}.

Expected wall time on this Mac: ~40 minutes for the attribution pass,
dominated by the per-child autograd.grad backward calls. Forward-sim is
~minutes. The attribution.py vectorization patch (commit c32ee41) is
load-bearing here; the pre-patch O(n^3)-wall version would have taken
hours at this n.
"""
from __future__ import annotations

import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from eonet_cascades.config import DataConfig
from eonet_cascades.data.store import EventStore
from eonet_cascades.interpret.attribution import compute_attribution_matrix
from eonet_cascades.interpret.forward_sim_matrix import compute_transition_matrix
from eonet_cascades.models.neural_hawkes import NeuralHawkes

RUN_DIR = Path("runs/tier1/20260525_162056")
SLICE_START = datetime(2024, 7, 1, tzinfo=UTC)
SLICE_END = datetime(2024, 8, 1, tzinfo=UTC)
SAMPLE_N = 5000
SEED = 0
# Marks that the v1 random sample missed and that we explicitly preserve here.
STRATIFY_KEEP_ALL = ["dust_haze", "landslide"]


def _stratified_sample(df: pl.DataFrame, n: int, keep_all: list[str], seed: int) -> pl.DataFrame:
    """Take all rows whose mark is in keep_all, then fill the remaining slots
    by uniform-random draw from the rest. Total rows = min(n, df.height)."""
    keep_df = df.filter(pl.col("mark").is_in(keep_all))
    rest_df = df.filter(~pl.col("mark").is_in(keep_all))
    n_keep = keep_df.height
    n_rest_target = max(0, n - n_keep)
    n_rest_actual = min(n_rest_target, rest_df.height)
    if n_rest_actual > 0:
        rest_sample = rest_df.sample(n_rest_actual, seed=seed)
    else:
        rest_sample = rest_df.head(0)
    out = pl.concat([keep_df, rest_sample], how="vertical")
    return out


def main() -> None:
    ckpt = torch.load(RUN_DIR / "checkpoint_best.pt", weights_only=False)
    mark_names: list[str] = ckpt["mark_names"]
    n_marks = len(mark_names)
    expected = [
        "dust_haze", "earthquake", "flood", "landslide",
        "severe_storm", "tornado", "wildfire",
    ]
    assert mark_names == expected, f"mark order mismatch: {mark_names} vs {expected}"
    print(f"checkpoint mark_names OK ({n_marks} marks)")

    ckpt_cfg = ckpt.get("config", {})
    # NOTE: any checkpoint missing 'config' is assumed linear; an MLP-head
    # checkpoint without a config dict would silently load the wrong architecture.
    mark_head_str = ckpt_cfg.get("mark_head", "linear")
    hidden_dim_val = ckpt_cfg.get("hidden_dim", 64)
    model = NeuralHawkes(
        n_marks=n_marks, hidden_dim=hidden_dim_val, mark_head=mark_head_str
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cfg = DataConfig()
    # Snapshot onto the source volume rather than /tmp — the boot disk has 200 MB
    # to spare and the DuckDB is ~1.1 GB. The source volume (external) has TB free.
    src = Path(cfg.duckdb_path)
    snap = src.parent / "run_task13_v2_snap.duckdb"
    print(f"snapshotting {src} -> {snap}")
    shutil.copy2(src, snap)
    store = EventStore(snap, read_only=True)
    df = store.query_events(time_start=SLICE_START, time_end=SLICE_END)
    print(f"raw Jul 2024 slice: {df.height} events")

    raw_counts = df.group_by("mark").len().sort("len", descending=True)
    print("raw mark counts in slice:")
    print(raw_counts)

    df = _stratified_sample(df, SAMPLE_N, STRATIFY_KEEP_ALL, seed=SEED)
    print(f"stratified sample: {df.height} events "
          f"(forced-keep {STRATIFY_KEEP_ALL}, seed={SEED})")
    counts = df.group_by("mark").len().sort("len", descending=True)
    print("mark counts in sample:")
    print(counts)

    for keep in STRATIFY_KEEP_ALL:
        present = int(df.filter(pl.col("mark") == keep).height)
        if present == 0:
            print(f"WARNING: stratified mark '{keep}' is absent from the raw slice")
        else:
            print(f"  '{keep}' in sample: {present}")

    mark_to_idx = {m: i for i, m in enumerate(mark_names)}
    times_np = df["time_start"].to_numpy().astype("datetime64[us]")
    t0 = times_np.min()
    t_days = (times_np - t0).astype("timedelta64[us]").astype("float32") / (86_400 * 1e6)
    order = np.argsort(t_days)
    t_days = t_days[order]
    lons = df["longitude"].to_numpy().astype("float32")[order]
    lats = df["latitude"].to_numpy().astype("float32")[order]
    marks_np = np.array([mark_to_idx[m] for m in df["mark"].to_list()], dtype=np.int64)[order]
    present_marks = sorted(set(marks_np.tolist()))
    print(f"present mark indices in sample: {present_marks} "
          f"({[mark_names[i] for i in present_marks]})")

    t_in = torch.tensor(t_days)
    lo_in = torch.tensor(lons)
    la_in = torch.tensor(lats)
    mk_in = torch.tensor(marks_np)

    print(f"\n=== attribution (n={df.height}) ===")
    t_start = time.time()
    a_tensor = compute_attribution_matrix(model, t_in, lo_in, la_in, mk_in, n_marks=n_marks)
    print(f"  done in {time.time() - t_start:.1f}s")

    print("\n=== forward-sim transition matrix ===")
    t_start = time.time()
    t_tensor = compute_transition_matrix(
        model, n_marks=n_marks, bbox=cfg.bbox, n_trajectories=200, window_days=14.0,
    )
    print(f"  done in {time.time() - t_start:.1f}s")

    a_np = a_tensor.numpy()
    t_np = t_tensor.numpy()

    def write_matrix_csv(matrix: np.ndarray, path: Path) -> None:
        cols = {mark_names[i]: matrix[:, i] for i in range(n_marks)}
        df_out = pl.DataFrame({"parent": mark_names, **cols})
        df_out.write_csv(path)

    write_matrix_csv(a_np, RUN_DIR / "attribution_matrix.csv")
    write_matrix_csv(t_np, RUN_DIR / "forward_sim_matrix.csv")

    for name, matrix in [("attribution_matrix", a_np), ("forward_sim_matrix", t_np)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap="viridis", vmin=0)
        ax.set_xticks(range(n_marks))
        ax.set_yticks(range(n_marks))
        ax.set_xticklabels(mark_names, rotation=45, ha="right")
        ax.set_yticklabels(mark_names)
        ax.set_xlabel("child")
        ax.set_ylabel("parent")
        title = ("Tier 1 attribution matrix (n=5000, stratified)"
                 if name == "attribution_matrix"
                 else "Tier 1 forward-sim transition matrix")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(RUN_DIR / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"saved {RUN_DIR / (name + '.png')}")

    def print_matrix(matrix: np.ndarray, title: str) -> None:
        print(title)
        header = "parent          " + " ".join(f"{m[:12]:>13s}" for m in mark_names)
        print(header)
        for p in range(n_marks):
            row = (f"{mark_names[p]:<14s}  "
                   + " ".join(f"{matrix[p, c]:>13.4f}" for c in range(n_marks)))
            print(row)

    print("\n=== diagnostics ===")
    print_matrix(a_np, "Attribution matrix (parent rows × child cols):")
    print()
    print_matrix(t_np, "Forward-sim matrix (parent rows × child cols, row-normalized):")

    print("\nAttribution row sums:")
    for i, m in enumerate(mark_names):
        print(f"  {m}: {a_np[i].sum():.3f}")
    print("\nForward-sim row sums:")
    for i, m in enumerate(mark_names):
        print(f"  {m}: {t_np[i].sum():.3f}")

    def top3(matrix: np.ndarray, label: str) -> None:
        flat = [(mark_names[p], mark_names[c], matrix[p, c])
                for p in range(n_marks) for c in range(n_marks)]
        flat.sort(key=lambda x: -x[2])
        print(f"\nTop 3 {label} (parent → child):")
        for p, c, v in flat[:3]:
            print(f"  {p:13s} → {c:13s}  {v:.4f}")

    top3(a_np, "attribution")
    top3(t_np, "forward-sim")

    print("\nPer-row top-child:")
    print(f"{'parent':<14}  {'attr top child':<20}  {'fwd top child':<20}")
    for p in range(n_marks):
        ac = int(np.argmax(a_np[p]))
        tc = int(np.argmax(t_np[p]))
        a_val = a_np[p, ac]
        t_val = t_np[p, tc]
        print(f"{mark_names[p]:<14}  {mark_names[ac]:<14} ({a_val:.3f})    "
              f"{mark_names[tc]:<14} ({t_val:.3f})")

    diag = float(np.diag(a_np).sum() / max(a_np.sum(), 1e-12))
    print(f"\nattribution diagonal mass / total: {diag:.3f}")

    if np.allclose(a_np, 0):
        print("WARNING: attribution matrix is all zero")
    if np.isnan(a_np).any() or np.isnan(t_np).any():
        print("WARNING: NaN entries detected")
    if a_np.shape[0] > 1:
        _, s, _ = np.linalg.svd(a_np)
        cond = s[0] / max(s[1], 1e-12)
        print(f"attribution SVD top-2 singular ratio s1/s2: {cond:.2f} "
              f"(>>1 ⇒ near rank-1)")


if __name__ == "__main__":
    main()
