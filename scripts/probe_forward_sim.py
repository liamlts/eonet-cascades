"""Forward-sim degeneracy probe.

Question: every row of the n=5000 forward-sim transition matrix peaks at
wildfire in 0.84-0.92. Three hypotheses for why (from the Task 14 notebook
commentary):
    (a) class imbalance swamps the conditioning
    (b) under-training
    (c) bbox-center cold-start seed gives the model no useful context

This script teases (a) and (c) apart by inspecting the per-mark intensity
lambda_k DIRECTLY at the moment forward-sim would sample its first child,
for several seed configurations:

    Seed A: single parent event at bbox center (current forward-sim seed)
    Seed B: N=50 real historical events from the Jul-2024 val slice, then
            the parent event appended.

If lambda_k(parent=p) -- the K-vector right after the seed -- is essentially
independent of p, the model is not parent-conditioning at all and forward-sim
can't be fixed by changing the seed. If lambda_k differs across p in absolute
terms but the row-normalized P(k) = lambda_k / sum_k lambda_k is still
wildfire-dominated, the issue is marginal-rate domination (a). If the
row-normalized P(k) differs meaningfully across p, the cold-start seed (c) is
the bottleneck and we can fix forward-sim by warm-starting it.
"""
from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from eonet_cascades.config import DataConfig
from eonet_cascades.data.store import EventStore
from eonet_cascades.interpret.forward_sim_matrix import _lambda_k_at
from eonet_cascades.models.neural_hawkes import NeuralHawkes

RUN_DIR = Path("runs/tier1_mlp/20260526_141553")
SLICE_START = datetime(2024, 7, 1, tzinfo=UTC)
SLICE_END = datetime(2024, 8, 1, tzinfo=UTC)
N_HIST = 50
T_QUERY_DAYS_AFTER_SEED = 0.5
SEED = 0


def main() -> None:
    ckpt = torch.load(RUN_DIR / "checkpoint_best.pt", weights_only=False)
    mark_names: list[str] = ckpt["mark_names"]
    n_marks = len(mark_names)
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
    cx = 0.5 * (cfg.bbox[0] + cfg.bbox[2])
    cy = 0.5 * (cfg.bbox[1] + cfg.bbox[3])

    print(f"K = {n_marks}, marks = {mark_names}")
    print(f"bbox center: ({cx:.2f}, {cy:.2f})")

    # ----- Seed A: bbox-center single-event seed (the current forward-sim seed) -----
    # Time sweep so we catch any parent-conditioning that may be present right at the
    # event and decay away before t=0.5 days (the default forward-sim query time).
    t_sweep = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0]
    print("\n=== Seed A: bbox-center, single parent event ===")
    seed_a_per_t = {}
    for t_q in t_sweep:
        rows = []
        with torch.no_grad():
            for p in range(n_marks):
                times = torch.tensor([0.0], dtype=torch.float32)
                lons = torch.tensor([cx], dtype=torch.float32)
                lats = torch.tensor([cy], dtype=torch.float32)
                marks = torch.tensor([p], dtype=torch.long)
                lam_k = _lambda_k_at(model, times, lons, lats, marks, t=t_q)
                rows.append(lam_k.detach().numpy())
        seed_a_per_t[t_q] = np.array(rows)

    seed_a = seed_a_per_t[T_QUERY_DAYS_AFTER_SEED]
    print(f"\nlambda_k (rows=parent, cols=child) at t={T_QUERY_DAYS_AFTER_SEED} days, absolute:")
    print_table(seed_a, mark_names)
    print(f"\nlambda_k / sum_k at t={T_QUERY_DAYS_AFTER_SEED} days (what forward-sim's multinomial sees):")
    row_norm_a = seed_a / seed_a.sum(axis=1, keepdims=True)
    print_table(row_norm_a, mark_names)

    print("\nTime sweep: total/max |row - row_mean| across parents at each query time t:")
    print(f"{'t (days)':>10}  {'total |dev|':>14}  {'max |dev|':>12}")
    for t_q in t_sweep:
        m = seed_a_per_t[t_q]
        rn = m / m.sum(axis=1, keepdims=True)
        total_dev = float(np.abs(rn - rn.mean(axis=0, keepdims=True)).sum())
        max_dev = float(np.abs(rn - rn.mean(axis=0, keepdims=True)).max())
        print(f"{t_q:>10.4f}  {total_dev:>14.6f}  {max_dev:>12.6f}")

    # ----- Seed B: real historical seed (last N=50 events from val) -----
    # Snapshot to a sibling on the source volume — the boot disk is tight, the
    # external Seagate is not.
    print(f"\n=== Seed B: N={N_HIST} real historical events + parent event ===")
    src = Path(cfg.duckdb_path)
    snap = src.parent / "probe_snap.duckdb"
    print(f"snapshotting DuckDB to {snap} (same volume as source)...")
    shutil.copy2(src, snap)
    store = EventStore(snap, read_only=True)
    df = store.query_events(time_start=SLICE_START, time_end=SLICE_END)
    df = df.sort("time_start").tail(N_HIST)
    print(f"loaded {df.height} historical events from {SLICE_START.date()}..{SLICE_END.date()}")

    mark_to_idx = {m: i for i, m in enumerate(mark_names)}
    times_np = df["time_start"].to_numpy().astype("datetime64[us]")
    t0 = times_np.min()
    t_days = (times_np - t0).astype("timedelta64[us]").astype("float32") / (86_400 * 1e6)
    lons_np = df["longitude"].to_numpy().astype("float32")
    lats_np = df["latitude"].to_numpy().astype("float32")
    marks_np = np.array([mark_to_idx[m] for m in df["mark"].to_list()], dtype=np.int64)

    hist_end_day = float(t_days.max())
    parent_t = hist_end_day + 0.01  # right after the last historical event
    query_t = parent_t + T_QUERY_DAYS_AFTER_SEED

    seed_b_table = []
    with torch.no_grad():
        for p in range(n_marks):
            t_seq = np.concatenate([t_days, np.array([parent_t], dtype=np.float32)])
            lon_seq = np.concatenate([lons_np, np.array([cx], dtype=np.float32)])
            lat_seq = np.concatenate([lats_np, np.array([cy], dtype=np.float32)])
            mark_seq = np.concatenate([marks_np, np.array([p], dtype=np.int64)])
            lam_k = _lambda_k_at(
                model,
                torch.tensor(t_seq),
                torch.tensor(lon_seq),
                torch.tensor(lat_seq),
                torch.tensor(mark_seq),
                t=query_t,
            )
            seed_b_table.append(lam_k.detach().numpy())
    seed_b = np.array(seed_b_table)

    print("\nlambda_k (rows=parent, cols=child), absolute:")
    print_table(seed_b, mark_names)
    print("\nlambda_k / sum_k (row-normalized):")
    row_norm_b = seed_b / seed_b.sum(axis=1, keepdims=True)
    print_table(row_norm_b, mark_names)

    print("\nPer-parent: max row deviation from the wildfire-marginal row:")
    wf_row_b = row_norm_b[mark_names.index("wildfire")]
    for p in range(n_marks):
        max_dev = float(np.abs(row_norm_b[p] - wf_row_b).max())
        print(f"  parent={mark_names[p]:<14}  max |row - wildfire_row| = {max_dev:.4f}")

    # Clean up the snapshot.
    try:
        snap.unlink()
    except OSError:
        pass

    # ----- Verdict -----
    print("\n=== verdict ===")
    a_total_dev = float(np.abs(row_norm_a - row_norm_a.mean(axis=0, keepdims=True)).sum())
    b_total_dev = float(np.abs(row_norm_b - row_norm_b.mean(axis=0, keepdims=True)).sum())
    print(f"Seed A (cold) total |row - row_mean| across parents: {a_total_dev:.4f}")
    print(f"Seed B (warm) total |row - row_mean| across parents: {b_total_dev:.4f}")
    if a_total_dev < 0.01 and b_total_dev < 0.01:
        print("=> model is NOT conditioning on parent mark in either case; the model itself")
        print("   is the bottleneck (under-training or insufficient capacity).")
    elif b_total_dev > 5 * a_total_dev:
        print("=> warm-start seed produces meaningfully more parent-conditioning. The")
        print("   bbox-center cold-start seed is the bottleneck; forward-sim should be")
        print("   re-implemented to seed from a real historical context.")
    else:
        print("=> warm-start does not help materially. Either the marginal-rate domination")
        print("   (hypothesis a) or model under-training (hypothesis b) dominates.")


def print_table(matrix: np.ndarray, mark_names: list[str]) -> None:
    header = " " * 14 + " ".join(f"{m[:11]:>12s}" for m in mark_names)
    print(header)
    for p in range(matrix.shape[0]):
        row = f"{mark_names[p]:<14s}" + " ".join(f"{matrix[p, c]:>12.4f}" for c in range(matrix.shape[1]))
        print(row)


if __name__ == "__main__":
    main()
