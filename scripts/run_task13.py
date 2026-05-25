"""Task 13: compute Tier 1 attribution + forward-sim transition matrices for the trained checkpoint."""
from __future__ import annotations

import shutil
import tempfile
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
SAMPLE_N = 2000
SEED = 0


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

    model = NeuralHawkes(n_marks=n_marks, hidden_dim=64)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cfg = DataConfig()
    print(f"snapshotting {cfg.duckdb_path} ...")
    snap = Path(tempfile.mkdtemp()) / "events.duckdb"
    shutil.copy2(cfg.duckdb_path, snap)
    store = EventStore(snap, read_only=True)
    df = store.query_events(time_start=SLICE_START, time_end=SLICE_END)
    print(f"raw Jul 2024 slice: {df.height} events")
    df = df.sample(min(SAMPLE_N, df.height), seed=SEED)
    print(f"sampled: {df.height} events (seed={SEED})")
    counts = df.group_by("mark").len().sort("len", descending=True)
    print("mark counts in sample:")
    print(counts)

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

    print("\n=== attribution ===")
    t_start = time.time()
    A = compute_attribution_matrix(model, t_in, lo_in, la_in, mk_in, n_marks=n_marks)
    print(f"  done in {time.time() - t_start:.1f}s")

    print("\n=== forward-sim transition matrix ===")
    t_start = time.time()
    T = compute_transition_matrix(
        model, n_marks=n_marks, bbox=cfg.bbox, n_trajectories=200, window_days=14.0,
    )
    print(f"  done in {time.time() - t_start:.1f}s")

    A_np = A.numpy()
    T_np = T.numpy()

    def write_matrix_csv(M: np.ndarray, path: Path) -> None:
        # Header: empty + mark_names; first column = parent label.
        cols = {mark_names[i]: M[:, i] for i in range(n_marks)}
        df_out = pl.DataFrame({"parent": mark_names, **cols})
        df_out.write_csv(path)

    write_matrix_csv(A_np, RUN_DIR / "attribution_matrix.csv")
    write_matrix_csv(T_np, RUN_DIR / "forward_sim_matrix.csv")

    for name, M in [("attribution_matrix", A_np), ("forward_sim_matrix", T_np)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(n_marks))
        ax.set_yticks(range(n_marks))
        ax.set_xticklabels(mark_names, rotation=45, ha="right")
        ax.set_yticklabels(mark_names)
        ax.set_xlabel("child")
        ax.set_ylabel("parent")
        title = "Tier 1 attribution matrix" if name == "attribution_matrix" else "Tier 1 forward-sim transition matrix"
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(RUN_DIR / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"saved {RUN_DIR / (name + '.png')}")

    def print_matrix(M: np.ndarray, title: str) -> None:
        print(title)
        header = "parent          " + " ".join(f"{m[:12]:>13s}" for m in mark_names)
        print(header)
        for p in range(n_marks):
            row = f"{mark_names[p]:<14s}  " + " ".join(f"{M[p,c]:>13.4f}" for c in range(n_marks))
            print(row)

    print("\n=== diagnostics ===")
    print_matrix(A_np, "Attribution matrix (parent rows × child cols):")
    print()
    print_matrix(T_np, "Forward-sim matrix (parent rows × child cols, row-normalized):")

    print("\nAttribution row sums:")
    for i, m in enumerate(mark_names):
        print(f"  {m}: {A_np[i].sum():.3f}")
    print("\nForward-sim row sums:")
    for i, m in enumerate(mark_names):
        print(f"  {m}: {T_np[i].sum():.3f}")

    def top3(M: np.ndarray, label: str) -> None:
        flat = [(mark_names[p], mark_names[c], M[p, c])
                for p in range(n_marks) for c in range(n_marks)]
        flat.sort(key=lambda x: -x[2])
        print(f"\nTop 3 {label} (parent → child):")
        for p, c, v in flat[:3]:
            print(f"  {p:13s} → {c:13s}  {v:.4f}")

    top3(A_np, "attribution")
    top3(T_np, "forward-sim")

    print("\nPer-row top-child:")
    print(f"{'parent':<14}  {'attr top child':<20}  {'fwd top child':<20}")
    for p in range(n_marks):
        ac = int(np.argmax(A_np[p]))
        tc = int(np.argmax(T_np[p]))
        a_val = A_np[p, ac]
        t_val = T_np[p, tc]
        print(f"{mark_names[p]:<14}  {mark_names[ac]:<14} ({a_val:.3f})    "
              f"{mark_names[tc]:<14} ({t_val:.3f})")

    diag = float(np.diag(A_np).sum() / max(A_np.sum(), 1e-12))
    print(f"\nattribution diagonal mass / total: {diag:.3f}")

    if np.allclose(A_np, 0):
        print("WARNING: attribution matrix is all zero")
    if np.isnan(A_np).any() or np.isnan(T_np).any():
        print("WARNING: NaN entries detected")
    if A_np.shape[0] > 1:
        u, s, vh = np.linalg.svd(A_np)
        cond = s[0] / max(s[1], 1e-12)
        print(f"attribution SVD top-2 singular ratio s1/s2: {cond:.2f} "
              f"(>>1 ⇒ near rank-1)")


if __name__ == "__main__":
    main()
