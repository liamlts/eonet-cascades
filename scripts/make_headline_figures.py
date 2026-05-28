"""Generate headline figures for the four-run negative-chain writeup.

Outputs to docs/figures/:
  * fig1_kk_grid.png       — 4-panel forward-sim K×K matrices (row-normalized)
  * fig2_convergence.png   — val NLL vs epoch for all 4 runs
  * fig3_marginal_bars.png — wildfire-dominant vs flat marginals across runs
  * fig4_row_dev_bars.png  — primary acceptance criterion across runs

Cold-start probe only (one bbox-center parent event); no Seagate dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from eonet_cascades.models.neural_hawkes import NeuralHawkes


REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Bbox center (matches the probe script's SLICE coordinates).
CX, CY = -97.5, 32.0
T_QUERY = 0.5  # days after the parent event


@dataclass
class RunSpec:
    label: str
    short: str
    run_dir: Path
    val_nll: float | None = None
    color: str = "tab:blue"


RUNS = [
    RunSpec(
        label="Tier 1\n(linear, default)",
        short="tier1",
        run_dir=REPO / "runs/tier1/20260525_162056",
        color="#1f3a5f",
    ),
    RunSpec(
        label="Tier 1.5\n(rebalance + stratify)",
        short="tier1_5",
        run_dir=REPO / "runs/tier1_5/20260526_043203",
        color="#c75b12",
    ),
    RunSpec(
        label="Tier 1-MLP\n(MLP head)",
        short="tier1_mlp",
        run_dir=REPO / "runs/tier1_mlp/20260526_141553",
        color="#1f6b3a",
    ),
    RunSpec(
        label="Tier 1-aux\n(MLP + aux CE λ=1.0)",
        short="tier1_aux",
        run_dir=REPO / "runs/tier1_aux/20260527_224337",
        color="#2c7a7b",
    ),
]


def load_model(run_dir: Path) -> tuple[NeuralHawkes, list[str], dict]:
    ckpt = torch.load(run_dir / "checkpoint_best.pt", weights_only=False)
    cfg = ckpt.get("config", {})
    mark_head = cfg.get("mark_head", "linear")
    hidden_dim = cfg.get("hidden_dim", 64)
    n_marks = cfg["n_marks"]
    mark_names = ckpt["mark_names"]
    model = NeuralHawkes(
        n_marks=n_marks, hidden_dim=hidden_dim, mark_head=mark_head
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, mark_names, cfg


def lambda_k_at_query(model: NeuralHawkes, parent_mark: int, t_query: float) -> torch.Tensor:
    """Cold-start probe: single parent event at (cx, cy, t=0), evaluate λ_k(h(t_query))."""
    times = torch.tensor([0.0], dtype=torch.float32)
    lons = torch.tensor([CX], dtype=torch.float32)
    lats = torch.tensor([CY], dtype=torch.float32)
    marks = torch.tensor([parent_mark], dtype=torch.long)

    with torch.no_grad():
        # Walk the model to get the post-event hidden state.
        n = 1
        hidden_dim = model.hidden_dim
        c_post = torch.zeros(1, hidden_dim)
        c_bar = torch.zeros(1, hidden_dim)
        delta = torch.ones(1, hidden_dim)
        o = torch.zeros(1, hidden_dim)
        t_last = torch.zeros(1)
        # Single event step
        t_i = times[0:1]
        dt = (t_i - t_last).clamp(min=0.0).unsqueeze(-1)
        h_at_t, _ = model.cell.evolve(c_post, c_bar, delta, o, dt)
        ev_inp = model._event_input(lons[0:1], lats[0:1], marks[0:1])
        _, c_post, c_bar, delta, o = model.cell.update(ev_inp, h_at_t, c_post, c_bar)
        t_last = t_i
        # Query at t_query days after the parent event.
        dt_q = torch.tensor([[t_query]], dtype=torch.float32)
        h_at_q, _ = model.cell.evolve(c_post, c_bar, delta, o, dt_q)
        lam_k = model._lambda_k(h_at_q)
        return lam_k.squeeze(0)  # (K,)


def compute_kk_matrix(model: NeuralHawkes, n_marks: int) -> np.ndarray:
    """Cold-start K×K probe: each parent mark, row-normalized λ_k."""
    rows = []
    for p in range(n_marks):
        lam_k = lambda_k_at_query(model, p, T_QUERY)
        row = (lam_k / lam_k.sum()).numpy()
        rows.append(row)
    return np.stack(rows, axis=0)


# ---------- Figure 1: K×K grid ----------
def fig1_kk_grid():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    fig.suptitle(
        "Forward-sim K×K matrices across four runs: rows are parent marks, columns are child marks.\n"
        "All four are row-degenerate — the parent mark does not condition the output composition.",
        fontsize=12, y=1.02,
    )

    all_matrices = []
    mark_names = None
    for ax, run in zip(axes, RUNS, strict=True):
        model, names, cfg = load_model(run.run_dir)
        if mark_names is None:
            mark_names = names
        K = len(names)
        M = compute_kk_matrix(model, K)
        all_matrices.append((run, M, names))

        im = ax.imshow(M, aspect="equal", cmap="viridis", vmin=0, vmax=max(0.5, M.max()))
        ax.set_title(run.label, fontsize=10, fontweight="bold")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        # Abbreviated mark labels
        short_names = [n[:5] for n in names]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("parent mark", fontsize=9)
        ax.set_xlabel("child mark", fontsize=9)

        # Annotate the row-deviation (max across pairwise row differences)
        row_mean = M.mean(axis=0)
        row_dev = float(np.abs(M - row_mean[None, :]).sum())
        ax.text(
            0.5, -0.32,
            f"row-dev (total): {row_dev:.4f}",
            transform=ax.transAxes,
            ha="center", fontsize=8, color="#c75b12" if row_dev < 0.1 else "#1f6b3a",
            fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("P(child | parent), row-normalized", fontsize=9)

    out = FIGS / "fig1_kk_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1: {out}")
    return all_matrices


# ---------- Figure 2: convergence curves ----------
def fig2_convergence():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    for run in RUNS:
        curves_path = run.run_dir / "train_curves.csv"
        if not curves_path.exists():
            print(f"  warn: missing {curves_path}, skipping {run.short}")
            continue
        df = pl.read_csv(curves_path)
        epochs = df["epoch"].to_numpy()
        # Use train_nll_hawkes if present (Tier 1-aux), else fallback to val_nll for the comparable line
        val_nll = df["val_nll"].to_numpy()
        ax.plot(epochs, val_nll, "-o", color=run.color, linewidth=2, markersize=5,
                label=run.label.replace("\n", " — "))
        # Final value annotation
        ax.annotate(f"{val_nll[-1]:.2f}", xy=(epochs[-1], val_nll[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=run.color, fontweight="bold", va="center")

    # Acceptance threshold line
    ax.axhline(4.41, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(14.3, 4.41, "  ≤ 4.41\n  (secondary\n  acceptance)",
            fontsize=8, va="center", color="black", alpha=0.7)

    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel("val NLL / event (pure Hawkes)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Convergence — Tier 1-MLP achieves best val NLL by 4× the convergence rate",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, which="both", alpha=0.3)

    out = FIGS / "fig2_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2: {out}")


# ---------- Figure 3: marginal bars ----------
def fig3_marginal_bars(all_matrices):
    """all_matrices: list of (RunSpec, K×K matrix, mark_names) from fig1.

    Some runs have K=7 (the standard subset) and Tier 1.5 has K=8 because the
    stratified subsample forced the singleton volcanic_eruption event in.
    To plot the marginals on a shared x-axis, project onto the K=7 common
    subset (dropping volcanic_eruption from Tier 1.5 and renormalizing).
    """
    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5), constrained_layout=True)

    # Pick the most common mark list (K=7) as the canonical subset.
    canonical = sorted(
        (names for _, _, names in all_matrices),
        key=lambda ns: (len(ns), ns),
    )[0]
    K = len(canonical)
    width = 0.2
    x = np.arange(K)

    for i, (run, M, names) in enumerate(all_matrices):
        marginal = M[0]  # row 0 = same as any row by construction
        if list(names) != list(canonical):
            # Project onto the canonical subset: keep only marks that appear in canonical.
            keep_idx = [names.index(n) for n in canonical]
            marginal = marginal[keep_idx]
            marginal = marginal / marginal.sum()  # renormalize after dropping
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, marginal, width, color=run.color,
                      label=run.label.replace("\n", " — "), edgecolor="white", linewidth=0.5)
    mark_names = canonical

    ax.set_xticks(x)
    ax.set_xticklabels(mark_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("P(child | any parent)", fontsize=11)
    ax.set_title(
        "Mark-head output marginal across runs — interventions DO shift the marginal,\n"
        "but the parent-conditional structure (K rows) never becomes parent-dependent",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)

    out = FIGS / "fig3_marginal_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3: {out}")


# ---------- Figure 4: row-deviation across runs ----------
def fig4_row_dev_bars(all_matrices):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    labels = []
    row_devs = []
    colors = []
    for run, M, _ in all_matrices:
        row_mean = M.mean(axis=0)
        row_dev = float(np.abs(M - row_mean[None, :]).sum())
        labels.append(run.label.replace("\n", " — "))
        row_devs.append(row_dev)
        colors.append(run.color)

    x = np.arange(len(labels))
    bars = ax.bar(x, row_devs, color=colors, edgecolor="white", linewidth=1)

    # Threshold line
    ax.axhline(0.1, color="black", linestyle="--", linewidth=1.5)
    ax.text(len(labels) - 0.5, 0.1, "  > 0.1 (primary acceptance)",
            fontsize=9, va="bottom", color="black", fontweight="bold")

    for i, (bar, dev) in enumerate(zip(bars, row_devs, strict=True)):
        ax.text(bar.get_x() + bar.get_width() / 2, dev + 0.0005,
                f"{dev:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
    ax.set_ylabel("total |row − row_mean| across parents", fontsize=11)
    ax.set_title(
        "Primary acceptance criterion (row-deviation > 0.1) across four runs.\n"
        "All four runs land 3+ orders of magnitude below the threshold.",
        fontsize=11,
    )
    ax.set_ylim(0, max(0.12, max(row_devs) * 1.5))
    ax.grid(True, axis="y", alpha=0.3)

    out = FIGS / "fig4_row_dev_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig4: {out}")


if __name__ == "__main__":
    print("Generating headline figures...")
    matrices = fig1_kk_grid()
    fig2_convergence()
    fig3_marginal_bars(matrices)
    fig4_row_dev_bars(matrices)
    print(f"\nAll figures written to {FIGS}")
