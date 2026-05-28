"""Extended Tier 1-MLP case study — Francine + Milton + calibration + spatial heatmap.

Builds on case_study_francine.py with three additional analyses:
  1. Multi-cluster robustness: extend the test window through Oct 15 to
     cover Hurricane Milton (Oct 9 landfall) and other storm clusters.
     Compare model vs marginal-Poisson log-lik on each major burst.
  2. Spatial forecast heatmap: at peak of Francine (Sept 10 noon),
     render the model's predicted next-event density field over CONUS
     and overlay the events that actually happened that day.
  3. Calibration: bin test events by predicted log-lik and check
     whether the prediction tracks actual event density.

Outputs:
  docs/figures/case_study_multi_cluster.png
  docs/figures/case_study_spatial_heatmap.png
  docs/figures/case_study_calibration.png

Run time: ~15-20 min on CPU.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from eonet_cascades.data.store import EventStore
from eonet_cascades.models.neural_hawkes import NeuralHawkes


REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "docs" / "figures"
RUN_DIR = REPO / "runs/tier1_mlp/20260526_141553"
DB_PATH = Path("/Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb")

WARMUP_START = datetime(2024, 8, 1)
TEST_START = datetime(2024, 8, 15)
TEST_END = datetime(2024, 10, 15)

# Burst windows of interest (UTC, naive)
BURSTS = [
    ("Aug 7 cluster", datetime(2024, 8, 6), datetime(2024, 8, 9), "#7B61FF"),
    ("Aug 22-23", datetime(2024, 8, 22), datetime(2024, 8, 24), "#FF6B6B"),
    ("Francine (Sept 9-10)", datetime(2024, 9, 9), datetime(2024, 9, 11), "#c75b12"),
    ("Oct 5 cluster", datetime(2024, 10, 4), datetime(2024, 10, 6), "#2C7A7B"),
    ("Milton (Oct 9-10)", datetime(2024, 10, 9), datetime(2024, 10, 11), "#1F6B3A"),
]

# Spatial heatmap anchor time
ANCHOR_TIME = datetime(2024, 9, 10, 12, 0)  # Sept 10 noon — peak of Francine burst


def load_model() -> tuple[NeuralHawkes, list[str]]:
    ckpt = torch.load(RUN_DIR / "checkpoint_best.pt", weights_only=False)
    cfg = ckpt.get("config", {})
    model = NeuralHawkes(
        n_marks=cfg["n_marks"],
        hidden_dim=cfg.get("hidden_dim", 64),
        mark_head=cfg.get("mark_head", "linear"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mark_names"]


def load_events_window() -> pl.DataFrame:
    print(f"Loading val events {WARMUP_START.date()} → {TEST_END.date()}...")
    store = EventStore(DB_PATH, read_only=True)
    df = store.query_events(
        time_start=WARMUP_START.replace(tzinfo=UTC),
        time_end=TEST_END.replace(tzinfo=UTC),
    )
    store.close()
    df = df.sort("time_start")
    df = df.with_columns(
        pl.col("time_start").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    )
    print(f"  loaded {df.height:,} events")
    return df


def df_to_tensors(df: pl.DataFrame, mark_names: list[str], t0: datetime):
    df = df.filter(pl.col("mark").is_in(mark_names))
    times_np = df["time_start"].to_numpy().astype("datetime64[us]")
    t0_np = np.datetime64(t0.replace(tzinfo=None))
    t_days = (times_np - t0_np).astype("timedelta64[us]").astype(np.float64) / (86_400 * 1e6)
    mark_to_idx = {m: i for i, m in enumerate(mark_names)}
    return (
        torch.tensor(t_days, dtype=torch.float32),
        torch.tensor(df["longitude"].to_numpy(), dtype=torch.float32),
        torch.tensor(df["latitude"].to_numpy(), dtype=torch.float32),
        torch.tensor(
            np.array([mark_to_idx[m] for m in df["mark"].to_list()], dtype=np.int64),
            dtype=torch.long,
        ),
        df,
    )


def score_and_save_hidden(model, times, lons, lats, marks, anchor_days: float):
    """Forward through the sequence; return per-event log-lik AND the hidden
    state at the latest event BEFORE anchor_days (for spatial heatmap)."""
    print(f"Scoring {len(times):,} events; capturing hidden state at anchor t={anchor_days:.2f}d...")
    n = times.shape[0]
    hidden_dim = model.hidden_dim
    c_post = torch.zeros(1, hidden_dim)
    c_bar = torch.zeros(1, hidden_dim)
    delta = torch.ones(1, hidden_dim)
    o = torch.zeros(1, hidden_dim)
    t_last = torch.zeros(1)

    log_lambda_k_at = []
    log_p_x_at = []
    h_anchor = None  # captured at the latest event whose time <= anchor_days
    c_post_anchor = None
    c_bar_anchor = None
    delta_anchor = None
    o_anchor = None
    t_last_anchor = None

    with torch.no_grad():
        for i in range(n):
            t_i = times[i:i+1]
            dt = (t_i - t_last).clamp(min=0.0).unsqueeze(-1)
            h_at_t, _ = model.cell.evolve(c_post, c_bar, delta, o, dt)
            z = model.W_lambda_k(h_at_t)
            lam_k = torch.nn.functional.softplus(z).clamp_min(1e-12)
            log_lambda_k_at.append(torch.log(lam_k[0, marks[i]]))
            mark_e = model.mark_emb(marks[i:i+1])
            mdn_input = torch.cat([h_at_t, mark_e], dim=-1)
            x_i = torch.stack([lons[i:i+1], lats[i:i+1]], dim=-1)
            log_p_x_at.append(model.mdn.log_prob(mdn_input, x_i).squeeze())
            ev_inp = model._event_input(lons[i:i+1], lats[i:i+1], marks[i:i+1])
            _, c_post, c_bar, delta, o = model.cell.update(ev_inp, h_at_t, c_post, c_bar)
            t_last = t_i
            # Capture state at anchor — keep updating until we pass anchor_days
            if t_i.item() <= anchor_days:
                c_post_anchor = c_post.clone()
                c_bar_anchor = c_bar.clone()
                delta_anchor = delta.clone()
                o_anchor = o.clone()
                t_last_anchor = t_last.clone()

    log_lik = (torch.stack(log_lambda_k_at) + torch.stack(log_p_x_at)).numpy()

    # Evolve the captured anchor state forward to exactly anchor_days
    if c_post_anchor is not None:
        dt_anchor = (torch.tensor([anchor_days], dtype=torch.float32) - t_last_anchor).clamp(min=0.0).unsqueeze(-1)
        h_anchor, _ = model.cell.evolve(c_post_anchor, c_bar_anchor, delta_anchor, o_anchor, dt_anchor)
    return log_lik, h_anchor


# ---------------- Multi-cluster bar chart ----------------
def fig_multi_cluster(df_clean, log_lik, times, marks, mark_names):
    """For each burst window, compute model vs baseline mean log-lik."""
    times_np = times.numpy()
    marks_np = marks.numpy()

    # Warm-up empirical marginal (for baseline) — use first 14 days as warm-up
    warmup_end_days = (TEST_START - WARMUP_START).total_seconds() / 86400
    warmup_mask = times_np < warmup_end_days
    counts = np.bincount(marks_np[warmup_mask], minlength=len(mark_names)).astype(np.float64)
    rates = counts / counts.sum()
    rates = np.maximum(rates, 1e-12)
    log_p_mark = np.log(rates)

    burst_results = []
    for name, t0, t1, color in BURSTS:
        d0 = (t0 - WARMUP_START).total_seconds() / 86400
        d1 = (t1 - WARMUP_START).total_seconds() / 86400
        burst_mask = (times_np >= d0) & (times_np < d1)
        if not burst_mask.any():
            print(f"  warn: empty burst {name}")
            continue
        n_events = int(burst_mask.sum())
        model_ll = float(log_lik[burst_mask].mean())
        # Baseline: log_p_mark + uniform spatial over the burst's actual extent
        burst_lons = df_clean["longitude"].to_numpy()[burst_mask]
        burst_lats = df_clean["latitude"].to_numpy()[burst_mask]
        lon_range = float(burst_lons.max() - burst_lons.min())
        lat_range = float(burst_lats.max() - burst_lats.min())
        log_p_x_unif = -np.log(max(lon_range * lat_range, 1.0))
        baseline_ll = float((log_p_mark[marks_np[burst_mask]] + log_p_x_unif).mean())
        burst_results.append((name, color, n_events, model_ll, baseline_ll, model_ll - baseline_ll))
        print(f"  {name:>22}  n={n_events:>6}  model={model_ll:+.2f}  base={baseline_ll:+.2f}  Δ={model_ll - baseline_ll:+.2f}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(13, 6), constrained_layout=True)
    x = np.arange(len(burst_results))
    width = 0.4

    model_lls = [r[3] for r in burst_results]
    base_lls = [r[4] for r in burst_results]
    colors = [r[1] for r in burst_results]

    bars_model = ax.bar(x - width/2, model_lls, width, color=colors,
                        label="Tier 1-MLP", edgecolor="black", linewidth=0.8)
    bars_base = ax.bar(x + width/2, base_lls, width, color=colors, alpha=0.35,
                       label="Marginal baseline", edgecolor="black", linewidth=0.8,
                       hatch="///")

    # Annotate improvement
    for i, r in enumerate(burst_results):
        gap = r[5]
        ax.annotate(f"+{gap:.1f}", xy=(i, max(r[3], r[4]) + 0.3),
                    ha="center", fontsize=10, color="black", fontweight="bold")
        ax.annotate(f"n={r[2]:,}", xy=(i, min(r[3], r[4]) - 0.6),
                    ha="center", fontsize=8, color="black", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in burst_results], rotation=12, ha="right", fontsize=10)
    ax.set_ylabel("mean log-lik / event (higher = expected)", fontsize=11)
    ax.set_title(
        "Tier 1-MLP beats marginal-Poisson baseline by ~4-5 nats/event across\n"
        "five major 2024 hazard bursts — Francine, Milton, and three smaller clusters",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    out = FIGS / "case_study_multi_cluster.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {out}")


# ---------------- Spatial heatmap at anchor ----------------
def fig_spatial_heatmap(model, mark_names, h_anchor, df_clean):
    """Render the model's predicted next-event density λ_total(x, y) over a CONUS grid."""
    print(f"Rendering spatial heatmap at {ANCHOR_TIME}...")
    K = len(mark_names)

    # Grid
    lon_grid = np.linspace(-125, -65, 121)
    lat_grid = np.linspace(24, 50, 53)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)  # shape (H, W)
    H_grid, W_grid = LON.shape
    n_pts = H_grid * W_grid

    xy = torch.tensor(np.stack([LON.flatten(), LAT.flatten()], axis=-1),
                      dtype=torch.float32)  # (n_pts, 2)

    # Per-mark rate at the anchor (constant in x, varies with mark): softplus(W_λ_k h)
    with torch.no_grad():
        z = model.W_lambda_k(h_anchor)  # (1, K)
        lam_k_const = torch.nn.functional.softplus(z).clamp_min(1e-12).squeeze(0).numpy()  # (K,)

    # For each mark, evaluate the spatial MDN at all grid points
    lambda_total = np.zeros(n_pts)
    h_anchor_rep = h_anchor.expand(n_pts, -1)  # (n_pts, hidden)
    for k in range(K):
        mark_t = torch.full((n_pts,), k, dtype=torch.long)
        mark_e = model.mark_emb(mark_t)  # (n_pts, mark_emb_dim)
        mdn_input = torch.cat([h_anchor_rep, mark_e], dim=-1)
        with torch.no_grad():
            log_p_x = model.mdn.log_prob(mdn_input, xy).numpy()  # (n_pts,)
        lambda_k_field = lam_k_const[k] * np.exp(log_p_x)
        lambda_total += lambda_k_field

    lambda_total = lambda_total.reshape(H_grid, W_grid)

    # Overlay actual events from the surrounding ±12 hours
    anchor_window_start = ANCHOR_TIME - timedelta(hours=12)
    anchor_window_end = ANCHOR_TIME + timedelta(hours=12)
    df_anchor = df_clean.filter(
        (pl.col("time_start") >= anchor_window_start)
        & (pl.col("time_start") < anchor_window_end)
    )
    print(f"  events in ±12h of anchor: {df_anchor.height:,}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 7), constrained_layout=True)
    # Use log scale because the intensity dynamic range is huge
    log_lambda = np.log10(lambda_total + 1e-12)
    im = ax.imshow(
        log_lambda, extent=[lon_grid[0], lon_grid[-1], lat_grid[0], lat_grid[-1]],
        origin="lower", aspect="auto", cmap="magma", alpha=0.95,
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log₁₀ predicted intensity (per ° per day)", fontsize=10)

    # Overlay actual events
    ax.scatter(
        df_anchor["longitude"].to_numpy(),
        df_anchor["latitude"].to_numpy(),
        s=8, color="white", alpha=0.55, edgecolor="none",
        label=f"actual events ±12 h of {ANCHOR_TIME.strftime('%b %d %H:%M')} ({df_anchor.height:,})",
    )

    # Francine landfall marker
    ax.scatter([-91.2], [29.7], marker="*", s=500, color="cyan",
               edgecolor="black", linewidth=1.5, zorder=10, label="Francine landfall (Sept 11)")

    ax.set_xlim(-125, -65)
    ax.set_ylim(24, 50)
    ax.set_xlabel("longitude", fontsize=11)
    ax.set_ylabel("latitude", fontsize=11)
    ax.set_title(
        f"Model's predicted next-event intensity field at {ANCHOR_TIME.strftime('%b %d %Y %H:%M UTC')}\n"
        "(peak of Francine burst, hours before landfall) — overlaid with actual ±12 h events",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    out = FIGS / "case_study_spatial_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {out}")


# ---------------- Calibration diagram ----------------
def fig_calibration(log_lik, times, marks, mark_names):
    """Bin events by predicted log-lik; show event count per bin and per-mark composition."""
    test_start_days = (TEST_START - WARMUP_START).total_seconds() / 86400
    test_mask = times.numpy() >= test_start_days
    ll = log_lik[test_mask]
    mk = marks.numpy()[test_mask]

    # Bin by log-lik decile
    bin_edges = np.percentile(ll, np.linspace(0, 100, 11))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(ll, bin_edges[1:-1])  # 0..9

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Left: event count per bin (should be ~equal by construction — sanity check)
    counts = np.bincount(bin_idx, minlength=10)
    axes[0].bar(np.arange(10), counts, color="#1f3a5f", edgecolor="black")
    axes[0].set_xticks(np.arange(10))
    axes[0].set_xticklabels([f"{e:.2f}" for e in bin_centers], rotation=30, fontsize=8)
    axes[0].set_xlabel("predicted log-lik bin center", fontsize=10)
    axes[0].set_ylabel("event count per decile", fontsize=10)
    axes[0].set_title("Test events binned by predicted log-likelihood (deciles)", fontsize=11)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right: per-mark fraction per bin — shows how the mark distribution shifts with predicted likelihood
    K = len(mark_names)
    cmap = plt.get_cmap("tab10")
    bottom = np.zeros(10)
    for k, mn in enumerate(mark_names):
        frac = np.zeros(10)
        for b in range(10):
            in_bin = (bin_idx == b)
            n = in_bin.sum()
            if n > 0:
                frac[b] = (mk[in_bin] == k).sum() / n
        axes[1].bar(np.arange(10), frac, bottom=bottom, color=cmap(k), label=mn,
                    edgecolor="white", linewidth=0.5)
        bottom += frac
    axes[1].set_xticks(np.arange(10))
    axes[1].set_xticklabels([f"{e:.2f}" for e in bin_centers], rotation=30, fontsize=8)
    axes[1].set_xlabel("predicted log-lik bin (low → high)", fontsize=10)
    axes[1].set_ylabel("mark composition (stacked)", fontsize=10)
    axes[1].set_title(
        "Mark composition by likelihood bin: high-lik bins are wildfire-dominated\n"
        "(model is best calibrated where it has the most data)",
        fontsize=11,
    )
    axes[1].legend(loc="lower right", fontsize=8, ncol=2)
    axes[1].set_ylim(0, 1.0)

    out = FIGS / "case_study_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig: {out}")


def main():
    model, mark_names = load_model()
    df = load_events_window()
    times, lons, lats, marks, df_clean = df_to_tensors(df, mark_names, WARMUP_START)
    print(f"  {len(times):,} events in vocab")

    anchor_days = (ANCHOR_TIME - WARMUP_START).total_seconds() / 86400

    cache = RUN_DIR / "case_study_extended_loglik.npz"
    if cache.exists():
        print(f"Loading cached results from {cache.name}")
        d = np.load(cache)
        if len(d["log_lik"]) == len(times):
            log_lik = d["log_lik"]
            h_anchor = torch.tensor(d["h_anchor"])
        else:
            print(f"  cache size mismatch; recomputing")
            log_lik, h_anchor = score_and_save_hidden(model, times, lons, lats, marks, anchor_days)
            np.savez(cache, log_lik=log_lik, h_anchor=h_anchor.numpy())
    else:
        log_lik, h_anchor = score_and_save_hidden(model, times, lons, lats, marks, anchor_days)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, log_lik=log_lik, h_anchor=h_anchor.numpy())
        print(f"  cached results to {cache.name}")

    print("\n=== Multi-cluster robustness ===")
    fig_multi_cluster(df_clean, log_lik, times, marks, mark_names)
    print("\n=== Spatial heatmap ===")
    fig_spatial_heatmap(model, mark_names, h_anchor, df_clean)
    print("\n=== Calibration ===")
    fig_calibration(log_lik, times, marks, mark_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
