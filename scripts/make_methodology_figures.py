"""Four methodology / model-property figures for the writeup.

  fig_geography.png      — per-mark spatial density grid; shows what the
                            model learned about WHERE each hazard happens
  fig_hawkes_decay.png   — λ_total(t) trajectory after a single seed event,
                            for different seed marks; educational decay-curve
  fig_effective_rank.png — PCA cumulative variance of mark-head outputs
                            across a batch of hidden states; quantifies the
                            encoder bottleneck (Tier 1.1 from future-work)
  fig_architecture.png   — block diagram of Tier 1-MLP for methods sections

Uses cached hidden states from the press-quality run for figures 1 & 3
(no model re-scoring needed). Run time: ~3 min CPU.
"""
from __future__ import annotations

import ssl as _ssl

import certifi as _certifi

_ssl._create_default_https_context = lambda: _ssl.create_default_context(cafile=_certifi.where())

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from eonet_cascades.models.neural_hawkes import NeuralHawkes


REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "docs" / "figures"
RUN_DIR = REPO / "runs/tier1_mlp/20260526_141553"

LON_MIN, LON_MAX = -125.0, -65.0
LAT_MIN, LAT_MAX = 24.0, 50.0


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


def load_anchor_hidden_states() -> torch.Tensor:
    """Stack cached hidden states from the press-quality run (~20 anchors)."""
    cache = RUN_DIR / "case_study_press_quality_h.npz"
    d = np.load(cache, allow_pickle=False)
    states = []
    for k in d.files:
        if k.startswith("h_"):
            states.append(torch.tensor(d[k]))
    return torch.cat(states, dim=0)  # (n_anchors, hidden_dim)


def _style_map_ax(ax, dark: bool = True):
    bg_land = "#1a1a2e" if dark else "#f5f5f5"
    bg_water = "#0a0a1a" if dark else "#e0e8ee"
    coast = "#888888" if dark else "#333333"
    borders = "#555577" if dark else "#666666"
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=bg_land, zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=bg_water, zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=bg_water,
                   edgecolor="#222244", linewidth=0.3, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor=coast, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor=borders, linewidth=0.3, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor=coast, linewidth=0.6, zorder=3)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())


# =========================================================
# Figure 1 — per-mark spatial density grid
# =========================================================
def fig_geography(model: NeuralHawkes, mark_names: list[str], h_batch: torch.Tensor):
    print("Rendering fig_geography (per-mark spatial density)...")
    lon_grid = np.linspace(LON_MIN, LON_MAX, 121)
    lat_grid = np.linspace(LAT_MIN, LAT_MAX, 53)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    n_pts = LON.size
    xy = torch.tensor(np.stack([LON.flatten(), LAT.flatten()], axis=-1),
                      dtype=torch.float32)

    K = len(mark_names)

    # Sample-average p(x|h,k) over the batch of typical hidden states.
    # For each (k, x), average exp(log_p_x) across h in the batch.
    per_mark_density = np.zeros((K, *LON.shape))
    h_batch_first = h_batch[:30]  # cap at 30 to keep render fast
    print(f"  averaging over {h_batch_first.shape[0]} hidden states per mark...")
    for k in range(K):
        mark_t = torch.full((n_pts,), k, dtype=torch.long)
        mark_e = model.mark_emb(mark_t)
        acc = np.zeros(n_pts, dtype=np.float64)
        with torch.no_grad():
            for h_one in h_batch_first:
                h_rep = h_one.unsqueeze(0).expand(n_pts, -1)
                mdn_input = torch.cat([h_rep, mark_e], dim=-1)
                log_p_x = model.mdn.log_prob(mdn_input, xy).numpy()
                acc += np.exp(log_p_x)
        acc /= h_batch_first.shape[0]
        per_mark_density[k] = acc.reshape(LON.shape)

    fig = plt.figure(figsize=(20, 11), facecolor="#0a0a1a")
    gs = GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.06,
                  left=0.02, right=0.98, top=0.90, bottom=0.06)

    for k, mark_name in enumerate(mark_names):
        row, col = divmod(k, 4)
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        _style_map_ax(ax, dark=True)
        field = per_mark_density[k]
        vmax = field.max()
        vmin = max(field.min(), vmax * 1e-3)
        im = ax.imshow(
            field, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower", transform=ccrs.PlateCarree(),
            cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax),
            alpha=0.92, zorder=2,
        )
        ax.text(0.5, 1.04, mark_name, ha="center", va="bottom",
                transform=ax.transAxes, fontsize=15, fontweight="bold", color="white")
        ax.text(0.5, 1.005, f"average p(x | h, k) over {h_batch_first.shape[0]} hidden states",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=8, color="#888888", style="italic")

    fig.suptitle(
        "Where does the model think each hazard happens?",
        fontsize=24, fontweight="bold", color="white", y=0.98,
    )
    fig.text(0.5, 0.94,
             "Per-mark spatial density learned by Tier 1-MLP's MDN head. "
             "Bright = high predicted probability of an event of that type at that location.",
             ha="center", fontsize=13, color="#cccccc")
    fig.text(0.5, 0.02,
             "Each panel averages the model's mark-conditional spatial density p(x | h, k) "
             "over 30 hidden states sampled from validation history.",
             ha="center", fontsize=10, color="#aaaaaa", style="italic")

    out = FIGS / "fig_geography.png"
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# =========================================================
# Figure 2 — Hawkes self-excitation explainer
# =========================================================
def fig_hawkes_decay(model: NeuralHawkes, mark_names: list[str]):
    print("Rendering fig_hawkes_decay (self-excitation explainer)...")
    # Seed at center of CONUS
    cx, cy = -97.5, 32.0
    n_t = 80
    t_query = np.logspace(-3, np.log10(14), n_t)  # 0.001 → 14 days

    fig, ax = plt.subplots(1, 1, figsize=(11, 6), constrained_layout=True,
                           facecolor="white")

    cmap = plt.get_cmap("tab10")
    for k, m in enumerate(mark_names):
        lam_total = np.zeros(n_t)
        # Walk model with single seed event of mark k
        with torch.no_grad():
            for ti, t in enumerate(t_query):
                hidden_dim = model.hidden_dim
                c_post = torch.zeros(1, hidden_dim)
                c_bar = torch.zeros(1, hidden_dim)
                delta = torch.ones(1, hidden_dim)
                o = torch.zeros(1, hidden_dim)
                # Event at t=0
                t_event = torch.zeros(1)
                dt0 = (t_event - torch.zeros(1)).clamp(min=0.0).unsqueeze(-1)
                h_at_0, _ = model.cell.evolve(c_post, c_bar, delta, o, dt0)
                marks_seed = torch.tensor([k], dtype=torch.long)
                ev_inp = model._event_input(
                    torch.tensor([cx]), torch.tensor([cy]), marks_seed)
                _, c_post, c_bar, delta, o = model.cell.update(ev_inp, h_at_0, c_post, c_bar)
                # Query at time t
                dt = torch.tensor([[t]], dtype=torch.float32)
                h_q, _ = model.cell.evolve(c_post, c_bar, delta, o, dt)
                z = model.W_lambda_k(h_q)
                lam_k = torch.nn.functional.softplus(z).clamp_min(1e-12).squeeze(0).numpy()
                lam_total[ti] = lam_k.sum()
        ax.semilogx(t_query, lam_total, "-", color=cmap(k), linewidth=2.2, label=m)

    ax.set_xlabel("time after seed event (days, log scale)", fontsize=12)
    ax.set_ylabel("total intensity λ_total(t) (events / day)", fontsize=12)
    ax.set_title(
        "How long does the model \"remember\" an event?\n"
        "Total intensity decay after a single seed event at CONUS center",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10, frameon=True, title="seed event type")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(1, color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(1.05, ax.get_ylim()[1] * 0.95, "1 day", color="grey",
            fontsize=9, va="top", alpha=0.7)
    ax.axvline(7, color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(7.2, ax.get_ylim()[1] * 0.95, "1 week", color="grey",
            fontsize=9, va="top", alpha=0.7)

    out = FIGS / "fig_hawkes_decay.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# =========================================================
# Figure 3 — Effective-rank diagnostic
# =========================================================
def fig_effective_rank(model: NeuralHawkes, mark_names: list[str], h_batch: torch.Tensor):
    print("Rendering fig_effective_rank...")
    K = len(mark_names)

    # Compute mark-head outputs on the batch of hidden states
    with torch.no_grad():
        z = model.W_lambda_k(h_batch)  # (n, K) raw logits
        lam_k = torch.nn.functional.softplus(z).clamp_min(1e-12)  # (n, K) intensities
        # Row-normalized (what forward-sim's multinomial sees)
        p_mark = lam_k / lam_k.sum(dim=-1, keepdim=True)
    z_np = z.numpy()
    lam_np = lam_k.numpy()
    p_np = p_mark.numpy()

    # PCA on each output type
    def cum_explained_variance(X: np.ndarray) -> np.ndarray:
        X_centered = X - X.mean(axis=0, keepdims=True)
        # Use SVD
        _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
        var = s ** 2
        return np.cumsum(var) / var.sum()

    cev_z = cum_explained_variance(z_np)
    cev_lam = cum_explained_variance(lam_np)
    cev_p = cum_explained_variance(p_np)

    # Effective rank = exp(entropy of normalized eigenvalues)
    def effective_rank(X: np.ndarray) -> float:
        X_centered = X - X.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
        var = s ** 2
        p = var / var.sum()
        p = p[p > 1e-12]
        entropy = -float((p * np.log(p)).sum())
        return float(np.exp(entropy))

    er_z = effective_rank(z_np)
    er_lam = effective_rank(lam_np)
    er_p = effective_rank(p_np)

    print(f"  effective rank of raw logits z:           {er_z:.3f} (of K={K})")
    print(f"  effective rank of intensities λ_k:        {er_lam:.3f}")
    print(f"  effective rank of row-normalized p(k|h):  {er_p:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                             facecolor="white")

    # Left: cumulative explained variance
    x = np.arange(1, K + 1)
    axes[0].plot(x, cev_z, "-o", linewidth=2.2, markersize=7, color="#1f3a5f",
                 label=f"raw logits z (eff. rank {er_z:.2f})")
    axes[0].plot(x, cev_lam, "-s", linewidth=2.2, markersize=7, color="#c75b12",
                 label=f"intensities λ_k (eff. rank {er_lam:.2f})")
    axes[0].plot(x, cev_p, "-^", linewidth=2.2, markersize=7, color="#1f6b3a",
                 label=f"row-normalized p(k|h) (eff. rank {er_p:.2f})")
    axes[0].axhline(0.95, color="grey", linestyle="--", alpha=0.5)
    axes[0].text(K + 0.05, 0.95, "95%", color="grey", va="center", fontsize=9)
    axes[0].set_xlabel("number of principal components", fontsize=11)
    axes[0].set_ylabel("cumulative explained variance", fontsize=11)
    axes[0].set_title("Mark-head outputs lie on a low-dimensional manifold", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 1.01)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Right: effective rank as bars
    bar_labels = ["raw logits\nz", "intensities\nλ_k", "row-normalized\np(k|h)"]
    bar_values = [er_z, er_lam, er_p]
    colors = ["#1f3a5f", "#c75b12", "#1f6b3a"]
    bars = axes[1].bar(bar_labels, bar_values, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].axhline(K, color="grey", linestyle="--", alpha=0.7)
    axes[1].text(2.4, K, f"  max possible: K={K}", color="grey", va="center", fontsize=9)
    for bar, val in zip(bars, bar_values, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                     f"{val:.2f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("effective rank", fontsize=11)
    axes[1].set_title(
        f"Encoder bottleneck: row-normalized p(k|h) has eff. rank {er_p:.2f} ≪ K={K}",
        fontsize=12, fontweight="bold",
    )
    axes[1].set_ylim(0, K + 0.5)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Quantifying H6: the mark-head output collapses to a low-dimensional subspace",
        fontsize=14, fontweight="bold", y=1.02,
    )

    out = FIGS / "fig_effective_rank.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# =========================================================
# Figure 4 — Architecture diagram
# =========================================================
def fig_architecture():
    print("Rendering fig_architecture...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 8.5), constrained_layout=True,
                           facecolor="white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Color palette
    EVENT = "#FFC857"   # warm yellow for input
    EMB = "#9DD9D2"     # mint for embeddings
    CTLSTM = "#7B61FF"  # purple for the LSTM
    MARK = "#FF6B6B"    # coral for mark head
    SPATIAL = "#2C7A7B" # teal for spatial head
    OUTPUT = "#1F3A5F"  # navy for outputs

    def box(x, y, w, h, text, color, fontsize=11, edgecolor="black", fontweight="normal"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            linewidth=1.3, edgecolor=edgecolor, facecolor=color, alpha=0.92,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight)

    def arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.18), color="black"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5,
                       "shrinkA": 5, "shrinkB": 5},
        )
        if label:
            ax.text((x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1],
                    label, ha="center", va="center", fontsize=9,
                    color=color, style="italic",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    # Title
    ax.text(7, 8.5, "Tier 1-MLP — Neural Hawkes architecture",
            ha="center", fontsize=18, fontweight="bold")
    ax.text(7, 8.05,
            "CTLSTM hidden state h(t) drives a per-mark intensity head and a "
            "mark-conditional spatial density head",
            ha="center", fontsize=11, style="italic", color="#555555")

    # Input
    box(0.3, 6.4, 2.6, 1.2,
        "event sequence\n(t_i, lon_i, lat_i, k_i)", EVENT,
        fontsize=11, fontweight="bold")
    ax.text(1.6, 6.2, "for each event i", ha="center", fontsize=8.5,
            color="#555555", style="italic")

    # Embeddings
    box(3.6, 7.0, 2.4, 0.9, "mark_emb(k_i)", EMB, fontsize=10)
    box(3.6, 5.9, 2.4, 0.9, "spatial_emb(lon, lat)", EMB, fontsize=10)

    # Concat and CTLSTM
    box(6.7, 6.4, 1.5, 1.2, "concat", "#dddddd", fontsize=10)
    box(8.7, 5.6, 3.3, 2.0,
        "CTLSTM cell\n(Mei & Eisner 2017)\n\nevolve & update", CTLSTM,
        fontsize=11, fontweight="bold")
    ax.text(10.35, 5.4, "continuous-time hidden state h(t)\n"
                       "between events: closed-form decay",
            ha="center", fontsize=8.5, color="#555555", style="italic")

    # h(t) carrier line
    box(12.4, 6.3, 1.4, 0.6, "h(t)", "#222222", fontsize=12, fontweight="bold",
        edgecolor="#222222")
    ax.text(13.1, 6.05, "(hidden_dim=64)", ha="center", fontsize=8.5,
            color="#555555", style="italic")

    # Mark head (top branch)
    box(8.7, 3.5, 3.3, 1.3,
        "mark head W_λ_k\n(MLP: 64 → 32 → 8)", MARK,
        fontsize=11, fontweight="bold")
    box(12.4, 3.6, 1.4, 1.1, "softplus", "#dddddd", fontsize=10)

    # Per-mark rates output
    box(8.7, 1.7, 3.3, 1.3,
        "per-mark intensities\nλ_k(t | h(t))",
        OUTPUT, fontsize=11, edgecolor="white", fontweight="bold")
    ax.text(10.35, 1.4, "(K=7 values, one per hazard type)",
            ha="center", fontsize=8.5, color="#555555", style="italic")
    # invert white text color
    for child in ax.texts[-1:]:
        pass
    # Re-add text with white color since the navy box hides it
    ax.patches[-1]  # last drawn = output box
    ax.text(10.35, 2.35, "per-mark intensities\nλ_k(t | h(t))", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")

    # Spatial head (bottom branch, takes h + mark_emb)
    box(3.0, 3.0, 3.6, 1.3,
        "MDN spatial head\n(8-component mixture of Gaussians)", SPATIAL,
        fontsize=10, fontweight="bold")

    box(3.0, 1.0, 3.6, 1.3,
        "spatial density\np(x | h(t), k)",
        OUTPUT, fontsize=11, edgecolor="white", fontweight="bold")
    ax.text(4.8, 1.65, "spatial density\np(x | h(t), k)", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")

    # Arrows: event → embeddings
    arrow(2.9, 7.4, 3.6, 7.4)
    arrow(2.9, 6.4, 3.6, 6.3)
    # Embeddings → concat
    arrow(6.0, 7.4, 6.7, 7.0)
    arrow(6.0, 6.3, 6.7, 6.5)
    # Concat → CTLSTM
    arrow(8.2, 6.8, 8.7, 6.7)
    # CTLSTM → h(t)
    arrow(12.0, 6.6, 12.4, 6.6)
    # h(t) → mark head (down) and spatial head (left)
    arrow(13.1, 6.3, 12.0, 4.5, color="#FF6B6B")
    arrow(12.4, 6.4, 6.6, 3.8, color="#2C7A7B")
    # Mark head → softplus
    arrow(12.0, 4.15, 12.4, 4.15, color="#FF6B6B")
    # Softplus → λ_k output
    arrow(13.1, 3.5, 12.0, 3.0, color="#FF6B6B")
    # MDN → p(x|h,k) output
    arrow(4.8, 3.0, 4.8, 2.3, color="#2C7A7B")

    # Mark embedding also feeds the MDN
    arrow(4.8, 7.0, 4.8, 4.3, color="#9DD9D2",
          label="mark_emb also feeds MDN", label_offset=(0, 0))

    # Likelihood box at bottom
    box(8.0, 0.05, 6.0, 0.85,
        "joint Hawkes log-likelihood:  log L = Σᵢ [log λ_{kᵢ}(tᵢ) + log p(xᵢ | h, kᵢ)] − ∫ Σₖ λₖ(t) dt",
        "#fff8e7", fontsize=10.5, edgecolor="#aaa", fontweight="bold")

    # Legend
    legend_y = 0.05
    legend_x = 0.3
    ax.text(legend_x, legend_y + 0.55, "Legend:", fontsize=9, fontweight="bold")
    for i, (clr, name) in enumerate([
        (EVENT, "raw events"),
        (EMB, "embeddings"),
        (CTLSTM, "recurrent core"),
        (MARK, "mark head (the H6 bottleneck)"),
        (SPATIAL, "spatial head"),
        (OUTPUT, "model outputs"),
    ]):
        rect = mpatches.FancyBboxPatch(
            (legend_x + i * 1.05, legend_y - 0.2), 0.6, 0.35,
            boxstyle="round,pad=0.02",
            linewidth=0.8, edgecolor="black", facecolor=clr, alpha=0.92,
        )
        ax.add_patch(rect)
        ax.text(legend_x + i * 1.05 + 0.7, legend_y - 0.02, name, ha="left", va="center",
                fontsize=7.5)

    out = FIGS / "fig_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


def main():
    model, mark_names = load_model()
    h_batch = load_anchor_hidden_states()
    # Strip any singleton dim so shape is (N, hidden_dim)
    if h_batch.dim() == 3 and h_batch.shape[1] == 1:
        h_batch = h_batch.squeeze(1)
    print(f"Loaded {h_batch.shape[0]} hidden states of dim {h_batch.shape[1]}.")

    fig_geography(model, mark_names, h_batch)
    fig_hawkes_decay(model, mark_names)
    fig_effective_rank(model, mark_names, h_batch)
    fig_architecture()

    print("\nDone.")


if __name__ == "__main__":
    main()
