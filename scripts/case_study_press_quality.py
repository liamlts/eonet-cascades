"""Press-quality Tier 1-MLP case-study figures.

Outputs:
  docs/figures/case_study_hero.png    — side-by-side Sept 1 vs Sept 10
                                         on a proper geographic map with
                                         state borders, model intensity
                                         field, actual events overlaid,
                                         and Francine's storm track
  docs/figures/case_study_francine.gif — 20-frame animation Sept 5 → Sept 15
                                         showing the model's intensity
                                         field shifting in real time as the
                                         storm intensifies

Re-uses scoring infrastructure from case_study_extended.py; this script
just adds the multi-anchor hidden-state capture and the cartopy rendering.

Run time: ~15-20 min CPU.
"""
from __future__ import annotations

# Cartopy downloads Natural Earth data on first use; macOS python.org installs
# often fail SSL verification. Use certifi's CA bundle (works in the venv).
import ssl as _ssl

import certifi as _certifi

_ssl._create_default_https_context = lambda: _ssl.create_default_context(cafile=_certifi.where())

from datetime import UTC, datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.colors import LogNorm

from eonet_cascades.data.store import EventStore
from eonet_cascades.models.neural_hawkes import NeuralHawkes


REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "docs" / "figures"
RUN_DIR = REPO / "runs/tier1_mlp/20260526_141553"
DB_PATH = Path("/Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb")

WINDOW_START = datetime(2024, 8, 1)
WINDOW_END = datetime(2024, 9, 16)

# Anchor times: hero figure (Sept 1 noon + Sept 10 noon) + animation frames
HERO_TIMES = [datetime(2024, 9, 1, 12, 0), datetime(2024, 9, 10, 12, 0)]
ANIM_TIMES = [datetime(2024, 9, 5) + timedelta(hours=12 * i) for i in range(20)]  # Sept 5 → Sept 14 18:00
ALL_ANCHORS = sorted(set(HERO_TIMES + ANIM_TIMES))

# CONUS bbox
LON_MIN, LON_MAX = -125.0, -65.0
LAT_MIN, LAT_MAX = 24.0, 50.0

# Francine track (approximate, from NHC reports)
FRANCINE_TRACK = [
    (datetime(2024, 9,  8, 12), -90.0, 21.5),  # tropical storm in Bay of Campeche
    (datetime(2024, 9,  9, 12), -91.5, 24.0),  # strengthening
    (datetime(2024, 9, 10, 12), -91.5, 27.0),  # category 1 hurricane
    (datetime(2024, 9, 11, 12), -91.2, 29.7),  # category 2 landfall
    (datetime(2024, 9, 12, 12), -90.0, 32.0),  # weakening tropical depression
    (datetime(2024, 9, 13, 12), -87.0, 33.0),  # remnants moving NE
]


def load_model_and_events():
    ckpt = torch.load(RUN_DIR / "checkpoint_best.pt", weights_only=False)
    cfg = ckpt.get("config", {})
    model = NeuralHawkes(
        n_marks=cfg["n_marks"],
        hidden_dim=cfg.get("hidden_dim", 64),
        mark_head=cfg.get("mark_head", "linear"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mark_names = ckpt["mark_names"]

    print(f"Loading val events {WINDOW_START.date()} → {WINDOW_END.date()}...")
    store = EventStore(DB_PATH, read_only=True)
    df = store.query_events(
        time_start=WINDOW_START.replace(tzinfo=UTC),
        time_end=WINDOW_END.replace(tzinfo=UTC),
    )
    store.close()
    df = df.sort("time_start").with_columns(
        pl.col("time_start").dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    )
    df = df.filter(pl.col("mark").is_in(mark_names))
    times_np = df["time_start"].to_numpy().astype("datetime64[us]")
    t0_np = np.datetime64(WINDOW_START.replace(tzinfo=None))
    t_days = (times_np - t0_np).astype("timedelta64[us]").astype(np.float64) / (86_400 * 1e6)
    mark_to_idx = {m: i for i, m in enumerate(mark_names)}
    return (
        model, mark_names, df,
        torch.tensor(t_days, dtype=torch.float32),
        torch.tensor(df["longitude"].to_numpy(), dtype=torch.float32),
        torch.tensor(df["latitude"].to_numpy(), dtype=torch.float32),
        torch.tensor(
            np.array([mark_to_idx[m] for m in df["mark"].to_list()], dtype=np.int64),
            dtype=torch.long,
        ),
    )


def forward_with_multi_anchor(model, times, lons, lats, marks, anchor_days_list):
    """Walk the LSTM through every event, capturing hidden-state-evolved-to-anchor
    for every requested anchor time. Returns dict {anchor_days: h_tensor}."""
    n = times.shape[0]
    hidden_dim = model.hidden_dim
    c_post = torch.zeros(1, hidden_dim)
    c_bar = torch.zeros(1, hidden_dim)
    delta = torch.ones(1, hidden_dim)
    o = torch.zeros(1, hidden_dim)
    t_last = torch.zeros(1)

    anchor_set = sorted(set(anchor_days_list))
    next_anchor_idx = 0
    out: dict[float, torch.Tensor] = {}

    print(f"Scoring {n:,} events; capturing {len(anchor_set)} hidden-state anchors...")
    with torch.no_grad():
        for i in range(n):
            t_i = times[i:i+1]
            # Capture any anchors that fall between t_last and t_i
            while (next_anchor_idx < len(anchor_set)
                   and anchor_set[next_anchor_idx] <= float(t_i.item())):
                anchor_t = anchor_set[next_anchor_idx]
                dt_q = (torch.tensor([anchor_t], dtype=torch.float32) - t_last).clamp(min=0.0).unsqueeze(-1)
                h_at_q, _ = model.cell.evolve(c_post, c_bar, delta, o, dt_q)
                out[anchor_t] = h_at_q.clone()
                next_anchor_idx += 1

            # Process event i
            dt = (t_i - t_last).clamp(min=0.0).unsqueeze(-1)
            h_at_t, _ = model.cell.evolve(c_post, c_bar, delta, o, dt)
            ev_inp = model._event_input(lons[i:i+1], lats[i:i+1], marks[i:i+1])
            _, c_post, c_bar, delta, o = model.cell.update(ev_inp, h_at_t, c_post, c_bar)
            t_last = t_i

        # Any remaining anchors that are AFTER the last event
        while next_anchor_idx < len(anchor_set):
            anchor_t = anchor_set[next_anchor_idx]
            dt_q = (torch.tensor([anchor_t], dtype=torch.float32) - t_last).clamp(min=0.0).unsqueeze(-1)
            h_at_q, _ = model.cell.evolve(c_post, c_bar, delta, o, dt_q)
            out[anchor_t] = h_at_q.clone()
            next_anchor_idx += 1

    return out


def lambda_field(model, h_anchor: torch.Tensor, n_marks: int,
                 lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    """Compute λ_total(x, y) on the lon/lat grid at hidden state h_anchor."""
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    H_grid, W_grid = LON.shape
    n_pts = H_grid * W_grid

    xy = torch.tensor(np.stack([LON.flatten(), LAT.flatten()], axis=-1),
                      dtype=torch.float32)

    with torch.no_grad():
        z = model.W_lambda_k(h_anchor)
        lam_k_const = torch.nn.functional.softplus(z).clamp_min(1e-12).squeeze(0).numpy()

    lambda_total = np.zeros(n_pts)
    h_rep = h_anchor.expand(n_pts, -1)
    for k in range(n_marks):
        mark_t = torch.full((n_pts,), k, dtype=torch.long)
        mark_e = model.mark_emb(mark_t)
        mdn_input = torch.cat([h_rep, mark_e], dim=-1)
        with torch.no_grad():
            log_p_x = model.mdn.log_prob(mdn_input, xy).numpy()
        lambda_total += lam_k_const[k] * np.exp(log_p_x)

    return lambda_total.reshape(H_grid, W_grid)


def _style_map_ax(ax):
    """Apply a consistent press-quality map style to a cartopy GeoAxes."""
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#1a1a2e", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#0a0a1a", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),
                   facecolor="#0a0a1a", edgecolor="#222244",
                   linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#888888", linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#555577", linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#888888", linewidth=0.7, zorder=3)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())


def hero_figure(model, mark_names, df_window, h_by_anchor):
    """Side-by-side Sept 1 vs Sept 10."""
    print("Rendering hero figure...")
    lon_grid = np.linspace(LON_MIN, LON_MAX, 181)  # ~0.33° resolution
    lat_grid = np.linspace(LAT_MIN, LAT_MAX, 79)
    K = len(mark_names)

    fig = plt.figure(figsize=(18, 8.5), facecolor="black")
    fig.patch.set_facecolor("#0a0a1a")

    anchor_days_list = [
        (t - WINDOW_START).total_seconds() / 86400.0 for t in HERO_TIMES
    ]
    # Compute fields with shared dynamic range
    fields = []
    for t_anchor, anchor_d in zip(HERO_TIMES, anchor_days_list, strict=True):
        h = h_by_anchor[anchor_d]
        field = lambda_field(model, h, K, lon_grid, lat_grid)
        fields.append((t_anchor, field))

    vmax = max(f.max() for _, f in fields)
    vmin = max(min(f.min() for _, f in fields), 1e-3)

    axes = []
    for i, (t_anchor, field) in enumerate(fields):
        ax = fig.add_subplot(1, 2, i + 1, projection=ccrs.PlateCarree())
        _style_map_ax(ax)

        im = ax.imshow(
            field, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower", transform=ccrs.PlateCarree(),
            cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax),
            alpha=0.85, zorder=2,
        )

        # Actual events ±12h of anchor
        t_lo = t_anchor - timedelta(hours=12)
        t_hi = t_anchor + timedelta(hours=12)
        df_events = df_window.filter(
            (pl.col("time_start") >= t_lo) & (pl.col("time_start") < t_hi)
        )
        ax.scatter(
            df_events["longitude"].to_numpy(),
            df_events["latitude"].to_numpy(),
            s=4, color="white", alpha=0.7, edgecolor="none",
            transform=ccrs.PlateCarree(), zorder=4,
        )

        # Francine track on the right panel
        if i == 1:
            tlons = [p[1] for p in FRANCINE_TRACK]
            tlats = [p[2] for p in FRANCINE_TRACK]
            ax.plot(tlons, tlats, color="#00ffff", linewidth=2.5,
                    transform=ccrs.PlateCarree(), zorder=5,
                    path_effects=[patheffects.Stroke(linewidth=4, foreground="black"),
                                  patheffects.Normal()])
            # Landfall marker
            ax.scatter([-91.2], [29.7], marker="*", s=600, color="#00ffff",
                       edgecolor="black", linewidth=1.5, zorder=6,
                       transform=ccrs.PlateCarree())
            ax.annotate(
                "Francine landfall\nSept 11, 17:00 UTC",
                xy=(-91.2, 29.7), xytext=(-83, 33),
                fontsize=11, color="#00ffff", fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#00ffff", "lw": 2,
                           "shrinkA": 4, "shrinkB": 8},
                path_effects=[patheffects.Stroke(linewidth=2.5, foreground="black"),
                              patheffects.Normal()],
                transform=ccrs.PlateCarree(),
            )

        # Panel title
        title_lines = [
            f"{t_anchor.strftime('%B %d, %Y — %H:%M UTC')}",
            "calm — pre-storm baseline" if i == 0 else "peak — Francine intensifies",
        ]
        ax.text(
            0.5, 1.05, title_lines[0],
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=15, fontweight="bold", color="white",
        )
        ax.text(
            0.5, 1.01, title_lines[1],
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=11, color="#aaaaaa", style="italic",
        )
        axes.append(ax)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(
        "model's predicted next-event intensity\n(log scale, events / °² / day)",
        fontsize=10, color="white",
    )
    cbar.ax.tick_params(colors="white")

    # Suptitle
    fig.suptitle(
        "Tier 1-MLP forecasts the 2024 hurricane season",
        fontsize=22, fontweight="bold", color="white", y=0.98,
    )
    fig.text(
        0.5, 0.93,
        "Same Neural Hawkes model. Two days. The LSTM hidden state shifts the predicted intensity field toward Francine's path between Sept 1 (calm) and Sept 10 (peak storm).",
        ha="center", fontsize=12, color="#cccccc",
    )
    fig.text(
        0.5, 0.04,
        "Bright = high predicted event rate. White dots = actual events within ±12 hours of the anchor time. Cyan = Hurricane Francine's track + landfall.",
        ha="center", fontsize=10, color="#aaaaaa", style="italic",
    )

    out = FIGS / "case_study_hero.png"
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  hero → {out}")


def animation_gif(model, mark_names, df_window, h_by_anchor):
    """20-frame GIF Sept 5 → Sept 14 evolution."""
    print("Rendering animation...")
    lon_grid = np.linspace(LON_MIN, LON_MAX, 121)  # slightly coarser for speed
    lat_grid = np.linspace(LAT_MIN, LAT_MAX, 53)
    K = len(mark_names)

    # Pre-compute all frames' intensity fields
    anim_days = [(t - WINDOW_START).total_seconds() / 86400.0 for t in ANIM_TIMES]
    frames = []
    for t_anchor, d in zip(ANIM_TIMES, anim_days, strict=True):
        h = h_by_anchor[d]
        field = lambda_field(model, h, K, lon_grid, lat_grid)
        # Events in the ±6h window
        df_ev = df_window.filter(
            (pl.col("time_start") >= t_anchor - timedelta(hours=6))
            & (pl.col("time_start") < t_anchor + timedelta(hours=6))
        )
        frames.append({
            "time": t_anchor,
            "field": field,
            "ev_lon": df_ev["longitude"].to_numpy(),
            "ev_lat": df_ev["latitude"].to_numpy(),
        })

    vmax = max(f["field"].max() for f in frames)
    vmin = max(min(f["field"].min() for f in frames), 1e-3)

    fig = plt.figure(figsize=(12, 7), facecolor="#0a0a1a")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    _style_map_ax(ax)

    im = ax.imshow(
        frames[0]["field"], extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        origin="lower", transform=ccrs.PlateCarree(),
        cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax),
        alpha=0.85, zorder=2,
    )
    scat = ax.scatter([], [], s=8, color="white", alpha=0.85,
                       edgecolor="none", transform=ccrs.PlateCarree(), zorder=4)
    # Static Francine track (always visible)
    tlons = [p[1] for p in FRANCINE_TRACK]
    tlats = [p[2] for p in FRANCINE_TRACK]
    ax.plot(tlons, tlats, color="#00ffff", linewidth=2,
            transform=ccrs.PlateCarree(), zorder=5,
            path_effects=[patheffects.Stroke(linewidth=3, foreground="black"),
                          patheffects.Normal()])
    landfall_t = datetime(2024, 9, 11, 17, 0)

    title = ax.text(
        0.5, 1.04, "", ha="center", va="bottom",
        transform=ax.transAxes, fontsize=14, fontweight="bold", color="white",
    )
    subtitle = ax.text(
        0.5, 1.005, "", ha="center", va="bottom",
        transform=ax.transAxes, fontsize=11, color="#aaaaaa", style="italic",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("predicted intensity (log scale)", fontsize=10, color="white")
    cbar.ax.tick_params(colors="white")

    fig.text(0.5, 0.02,
             "Tier 1-MLP — Hurricane Francine in real time. White dots = actual events ±6 h. Cyan = storm track.",
             ha="center", fontsize=10, color="#aaaaaa")

    def update(idx):
        f = frames[idx]
        im.set_data(f["field"])
        scat.set_offsets(np.column_stack([f["ev_lon"], f["ev_lat"]]))
        title.set_text(f["time"].strftime("%B %d, %Y — %H:%M UTC"))
        delta_landfall = (f["time"] - landfall_t).total_seconds() / 3600
        if delta_landfall < 0:
            subtitle.set_text(f"{-delta_landfall:.0f} hours before Francine landfall")
        else:
            subtitle.set_text(f"{delta_landfall:.0f} hours after Francine landfall")
        return im, scat, title, subtitle

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=400, blit=False,
    )
    out = FIGS / "case_study_francine.gif"
    anim.save(out, writer=animation.PillowWriter(fps=2.5))
    plt.close(fig)
    print(f"  gif → {out}")


def main():
    model, mark_names, df_window, times, lons, lats, marks = load_model_and_events()
    print(f"  {len(times):,} events in vocab")

    anchor_days_list = [(t - WINDOW_START).total_seconds() / 86400.0 for t in ALL_ANCHORS]

    cache = RUN_DIR / "case_study_press_quality_h.npz"
    if cache.exists():
        print(f"Loading cached hidden states from {cache.name}")
        d = np.load(cache, allow_pickle=False)
        keys = list(d.files)
        h_by_anchor = {}
        for k in keys:
            if k.startswith("h_"):
                anchor_d = float(k[2:])
                h_by_anchor[anchor_d] = torch.tensor(d[k])
    else:
        h_by_anchor = forward_with_multi_anchor(
            model, times, lons, lats, marks, anchor_days_list
        )
        np.savez(
            cache,
            **{f"h_{d:.6f}": h.numpy() for d, h in h_by_anchor.items()},
        )
        print(f"  cached hidden states to {cache.name}")

    hero_figure(model, mark_names, df_window, h_by_anchor)
    animation_gif(model, mark_names, df_window, h_by_anchor)
    print("\nDone.")


if __name__ == "__main__":
    main()
