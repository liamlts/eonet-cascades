"""Storm-only versions of the press-quality case-study figures.

Restricts the intensity field to flood + severe_storm + tornado marks only
(drops the wildfire baseline that swamped the original visualization).
The remaining signal is the storm-related predictions — exactly the
component that should respond to Hurricane Francine.

Outputs:
  docs/figures/case_study_hero_storm.png    — side-by-side Sept 1 vs Sept 10
                                                storm-only intensity
  docs/figures/case_study_francine_storm.mp4 — animation as proper mp4 video

Re-uses the cached hidden states from case_study_press_quality_h.npz.

Run time: ~5 min CPU (no model re-scoring needed).
"""
from __future__ import annotations

import ssl as _ssl

import certifi as _certifi

_ssl._create_default_https_context = lambda: _ssl.create_default_context(cafile=_certifi.where())

from datetime import UTC, datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio_ffmpeg
import matplotlib
import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from matplotlib.colors import LogNorm

from eonet_cascades.data.store import EventStore
from eonet_cascades.models.neural_hawkes import NeuralHawkes

# Tell matplotlib where ffmpeg lives (provided by imageio-ffmpeg)
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "docs" / "figures"
RUN_DIR = REPO / "runs/tier1_mlp/20260526_141553"
DB_PATH = Path("/Volumes/Seagate_Ext/eonet-cascades-data/events.duckdb")

WINDOW_START = datetime(2024, 8, 1)
WINDOW_END = datetime(2024, 9, 16)

HERO_TIMES = [datetime(2024, 9, 1, 12, 0), datetime(2024, 9, 10, 12, 0)]
ANIM_TIMES = [datetime(2024, 9, 5) + timedelta(hours=12 * i) for i in range(20)]

LON_MIN, LON_MAX = -125.0, -65.0
LAT_MIN, LAT_MAX = 24.0, 50.0

FRANCINE_TRACK = [
    (datetime(2024, 9,  8, 12), -90.0, 21.5),
    (datetime(2024, 9,  9, 12), -91.5, 24.0),
    (datetime(2024, 9, 10, 12), -91.5, 27.0),
    (datetime(2024, 9, 11, 12), -91.2, 29.7),
    (datetime(2024, 9, 12, 12), -90.0, 32.0),
    (datetime(2024, 9, 13, 12), -87.0, 33.0),
]

# Marks to keep — storm-related, in the order of mark_names list expected
STORM_MARK_NAMES = {"severe_storm", "flood", "tornado"}


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
    return model, mark_names, df


def storm_lambda_field(model, h_anchor: torch.Tensor, mark_names: list[str],
                       lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    """Compute λ_storm(x, y) = Σ_{k ∈ storm marks} λ_k(x, y).

    Drops wildfire and other non-storm marks from the intensity sum so the
    figure isolates the storm-related component of the model's prediction.
    """
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    H_grid, W_grid = LON.shape
    n_pts = H_grid * W_grid
    xy = torch.tensor(np.stack([LON.flatten(), LAT.flatten()], axis=-1),
                      dtype=torch.float32)

    storm_indices = [i for i, m in enumerate(mark_names) if m in STORM_MARK_NAMES]
    assert len(storm_indices) > 0, "no storm marks found in vocab"

    with torch.no_grad():
        z = model.W_lambda_k(h_anchor)
        lam_k_const = torch.nn.functional.softplus(z).clamp_min(1e-12).squeeze(0).numpy()

    lambda_storm = np.zeros(n_pts)
    h_rep = h_anchor.expand(n_pts, -1)
    for k in storm_indices:
        mark_t = torch.full((n_pts,), k, dtype=torch.long)
        mark_e = model.mark_emb(mark_t)
        mdn_input = torch.cat([h_rep, mark_e], dim=-1)
        with torch.no_grad():
            log_p_x = model.mdn.log_prob(mdn_input, xy).numpy()
        lambda_storm += lam_k_const[k] * np.exp(log_p_x)

    return lambda_storm.reshape(H_grid, W_grid)


def _style_map_ax(ax):
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


def hero_figure_storm(model, mark_names, df_window, h_by_anchor):
    print("Rendering storm-only hero figure...")
    lon_grid = np.linspace(LON_MIN, LON_MAX, 181)
    lat_grid = np.linspace(LAT_MIN, LAT_MAX, 79)

    anchor_days_list = [(t - WINDOW_START).total_seconds() / 86400.0 for t in HERO_TIMES]
    fields = []
    for t_anchor, anchor_d in zip(HERO_TIMES, anchor_days_list, strict=True):
        h = h_by_anchor[anchor_d]
        field = storm_lambda_field(model, h, mark_names, lon_grid, lat_grid)
        fields.append((t_anchor, field))
        print(f"  {t_anchor}: storm-intensity range [{field.min():.4e}, {field.max():.4e}]")

    vmax = max(f.max() for _, f in fields)
    vmin = max(min(f.min() for _, f in fields), vmax * 1e-3)

    fig = plt.figure(figsize=(18, 8.5), facecolor="#0a0a1a")

    for i, (t_anchor, field) in enumerate(fields):
        ax = fig.add_subplot(1, 2, i + 1, projection=ccrs.PlateCarree())
        _style_map_ax(ax)

        im = ax.imshow(
            field, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            origin="lower", transform=ccrs.PlateCarree(),
            cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax),
            alpha=0.9, zorder=2,
        )

        # Overlay actual STORM-mark events from ±12h
        t_lo = t_anchor - timedelta(hours=12)
        t_hi = t_anchor + timedelta(hours=12)
        df_storm = df_window.filter(
            (pl.col("time_start") >= t_lo)
            & (pl.col("time_start") < t_hi)
            & (pl.col("mark").is_in(list(STORM_MARK_NAMES)))
        )
        ax.scatter(
            df_storm["longitude"].to_numpy(),
            df_storm["latitude"].to_numpy(),
            s=18, color="white", alpha=0.9, edgecolor="cyan", linewidth=0.5,
            transform=ccrs.PlateCarree(), zorder=4,
        )
        n_storm = df_storm.height

        if i == 1:
            tlons = [p[1] for p in FRANCINE_TRACK]
            tlats = [p[2] for p in FRANCINE_TRACK]
            ax.plot(tlons, tlats, color="#00ffff", linewidth=2.5,
                    transform=ccrs.PlateCarree(), zorder=5,
                    path_effects=[patheffects.Stroke(linewidth=4, foreground="black"),
                                  patheffects.Normal()])
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

        ax.text(0.5, 1.05, t_anchor.strftime("%B %d, %Y — %H:%M UTC"),
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=15, fontweight="bold", color="white")
        ax.text(0.5, 1.01,
                "calm — pre-storm baseline" if i == 0 else "peak — Francine intensifies",
                ha="center", va="bottom", transform=ax.transAxes,
                fontsize=11, color="#aaaaaa", style="italic")
        ax.text(0.02, 0.02, f"actual storm events (±12 h): {n_storm}",
                transform=ax.transAxes, color="white", fontsize=10,
                bbox={"facecolor": "#0a0a1a", "edgecolor": "#444444", "pad": 4})

    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.62])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(
        "predicted storm-event intensity\n(severe_storm + flood + tornado, log scale)",
        fontsize=10, color="white",
    )
    cbar.ax.tick_params(colors="white")

    fig.suptitle(
        "Tier 1-MLP — predicting storm-related events during Hurricane Francine",
        fontsize=22, fontweight="bold", color="white", y=0.98,
    )
    fig.text(
        0.5, 0.93,
        "Wildfires dropped from the intensity field — only the storm-related marks remain. The Sept 10 panel shows the model's prediction lighting up the entire Gulf and Eastern US ahead of Francine's landfall.",
        ha="center", fontsize=12, color="#cccccc",
    )
    fig.text(
        0.5, 0.04,
        "Bright = high predicted storm intensity. White-with-cyan-rim dots = actual severe_storm / flood / tornado events ±12 h. Cyan curve = Francine's track + landfall.",
        ha="center", fontsize=10, color="#aaaaaa", style="italic",
    )

    out = FIGS / "case_study_hero_storm.png"
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  hero → {out}")


def animation_mp4_storm(model, mark_names, df_window, h_by_anchor):
    print("Rendering storm-only animation (mp4)...")
    lon_grid = np.linspace(LON_MIN, LON_MAX, 121)
    lat_grid = np.linspace(LAT_MIN, LAT_MAX, 53)

    anim_days = [(t - WINDOW_START).total_seconds() / 86400.0 for t in ANIM_TIMES]
    frames = []
    for t_anchor, d in zip(ANIM_TIMES, anim_days, strict=True):
        h = h_by_anchor[d]
        field = storm_lambda_field(model, h, mark_names, lon_grid, lat_grid)
        df_ev = df_window.filter(
            (pl.col("time_start") >= t_anchor - timedelta(hours=6))
            & (pl.col("time_start") < t_anchor + timedelta(hours=6))
            & (pl.col("mark").is_in(list(STORM_MARK_NAMES)))
        )
        frames.append({
            "time": t_anchor,
            "field": field,
            "ev_lon": df_ev["longitude"].to_numpy(),
            "ev_lat": df_ev["latitude"].to_numpy(),
            "n_ev": df_ev.height,
        })

    vmax = max(f["field"].max() for f in frames)
    vmin = max(min(f["field"].min() for f in frames), vmax * 1e-3)

    fig = plt.figure(figsize=(12, 7), facecolor="#0a0a1a")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    _style_map_ax(ax)

    im = ax.imshow(
        frames[0]["field"], extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        origin="lower", transform=ccrs.PlateCarree(),
        cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax),
        alpha=0.9, zorder=2,
    )
    scat = ax.scatter([], [], s=22, color="white", alpha=0.9,
                      edgecolor="cyan", linewidth=0.6,
                      transform=ccrs.PlateCarree(), zorder=4)
    tlons = [p[1] for p in FRANCINE_TRACK]
    tlats = [p[2] for p in FRANCINE_TRACK]
    ax.plot(tlons, tlats, color="#00ffff", linewidth=2,
            transform=ccrs.PlateCarree(), zorder=5,
            path_effects=[patheffects.Stroke(linewidth=3, foreground="black"),
                          patheffects.Normal()])
    landfall_t = datetime(2024, 9, 11, 17, 0)
    ax.scatter([-91.2], [29.7], marker="*", s=500, color="#00ffff",
               edgecolor="black", linewidth=1.5, zorder=6,
               transform=ccrs.PlateCarree())

    title = ax.text(0.5, 1.04, "", ha="center", va="bottom",
                    transform=ax.transAxes, fontsize=14, fontweight="bold", color="white")
    subtitle = ax.text(0.5, 1.005, "", ha="center", va="bottom",
                       transform=ax.transAxes, fontsize=11, color="#aaaaaa", style="italic")
    ev_count_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                             color="white", fontsize=9,
                             bbox={"facecolor": "#0a0a1a", "edgecolor": "#444444", "pad": 3})

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("storm intensity (log scale)\nsevere_storm + flood + tornado",
                   fontsize=10, color="white")
    cbar.ax.tick_params(colors="white")

    fig.text(0.5, 0.02,
             "Tier 1-MLP — Hurricane Francine in real time, storm-marks only. White-cyan dots = actual storm events ±6 h.",
             ha="center", fontsize=10, color="#aaaaaa")
    fig.suptitle(
        "Watching the model track Francine",
        fontsize=18, fontweight="bold", color="white", y=0.97,
    )

    def update(idx):
        f = frames[idx]
        im.set_data(f["field"])
        scat.set_offsets(np.column_stack([f["ev_lon"], f["ev_lat"]])
                         if len(f["ev_lon"]) > 0 else np.empty((0, 2)))
        title.set_text(f["time"].strftime("%B %d, %Y — %H:%M UTC"))
        delta_landfall = (f["time"] - landfall_t).total_seconds() / 3600
        if delta_landfall < 0:
            subtitle.set_text(f"{-delta_landfall:.0f} hours before Francine landfall")
        else:
            subtitle.set_text(f"{delta_landfall:.0f} hours after Francine landfall")
        ev_count_text.set_text(f"actual storm events in window: {f['n_ev']}")
        return im, scat, title, subtitle, ev_count_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=400, blit=False,
    )
    out = FIGS / "case_study_francine_storm.mp4"
    writer = animation.FFMpegWriter(
        fps=3, bitrate=2400, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(out, writer=writer, dpi=140)
    plt.close(fig)
    print(f"  mp4 → {out}")


def main():
    model, mark_names, df_window = load_model_and_events()

    cache = RUN_DIR / "case_study_press_quality_h.npz"
    print(f"Loading cached hidden states from {cache.name}")
    d = np.load(cache, allow_pickle=False)
    h_by_anchor = {}
    for k in d.files:
        if k.startswith("h_"):
            h_by_anchor[float(k[2:])] = torch.tensor(d[k])
    print(f"  {len(h_by_anchor)} anchors loaded")

    hero_figure_storm(model, mark_names, df_window, h_by_anchor)
    animation_mp4_storm(model, mark_names, df_window, h_by_anchor)
    print("\nDone.")


if __name__ == "__main__":
    main()
