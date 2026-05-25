"""Cascade-graph extraction from a fitted ParametricHawkes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl

from eonet_cascades.models.hawkes import HawkesParams


def excitation_to_dataframe(params: HawkesParams, mark_names: list[str]) -> pl.DataFrame:
    """Flatten (alpha, beta, sigma) into a long DataFrame keyed by (parent_mark, child_mark)."""
    n_marks = params.K
    if len(mark_names) != n_marks:
        raise ValueError(f"mark_names length {len(mark_names)} != n_marks={n_marks}")
    rows = []
    for i in range(n_marks):
        for j in range(n_marks):
            rows.append(
                {
                    "parent_mark": mark_names[i],
                    "child_mark": mark_names[j],
                    "alpha": float(params.alpha[i, j]),
                    "beta": float(params.beta[i, j]),
                    "sigma": float(params.sigma[i, j]),
                }
            )
    return pl.DataFrame(rows)


def plot_excitation_heatmap(
    params: HawkesParams,
    mark_names: list[str],
    title: str = "Cross-mark excitation alpha",
):
    """Render the alpha matrix as a heatmap. Rows are parents, columns are children."""
    n_marks = params.K
    fig, ax = plt.subplots(figsize=(0.6 * n_marks + 2, 0.6 * n_marks + 2))
    vmax = max(float(params.alpha.max()), 1e-6)
    # Try rocket_r (seaborn); fall back to viridis if not available.
    try:
        cmap = "rocket_r"
        im = ax.imshow(params.alpha, vmin=0.0, vmax=vmax, cmap=cmap, aspect="equal")
    except (ValueError, KeyError):
        im = ax.imshow(params.alpha, vmin=0.0, vmax=vmax, cmap="viridis", aspect="equal")
    ax.set_xticks(range(n_marks))
    ax.set_yticks(range(n_marks))
    ax.set_xticklabels(mark_names, rotation=45, ha="right")
    ax.set_yticklabels(mark_names)
    ax.set_xlabel("child mark")
    ax.set_ylabel("parent mark")
    ax.set_title(title)
    for i in range(n_marks):
        for j in range(n_marks):
            val = params.alpha[i, j]
            if val > 0.01:
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if val > vmax / 2 else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, label="alpha (branching ratio)")
    fig.tight_layout()
    return fig
