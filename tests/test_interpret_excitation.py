"""Excitation-matrix extraction and plotting tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from eonet_cascades.interpret.excitation import (
    excitation_to_dataframe,
    plot_excitation_heatmap,
)
from eonet_cascades.models.hawkes import HawkesParams


def test_excitation_to_dataframe_shape():
    n_marks = 3
    p = HawkesParams(
        mu=np.zeros(n_marks),
        alpha=np.array([[0.1, 0.2, 0.0], [0.0, 0.3, 0.4], [0.5, 0.0, 0.6]]),
        beta=np.ones((n_marks, n_marks)),
        sigma=np.ones((n_marks, n_marks)),
    )
    mark_names = ["wildfire", "flood", "earthquake"]
    df = excitation_to_dataframe(p, mark_names)
    assert df.shape == (n_marks * n_marks, 5)  # parent_mark, child_mark, alpha, beta, sigma
    # Diagonal entries should appear.
    diag = df.filter(pl.col("parent_mark") == pl.col("child_mark"))
    assert diag.height == n_marks


def test_plot_excitation_heatmap_returns_figure(tmp_path):
    n_marks = 4
    p = HawkesParams(
        mu=np.zeros(n_marks),
        alpha=np.random.default_rng(0).uniform(0, 0.5, (n_marks, n_marks)),
        beta=np.ones((n_marks, n_marks)),
        sigma=np.ones((n_marks, n_marks)),
    )
    mark_names = [f"m{i}" for i in range(n_marks)]
    fig = plot_excitation_heatmap(p, mark_names)
    out = tmp_path / "alpha.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000  # rendered SOMETHING
