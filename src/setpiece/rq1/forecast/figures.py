"""Forecast reliability plot (Task 20)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from setpiece.rq1.forecast.evaluate import reliability_table


def plot_reliability(y: np.ndarray, p: np.ndarray, title: str):
    """Reliability diagram: predicted vs observed fraction per bin."""
    rt = reliability_table(y, p, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    # Only plot bins with data.
    rt = rt[rt["n"] > 0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.plot(rt["mean_p"], rt["frac_pos"], "o-", color="steelblue")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
