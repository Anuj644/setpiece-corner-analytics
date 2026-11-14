"""Forecast reliability plot (Task 20)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from setpiece.rq1.forecast.evaluate import reliability_table


def plot_reliability(y: np.ndarray, p: np.ndarray, title: str):
    """Reliability diagram: mean predicted probability against observed fraction, per bin.

    Bins without a test corner are left out; each point is labelled with its bin's corner count.
    """
    rt = reliability_table(y, p)
    rt = rt[rt["n"] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.plot(rt["mean_p"], rt["frac_pos"], "o-", color="steelblue")
    for _, row in rt.iterrows():
        ax.annotate(f"{int(row['n'])}", (row["mean_p"], row["frac_pos"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")  # fmt: skip
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
