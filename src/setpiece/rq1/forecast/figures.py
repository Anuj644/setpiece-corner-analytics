"""Forecast reliability plot (Task 20)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from setpiece.rq1.forecast.evaluate import reliability_table

COLORS = {"model": "tab:blue", "league_side": "tab:green", "team_side_shrunk": "tab:orange"}


def plot_reliability(
    y: np.ndarray,
    preds: dict[str, np.ndarray],
    title: str,
    chosen: str | None = None,
):
    """Reliability diagram of every method on the same test corners.

    ``preds`` maps a method name to its predicted probabilities for the corners with outcomes
    ``y``. Each method is a line of mean predicted probability against observed fraction per
    bin (bins without a test corner are left out); the chosen method is drawn thick, marked in the
    legend and labelled with the corner count of each of its bins.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    for name, p in preds.items():
        rt = reliability_table(y, p)
        rt = rt[rt["n"] > 0]
        is_chosen = name == chosen
        ax.plot(
            rt["mean_p"], rt["frac_pos"], "o-", color=COLORS.get(name),
            linewidth=2.5 if is_chosen else 1.0, markersize=5 if is_chosen else 3,
            alpha=1.0 if is_chosen else 0.7, label=f"{name} (chosen)" if is_chosen else name,
        )  # fmt: skip
        if is_chosen:
            for _, row in rt.iterrows():
                ax.annotate(f"{int(row['n'])}", (row["mean_p"], row["frac_pos"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")  # fmt: skip
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction")
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig
