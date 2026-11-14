"""RQ1 figure generation: endpoints and attribute bar charts (Task 15)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from setpiece.plots.pitch import draw_half_pitch


def figure_title(
    team_name: str,
    slice_name: str,
    n: int,
    date_min: str,
    date_max: str,
) -> str:
    """Formatted title string for RQ1 figures."""
    return f"{team_name} — {slice_name} — n={n} corners ({date_min}–{date_max})"


def plot_endpoints(rq1_team: pd.DataFrame, ax, title: str) -> None:
    """Scatter of mirrored corner endpoints on a half-pitch."""
    draw_half_pitch(ax)
    valid = rq1_team.dropna(subset=["pass_end_x", "end_y_mirrored"])
    ax.scatter(valid["pass_end_x"], valid["end_y_mirrored"], alpha=0.5, s=20)
    ax.set_title(title, fontsize=9)


def plot_attribute_bars(
    summary_team: pd.DataFrame,
    attribute: str,
    title: str,
):
    """Horizontal bar chart of category shares for one attribute."""
    data = summary_team[summary_team["attribute"] == attribute].copy()
    data = data.sort_values("pct", ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(data["category"], data["pct"], color="steelblue")
    ax.set_xlabel("Share (%)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
