"""RQ1 figures: mirrored endpoints and one bar chart per attribute (Task 15, spec §6.3)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setpiece.config import Config
from setpiece.plots.pitch import draw_half_pitch
from setpiece.rq1.summaries import ATTRIBUTES


def figure_title(
    team_name: str,
    slice_name: str,
    n: int,
    date_min: str,
    date_max: str,
) -> str:
    """Formatted title string for RQ1 figures."""
    return f"{team_name} — {slice_name} — n={n} corners ({date_min}–{date_max})"


def plot_endpoints(rq1_team: pd.DataFrame, ax, title: str, cfg: Config) -> None:
    """Scatter of mirrored corner endpoints on a half-pitch."""
    draw_half_pitch(ax, cfg)
    valid = rq1_team.dropna(subset=["pass_end_x", "end_y_mirrored"])
    ax.scatter(valid["pass_end_x"].astype(float), valid["end_y_mirrored"].astype(float),
               alpha=0.5, s=20)  # fmt: skip
    ax.set_title(title, fontsize=8)


def plot_attribute_bars(
    summary_team: pd.DataFrame,
    attribute: str,
    title: str,
):
    """Horizontal bar chart of category shares (%) for one attribute.

    Each bar carries its ``n/denominator`` label and a whisker for the 95% match-bootstrap interval
    (``ci_lo`` to ``ci_hi``). The whisker is a plain segment, not an error bar: a percentile
    interval need not contain the point estimate, and it is drawn as computed.
    """
    data = summary_team[summary_team["attribute"] == attribute].sort_values("pct", ascending=True)
    positions = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(positions, data["pct"].astype(float), color="steelblue")
    ax.hlines(positions, data["ci_lo"].astype(float), data["ci_hi"].astype(float),
              color="black", linewidth=1.2, zorder=3)  # fmt: skip
    ax.plot(data["ci_lo"].astype(float), positions, "|", color="black", zorder=3)
    ax.plot(data["ci_hi"].astype(float), positions, "|", color="black", zorder=3)
    for y, (_, row) in zip(positions, data.iterrows(), strict=True):
        x = max((v for v in (row["pct"], row["ci_hi"]) if pd.notna(v)), default=0.0)
        ax.text(x, y, f"  {row['n']}/{row['denominator']}", va="center", fontsize=7)
    ax.set_yticks(positions, data["category"].astype(str))
    ax.set_xlabel("Share of corners (%); whiskers: 95% match-bootstrap interval")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    return fig

def save_team_figures(
    rq1_team: pd.DataFrame,
    summary_team: pd.DataFrame,
    slice_name: str,
    out_dir: Path,
    cfg: Config,
) -> list[Path]:
    """Write the endpoint scatter and one bar chart per attribute for one team-season.

    Files are ``<comp>_<season>_<team_id>_endpoints.svg`` and
    ``<comp>_<season>_<team_id>_<attribute>.svg``; every title is ``figure_title`` (team, slice,
    n, first and last corner date) so a figure carries its own sample size.
    """
    first = rq1_team.iloc[0]
    stem = f"{first['competition_id']}_{first['season_id']}_{first['team_id']}"
    dates = rq1_team["match_date"].dropna().sort_values()
    title = figure_title(
        str(first["team_name"]), slice_name, len(rq1_team),
        str(dates.iloc[0]) if len(dates) else "?",
        str(dates.iloc[-1]) if len(dates) else "?",
    )  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    fig, ax = plt.subplots(figsize=(6, 8))
    plot_endpoints(rq1_team, ax, title, cfg)
    written.append(out_dir / f"{stem}_endpoints.svg")
    fig.savefig(written[-1])
    plt.close(fig)

    for attribute in ATTRIBUTES:
        fig = plot_attribute_bars(summary_team, attribute, f"{title}\n{attribute}")
        written.append(out_dir / f"{stem}_{attribute}.svg")
        fig.savefig(written[-1])
        plt.close(fig)
    return written
