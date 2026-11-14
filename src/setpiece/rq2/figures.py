"""RQ2 figures: shots per corner by team and end-reason shares by team (Task 26, spec 6.3).

One pair of charts per configured slice. Every title is ``figure_title`` (what is shown, slice, n
and the date range of the corners), so a chart carries its own sample size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from setpiece.config import Config
from setpiece.rq1.figures import figure_title

_INK = "#2b4c7e"  # the one hue of the shots chart
_MUTED = "#6b7280"

# A colour belongs to its end reason, whichever reasons a chart shows (Okabe-Ito based).
END_REASON_COLOURS: dict[str, str] = {
    "goal": "#009E73",
    "stoppage": "#E69F00",
    "restart": "#0072B2",
    "new_corner": "#56B4E9",
    "possession_change": "#CC79A7",
    "time_cap": "#D55E00",
    "period_end": "#999999",
    "timeline_end": "#444444",
}
_OTHER_COLOUR = "#BBBBBB"  # a reason this module has not heard of


def _label(name, team_id, eligible, total, detailed) -> str:
    """The tick label of a team: its name (``Team <id>`` if it has none) and its n.

    ``(n=<eligible>)``; with ``detailed`` (some corner of the slice is incomplete)
    ``(<eligible> eligible of <total> corners, <k> incomplete)``, spec 7.
    """
    who = name if not pd.isna(name) else f"Team {team_id}"
    if detailed:
        return f"{who} ({eligible} eligible of {total} corners, {total - eligible} incomplete)"
    return f"{who} (n={eligible})"


def slice_title(
    label: str, slice_name: str, eligible: int, total: int, first: str, last: str
) -> str:
    """``figure_title`` of a slice's chart; says "<eligible> eligible of <total>" if any are not.

    ``eligible`` counts the complete chains and ``total`` all of them, so a reader sees what was
    left out next to the denominator (spec 7), e.g. ``n=230 eligible of 236 corners (6 incomplete;
    <first>-<last>)``.
    """
    if eligible == total:
        return figure_title(label, slice_name, total, first, last)
    return (
        f"{label} — {slice_name} — n={eligible} eligible of {total} corners "
        f"({total - eligible} incomplete; {first}–{last})"
    )


def plot_shots_per_corner(summary: pd.DataFrame, title: str) -> Figure:
    """Shots per eligible corner by team, with 95% interval whiskers, the highest team on top.

    ``summary`` is ``summarise_rq2`` output (the ``shots_per_corner`` rows are used). A team
    without an eligible corner has no value and no row. A team whose interval is NaN (too few
    matches with a complete chain) is a hollow marker without whiskers. The whiskers are drawn
    from ``ci_lo`` to ``ci_hi``, not as offsets from the value: a percentile interval can miss it.
    If any chain of the slice is incomplete the tick labels say how many corners were left out.
    """
    shots = summary[summary["measure"] == "shots_per_corner"]
    detailed = bool(shots["n_incomplete"].sum() > 0)
    data = shots[shots["value"].notna()].sort_values("value", kind="stable").reset_index(drop=True)
    value, lo, hi = (
        data[c].to_numpy(dtype=float, na_value=np.nan) for c in ("value", "ci_lo", "ci_hi")
    )
    y = np.arange(len(data))
    has_interval = ~(np.isnan(lo) | np.isnan(hi))

    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.3 * len(data) + 1.6)), layout="constrained")
    if len(data) == 0:
        ax.text(
            0.5,
            0.5,
            "no eligible corners",
            ha="center",
            va="center",
            color=_MUTED,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)
        return fig
    ax.hlines(y[has_interval], lo[has_interval], hi[has_interval], color=_INK, linewidth=1.5)
    ax.plot(value[has_interval], y[has_interval], "o", color=_INK, markersize=5)
    if (~has_interval).any():
        ax.plot(
            value[~has_interval], y[~has_interval], "o", markerfacecolor="none", color=_INK,
            markersize=5, label="no interval (too few matches)",
        )  # fmt: skip
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            _label(r.team_name, r.team_id, r.denominator, r.n_corners, detailed)
            for r in data.itertuples()
        ],
        fontsize=7,
    )
    ax.set_xlim(left=0)
    ax.set_xlabel("Shots per eligible corner (attacking team; 95% match-bootstrap interval)")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=9)
    return fig


def plot_end_reasons(end_reasons: pd.DataFrame, title: str) -> Figure:
    """Share of each team's corner phases by ``end_reason`` as one stacked bar per team.

    ``end_reasons`` is ``end_reason_table`` output with a ``team_name`` column; every chain counts,
    incomplete ones included (``outcome_status`` tells them apart). Teams are in name order; each
    reason has a fixed colour. If any chain of the slice is incomplete the tick labels say how
    many corners were left out of the eligible n. No rows give an empty page that says so.
    """
    if end_reasons.empty:
        fig, ax = plt.subplots(figsize=(7, 3), layout="constrained")
        ax.text(
            0.5, 0.5, "no corners", ha="center", va="center", color=_MUTED, transform=ax.transAxes
        )
        ax.set_axis_off()
        ax.set_title(title, fontsize=9)
        return fig
    counts = (
        end_reasons.groupby(["team_id", "end_reason"], dropna=False)["n"]
        .sum()
        .unstack(fill_value=0)
    )
    names = end_reasons.drop_duplicates("team_id").set_index("team_id")["team_name"]
    teams = sorted(counts.index, key=lambda t: (str(names.get(t, "")), t))
    known = [r for r in END_REASON_COLOURS if r in counts.columns]
    reasons = known + sorted(r for r in counts.columns if r not in END_REASON_COLOURS)
    totals = counts.sum(axis=1)
    incomplete = end_reasons[end_reasons["outcome_status"] != "complete"]
    incomplete = incomplete.groupby("team_id")["n"].sum().reindex(totals.index, fill_value=0)
    detailed = bool(incomplete.sum() > 0)

    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.3 * len(teams) + 1.6)), layout="constrained")
    left = np.zeros(len(teams))
    y = np.arange(len(teams))
    for reason in reasons:
        share = np.array([counts.loc[t, reason] / totals[t] for t in teams], dtype=float)
        ax.barh(
            y, share, left=left, height=0.7, color=END_REASON_COLOURS.get(reason, _OTHER_COLOUR),
            edgecolor="white", linewidth=0.8, label=reason.replace("_", " "),
        )  # fmt: skip
        left += share
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            _label(names.get(t, pd.NA), t, int(totals[t] - incomplete[t]), int(totals[t]), detailed)
            for t in teams
        ],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the team's corner phases")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=9)
    return fig


def _date_range(rq1: pd.DataFrame, competition_id: int, season_id: int) -> tuple[str, str]:
    """First and last ``match_date`` of the slice's corners, ``("?", "?")`` if there is none."""
    in_slice = (rq1["competition_id"] == competition_id) & (rq1["season_id"] == season_id)
    dates = rq1.loc[in_slice, "match_date"].dropna().sort_values()
    return (str(dates.iloc[0]), str(dates.iloc[-1])) if len(dates) else ("?", "?")


def save_rq2_figures(
    summary: pd.DataFrame,
    end_reasons: pd.DataFrame,
    rq1: pd.DataFrame,
    cfg: Config,
    out_dir: Path,
) -> list[Path]:
    """Per slice, write ``<comp>_<season>_shots_per_corner.svg`` and ``..._end_reasons.svg``.

    ``summary`` is ``summarise_rq2`` output of the main table, ``end_reasons`` ``end_reason_table``
    output and ``rq1`` ``derived/rq1_corners`` (for the date range). The n of the shots chart is the
    slice's eligible (complete) corners; if some chains are incomplete both titles (and the team
    labels) say "<eligible> eligible of <all> corners (<k> incomplete)". A slice with no team in
    ``summary`` gets no figures. Returns the paths written, in slice order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = summary[["competition_id", "season_id", "team_id", "team_name"]].drop_duplicates()
    written = []
    for sl in cfg.slices:
        in_summary = (summary["competition_id"] == sl.competition_id) & (
            summary["season_id"] == sl.season_id
        )
        if not in_summary.any():
            continue
        first, last = _date_range(rq1, sl.competition_id, sl.season_id)
        stem = f"{sl.competition_id}_{sl.season_id}"

        shots = summary[in_summary]
        counted = shots[shots["measure"] == "shots_per_corner"]
        eligible, total = int(counted["denominator"].sum()), int(counted["n_corners"].sum())
        title = slice_title(
            "Shots per eligible corner by team", sl.name, eligible, total, first, last
        )
        fig = plot_shots_per_corner(shots, title)
        written.append(out_dir / f"{stem}_shots_per_corner.svg")
        fig.savefig(written[-1])
        plt.close(fig)

        in_ends = (end_reasons["competition_id"] == sl.competition_id) & (
            end_reasons["season_id"] == sl.season_id
        )
        ends = end_reasons[in_ends].merge(
            names, on=["competition_id", "season_id", "team_id"], how="left", validate="many_to_one"
        )
        title = slice_title(
            "How the phase after the corner ends, by team", sl.name, eligible, total, first, last
        )
        fig = plot_end_reasons(ends, title)
        written.append(out_dir / f"{stem}_end_reasons.svg")
        fig.savefig(written[-1])
        plt.close(fig)
    return written
