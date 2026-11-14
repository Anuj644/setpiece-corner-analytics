"""RQ1 review exports for gate G2: short-corner borderline sample, side outliers, technique by height
(Task 13).

``run_review_export_rq1`` writes CSVs and one half-pitch PNG per sampled borderline corner to
``reports/review/rq1/``. The PNG shows the delivery and nothing else (no length, no rule output),
so a labeller decides ``short`` or ``not_short`` from the picture alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setpiece.config import Config
from setpiece.io import read_table
from setpiece.plots.pitch import draw_half_pitch
from setpiece.rq1.corners import PITCH_WIDTH

# A corner is taken from the corner flag, within this distance of the touchline (y = 0 or 80).
# One that starts further in is a side outlier to look at (spec §6.2).
CORNER_FLAG_MARGIN = 2.0


def short_borderline(rq1: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Ground-pass corners with ``lo <= pass_length <= hi``, sorted by length.

    Includes empty ``label_claude`` and ``label_human`` columns for review.
    """
    mask = (
        (rq1["pass_height"] == "Ground Pass")
        & rq1["pass_length"].notna()
        & (rq1["pass_length"] >= lo)
        & (rq1["pass_length"] <= hi)
    )
    b = rq1[mask].sort_values("pass_length", kind="stable").copy()
    b["label_claude"] = pd.NA
    b["label_human"] = pd.NA
    return b


def sample_borderline(borderline: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``min(n, available)`` rows, spread evenly over the length range, the same for the same seed.

    The length range is cut into ``n`` equal-width bins (half-open, the longest corner in the last
    one). Each pass takes one random row from every bin that still has rows, until ``n`` rows are
    chosen, so sparse bins give up their one row and crowded bins fill the rest.
    """
    if len(borderline) <= n:
        return borderline.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    lengths = borderline["pass_length"].to_numpy(dtype=float)
    inner_edges = np.linspace(lengths.min(), lengths.max(), n + 1)[1:-1]
    bins = np.digitize(lengths, inner_edges)  # 0 .. n-1
    pools = [rng.permutation(np.flatnonzero(bins == b)).tolist() for b in range(n)]

    picked: list[int] = []
    while len(picked) < n:
        for pool in pools:
            if pool and len(picked) < n:
                picked.append(pool.pop())
    return borderline.iloc[sorted(picked)].reset_index(drop=True)


def side_outliers(rq1: pd.DataFrame) -> pd.DataFrame:
    """Corners that start away from the corner flags: ``margin < start_y < width - margin``."""
    y = rq1["start_y"]
    mask = y.notna() & (y > CORNER_FLAG_MARGIN) & (y < PITCH_WIDTH - CORNER_FLAG_MARGIN)
    return rq1[mask].copy()


def technique_by_height(rq1: pd.DataFrame) -> pd.DataFrame:
    """Per slice, the ``pass_height`` x ``technique`` count table (a null height is kept)."""
    return (
        rq1.groupby(
            ["competition_id", "season_id", "pass_height", "technique"], dropna=False, sort=True
        )
        .size()
        .reset_index(name="count")
    )


def plot_corner_for_review(row: pd.Series, cfg: Config, path: Path) -> None:
    """Half-pitch PNG of one corner: where it was taken and where it went, nothing else."""
    fig, ax = plt.subplots(figsize=(5, 6))
    draw_half_pitch(ax, cfg)
    sx, sy = float(row["start_x"]), float(row["start_y"])
    ex, ey = float(row["pass_end_x"]), float(row["pass_end_y"])
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy), arrowprops={"arrowstyle": "->", "color": "C3"})
    ax.plot([sx], [sy], "o", color="C0", markersize=5)
    ax.set_title(str(row["event_id"]), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run_review_export_rq1(cfg: Config) -> dict:
    """Write the G2 review material for RQ1 to ``reports/review/rq1/`` and return its counts.

    Reads ``derived/rq1_corners`` (written by the ``rq1`` stage). The borderline range and sample
    size are ``short_review_range`` and ``short_review_sample``; the draw is seeded with
    ``cfg.seed``. Files: ``short_borderline_all.csv``, ``short_borderline_sample.csv`` (with empty
    ``label_claude`` / ``label_human`` columns), ``side_outliers.csv``, ``technique_by_height.csv``
    and ``short_<event_id>.png`` for every sampled corner.
    """
    try:
        rq1 = read_table("derived/rq1_corners", cfg)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; run `python -m setpiece rq1` first") from exc

    lo, hi = cfg.p("short_review_range")
    borderline = short_borderline(rq1, lo, hi)
    sample = sample_borderline(borderline, cfg.p("short_review_sample"), cfg.seed)
    outliers = side_outliers(rq1)

    out = cfg.reports_dir / "review" / "rq1"
    out.mkdir(parents=True, exist_ok=True)
    borderline.to_csv(out / "short_borderline_all.csv", index=False)
    sample.to_csv(out / "short_borderline_sample.csv", index=False)
    outliers.to_csv(out / "side_outliers.csv", index=False)
    technique_by_height(rq1).to_csv(out / "technique_by_height.csv", index=False)
    for _, row in sample.iterrows():
        plot_corner_for_review(row, cfg, out / f"short_{row['event_id']}.png")

    return {
        "corners": len(rq1),
        "borderline": len(borderline),
        "sampled": len(sample),
        "side_outliers": len(outliers),
    }
