"""RQ1 review exports for gate G2: short-corner borderline sample, side outliers, technique by height
(Task 13).

``run_review_export_rq1`` writes one half-pitch PNG per sampled borderline corner and a blind label
sheet to ``reports/review/rq1/``: the PNG shows the delivery and nothing else, and the sheet holds
only ``event_id``, the PNG name and empty label columns, in a seeded shuffled order, so a labeller
decides ``short`` or ``not_short`` from the picture alone. Everything that reveals the rule (the
pass length, ``short_corner_proxy``, the order by length) is in ``reports/review_key/rq1/``, which
is not handed to the labellers.

The sample is drawn from the core slices only (``Slice.core``), so the 60 corners stay the same
when the corpus is expanded.
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


KEY_FILES = ("short_borderline_all.csv", "short_borderline_key.csv")

# What side_outliers.csv shows: which corners to look at, not how long they were or what the rule
# says about them, since it sits in the labellers' folder and an outlier can be a borderline pass.
SIDE_OUTLIER_COLUMNS = [
    "competition_id", "season_id", "match_id", "match_date", "team_name", "event_id",
    "start_x", "start_y", "corner_side",
]  # fmt: skip


def core_only(rq1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The rows of ``rq1`` that belong to a core slice (``Slice.core``), in their order."""
    core = {(s.competition_id, s.season_id) for s in cfg.slices if s.core}
    keep = [(int(c), int(s)) in core for c, s in zip(rq1["competition_id"], rq1["season_id"])]
    return rq1[keep]


def review_png_name(event_id: str) -> str:
    return f"short_{event_id}.png"


def blind_review_sheet(sample: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(sheet, shuffled sample)``: the label sheet and the sample in the sheet's order.

    The sheet has ``event_id``, ``png`` and empty ``label_claude`` / ``label_human`` columns. Its
    order is a permutation drawn from ``seed`` (not the length order of ``sample``), so neither the
    order of the rows nor any other column says how long a corner is.
    """
    order = np.random.default_rng([seed, 1]).permutation(len(sample))
    shuffled = sample.iloc[order].reset_index(drop=True)
    sheet = pd.DataFrame(
        {
            "event_id": shuffled["event_id"],
            "png": [review_png_name(e) for e in shuffled["event_id"]],
            "label_claude": pd.Series(pd.NA, index=shuffled.index, dtype="string"),
            "label_human": pd.Series(pd.NA, index=shuffled.index, dtype="string"),
        }
    )
    return sheet, shuffled


def run_review_export_rq1(cfg: Config) -> dict:
    """Write the G2 review material for RQ1 and return its counts.

    Reads ``derived/rq1_corners`` (written by the ``rq1`` stage). The borderline range and sample
    size are ``short_review_range`` and ``short_review_sample``; the draw is seeded with
    ``cfg.seed`` and taken from the core slices.

    ``reports/review/rq1/`` (for the labellers): ``short_borderline_sample.csv`` (the blind sheet),
    ``short_<event_id>.png`` for every sampled corner, ``side_outliers.csv`` and
    ``technique_by_height.csv``. ``reports/review_key/rq1/`` (not for the labellers):
    ``short_borderline_key.csv`` (the sample with its lengths, in the sheet's order) and
    ``short_borderline_all.csv`` (the whole core pool, by length). Old ``short_*.png`` files are
    deleted first, so the folder holds exactly the PNGs of this sample.
    """
    try:
        rq1 = read_table("derived/rq1_corners", cfg)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; run `python -m setpiece rq1` first") from exc

    lo, hi = cfg.p("short_review_range")
    pool = short_borderline(core_only(rq1, cfg), lo, hi)
    sample = sample_borderline(pool, cfg.p("short_review_sample"), cfg.seed)
    sheet, keyed = blind_review_sheet(sample, cfg.seed)
    outliers = side_outliers(rq1)

    out = cfg.reports_dir / "review" / "rq1"
    key_dir = cfg.reports_dir / "review_key" / "rq1"
    out.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("short_*.png"), *(out / name for name in KEY_FILES)]:
        stale.unlink(missing_ok=True)  # key files lived here before they were moved out

    sheet.to_csv(out / "short_borderline_sample.csv", index=False)
    outliers[SIDE_OUTLIER_COLUMNS].to_csv(out / "side_outliers.csv", index=False)
    technique_by_height(rq1).to_csv(out / "technique_by_height.csv", index=False)
    keyed.to_csv(key_dir / "short_borderline_key.csv", index=False)
    pool.to_csv(key_dir / "short_borderline_all.csv", index=False)
    for _, row in keyed.iterrows():
        plot_corner_for_review(row, cfg, out / review_png_name(row["event_id"]))

    return {
        "corners": len(rq1),
        "borderline": len(pool),
        "sampled": len(sample),
        "side_outliers": len(outliers),
    }