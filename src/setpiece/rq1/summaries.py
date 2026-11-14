"""RQ1 team-season distribution summaries with bootstrap CIs (Task 14).

Long-format table: one row per (team-season, attribute, category).
Every published share carries its numerator, denominator and CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from setpiece.config import Config
from setpiece.stats import match_bootstrap_ratio

ATTRIBUTES: list[str] = [
    "technique",
    "pass_height",
    "corner_side",
    "short_corner_proxy",
    "coarse_zone",
    "fine_zone",
    "pass_body_part",
]

SUMMARY_COLUMNS: list[str] = [
    "competition_id",
    "season_id",
    "team_id",
    "team_name",
    "matches_in_sample",
    "sample_label",
    "attribute",
    "category",
    "n",
    "denominator",
    "pct",
    "ci_lo",
    "ci_hi",
]


def _per_match_counts(
    rq1_team: pd.DataFrame, attribute: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-match (numerator, denominator) arrays for bootstrap.

    For a share, numerator = count of category in that match,
    denominator = total corners in that match.
    """
    per_match = rq1_team.groupby("match_id")[attribute]
    # denominator = corners per match
    den = per_match.size().to_numpy()
    return den


def summarise_rq1(
    rq1: pd.DataFrame,
    samples: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Long-format summary table with bootstrap CIs on category shares.

    Parameters
    ----------
    rq1:
        ``rq1_corners`` DataFrame (from ``build_rq1_corners``).
    samples:
        ``team_sample_table`` output (Task 11).
    cfg:
        Config (for ``bootstrap_reps`` and ``seed``).

    Returns
    -------
    DataFrame with columns ``SUMMARY_COLUMNS``.
    """
    n_reps = cfg.p("bootstrap_reps")
    seed = cfg.seed

    rows: list[dict] = []

    for _, srow in samples.iterrows():
        comp_id = srow["competition_id"]
        seas_id = srow["season_id"]
        team_id = srow["team_id"]
        team_name = srow["team_name"]
        matches_in_sample = srow["matches_in_sample"]
        sample_label = srow["sample_label"]

        team_corners = rq1[
            (rq1["competition_id"] == comp_id)
            & (rq1["season_id"] == seas_id)
            & (rq1["team_id"] == team_id)
        ]

        denominator = len(team_corners)

        for attr in ATTRIBUTES:
            if denominator == 0:
                # No corners for this team-season.
                rows.append(
                    {
                        "competition_id": comp_id,
                        "season_id": seas_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "matches_in_sample": matches_in_sample,
                        "sample_label": sample_label,
                        "attribute": attr,
                        "category": "none",
                        "n": 0,
                        "denominator": 0,
                        "pct": np.nan,
                        "ci_lo": np.nan,
                        "ci_hi": np.nan,
                    }
                )
                continue

            # Fill nulls with "Unknown".
            cats = team_corners[attr].fillna("Unknown")
            value_counts = cats.value_counts()

            # Per-match denominator for bootstrap.
            per_match = team_corners.assign(_cat=cats.values).groupby("match_id")

            for cat, n_cat in value_counts.items():
                pct_val = (n_cat / denominator) * 100.0

                # Per-match counts for this category for bootstrap.
                per_match_num = (
                    per_match["_cat"]
                    .apply(lambda x, c=cat: int((x == c).sum()))
                    .to_numpy()
                )
                per_match_den = per_match.size().to_numpy()

                lo, hi = match_bootstrap_ratio(
                    per_match_num, per_match_den, n_reps, seed
                )
                # Convert to percentage.
                lo_pct = lo * 100.0 if not np.isnan(lo) else np.nan
                hi_pct = hi * 100.0 if not np.isnan(hi) else np.nan

                rows.append(
                    {
                        "competition_id": comp_id,
                        "season_id": seas_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "matches_in_sample": matches_in_sample,
                        "sample_label": sample_label,
                        "attribute": attr,
                        "category": cat,
                        "n": int(n_cat),
                        "denominator": int(denominator),
                        "pct": pct_val,
                        "ci_lo": lo_pct,
                        "ci_hi": hi_pct,
                    }
                )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
