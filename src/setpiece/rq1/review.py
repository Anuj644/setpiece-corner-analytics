"""RQ1 review exports: short-borderline sampling, side outliers, technique-by-height (Task 13)."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    b = rq1[mask].sort_values("pass_length").copy()
    b["label_claude"] = pd.NA
    b["label_human"] = pd.NA
    return b


def sample_borderline(borderline: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``min(n, available)`` rows spread evenly over the length range.

    Uses equal-width length bins and a seeded draw within each bin.
    """
    if len(borderline) <= n:
        return borderline.copy()

    rng = np.random.default_rng(seed)
    lengths = borderline["pass_length"].values
    lo = lengths.min()
    hi = lengths.max()

    # Equal-width bins across the length range.
    n_bins = min(n, len(borderline))
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    sampled_indices = []
    for i in range(n_bins):
        mask = (lengths >= bin_edges[i]) & (lengths <= bin_edges[i + 1])
        bin_indices = np.where(mask)[0]
        if len(bin_indices) == 0:
            continue
        # Draw one per bin (or all if fewer than needed per bin).
        pick = rng.choice(bin_indices, size=1, replace=False)
        sampled_indices.extend(pick.tolist())

    # If we have more bins than n, truncate. If fewer, fill up randomly.
    if len(sampled_indices) > n:
        sampled_indices = sampled_indices[:n]
    elif len(sampled_indices) < n:
        remaining = set(range(len(borderline))) - set(sampled_indices)
        extra = rng.choice(list(remaining), size=n - len(sampled_indices), replace=False)
        sampled_indices.extend(extra.tolist())

    return borderline.iloc[sorted(set(sampled_indices))].reset_index(drop=True)


def side_outliers(rq1: pd.DataFrame) -> pd.DataFrame:
    """Corners with ``2 < start_y < 78`` (not at a corner flag)."""
    mask = rq1["start_y"].notna() & (rq1["start_y"] > 2) & (rq1["start_y"] < 78)
    return rq1[mask].copy()


def technique_by_height(rq1: pd.DataFrame) -> pd.DataFrame:
    """Per slice, the ``pass_height`` x ``technique`` count table."""
    return (
        rq1.groupby(["competition_id", "season_id", "pass_height", "technique"])
        .size()
        .reset_index(name="count")
    )
