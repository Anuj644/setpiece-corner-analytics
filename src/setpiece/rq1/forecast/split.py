"""Forecast lane — data preparation and time split (Task 16).

``prepare``  drops rows with null ``ends_in_box`` or ``corner_side`` and
returns the cleaned frame plus the excluded row count.

``time_split`` partitions each slice's matches chronologically:
  * Matches are sorted by ``(match_date, match_id)``.
  * The number of test matches is ``ceil(test_share * n_matches)``.
  * Any match whose date equals the boundary date is moved to the **test**
    side, so matches on the same day are never split across train/test.
"""

from __future__ import annotations

import math

import pandas as pd


def prepare(corners: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with null ``ends_in_box`` or ``corner_side``.

    Returns
    -------
    clean:
        The filtered DataFrame (index reset).
    n_excluded:
        Number of rows dropped.
    """
    mask = corners["ends_in_box"].notna() & corners["corner_side"].notna()
    clean = corners[mask].reset_index(drop=True)
    n_excluded = int((~mask).sum())
    return clean, n_excluded


def time_split(
    corners: pd.DataFrame,
    test_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train / test split applied independently per slice.

    Parameters
    ----------
    corners:
        DataFrame with at least ``match_id``, ``match_date``,
        ``competition_id`` and ``season_id`` columns.
    test_share:
        Fraction of matches to place in the test set
        (``ceil(test_share * n_matches)`` matches per slice).

    Returns
    -------
    train, test : pd.DataFrame
        Disjoint subsets.  All matches on the boundary date go to test.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    slice_keys = ["competition_id", "season_id"]
    # Handle DataFrames that may not carry slice keys (single-slice tests).
    group_keys = [k for k in slice_keys if k in corners.columns]

    if group_keys:
        groups = corners.groupby(group_keys, sort=False)
    else:
        # Treat the whole frame as one slice.
        groups = [(None, corners)]

    for _, grp in groups:
        # Unique matches sorted by (match_date, match_id).
        match_df = (
            grp[["match_id", "match_date"]]
            .drop_duplicates("match_id")
            .sort_values(["match_date", "match_id"])
            .reset_index(drop=True)
        )
        n_matches = len(match_df)
        n_test = math.ceil(test_share * n_matches)

        # Boundary: the date of the n_test-th match from the end.
        boundary_date = match_df.iloc[n_matches - n_test]["match_date"]

        # All matches on or after the boundary date go to test.
        test_match_ids = set(
            match_df.loc[match_df["match_date"] >= boundary_date, "match_id"]
        )
        train_match_ids = set(match_df["match_id"]) - test_match_ids

        train_parts.append(grp[grp["match_id"].isin(train_match_ids)])
        test_parts.append(grp[grp["match_id"].isin(test_match_ids)])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test
