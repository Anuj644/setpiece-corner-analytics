"""Tests for forecast time-split and prepare (Task 16)."""

import pandas as pd
import pytest

from setpiece.rq1.forecast.split import prepare, time_split
from tests.rq1.forecast_helpers import synthetic_corners


def test_no_match_in_both_and_share():
    c = synthetic_corners(n_matches=40, teams=[1, 2, 3, 4], seed=1)
    tr, te = time_split(c, 0.25)
    assert set(tr.match_id).isdisjoint(te.match_id)
    assert te.match_id.nunique() >= 10
    assert tr.match_date.max() < te.match_date.min()


def test_same_date_matches_stay_together():
    c = synthetic_corners(n_matches=8, teams=[1, 2], seed=2, same_date_pairs=True)
    tr, te = time_split(c, 0.25)
    shared_dates = set(tr.match_date) & set(te.match_date)
    for d in shared_dates:
        pytest.fail(f"date {d} split across train/test")


def test_null_label_rows_excluded_and_counted():
    c = synthetic_corners(n_matches=4, teams=[1, 2], seed=3)
    c.loc[c.index[:2], "ends_in_box"] = pd.NA
    clean, n_ex = prepare(c)
    assert n_ex == 2 and clean.ends_in_box.notna().all()


def test_prepare_drops_null_corner_side():
    c = synthetic_corners(n_matches=4, teams=[1, 2], seed=10)
    c.loc[c.index[:3], "corner_side"] = None
    clean, n_ex = prepare(c)
    assert n_ex == 3
    assert clean.corner_side.notna().all()


def test_prepare_drops_both_null_types():
    c = synthetic_corners(n_matches=4, teams=[1, 2], seed=11)
    c.loc[c.index[0], "ends_in_box"] = pd.NA
    c.loc[c.index[1], "corner_side"] = None
    clean, n_ex = prepare(c)
    assert n_ex == 2


def test_time_split_ceiling():
    # 10 matches, 25% -> ceil(2.5) = 3 test matches
    c = synthetic_corners(n_matches=10, teams=[1, 2], seed=5)
    tr, te = time_split(c, 0.25)
    assert te.match_id.nunique() == 3


def test_time_split_all_matches_covered():
    c = synthetic_corners(n_matches=20, teams=[1, 2], seed=6)
    tr, te = time_split(c, 0.3)
    all_matches = set(c.match_id)
    assert set(tr.match_id) | set(te.match_id) == all_matches


def test_time_split_per_slice():
    """When multiple slices are present, split is applied per slice."""
    c1 = synthetic_corners(n_matches=20, teams=[1, 2], seed=7)
    c2 = synthetic_corners(n_matches=20, teams=[3, 4], seed=8)
    c2["competition_id"] = 2
    c2["season_id"] = 2
    # Bump match_ids so they don't collide.
    c2["match_id"] = c2["match_id"] + 100
    combined = pd.concat([c1, c2], ignore_index=True)
    tr, te = time_split(combined, 0.25)
    # Both slices contribute to train and test.
    assert tr["competition_id"].nunique() == 2
    assert te["competition_id"].nunique() == 2
