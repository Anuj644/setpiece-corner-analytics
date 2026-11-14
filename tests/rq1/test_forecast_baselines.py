"""Tests for forecast baselines with grouped-CV shrinkage (Task 17)."""

import numpy as np
import pandas as pd
import pytest
from forecast_helpers import synthetic_corners

from setpiece.rq1.forecast import baselines
from setpiece.rq1.forecast.baselines import (
    LeagueSide,
    ShrunkTeamSide,
    choose_m,
    grouped_folds,
    league_side_rates,
    shrunk_rate,
)
from setpiece.rq1.forecast.split import labels


def corners(rows):
    """``rows`` of (match_id, team_id, side, ends_in_box) as a corners frame."""
    return pd.DataFrame(rows, columns=["match_id", "team_id", "corner_side", "ends_in_box"])


def test_shrunk_rate_formula():
    assert shrunk_rate(3, 4, 2, 0.5) == pytest.approx(4 / 6)
    assert shrunk_rate(0, 0, 5, 0.3) == pytest.approx(0.3)
    assert shrunk_rate(10, 10, 1000, 0.3) == pytest.approx(0.3, abs=0.01)  # big m: the league rate
    assert shrunk_rate(3, 10, 0, 0.9) == pytest.approx(0.3)  # m = 0: the team's own rate
    assert shrunk_rate(0, 0, 0, 0.9) == 0.9  # nothing at all: the league rate


def test_league_side_rates_are_the_empirical_rates():
    tr = corners(
        [(1, 1, "left", True), (1, 1, "left", False), (2, 2, "left", True), (2, 2, "right", False)]
    )
    assert league_side_rates(tr) == {"left": pytest.approx(2 / 3), "right": 0.0}


def test_league_side_rate_of_a_side_without_corners_is_the_pooled_rate():
    tr = corners([(1, 1, "left", True), (1, 1, "left", False), (2, 2, "left", True)])
    assert league_side_rates(tr)["right"] == pytest.approx(2 / 3)


def test_league_side_rates_need_corners():
    with pytest.raises(ValueError, match="no corners"):
        league_side_rates(corners([]))


def test_null_labels_are_refused_not_filled():
    tr = corners([(1, 1, "left", True), (1, 1, "left", pd.NA)])
    with pytest.raises(ValueError, match="prepare"):
        labels(tr)
    with pytest.raises(ValueError, match="prepare"):
        league_side_rates(tr)


def test_unseen_team_gets_league_rate():
    tr = synthetic_corners(20, [1, 2], seed=4)
    model = ShrunkTeamSide.fit(tr, m=5)
    p = model.predict(pd.DataFrame({"team_id": [99], "corner_side": ["left"]}))
    assert p[0] == pytest.approx(league_side_rates(tr)["left"])


def test_a_known_team_s_unseen_side_gets_the_league_rate_of_that_side():
    # team 1 only ever took left corners; its right-side prediction must be the league right rate
    tr = corners(
        [(1, 1, "left", True), (1, 1, "left", True), (2, 2, "left", False), (2, 2, "right", False)]
        + [(3, 2, "right", True), (3, 2, "right", False)]
    )
    model = ShrunkTeamSide.fit(tr, m=2)
    both = pd.DataFrame({"team_id": [1, 1], "corner_side": ["left", "right"]})
    left, right = model.predict(both)
    assert right == pytest.approx(league_side_rates(tr)["right"])
    assert left == pytest.approx(shrunk_rate(2, 2, 2, league_side_rates(tr)["left"]))


def test_shrunk_team_side_predict_matches_the_formula():
    tr = corners([(1, 7, "left", True), (1, 7, "left", False), (2, 8, "left", False)])
    p_league = 1 / 3
    got = ShrunkTeamSide.fit(tr, m=4).predict(
        pd.DataFrame({"team_id": [7, 8], "corner_side": ["left", "left"]})
    )
    assert got[0] == pytest.approx(shrunk_rate(1, 2, 4, p_league))
    assert got[1] == pytest.approx(shrunk_rate(0, 1, 4, p_league))


def test_league_side_model_predicts_the_side_rate():
    tr = synthetic_corners(30, [1, 2, 3], seed=3)
    rates = league_side_rates(tr)
    p = LeagueSide.fit(tr).predict(pd.DataFrame({"team_id": [9, 9], "corner_side": ["right", "left"]}))
    assert p.tolist() == [rates["right"], rates["left"]]


def test_folds_never_split_matches():
    tr = synthetic_corners(30, [1, 2, 3], seed=5)
    folds = grouped_folds(tr, 5, 0)
    assert len(folds) == 5
    for a, b in folds:
        assert set(tr.iloc[a].match_id).isdisjoint(tr.iloc[b].match_id)


def test_folds_depend_on_the_seed_only():
    tr = synthetic_corners(30, [1, 2, 3], seed=5)

    def val_sets(seed):
        return [tuple(v) for _, v in grouped_folds(tr, 5, seed)]

    assert val_sets(1) == val_sets(1)
    assert val_sets(1) != val_sets(2)


def test_choose_m_picks_the_smaller_m_on_a_tie_whatever_the_grid_order(monkeypatch):
    class Constant:
        @classmethod
        def fit(cls, train, m):
            return cls()

        def predict(self, df):
            return np.full(len(df), 0.5)  # the same Brier score for every m

    monkeypatch.setattr(baselines, "ShrunkTeamSide", Constant)
    tr = synthetic_corners(30, [1, 2], seed=7)
    assert choose_m(tr, [5, 1, 2], n_folds=3, seed=0) == 1
    assert choose_m(tr, [1, 2, 5], n_folds=3, seed=0) == 1


def test_choose_m_prefers_strong_shrinkage_when_teams_do_not_differ():
    # every team has the same true rate: only noise separates them, so shrinking must win
    tr = synthetic_corners(80, [1, 2, 3, 4], seed=11)
    assert choose_m(tr, [0, 1000], n_folds=5, seed=0) == 1000
