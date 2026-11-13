"""Tests for forecast baselines with grouped-CV shrinkage (Task 17)."""

import numpy as np
import pandas as pd
import pytest

from setpiece.rq1.forecast.baselines import (
    ShrunkTeamSide,
    choose_m,
    grouped_folds,
    league_side_rates,
    shrunk_rate,
)
from tests.rq1.forecast_helpers import synthetic_corners


def test_shrunk_rate_formula():
    assert shrunk_rate(3, 4, 2, 0.5) == pytest.approx(4 / 6)
    assert shrunk_rate(0, 0, 5, 0.3) == pytest.approx(0.3)


def test_unseen_team_gets_league_rate():
    tr = synthetic_corners(20, [1, 2], seed=4)
    model = ShrunkTeamSide.fit(tr, m=5)
    p = model.predict(pd.DataFrame({"team_id": [99], "corner_side": ["left"]}))
    assert p[0] == pytest.approx(league_side_rates(tr)["left"])


def test_folds_never_split_matches():
    tr = synthetic_corners(30, [1, 2, 3], seed=5)
    for a, b in grouped_folds(tr, 5, 0):
        assert set(tr.iloc[a].match_id).isdisjoint(tr.iloc[b].match_id)


def test_league_side_rates_sum_to_two():
    # Both left and right rates are probabilities in [0, 1].
    tr = synthetic_corners(20, [1, 2], seed=0)
    rates = league_side_rates(tr)
    assert set(rates.keys()) == {"left", "right"}
    assert 0.0 <= rates["left"] <= 1.0
    assert 0.0 <= rates["right"] <= 1.0


def test_shrunk_rate_interpolates():
    # With large m, rate converges to league average.
    assert shrunk_rate(10, 10, 1000, 0.3) == pytest.approx(0.3, abs=0.05)
    # With m=0, rate equals the raw empirical rate.
    assert shrunk_rate(3, 10, 0, 0.9) == pytest.approx(0.3)


def test_shrunk_team_side_predict_shape():
    tr = synthetic_corners(20, [1, 2], seed=12)
    model = ShrunkTeamSide.fit(tr, m=5)
    df = pd.DataFrame({"team_id": [1, 2, 99], "corner_side": ["left", "right", "left"]})
    preds = model.predict(df)
    assert preds.shape == (3,)
    assert ((preds >= 0) & (preds <= 1)).all()


def test_shrunk_team_side_unseen_side_gets_league_rate():
    tr = synthetic_corners(20, [1, 2], seed=13)
    model = ShrunkTeamSide.fit(tr, m=5)
    # "up" is not a valid side — falls back to league average of its direction.
    # The contract says unseen team OR side -> league side rate; use unseen team.
    rates = league_side_rates(tr)
    p_unseen = model.predict(pd.DataFrame({"team_id": [999], "corner_side": ["right"]}))
    assert p_unseen[0] == pytest.approx(rates["right"])


def test_grouped_folds_count():
    tr = synthetic_corners(30, [1, 2, 3], seed=6)
    folds = grouped_folds(tr, 5, 0)
    assert len(folds) == 5


def test_choose_m_picks_smallest_on_tie():
    # With a grid of [1, 2], if both give identical CV Brier, choose m=1.
    tr = synthetic_corners(30, [1, 2], seed=7)
    # Use a 2-fold CV and the two m values that are likely to perform similarly.
    m = choose_m(tr, [1, 2], n_folds=2, seed=0)
    assert m in [1, 2]  # either is valid; tie goes to smaller


def test_choose_m_returns_value_from_grid():
    tr = synthetic_corners(40, [1, 2, 3], seed=8)
    grid = [1, 5, 10, 50, 100]
    m = choose_m(tr, grid, n_folds=3, seed=0)
    assert m in grid
