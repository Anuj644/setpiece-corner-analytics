"""Tests for L2 logistic regression forecast model (Task 18)."""

import pandas as pd
import pytest

from setpiece.rq1.forecast.model import fit_logreg
from tests.rq1.forecast_helpers import synthetic_corners


def test_learns_team_effect():
    tr = synthetic_corners(
        60,
        [1, 2],
        seed=6,
        p_by_team_side={
            (1, "left"): 0.9,
            (1, "right"): 0.9,
            (2, "left"): 0.2,
            (2, "right"): 0.2,
        },
    )
    m = fit_logreg(tr, [0.01, 0.1, 1, 10, 100], 5, 0)
    p = m.predict(pd.DataFrame({"team_id": [1, 2], "corner_side": ["left", "left"]}))
    assert p[0] > 0.7 and p[1] < 0.4


def test_unseen_team_predicts_without_error():
    tr = synthetic_corners(20, [1, 2], seed=7)
    p = fit_logreg(tr, [1.0], 3, 0).predict(
        pd.DataFrame({"team_id": [42], "corner_side": ["right"]})
    )
    assert 0 < p[0] < 1


def test_c_attribute_is_from_grid():
    tr = synthetic_corners(30, [1, 2, 3], seed=9)
    grid = [0.01, 0.1, 1.0]
    m = fit_logreg(tr, grid, 3, 0)
    assert m.C in grid


def test_predict_probabilities_in_range():
    tr = synthetic_corners(40, [1, 2, 3, 4], seed=10)
    m = fit_logreg(tr, [1.0], 3, 0)
    df = pd.DataFrame({"team_id": [1, 2, 3, 4], "corner_side": ["left", "right", "left", "right"]})
    p = m.predict(df)
    assert ((p >= 0) & (p <= 1)).all()


def test_handles_single_c_grid():
    tr = synthetic_corners(20, [1, 2], seed=11)
    m = fit_logreg(tr, [0.5], 2, 0)
    assert m.C == pytest.approx(0.5)
