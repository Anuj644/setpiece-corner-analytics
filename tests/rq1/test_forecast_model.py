"""Tests for L2 logistic regression forecast model (Task 18)."""

import pandas as pd
import pytest
from forecast_helpers import synthetic_corners

from setpiece.rq1.forecast.model import fit_logreg


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


def test_a_tie_in_cv_brier_goes_to_the_smaller_c_whatever_the_grid_order(monkeypatch):
    from setpiece.rq1.forecast import model

    monkeypatch.setattr(model, "brier", lambda y, p: 0.25)  # every C scores the same
    tr = synthetic_corners(30, [1, 2], seed=12)
    assert fit_logreg(tr, [10, 0.1, 1], 3, 0).C == 0.1
    assert fit_logreg(tr, [0.1, 1, 10], 3, 0).C == 0.1


def test_null_labels_are_refused():
    tr = synthetic_corners(20, [1, 2], seed=13)
    tr.loc[tr.index[0], "ends_in_box"] = pd.NA
    with pytest.raises(ValueError, match="prepare"):
        fit_logreg(tr, [1.0], 3, 0)
