"""Tests for forecast evaluation: Brier, CI, decision rule, reliability (Task 19)."""

import numpy as np
import pandas as pd
import pytest
from forecast_helpers import synthetic_corners

from setpiece.rq1.forecast.evaluate import (
    brier,
    brier_diff_ci,
    decide,
    reliability_table,
)
from setpiece.rq1.forecast.split import labels
from setpiece.stats import match_bootstrap_ratio


def test_brier_exact():
    assert brier(np.array([1, 0]), np.array([0.8, 0.2])) == pytest.approx(0.04)


def test_brier_perfect_and_worst_prediction():
    assert brier(np.array([1, 1, 0, 0]), np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert brier(np.array([1, 0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_decide_prefers_simpler_when_tied():
    assert (
        decide(
            0.20,
            {
                "league_side": (0.20, 0.0, -0.01, 0.01),
                "team_side_shrunk": (0.21, -0.01, -0.02, 0.0),
            },
        )
        == "league_side"
    )


def test_decide_ci_includes_zero_means_simpler():
    assert (
        decide(
            0.18,
            {
                "league_side": (0.19, -0.01, -0.03, 0.01),
                "team_side_shrunk": (0.20, -0.02, -0.04, -0.005),
            },
        )
        == "league_side"
    )


def test_decide_model_when_clearly_better():
    assert (
        decide(
            0.15,
            {
                "league_side": (0.20, -0.05, -0.07, -0.03),
                "team_side_shrunk": (0.19, -0.04, -0.06, -0.02),
            },
        )
        == "model"
    )


def test_decide_team_side_shrunk_when_only_it_is_not_clearly_beaten():
    result = decide(
        0.18,
        {
            "league_side": (0.25, -0.07, -0.09, -0.05),
            "team_side_shrunk": (0.19, -0.01, -0.03, 0.01),
        },
    )
    assert result == "team_side_shrunk"


def test_brier_diff_ci_point_estimate_is_the_exact_difference():
    c = synthetic_corners(30, [1, 2], seed=1)
    y = labels(c)
    p_model, p_base = np.full(len(c), 0.6), np.full(len(c), 0.5)
    diff, lo, hi = brier_diff_ci(c, p_model, p_base, n_reps=300, seed=0)
    assert diff == pytest.approx(brier(y, p_model) - brier(y, p_base))
    assert lo <= hi


def test_brier_diff_ci_is_the_match_bootstrap_of_the_per_match_sums():
    """Two matches with opposite results: the CI must span both, as whole matches are resampled."""
    c = pd.DataFrame(
        {"match_id": [1, 1, 2, 2], "ends_in_box": [True, True, False, False]}
    ).astype({"ends_in_box": "boolean"})
    p_model = np.array([0.9, 0.9, 0.9, 0.9])  # right on match 1, wrong on match 2
    p_base = np.full(4, 0.5)
    diff, lo, hi = brier_diff_ci(c, p_model, p_base, n_reps=500, seed=3)
    err = (labels(c) - p_model) ** 2 - (labels(c) - p_base) ** 2
    assert diff == pytest.approx(err.mean())
    assert (lo, hi) == match_bootstrap_ratio(np.array([err[:2].sum(), err[2:].sum()]),
                                             np.array([2, 2]), 500, 3)  # fmt: skip
    assert lo < 0 < hi  # one match favours the model, the other the baseline


def test_brier_diff_ci_is_negative_when_the_model_is_better_and_deterministic():
    c = synthetic_corners(30, [1, 2], seed=1)
    p_base = np.full(len(c), 0.5)
    first = brier_diff_ci(c, labels(c), p_base, n_reps=200, seed=1)  # a perfect model
    assert first[0] < 0 and first[2] < 0  # the whole CI is below zero
    assert first == brier_diff_ci(c, labels(c), p_base, n_reps=200, seed=1)


def test_brier_diff_ci_refuses_null_labels():
    c = synthetic_corners(4, [1, 2], seed=1)
    c.loc[c.index[0], "ends_in_box"] = pd.NA
    with pytest.raises(ValueError, match="prepare"):
        brier_diff_ci(c, np.full(len(c), 0.5), np.full(len(c), 0.5), 10, 0)


def test_reliability_table_columns_and_counts():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 100), rng.uniform(0, 1, 100)
    rt = reliability_table(y, p, n_bins=5)
    assert list(rt.columns) == ["bin", "mean_p", "frac_pos", "n"] and len(rt) == 5
    assert rt["n"].sum() == 100


def test_reliability_table_is_exact_on_a_known_case():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.55, 0.6, 1.0])  # 1.0 belongs to the last bin
    rt = reliability_table(y, p, n_bins=2)
    assert rt["n"].tolist() == [2, 3]
    assert rt["frac_pos"].tolist() == [0.0, 1.0]
    assert rt["mean_p"].tolist() == pytest.approx([0.075, (0.55 + 0.6 + 1.0) / 3])
