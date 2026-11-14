"""Tests for forecast evaluation: Brier, CI, decision rule, reliability (Task 19)."""

import numpy as np
import pytest

from setpiece.rq1.forecast.evaluate import (
    brier,
    brier_diff_ci,
    decide,
    reliability_table,
)
from tests.rq1.forecast_helpers import synthetic_corners


def test_brier_exact():
    assert brier(np.array([1, 0]), np.array([0.8, 0.2])) == pytest.approx(0.04)


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


def test_brier_perfect_prediction():
    y = np.array([1, 1, 0, 0])
    p = np.array([1.0, 1.0, 0.0, 0.0])
    assert brier(y, p) == pytest.approx(0.0)


def test_brier_worst_prediction():
    y = np.array([1, 0])
    p = np.array([0.0, 1.0])
    assert brier(y, p) == pytest.approx(1.0)


def test_brier_diff_ci_returns_triple():
    c = synthetic_corners(20, [1, 2], seed=0)
    p_model = np.full(len(c), 0.6)
    p_base = np.full(len(c), 0.5)
    diff, lo, hi = brier_diff_ci(c, p_model, p_base, n_reps=200, seed=0)
    assert lo <= diff <= hi or lo <= hi  # CI is ordered


def test_brier_diff_ci_negative_when_model_better():
    # Model perfectly predicts; baseline is 0.5 everywhere.
    c = synthetic_corners(30, [1, 2], seed=1)
    y = c["ends_in_box"].fillna(False).astype(float).to_numpy()
    p_model = y  # perfect
    p_base = np.full(len(c), 0.5)
    diff, _lo, _hi = brier_diff_ci(c, p_model, p_base, n_reps=200, seed=1)
    assert diff < 0  # model Brier - base Brier < 0


def test_decide_team_side_shrunk_simpler_than_model():
    # league_side is not applicable (model clearly beats it),
    # but team_side_shrunk's CI includes 0.
    result = decide(
        0.18,
        {
            "league_side": (0.25, -0.07, -0.09, -0.05),
            "team_side_shrunk": (0.19, -0.01, -0.03, 0.01),
        },
    )
    assert result == "team_side_shrunk"


def test_reliability_table_columns():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    p = rng.uniform(0, 1, 100)
    rt = reliability_table(y, p, n_bins=5)
    assert set(rt.columns) == {"bin", "mean_p", "frac_pos", "n"}
    assert len(rt) == 5


def test_reliability_table_bin_counts_sum_to_total():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(0, 1, 200)
    rt = reliability_table(y, p, n_bins=10)
    assert rt["n"].sum() == 200


def test_reliability_table_frac_pos_in_range():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 100)
    p = rng.uniform(0, 1, 100)
    rt = reliability_table(y, p)
    assert ((rt["frac_pos"] >= 0) & (rt["frac_pos"] <= 1)).all()
