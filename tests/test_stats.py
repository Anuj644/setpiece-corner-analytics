"""Tests for match-level bootstrap helpers (Task 10)."""

import numpy as np
import pandas as pd
import pytest

from setpiece.stats import match_bootstrap, match_bootstrap_ratio, rate


def test_ratio_bootstrap_matches_generic_on_small_case():
    df = pd.DataFrame({"match_id": [1] * 3 + [2] * 1, "y": [1, 1, 1, 0]})
    g = df.groupby("match_id").y
    lo, hi = match_bootstrap_ratio(g.sum().to_numpy(), g.size().to_numpy(), 2000, 1)
    assert (lo, hi) == (0.0, 1.0)
    assert all(np.isnan(match_bootstrap_ratio(np.array([]), np.array([]), 10, 1)))


def test_deterministic_with_seed():
    df = pd.DataFrame({"match_id": [1, 1, 2, 3, 3, 3], "y": [1, 0, 1, 0, 0, 1]})
    f = lambda d: d.y.mean()
    assert match_bootstrap(df, f, 500, 7) == match_bootstrap(df, f, 500, 7)


def test_resamples_whole_matches():
    # match 1: three ones; match 2: one zero -> resample means are 1.0, 0.0 or 0.75
    # (never row-level mixes)
    df = pd.DataFrame({"match_id": [1] * 3 + [2] * 1, "y": [1, 1, 1, 0]})
    lo, hi = match_bootstrap(df, lambda d: d.y.mean(), 2000, 1)
    assert lo == 0.0 and hi == 1.0


def test_empty_and_zero_den():
    assert all(
        np.isnan(
            match_bootstrap(
                pd.DataFrame({"match_id": [], "y": []}),
                lambda d: d.y.mean(),
                10,
                1,
            )
        )
    )
    assert np.isnan(rate(0, 0)) and rate(1, 4) == 0.25


def test_ratio_bootstrap_empty_arrays():
    result = match_bootstrap_ratio(np.array([]), np.array([]), 10, 42)
    assert len(result) == 2
    assert all(np.isnan(r) for r in result)


def test_ratio_bootstrap_single_match():
    # one match, 3 successes out of 4: should return (0.75, 0.75) since only
    # one resample value is possible
    num = np.array([3.0])
    den = np.array([4.0])
    lo, hi = match_bootstrap_ratio(num, den, 1000, 0)
    assert lo == pytest.approx(0.75) and hi == pytest.approx(0.75)


def test_ratio_bootstrap_ignores_zero_denominator_replicates():
    # Two matches; replicates that draw match 2 twice give den=0 and are dropped.
    # Only replicates where match 1 appears contribute.
    num = np.array([1.0, 0.0])
    den = np.array([1.0, 0.0])
    lo, hi = match_bootstrap_ratio(num, den, 2000, 7)
    # all valid replicates must give 1.0 or 0/something; since den[1]=0 is excluded,
    # valid replicates are those that draw match 1 at least once: ratio is 1.0
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_match_bootstrap_group_parameter():
    df = pd.DataFrame(
        {"team_id": [1, 1, 2, 2], "y": [1, 1, 0, 0]}
    )
    lo, hi = match_bootstrap(df, lambda d: d.y.mean(), 500, 0, group="team_id")
    assert lo == 0.0 and hi == 1.0


def test_rate_basic():
    assert rate(0, 5) == 0.0
    assert rate(5, 5) == 1.0
    assert np.isnan(rate(0, 0))
    assert rate(3, 4) == pytest.approx(0.75)
