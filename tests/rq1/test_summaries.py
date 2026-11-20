"""Tests for RQ1 team summaries (Task 14)."""

import pandas as pd
import pytest

from setpiece.rq1.corners import RQ1_CORNERS_COLUMNS, build_rq1_corners
from setpiece.rq1.summaries import summarise_rq1


def test_pct_sums_and_unknown_kept(rq1_two_teams, samples, cfg):
    rq1 = build_rq1_corners(rq1_two_teams, cfg)
    s = summarise_rq1(rq1, samples, cfg)
    g = s[(s.team_id == 1) & (s.attribute == "technique")]
    assert g.pct.sum() == pytest.approx(100.0)
    assert "Unknown" in set(g.category)
    assert (s.n <= s.denominator).all()
    assert (g.denominator == len(rq1.query("team_id == 1"))).all()


def test_ci_is_ordered_and_in_range(rq1_two_teams, samples, cfg):
    rq1 = build_rq1_corners(rq1_two_teams, cfg)
    s = summarise_rq1(rq1, samples, cfg)
    ok = s.dropna(subset=["ci_lo"])
    assert ((0 <= ok.ci_lo) & (ok.ci_lo <= ok.ci_hi) & (ok.ci_hi <= 100)).all()


def test_zero_denominator_rows(samples, cfg):
    empty = pd.DataFrame(columns=RQ1_CORNERS_COLUMNS)
    s = summarise_rq1(empty, samples, cfg)
    assert (s.denominator == 0).all() and s.pct.isna().all()


def test_team_with_no_corners_gets_none_category(rq1_two_teams, samples, cfg):
    rq1 = build_rq1_corners(rq1_two_teams, cfg)
    s = summarise_rq1(rq1, samples, cfg)
    # Team 3 has no corners in rq1_two_teams.
    g = s[s.team_id == 3]
    assert (g.category == "none").all()
    assert (g.n == 0).all()
    assert (g.denominator == 0).all()
