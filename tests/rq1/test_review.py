"""Tests for RQ1 review exports (Task 13)."""


from setpiece.rq1.corners import build_rq1_corners
from setpiece.rq1.review import (
    sample_borderline,
    short_borderline,
    side_outliers,
    technique_by_height,
)


def test_borderline_range(rq1_small, cfg):
    rq1 = build_rq1_corners(rq1_small, cfg)
    b = short_borderline(rq1, 8, 14)
    assert b.pass_length.between(8, 14).all() and (b.pass_height == "Ground Pass").all()
    assert {"label_claude", "label_human"} <= set(b.columns)


def test_side_outliers(rq1_small, cfg):
    rq1 = build_rq1_corners(rq1_small, cfg)
    assert set(side_outliers(rq1).event_id) == {"weird-start"}


def test_sample_is_seeded_and_capped(rq1_small, cfg):
    rq1 = build_rq1_corners(rq1_small, cfg)
    b = short_borderline(rq1, 8, 14)
    # Fewer than 60 available → return all.
    assert len(sample_borderline(b, 60, 0)) == len(b)
    # Deterministic with same seed.
    assert sample_borderline(b, 1, 0).equals(sample_borderline(b, 1, 0))


def test_technique_by_height(rq1_small, cfg):
    rq1 = build_rq1_corners(rq1_small, cfg)
    t = technique_by_height(rq1)
    assert "count" in t.columns
    assert "pass_height" in t.columns
    assert "technique" in t.columns
