"""Tests for RQ1 review exports (Task 13)."""

import numpy as np
import pandas as pd
import pytest
from helpers import corner_row, events_df

from setpiece.io import write_table
from setpiece.rq1.corners import build_rq1_corners
from setpiece.rq1.review import (
    run_review_export_rq1,
    sample_borderline,
    short_borderline,
    side_outliers,
    technique_by_height,
)


def ground_passes(lengths):
    """A ``rq1_corners`` frame of one ground-pass corner per length."""
    return events_df(
        [
            corner_row(event_id=f"g{i}", technique=None, pass_length=float(length))
            for i, length in enumerate(lengths)
        ]
    )


@pytest.fixture
def wide_band(cfg):
    """200 ground passes spread over 4-18, 86 of them in the 8-14 review band."""
    return build_rq1_corners(ground_passes(np.linspace(4, 18, 200)), cfg)


def test_borderline_range(rq1_small, cfg):
    rq1 = build_rq1_corners(rq1_small, cfg)
    b = short_borderline(rq1, 8, 14)
    assert b.pass_length.between(8, 14).all() and (b.pass_height == "Ground Pass").all()
    assert {"label_claude", "label_human"} <= set(b.columns)
    assert b.pass_length.is_monotonic_increasing


def test_side_outliers(rq1_small, cfg):
    rq1 = build_rq1_corners(rq1_small, cfg)
    assert set(side_outliers(rq1).event_id) == {"weird-start"}


def test_side_outliers_ignores_corner_flag_starts_and_null(cfg):
    rq1 = build_rq1_corners(
        events_df(
            [
                corner_row(event_id="flag-low", start_y=0.1),
                corner_row(event_id="flag-high", start_y=79.9),
                corner_row(event_id="nothing", start_y=None),
            ]
        ),
        cfg,
    )
    assert side_outliers(rq1).empty


def test_sample_is_seeded_and_capped(rq1_small, cfg):
    rq1 = build_rq1_corners(rq1_small, cfg)
    b = short_borderline(rq1, 8, 14)
    assert len(sample_borderline(b, 60, 0)) == len(b)  # fewer than 60 available: all of them
    assert sample_borderline(b, 1, 0).equals(sample_borderline(b, 1, 0))


@pytest.mark.parametrize("n", [1, 7, 30, 60])
def test_sample_has_exactly_n_rows_when_enough_are_available(wide_band, n):
    b = short_borderline(wide_band, 8, 14)
    assert len(b) > 60
    s = sample_borderline(b, n, 0)
    assert len(s) == n and s.event_id.is_unique


def test_sample_spreads_over_the_whole_length_range(wide_band):
    b = short_borderline(wide_band, 8, 14)
    s = sample_borderline(b, 6, 0)
    # one row per sixth of 8-14: each of the six equal-width bins is represented
    edges = np.linspace(b.pass_length.min(), b.pass_length.max(), 7)
    counts = np.histogram(s.pass_length, bins=edges)[0]
    assert (counts == 1).all()


def test_sample_depends_on_the_seed(wide_band):
    b = short_borderline(wide_band, 8, 14)
    assert not sample_borderline(b, 20, 0).event_id.equals(sample_borderline(b, 20, 1).event_id)
    assert sample_borderline(b, 20, 5).event_id.equals(sample_borderline(b, 20, 5).event_id)


def test_sample_of_identical_lengths_still_returns_n(cfg):
    b = short_borderline(build_rq1_corners(ground_passes([10.0] * 30), cfg), 8, 14)
    assert len(sample_borderline(b, 12, 0)) == 12


def test_technique_by_height(rq1_small, cfg):
    t = technique_by_height(build_rq1_corners(rq1_small, cfg))
    assert {"competition_id", "season_id", "pass_height", "technique", "count"} <= set(t.columns)
    ground = t[t.pass_height == "Ground Pass"]
    assert set(ground.technique) == {"Unknown"} and ground["count"].sum() == 4


def test_technique_by_height_keeps_a_null_height(cfg):
    rows = [corner_row(event_id="a"), corner_row(event_id="b", pass_height=None)]
    t = technique_by_height(build_rq1_corners(events_df(rows), cfg))
    assert t["count"].sum() == 2 and t.pass_height.isna().sum() == 1


def test_export_writes_csvs_and_one_png_per_sampled_corner(cfg_tmp, wide_band):
    write_table(wide_band, "derived/rq1_corners", cfg_tmp, None)
    counts = run_review_export_rq1(cfg_tmp)

    out = cfg_tmp.reports_dir / "review" / "rq1"
    lo, hi = cfg_tmp.p("short_review_range")
    n = cfg_tmp.p("short_review_sample")
    in_band = int(wide_band.pass_length.between(lo, hi).sum())
    assert counts["borderline"] == in_band and counts["sampled"] == min(n, in_band) == n

    sample = pd.read_csv(out / "short_borderline_sample.csv")
    assert len(sample) == n and {"label_claude", "label_human"} <= set(sample.columns)
    assert sample.label_claude.isna().all() and sample.label_human.isna().all()
    assert len(pd.read_csv(out / "short_borderline_all.csv")) == in_band
    for name in ("side_outliers.csv", "technique_by_height.csv"):
        assert (out / name).exists()
    pngs = sorted(p.name for p in out.glob("short_*.png"))
    assert pngs == sorted(f"short_{e}.png" for e in sample.event_id)
    assert all((out / p).stat().st_size > 0 for p in pngs)


def test_export_is_repeatable(cfg_tmp, wide_band):
    write_table(wide_band, "derived/rq1_corners", cfg_tmp, None)
    run_review_export_rq1(cfg_tmp)
    path = cfg_tmp.reports_dir / "review" / "rq1" / "short_borderline_sample.csv"
    first = path.read_text()
    run_review_export_rq1(cfg_tmp)
    assert path.read_text() == first


def test_export_without_the_rq1_table_says_what_to_run(cfg_tmp):
    with pytest.raises(FileNotFoundError, match="python -m setpiece rq1"):
        run_review_export_rq1(cfg_tmp)
