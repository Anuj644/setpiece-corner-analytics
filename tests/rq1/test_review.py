"""Tests for RQ1 review exports (Task 13)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from helpers import corner_row, events_df

from setpiece.config import Param, Slice
from setpiece.io import write_table
from setpiece.rq1.corners import build_rq1_corners
from setpiece.rq1.review import (
    blind_review_sheet,
    core_only,
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


def run_export(cfg, rq1):
    write_table(rq1, "derived/rq1_corners", cfg, None)
    counts = run_review_export_rq1(cfg)
    return counts, cfg.reports_dir / "review" / "rq1", cfg.reports_dir / "review_key" / "rq1"


def test_export_writes_a_blind_sheet_and_one_png_per_sampled_corner(cfg_tmp, wide_band):
    counts, out, _ = run_export(cfg_tmp, wide_band)
    lo, hi = cfg_tmp.p("short_review_range")
    n = cfg_tmp.p("short_review_sample")
    in_band = int(wide_band.pass_length.between(lo, hi).sum())
    assert counts["borderline"] == in_band and counts["sampled"] == min(n, in_band) == n

    sheet = pd.read_csv(out / "short_borderline_sample.csv")
    assert list(sheet.columns) == ["event_id", "png", "label_claude", "label_human"]
    assert len(sheet) == n and sheet.event_id.is_unique
    assert sheet.label_claude.isna().all() and sheet.label_human.isna().all()
    assert sorted(p.name for p in out.glob("short_*.png")) == sorted(sheet.png)
    assert all((out / p).stat().st_size > 0 for p in sheet.png)
    for name in ("side_outliers.csv", "technique_by_height.csv"):
        assert (out / name).exists()


def test_nothing_in_the_labellers_folder_shows_the_length_or_the_rule(cfg_tmp, wide_band):
    """A labeller handed the folder must decide from the pictures: no per-corner length or proxy."""
    # a ground pass of band length that starts away from the flag, so side_outliers.csv lists it
    wide_band.loc[wide_band.index[0], ["start_y", "pass_length"]] = [40.0, 10.0]
    _, out, _ = run_export(cfg_tmp, wide_band)
    assert (out / "side_outliers.csv").read_text().count("\n") > 1  # it does list a corner
    for csv in out.glob("*.csv"):
        header = csv.read_text().splitlines()[0]
        for revealing in ("pass_length", "short_corner_proxy", "pass_end_x", "pass_end_y"):
            assert revealing not in header, f"{csv.name} shows {revealing}"


def test_the_key_file_is_outside_the_review_folder_and_in_the_sheets_order(cfg_tmp, wide_band):
    _, out, key_dir = run_export(cfg_tmp, wide_band)
    assert key_dir != out and key_dir not in out.parents and out not in key_dir.parents
    sheet = pd.read_csv(out / "short_borderline_sample.csv")
    key = pd.read_csv(key_dir / "short_borderline_key.csv")
    assert key.event_id.tolist() == sheet.event_id.tolist()
    assert {"pass_length", "short_corner_proxy"} <= set(key.columns)
    assert key.pass_length.between(*cfg_tmp.p("short_review_range")).all()
    everything = pd.read_csv(key_dir / "short_borderline_all.csv")
    assert everything.pass_length.is_monotonic_increasing and set(key.event_id) <= set(everything.event_id)


def test_the_sheet_is_in_a_shuffled_order_not_the_length_order(cfg_tmp, wide_band):
    _, _, key_dir = run_export(cfg_tmp, wide_band)
    lengths = pd.read_csv(key_dir / "short_borderline_key.csv").pass_length
    assert not lengths.is_monotonic_increasing and not lengths.is_monotonic_decreasing


def test_blind_sheet_order_comes_from_the_seed(wide_band, cfg):
    sample = short_borderline(build_rq1_corners(ground_passes(np.linspace(8, 14, 40)), cfg), 8, 14)
    a, _ = blind_review_sheet(sample, seed=3)
    b, _ = blind_review_sheet(sample, seed=3)
    c, shuffled = blind_review_sheet(sample, seed=4)
    assert a.equals(b) and not a.event_id.equals(c.event_id)
    assert sorted(a.event_id) == sorted(sample.event_id)
    assert shuffled.event_id.tolist() == c.event_id.tolist()


def test_export_is_repeatable(cfg_tmp, wide_band):
    _, out, key_dir = run_export(cfg_tmp, wide_band)
    paths = [out / "short_borderline_sample.csv", key_dir / "short_borderline_key.csv"]
    first = [p.read_text() for p in paths]
    run_review_export_rq1(cfg_tmp)
    assert [p.read_text() for p in paths] == first


def test_old_pngs_and_the_old_key_file_are_deleted_before_the_new_ones_are_written(
    cfg_tmp, wide_band
):
    _, out, _ = run_export(cfg_tmp, wide_band)
    (out / "short_from-an-earlier-sample.png").write_bytes(b"png")
    (out / "short_borderline_all.csv").write_text("pass_length\n9.9\n")  # where it used to live
    (out / "my_notes.txt").write_text("keep")
    run_review_export_rq1(cfg_tmp)

    sheet = pd.read_csv(out / "short_borderline_sample.csv")
    assert sorted(p.name for p in out.glob("short_*.png")) == sorted(sheet.png)
    assert not (out / "short_borderline_all.csv").exists()
    assert (out / "my_notes.txt").read_text() == "keep"  # only the export's own files are removed


def test_a_smaller_sample_leaves_exactly_that_many_pngs(cfg_tmp, wide_band):
    run_export(cfg_tmp, wide_band)
    param = cfg_tmp.params["short_review_sample"]
    smaller = dataclasses.replace(
        cfg_tmp, params={**cfg_tmp.params, "short_review_sample": Param(10, param.status, "")}
    )
    counts = run_review_export_rq1(smaller)
    assert counts["sampled"] == 10
    assert len(list((cfg_tmp.reports_dir / "review" / "rq1").glob("short_*.png"))) == 10


def non_core_rows(cfg):
    """50 ground passes of band length in a slice the config marks as not core (an expansion)."""
    rows = [
        corner_row(
            event_id=f"x{i}", technique=None, pass_length=9.0 + i / 10,
            competition_id=12, season_id=27,
        )
        for i in range(50)
    ]  # fmt: skip
    return build_rq1_corners(events_df(rows), cfg)


@pytest.fixture
def cfg_with_expansion(cfg_tmp):
    return dataclasses.replace(
        cfg_tmp, slices=[*cfg_tmp.slices, Slice(12, 27, "Serie A 2015/16", "league", False, core=False)]
    )


def test_the_sample_pool_is_the_core_slices_only(cfg_with_expansion, wide_band, cfg):
    both = pd.concat([wide_band, non_core_rows(cfg)], ignore_index=True)
    counts, out, _ = run_export(cfg_with_expansion, both)
    sheet = pd.read_csv(out / "short_borderline_sample.csv")
    assert not sheet.event_id.str.startswith("x").any()
    assert counts["borderline"] == int(wide_band.pass_length.between(8, 14).sum())  # no x rows


def test_expanding_the_corpus_does_not_change_the_sample(cfg_tmp, cfg_with_expansion, wide_band, cfg):
    """After Task 9b the 60 corners labelled at G2 must still be the same 60."""
    _, out, _ = run_export(cfg_tmp, wide_band)
    before = (out / "short_borderline_sample.csv").read_text()
    both = pd.concat([wide_band, non_core_rows(cfg)], ignore_index=True)
    write_table(both, "derived/rq1_corners", cfg_with_expansion, None)
    run_review_export_rq1(cfg_with_expansion)
    assert (out / "short_borderline_sample.csv").read_text() == before


def test_core_only_keeps_core_rows_in_order(cfg_with_expansion, wide_band, cfg):
    both = pd.concat([non_core_rows(cfg), wide_band], ignore_index=True)
    kept = core_only(both, cfg_with_expansion)
    assert kept.event_id.tolist() == wide_band.event_id.tolist()


def test_export_without_the_rq1_table_says_what_to_run(cfg_tmp):
    with pytest.raises(FileNotFoundError, match="python -m setpiece rq1"):
        run_review_export_rq1(cfg_tmp)