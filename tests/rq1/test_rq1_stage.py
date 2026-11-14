"""Tests for the ``rq1`` stage: it must write every output and read each one back (Task 15)."""

import json

import pandas as pd

from setpiece.io import read_table
from setpiece.rq1.corners import RQ1_CORNERS_COLUMNS
from setpiece.rq1.summaries import ATTRIBUTES, SUMMARY_COLUMNS
from setpiece.stages.rq1 import run_rq1


def test_stage_writes_every_output_and_reads_it_back(cfg_tmp_with_events):
    cfg = cfg_tmp_with_events
    counts = run_rq1(cfg)

    corners = read_table("derived/rq1_corners", cfg)
    summary = read_table("derived/rq1_summary", cfg)
    samples = read_table("derived/team_samples", cfg)

    assert list(corners.columns) == RQ1_CORNERS_COLUMNS and len(corners) == 24
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert set(summary.attribute) == set(ATTRIBUTES)
    assert {1, 2} <= set(samples.team_id)
    assert counts["rq1_corners"] == 24
    manifest = json.loads((cfg.processed_dir / "manifests" / "rq1.json").read_text())
    assert manifest["stage"] == "rq1" and manifest["counts"]["rq1_corners"] == 24


def test_summary_category_is_text_whatever_the_attribute(cfg_tmp_with_events):
    """The crash: bool categories of ``short_corner_proxy`` next to str ones broke Arrow."""
    run_rq1(cfg_tmp_with_events)
    summary = read_table("derived/rq1_summary", cfg_tmp_with_events)
    assert all(isinstance(c, str) for c in summary.category)
    proxy = summary[summary.attribute == "short_corner_proxy"]
    assert set(proxy.category) <= {"True", "False", "Unknown"}


def test_stage_draws_every_attribute_for_every_team_with_corners(cfg_tmp_with_events):
    cfg = cfg_tmp_with_events
    run_rq1(cfg)
    fig_dir = cfg.reports_dir / "figures" / "rq1"
    figures = {p.name for p in fig_dir.glob("*.svg")}
    for team_id in (1, 2):
        assert f"9_281_{team_id}_endpoints.svg" in figures
        for attribute in ATTRIBUTES:
            assert f"9_281_{team_id}_{attribute}.svg" in figures


def test_every_figure_title_carries_team_slice_n_and_dates(cfg_tmp_with_events):
    """Spec §6.3: no figure without its sample size."""
    cfg = cfg_tmp_with_events
    run_rq1(cfg)
    fig_dir = cfg.reports_dir / "figures" / "rq1"
    for name in ("9_281_1_endpoints.svg", "9_281_1_technique.svg", "9_281_1_fine_zone.svg"):
        svg = (fig_dir / name).read_text(encoding="utf-8")
        for part in ("Team A", "1. Bundesliga 2023/24", "n=12 corners", "2023-10-01"):
            assert part in svg, f"{part!r} missing from {name}"


def test_stage_without_any_event_partition_still_writes_typed_tables(cfg_tmp, rq1_two_teams):
    """A slice with no events gives empty tables with their columns, never zero-column ones."""
    from setpiece.io import write_table

    write_table(rq1_two_teams.iloc[0:0], "events", cfg_tmp, (9, 281))
    run_rq1(cfg_tmp)
    corners = read_table("derived/rq1_corners", cfg_tmp)
    summary = read_table("derived/rq1_summary", cfg_tmp)
    assert corners.empty and list(corners.columns) == RQ1_CORNERS_COLUMNS
    assert summary.empty and list(summary.columns) == SUMMARY_COLUMNS
    assert isinstance(corners, pd.DataFrame)
