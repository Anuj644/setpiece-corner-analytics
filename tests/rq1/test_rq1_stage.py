"""Tests for the ``rq1`` stage: it must write every output and read each one back (Task 15)."""

import dataclasses
import gc
import json
import weakref

import pytest
from helpers import events_df

from setpiece import io
from setpiece.config import Slice
from setpiece.io import read_table, write_table
from setpiece.rq1.corners import RQ1_CORNERS_COLUMNS, RQ1_CORNERS_DTYPES
from setpiece.rq1.summaries import ATTRIBUTES, SUMMARY_COLUMNS, SUMMARY_DTYPES
from setpiece.stages import rq1 as rq1_stage
from setpiece.stages.rq1 import run_rq1


def fig_dir(cfg):
    return cfg.reports_dir / "figures" / "rq1"


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
    assert manifest["stage"] == "rq1" and manifest["counts"] == counts


def test_manifest_names_what_it_counts_and_records_the_events_read(cfg_tmp_with_events):
    counts = run_rq1(cfg_tmp_with_events)
    events_rows = len(read_table("events", cfg_tmp_with_events))
    assert counts["events"] == events_rows == 24
    assert counts["team_seasons"] == 2 and counts["teams_with_corners"] == 2
    assert counts["slices"] == {"9/281": {"events": 24, "corners": 24, "teams_with_corners": 2}}
    assert "teams" not in counts  # the old name could mean either of the two counts above


def test_a_team_that_played_but_took_no_corner_is_a_team_season_not_a_team_with_corners(
    cfg_tmp_with_events, stage_events
):
    only_team_1 = stage_events[stage_events.team_id == 1]  # team 2 still plays as the away side
    write_table(only_team_1, "events", cfg_tmp_with_events, (9, 281))
    counts = run_rq1(cfg_tmp_with_events)
    assert counts["team_seasons"] == 2 and counts["teams_with_corners"] == 1


def test_summary_category_is_text_whatever_the_attribute(cfg_tmp_with_events):
    """The crash: bool categories of ``short_corner_proxy`` next to str ones broke Arrow."""
    run_rq1(cfg_tmp_with_events)
    summary = read_table("derived/rq1_summary", cfg_tmp_with_events)
    assert all(isinstance(c, str) for c in summary.category)
    proxy = summary[(summary.attribute == "short_corner_proxy") & (summary.team_id == 1)]
    # per team: 3 short (True), 6 long ground or high (False), 3 without a height (Unknown)
    assert proxy.set_index("category").n.to_dict() == {"True": 3, "False": 6, "Unknown": 3}
    assert (proxy.denominator == 12).all()


def test_stage_draws_every_attribute_for_every_team_with_corners(cfg_tmp_with_events):
    cfg = cfg_tmp_with_events
    run_rq1(cfg)
    figures = {p.name for p in fig_dir(cfg).glob("*.svg")}
    for team_id in (1, 2):
        assert f"9_281_{team_id}_endpoints.svg" in figures
        for attribute in ATTRIBUTES:
            assert f"9_281_{team_id}_{attribute}.svg" in figures


def test_every_figure_title_carries_team_slice_n_and_the_date_range(cfg_tmp_with_events):
    """Spec §6.3: no figure without its sample size and dates (first and last, not in id order)."""
    cfg = cfg_tmp_with_events
    run_rq1(cfg)
    for name in ("9_281_1_endpoints.svg", "9_281_1_technique.svg", "9_281_1_fine_zone.svg"):
        svg = (fig_dir(cfg) / name).read_text(encoding="utf-8")
        for part in ("Team A", "1. Bundesliga 2023/24", "n=12 corners"):
            assert part in svg, f"{part!r} missing from {name}"
        assert 0 <= svg.index("2023-09-01") < svg.index("2023-10-01")  # first date, then last


def test_stage_with_an_empty_events_partition_writes_typed_empty_tables(
    cfg_tmp_with_events, stage_events
):
    """An empty partition is legitimate (a slice with no events): the tables keep columns and dtypes."""
    cfg = cfg_tmp_with_events
    write_table(stage_events.iloc[0:0], "events", cfg, (9, 281))
    counts = run_rq1(cfg)
    corners = read_table("derived/rq1_corners", cfg)
    summary = read_table("derived/rq1_summary", cfg)
    assert corners.empty and list(corners.columns) == RQ1_CORNERS_COLUMNS
    assert summary.empty and list(summary.columns) == SUMMARY_COLUMNS
    assert {c: str(d) for c, d in corners.dtypes.items()} == RQ1_CORNERS_DTYPES
    assert {c: str(d) for c, d in summary.dtypes.items()} == SUMMARY_DTYPES
    assert counts["slices"]["9/281"] == {"events": 0, "corners": 0, "teams_with_corners": 0}


def test_a_configured_slice_without_an_events_partition_stops_the_stage(cfg_tmp_with_events):
    """Ingest stopped early: do not write zero corners for the slices it never reached."""
    cfg = dataclasses.replace(
        cfg_tmp_with_events,
        slices=[*cfg_tmp_with_events.slices, Slice(11, 90, "La Liga 2020/21", "league", True)],
    )
    with pytest.raises(FileNotFoundError, match=r"11, 90.*python -m setpiece ingest"):
        run_rq1(cfg)
    assert not (cfg.processed_dir / "derived").exists()  # nothing was written
    assert not (cfg.processed_dir / "manifests" / "rq1.json").exists()


def test_without_any_events_table_the_stage_says_to_run_ingest(cfg_tmp):
    with pytest.raises(FileNotFoundError, match="python -m setpiece ingest"):
        run_rq1(dataclasses.replace(cfg_tmp, slices=cfg_tmp.slices[:1]))


def test_the_stage_deletes_the_figures_of_an_earlier_run(cfg_tmp_with_events):
    cfg = cfg_tmp_with_events
    fig_dir(cfg).mkdir(parents=True)
    stale = fig_dir(cfg) / "99_99_7_endpoints.svg"  # a team-season of a slice that left the config
    stale.write_text("<svg/>")
    run_rq1(cfg)
    assert not stale.exists()
    assert (fig_dir(cfg) / "9_281_1_endpoints.svg").exists()


def test_a_failed_run_keeps_the_figures_it_did_not_replace(cfg_tmp_with_events):
    cfg = dataclasses.replace(
        cfg_tmp_with_events,
        slices=[*cfg_tmp_with_events.slices, Slice(11, 90, "La Liga 2020/21", "league", True)],
    )
    fig_dir(cfg).mkdir(parents=True)
    old = fig_dir(cfg) / "9_281_1_endpoints.svg"
    old.write_text("<svg/>")
    with pytest.raises(FileNotFoundError):
        run_rq1(cfg)
    assert old.exists()


def test_the_previous_slice_is_freed_before_the_next_one_is_read(cfg_tmp_with_events, monkeypatch):
    """Peak memory is one slice of events, not two: the first frame is gone when the second loads."""
    cfg = dataclasses.replace(
        cfg_tmp_with_events,
        slices=[*cfg_tmp_with_events.slices, Slice(11, 90, "La Liga 2020/21", "league", True)],
    )
    write_table(events_df([]), "events", cfg, (11, 90))
    seen = []

    def spying_read(table, cfg_, slices=None):
        gc.collect()
        spying_read.alive_at_call.append([ref() is not None for ref in seen])
        frame = io.read_table(table, cfg_, slices)
        seen.append(weakref.ref(frame))
        return frame

    spying_read.alive_at_call = []
    monkeypatch.setattr(rq1_stage, "read_table", spying_read)
    run_rq1(cfg)
    assert spying_read.alive_at_call == [[], [False]]
