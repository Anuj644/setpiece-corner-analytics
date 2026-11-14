import dataclasses
import json

import pandas as pd
import pytest
from rq2_helpers import corner, ev_, tl

from setpiece.config import Slice
from setpiece.io import read_table, write_table
from setpiece.rq2 import chains
from setpiece.rq2.chains import (
    RQ2_COLUMNS,
    ChainResult,
    Restarts,
    build_rq2_outcomes,
    restarts_from,
)
from setpiece.rq2.diagnostics import opponent_action_within
from setpiece.stages import rq2 as rq2_stage
from setpiece.stages.rq2 import run_rq2

RS = Restarts(["Throw-in", "Goal Kick", "Free Kick", "Kick Off"], ["Free Kick", "Penalty"], ["Referee Ball-Drop"])

SLICES = [Slice(43, 106, "FIFA World Cup 2022", "tournament", True), Slice(9, 281, "1. Bundesliga 2023/24", "league", True, 34)]


def test_one_row_per_corner_no_cross_match(two_match_events):
    out = build_rq2_outcomes(two_match_events, 15, RS)
    corners = two_match_events.query("event_type == 'Pass' and pass_type == 'Corner'")
    assert len(out) == len(corners)
    for _, r in out.iterrows():
        ids = set(two_match_events.query("match_id == @r.match_id").event_id)
        assert set(r.included_event_ids) <= ids

def test_durations_nonnegative(two_match_events):
    assert (build_rq2_outcomes(two_match_events, 15, RS).sequence_duration_seconds.dropna() >= 0).all()


# --- build_rq2_outcomes -------------------------------------------------------------------


EXPECTED_DTYPES = {
    "match_id": "Int64", "corner_event_id": "string", "last_included_event_id": "string",
    "included_event_ids": "object", "n_events": "Int64", "sequence_duration_seconds": "Float64",
    "end_reason": "string", "restart_type": "string", "restart_team": "string", "shot_count": "Int64",
    "first_shot_event_id": "string", "goal": "boolean", "goal_event_id": "string",
    "lost_possession": "boolean", "penalty_awarded": "boolean", "opponent_action_count": "Int64",
    "outcome_status": "string", "competition_id": "Int64", "season_id": "Int64", "team_id": "Int64",
}


def test_columns_are_the_spec_columns_in_order():
    assert RQ2_COLUMNS == [
        "match_id", "corner_event_id", "last_included_event_id", "included_event_ids", "n_events",
        "sequence_duration_seconds", "end_reason", "restart_type", "restart_team", "shot_count",
        "first_shot_event_id", "goal", "goal_event_id", "lost_possession", "penalty_awarded",
        "opponent_action_count", "outcome_status", "competition_id", "season_id", "team_id",
    ]
    assert RQ2_COLUMNS[2:-3] == [f.name for f in dataclasses.fields(ChainResult)]


def test_dtypes(two_match_events):
    out = build_rq2_outcomes(two_match_events, 15, RS)
    assert list(out.columns) == RQ2_COLUMNS
    assert {c: str(t) for c, t in out.dtypes.items()} == EXPECTED_DTYPES
    assert out.index.equals(pd.RangeIndex(4))


def test_rows_follow_the_chain_rule_and_carry_the_corner_row_ids(two_match_events):
    out = build_rq2_outcomes(two_match_events, 15, RS).set_index("corner_event_id")
    assert list(out.index) == ["m1-c1", "m1-c2", "m2-c1", "m2-c2"]
    first = out.loc["m1-c1"]
    assert (first.match_id, first.competition_id, first.season_id, first.team_id) == (1, 43, 106, 10)
    assert list(first.included_event_ids) == ["m1-p1", "m1-s1", "m1-clr"]
    assert (first.last_included_event_id, first.n_events, first.sequence_duration_seconds) == ("m1-clr", 3, 6.0)
    assert (first.end_reason, first.restart_type, first.restart_team) == ("stoppage", "Goal Kick", "defending")
    assert (first.shot_count, first.first_shot_event_id, first.goal) == (1, "m1-s1", False)
    assert pd.isna(first.goal_event_id)
    assert (first.lost_possession, first.penalty_awarded, first.opponent_action_count) == (False, False, 1)
    assert first.outcome_status == "complete"

    repeat = out.loc["m1-c2"]  # the last events of match 1: the walk must not run into match 2
    assert (repeat.team_id, list(repeat.included_event_ids)) == (20, ["m1-p2"])
    assert (repeat.end_reason, repeat.outcome_status) == ("timeline_end", "incomplete")
    assert pd.isna(repeat.restart_type)

    scored = out.loc["m2-c1"]
    assert (scored.match_id, scored.competition_id, scored.season_id, scored.team_id) == (2, 9, 281, 30)
    assert list(scored.included_event_ids) == ["m2-opp", "m2-goal"]
    assert (scored.end_reason, scored.goal, scored.goal_event_id, scored.shot_count) == ("goal", True, "m2-goal", 1)
    assert (scored.opponent_action_count, scored.sequence_duration_seconds) == (1, 6.0)

    lost = out.loc["m2-c2"]
    assert (lost.team_id, list(lost.included_event_ids), lost.last_included_event_id) == (40, [], "m2-c2")
    assert (lost.end_reason, lost.lost_possession, lost.n_events, lost.sequence_duration_seconds) == ("possession_change", True, 0, 0.0)


def test_nulls_stay_null(two_match_events):
    out = build_rq2_outcomes(two_match_events, 15, RS)
    assert out.restart_type.isna().sum() == 3 and out.first_shot_event_id.isna().sum() == 2
    assert out.goal_event_id.isna().sum() == 3 and out.restart_team.isna().sum() == 3


def test_cap_is_passed_to_every_chain():
    ev = tl(corner(t=0, id="c"), ev_(t=12, type="Pass"), ev_(t=18, type="Shot", shot_outcome="Saved"), ev_(t=30, type="Half End"))
    by_cap = {cap: build_rq2_outcomes(ev, cap, RS).iloc[0] for cap in (10, 15, 20, None)}
    assert {cap: (r.shot_count, r.end_reason) for cap, r in by_cap.items()} == {
        10: (0, "time_cap"), 15: (0, "time_cap"), 20: (1, "period_end"), None: (1, "period_end")}


def test_input_row_order_and_index_do_not_matter(two_match_events):
    expected = build_rq2_outcomes(two_match_events, 15, RS)
    shuffled = two_match_events.sample(frac=1, random_state=3)
    assert list(shuffled.index) != list(two_match_events.index)
    pd.testing.assert_frame_equal(build_rq2_outcomes(shuffled, 15, RS), expected)


def test_events_are_not_modified(two_match_events):
    before = two_match_events.copy()
    build_rq2_outcomes(two_match_events.sample(frac=1, random_state=1), 15, RS)
    build_rq2_outcomes(two_match_events, 15, RS)
    pd.testing.assert_frame_equal(two_match_events, before)


def test_no_corners_gives_an_empty_frame_with_the_columns(two_match_events):
    for events in (two_match_events[two_match_events.pass_type != "Corner"], two_match_events.iloc[:0]):
        out = build_rq2_outcomes(events, 15, RS)
        assert len(out) == 0 and list(out.columns) == RQ2_COLUMNS
        assert {c: str(t) for c, t in out.dtypes.items()} == EXPECTED_DTYPES


def test_corner_with_null_ids_keeps_the_nulls():
    ev = tl(corner(t=0, id="c", team=None, poss=None), ev_(t=2, type="Half End")).assign(
        match_id=pd.array([None, None], dtype="Int64"))
    r = build_rq2_outcomes(ev, 15, RS).iloc[0]
    assert pd.isna(r.match_id) and pd.isna(r.team_id) and pd.isna(r.competition_id) and pd.isna(r.season_id)
    assert (r.corner_event_id, r.end_reason, r.outcome_status) == ("c", "period_end", "incomplete")


def test_new_corner_by_the_other_team_is_a_defending_restart():
    ev = tl(corner(t=0), corner(t=20, team=20, poss=6, poss_team=20, id="c2"))
    out = build_rq2_outcomes(ev, 15, RS)
    assert list(out.end_reason) == ["new_corner", "timeline_end"]
    assert (out.restart_type[0], out.restart_team[0]) == ("Corner", "defending")


# --- restarts_from ------------------------------------------------------------------------


def test_restarts_from_reads_the_config(cfg):
    restarts = restarts_from(cfg)
    assert isinstance(restarts, Restarts) and restarts == Restarts.from_config(cfg) == RS


def test_restarts_from_is_defined_in_chains():
    assert chains.restarts_from is restarts_from


# --- opponent_action_within ---------------------------------------------------------------


def test_opponent_action_per_corner_in_match_order(two_match_events):
    flags = opponent_action_within(two_match_events, 15)
    assert list(flags.index) == ["m1-c1", "m1-c2", "m2-c1", "m2-c2"]
    assert flags.dtype == bool and list(flags) == [True, False, True, False]
    assert list(opponent_action_within(two_match_events.sample(frac=1, random_state=5), 15)) == list(flags)


def opp(*events):
    """The flags for a one-match timeline that starts with a corner by team 10."""
    return list(opponent_action_within(tl(corner(t=100, id="c"), *events), 15))


def test_opponent_window_is_inclusive_and_never_looks_back():
    assert opp(ev_(t=115, type="Pass", team=20)) == [True]
    assert opp(ev_(t=115.001, type="Pass", team=20)) == [False]
    assert opp(ev_(t=100, type="Pass", team=20)) == [True]
    assert opp(ev_(t=99, type="Pass", team=20)) == [False]  # stamped before the corner: not clamped to 0


def test_opponent_action_needs_the_other_team_in_the_same_possession_and_period():
    assert opp(ev_(t=102, type="Pass", team=10)) == [False]
    assert opp(ev_(t=102, type="Pass", team=20, poss=6, poss_team=20)) == [False]
    assert opp(ev_(t=102, type="Pass", team=20, period=2)) == [False]
    assert opp(ev_(t=102, type="Pass", team=None)) == [False]
    assert opp(ev_(t=102, type="Pass", team=20, poss=None)) == [False]
    assert opp(ev_(t=None, type="Pass", team=20)) == [False]
    assert opp(ev_(t=102, type="Pass", team=20)) == [True]


def test_opponent_action_ignores_stoppages_and_event_types_in_between():
    assert opp(ev_(t=101, type="Clearance", team=10, out=True), ev_(t=110, type="Ball Receipt*", team=20)) == [True]
    assert opp(ev_(t=101, type="Foul Committed", team=10), ev_(t=105, type="Duel", team=20)) == [True]


def test_opponent_action_does_not_leave_the_match_or_look_at_earlier_events():
    before = tl(ev_(t=95, type="Pass", team=20), corner(t=100, id="c"))
    assert list(opponent_action_within(before, 15)) == [False]
    after = tl(corner(t=100, id="d"), ev_(t=101, type="Pass", team=20)).assign(match_id=pd.array([2, 2], dtype="Int64"))
    two = pd.concat([before.assign(event_id=["a0", "a1"]), after])
    assert list(opponent_action_within(two, 15)) == [False, True]


def test_opponent_action_seconds_parameter_and_no_corners(two_match_events):
    assert int(opponent_action_within(two_match_events, 2).sum()) == 0
    assert int(opponent_action_within(two_match_events, 3).sum()) == 1
    assert int(opponent_action_within(two_match_events, 300).sum()) == 2
    empty = opponent_action_within(two_match_events.iloc[:0], 15)
    assert len(empty) == 0 and empty.dtype == bool


# --- run_rq2 ------------------------------------------------------------------------------


@pytest.fixture
def rq2_cfg(cfg_tmp, two_match_events):
    """``cfg_tmp`` with the two fixture slices configured and their events written as partitions."""
    cfg = dataclasses.replace(cfg_tmp, slices=list(SLICES))
    for (comp, season), part in two_match_events.groupby(["competition_id", "season_id"]):
        write_table(part.reset_index(drop=True), "events", cfg, (int(comp), int(season)))
    return cfg


def test_run_rq2_writes_the_main_table_and_one_per_cap_sensitivity_entry(rq2_cfg):
    run_rq2(rq2_cfg)
    derived = rq2_cfg.processed_dir / "derived"
    assert sorted(p.name for p in derived.iterdir()) == [
        "rq2_outcomes.parquet", "rq2_outcomes_cap10.parquet", "rq2_outcomes_cap20.parquet",
        "rq2_outcomes_nocap.parquet"]
    out = read_table("derived/rq2_outcomes", rq2_cfg)
    assert list(out.columns) == RQ2_COLUMNS and len(out) == 4
    assert {c: str(t) for c, t in out.dtypes.items()} == EXPECTED_DTYPES
    assert list(out.corner_event_id) == ["m1-c1", "m1-c2", "m2-c1", "m2-c2"]
    assert list(out.competition_id) == [43, 43, 9, 9]


def test_run_rq2_reads_one_configured_slice_at_a_time(rq2_cfg, monkeypatch):
    seen = []
    real = rq2_stage.read_table

    def spy(table, cfg, slices=None):
        seen.append((table, slices))
        return real(table, cfg, slices)

    monkeypatch.setattr(rq2_stage, "read_table", spy)
    run_rq2(rq2_cfg)
    assert seen == [("events", [(43, 106)]), ("events", [(9, 281)])]


def test_run_rq2_variants_use_their_own_caps(cfg_tmp):
    one = tl(corner(t=0, id="c"), ev_(t=12, type="Pass"), ev_(t=18, type="Shot", shot_outcome="Saved"), ev_(t=30, type="Half End"))
    cfg = dataclasses.replace(cfg_tmp, slices=SLICES[:1])
    write_table(one, "events", cfg, (43, 106))
    counts = run_rq2(cfg)
    assert {name: c["shots"] for name, c in counts.items()} == {
        "rq2_outcomes": 0, "rq2_outcomes_cap10": 0, "rq2_outcomes_cap20": 1, "rq2_outcomes_nocap": 1}
    for name, shots in [("rq2_outcomes_cap10", 0), ("rq2_outcomes_cap20", 1), ("rq2_outcomes_nocap", 1)]:
        assert int(read_table(f"derived/{name}", cfg).shot_count.sum()) == shots


def test_run_rq2_counts_and_manifest(rq2_cfg):
    counts = run_rq2(rq2_cfg)
    assert list(counts) == ["rq2_outcomes", "rq2_outcomes_cap10", "rq2_outcomes_cap20", "rq2_outcomes_nocap"]
    assert counts["rq2_outcomes"] == {"corners": 4, "complete": 3, "shots": 2, "goals": 1}
    assert all(type(v) is int for c in counts.values() for v in c.values())
    manifest = json.loads((rq2_cfg.processed_dir / "manifests" / "rq2.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "rq2" and manifest["counts"] == counts


def test_run_rq2_rebuilds_whole_from_the_configured_slices(rq2_cfg):
    run_rq2(rq2_cfg)
    run_rq2(dataclasses.replace(rq2_cfg, slices=SLICES[1:]))  # the other partition stays on disk
    out = read_table("derived/rq2_outcomes_nocap", rq2_cfg)
    assert list(out.corner_event_id) == ["m2-c1", "m2-c2"]


def test_run_rq2_cap_sensitivity_comes_from_the_config(rq2_cfg):
    sensitivity = dataclasses.replace(rq2_cfg.params["cap_sensitivity"], value=[7.5, None])
    counts = run_rq2(dataclasses.replace(rq2_cfg, params={**rq2_cfg.params, "cap_sensitivity": sensitivity}))
    assert list(counts) == ["rq2_outcomes", "rq2_outcomes_cap7.5", "rq2_outcomes_nocap"]
    assert (rq2_cfg.processed_dir / "derived" / "rq2_outcomes_cap7.5.parquet").is_file()


def test_run_rq2_without_slices_raises(rq2_cfg):
    with pytest.raises(ValueError, match="no slices"):
        run_rq2(dataclasses.replace(rq2_cfg, slices=[]))


def test_run_rq2_slice_without_corners_still_writes_a_table(cfg_tmp, two_match_events):
    cfg = dataclasses.replace(cfg_tmp, slices=SLICES[:1])
    quiet = two_match_events[two_match_events.pass_type != "Corner"]
    write_table(quiet[quiet.competition_id == 43].reset_index(drop=True), "events", cfg, (43, 106))
    counts = run_rq2(cfg)
    assert counts["rq2_outcomes"] == {"corners": 0, "complete": 0, "shots": 0, "goals": 0}
    out = read_table("derived/rq2_outcomes", cfg)
    assert len(out) == 0 and list(out.columns) == RQ2_COLUMNS
