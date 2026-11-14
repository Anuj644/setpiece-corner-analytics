import dataclasses

import pytest
from rq2_helpers import corner, ev_, tl

from setpiece.config import Param
from setpiece.rq2.chains import ChainResult, Restarts, build_chain, parse_timestamp

CAP = 15
RS = Restarts(["Throw-in", "Goal Kick", "Free Kick", "Kick Off"], ["Free Kick", "Penalty"], ["Referee Ball-Drop"])

def test_period_end():
    ev = tl(corner(t=0), ev_(t=2, type="Clearance", team=20), ev_(t=3, type="Half End"))
    r = build_chain(ev, 0, CAP, RS)
    assert r.end_reason == "period_end" and r.n_events == 1 and r.outcome_status == "complete"

def test_new_corner_excluded():
    r = build_chain(tl(corner(t=0), ev_(t=3, type="Clearance", team=20), corner(t=30, id="c2")), 0, CAP, RS)
    assert r.end_reason == "new_corner" and "c2" not in r.included_event_ids

def test_possession_change_lost_and_not_lost():
    lost = tl(corner(t=0, poss=5), ev_(t=4, type="Pass", team=20, poss=6, poss_team=20))
    kept = tl(corner(t=0, poss=5), ev_(t=4, type="Pass", team=10, poss=6, poss_team=10))
    assert build_chain(lost, 0, CAP, RS).lost_possession is True
    assert build_chain(kept, 0, CAP, RS).lost_possession is False

def test_restart_excluded():
    r = build_chain(tl(corner(t=0), ev_(t=5, type="Pass", pass_type="Throw-in", team=10)), 0, CAP, RS)
    assert r.end_reason == "restart" and r.n_events == 0

def test_goal_kick_is_restart_not_lost_possession():
    # real data: every restart also starts a new possession, so the restart check must win
    ev = tl(corner(t=0, poss=5), ev_(t=6, type="Pass", pass_type="Goal Kick", team=20, poss=6, poss_team=20))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.restart_type, r.restart_team, r.lost_possession) == ("restart", "Goal Kick", "defending", False)

def test_attacking_throw_in_is_restart_for_attacking_team():
    ev = tl(corner(t=0, poss=5), ev_(t=6, type="Pass", pass_type="Throw-in", team=10, poss=6, poss_team=10))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.restart_team) == ("restart", "attacking")

def test_free_kick_shot_and_ball_drop_are_restarts():
    a = tl(corner(t=0, poss=5), ev_(t=9, type="Shot", shot_type="Free Kick", shot_outcome="Saved", team=10, poss=6, poss_team=10))
    b = tl(corner(t=0, poss=5), ev_(t=9, type="Referee Ball-Drop", team=20, poss=6, poss_team=20))
    assert build_chain(a, 0, CAP, RS).shot_count == 0 and build_chain(a, 0, CAP, RS).end_reason == "restart"
    assert build_chain(b, 0, CAP, RS).restart_type == "Referee Ball-Drop"

def test_stoppage_records_the_restart_that_follows():
    ev = tl(corner(t=0, poss=5), ev_(t=3, type="Clearance", team=20, poss=5, out=True),
            ev_(t=20, type="Pass", pass_type="Goal Kick", team=20, poss=6, poss_team=20))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.restart_type, r.restart_team, r.lost_possession) == ("stoppage", "Goal Kick", "defending", False)
    again = tl(corner(t=0, poss=5), ev_(t=3, type="Clearance", team=20, poss=5, out=True), corner(t=40, id="c2", poss=6))
    assert (build_chain(again, 0, CAP, RS).restart_type, build_chain(again, 0, CAP, RS).restart_team) == ("Corner", "attacking")

def test_no_cap_runs_to_end_of_possession():
    ev = tl(corner(t=0, poss=5), ev_(t=10, type="Pass", team=10, poss=5), ev_(t=40, type="Shot", team=10, poss=5, shot_outcome="Saved"),
            ev_(t=41, type="Goal Keeper", team=20, poss=6, poss_team=20))
    assert build_chain(ev, 0, CAP, RS).shot_count == 0 and build_chain(ev, 0, CAP, RS).end_reason == "time_cap"
    r = build_chain(ev, 0, None, RS)
    assert (r.shot_count, r.end_reason, r.lost_possession) == (1, "possession_change", True)

def test_time_cap():
    r = build_chain(tl(corner(t=0), ev_(t=10, type="Pass", team=10), ev_(t=16, type="Pass", team=10)), 0, CAP, RS)
    assert r.end_reason == "time_cap" and r.n_events == 1

def test_goal_included_and_counted_once_with_out():
    r = build_chain(tl(corner(t=0), ev_(t=2, type="Shot", team=10, shot_outcome="Goal", out=True, id="g")), 0, CAP, RS)
    assert (r.end_reason, r.goal, r.goal_event_id, r.shot_count) == ("goal", True, "g", 1)

def test_own_goal_for_counts_once():
    r = build_chain(tl(corner(t=0), ev_(t=2, type="Own Goal Against", team=20),
                       ev_(t=2, type="Own Goal For", team=10, id="og")), 0, CAP, RS)
    assert r.goal is True and r.goal_event_id == "og" and r.shot_count == 0

def test_stoppage_out_included():
    r = build_chain(tl(corner(t=0), ev_(t=3, type="Clearance", team=20, out=True, id="x")), 0, CAP, RS)
    assert r.end_reason == "stoppage" and r.last_included_event_id == "x" and r.opponent_action_count == 1

def test_foul_with_advantage_continues():
    r = build_chain(tl(corner(t=0), ev_(t=2, type="Foul Committed", team=20, foul_advantage=True),
                       ev_(t=3, type="Shot", team=10, shot_outcome="Saved")), 0, CAP, RS)
    assert r.shot_count == 1

def test_penalty_flagged_not_counted():
    r = build_chain(tl(corner(t=0), ev_(t=2, type="Foul Committed", team=20,
                                         details={"foul_committed": {"penalty": True}}),
                       ev_(t=60, type="Shot", team=10, shot_type="Penalty", shot_outcome="Goal")), 0, CAP, RS)
    assert r.penalty_awarded is True and r.goal is False and r.end_reason == "stoppage"

def test_timeline_end_incomplete():
    r = build_chain(tl(corner(t=0), ev_(t=2, type="Pass", team=10)), 0, CAP, RS)
    assert r.end_reason == "timeline_end" and r.outcome_status == "incomplete"

def test_null_possession_incomplete():
    r = build_chain(tl(corner(t=0), ev_(t=2, type="Pass", team=10, poss=None), ev_(t=3, type="Half End")), 0, CAP, RS)
    assert r.outcome_status == "incomplete"

def test_defender_shot_not_counted():
    r = build_chain(tl(corner(t=0), ev_(t=2, type="Shot", team=20, shot_outcome="Off T"),
                       ev_(t=3, type="Half End")), 0, CAP, RS)
    assert r.shot_count == 0


# --- Rulings and extra coverage -----------------------------------------------------------


def test_chain_result_has_the_outcome_fields_of_the_spec():
    assert [f.name for f in dataclasses.fields(ChainResult)] == [
        "last_included_event_id", "included_event_ids", "n_events", "sequence_duration_seconds",
        "end_reason", "restart_type", "restart_team", "shot_count", "first_shot_event_id", "goal",
        "goal_event_id", "lost_possession", "penalty_awarded", "opponent_action_count",
        "outcome_status",
    ]


def test_result_values_are_plain_python_types():
    ev = tl(corner(t=0), ev_(t=2, type="Shot", team=10, shot_outcome="Goal", id="g"))
    r = build_chain(ev, 0, CAP, RS)
    assert type(r.goal) is bool and type(r.lost_possession) is bool and type(r.penalty_awarded) is bool
    assert type(r.n_events) is int and type(r.shot_count) is int and type(r.opponent_action_count) is int
    assert type(r.sequence_duration_seconds) is float
    assert all(type(i) is str for i in r.included_event_ids) and type(r.last_included_event_id) is str


def test_parse_timestamp():
    assert parse_timestamp("00:00:00.000") == 0.0
    assert parse_timestamp("00:12:03.250") == pytest.approx(723.25)
    assert parse_timestamp("01:00:00.000") == 3600.0


def test_corner_alone_has_no_included_events():
    r = build_chain(tl(corner(t=0, id="c1")), 0, CAP, RS)
    assert (r.last_included_event_id, r.included_event_ids, r.n_events) == ("c1", [], 0)
    assert (r.sequence_duration_seconds, r.end_reason, r.outcome_status) == (0.0, "timeline_end", "incomplete")


def test_included_events_exclude_the_corner_and_measure_the_duration():
    ev = tl(corner(t=100, id="c1"), ev_(t=102, type="Pass", id="a"), ev_(t=104.5, type="Pass", team=20, id="b"),
            ev_(t=106, type="Half End"))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.included_event_ids, r.n_events, r.last_included_event_id) == (["a", "b"], 2, "b")
    assert r.sequence_duration_seconds == pytest.approx(4.5) and r.opponent_action_count == 1


def test_time_cap_is_strictly_greater():
    ev = tl(corner(t=0), ev_(t=15, type="Pass"), ev_(t=15.001, type="Pass"))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.n_events, r.sequence_duration_seconds) == ("time_cap", 1, 15.0)


def test_time_cap_ends_with_no_restart_info():
    r = build_chain(tl(corner(t=0), ev_(t=16, type="Pass")), 0, CAP, RS)
    assert (r.end_reason, r.restart_type, r.restart_team) == ("time_cap", None, None)


def test_restart_is_checked_before_the_time_cap():
    r = build_chain(tl(corner(t=0), ev_(t=16, type="Pass", pass_type="Throw-in")), 0, CAP, RS)
    assert (r.end_reason, r.restart_type) == ("restart", "Throw-in")


def test_negative_elapsed_time_is_zero():
    ev = tl(corner(t=10), ev_(t=8, type="Pass"), ev_(t=9, type="Half End"))
    r = build_chain(ev, 0, 0, RS)  # a zero cap still keeps a step backwards
    assert (r.n_events, r.sequence_duration_seconds, r.end_reason) == (1, 0.0, "period_end")
    later = tl(corner(t=10), ev_(t=8, type="Pass"), ev_(t=12, type="Pass"), ev_(t=13, type="Half End"))
    assert build_chain(later, 0, CAP, RS).sequence_duration_seconds == pytest.approx(2.0)


def test_null_timestamp_is_incomplete_and_skips_the_cap():
    ev = tl(corner(t=0), ev_(t=None, type="Pass"), ev_(t=3, type="Half End"))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.n_events, r.end_reason, r.outcome_status) == (1, "period_end", "incomplete")


def test_next_period_ends_the_chain():
    ev = tl(corner(t=0), ev_(t=1, type="Pass", period=2))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.n_events, r.restart_type, r.outcome_status) == ("period_end", 0, None, "complete")


def test_stoppage_look_ahead_stops_at_the_period_end():
    out = ev_(t=3, type="Clearance", team=20, out=True)
    next_period = tl(corner(t=0), out, ev_(t=1, type="Pass", pass_type="Goal Kick", team=20, period=2))
    half_end = tl(corner(t=0), out, ev_(t=4, type="Half End"), ev_(t=5, type="Pass", pass_type="Goal Kick", team=20))
    for ev in (next_period, half_end):
        r = build_chain(ev, 0, CAP, RS)
        assert (r.end_reason, r.restart_type, r.restart_team) == ("stoppage", None, None)


def test_stoppage_look_ahead_skips_other_events_and_has_no_cap():
    ev = tl(corner(t=0), ev_(t=3, type="Shot", shot_outcome="Off T"), ev_(t=4, type="Goal Keeper", team=20, out=True),
            ev_(t=30, type="Ball Out", team=20), ev_(t=50, type="Pass", pass_type="Throw-in", team=10, poss=6))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.restart_type, r.restart_team) == ("stoppage", "Throw-in", "attacking")


def test_stoppage_after_foul_offside_and_advantage_flag():
    for kind, adv in [("Foul Committed", False), ("Foul Committed", None), ("Offside", None)]:
        r = build_chain(tl(corner(t=0), ev_(t=2, type=kind, team=20, foul_advantage=adv, id="x")), 0, CAP, RS)
        assert (r.end_reason, r.last_included_event_id) == ("stoppage", "x"), (kind, adv)


def test_goal_by_the_defending_team_does_not_end_the_chain():
    ev = tl(corner(t=0), ev_(t=2, type="Shot", team=20, shot_outcome="Goal"), ev_(t=3, type="Half End"))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.goal, r.goal_event_id, r.shot_count, r.end_reason) == (False, None, 0, "period_end")


def test_shots_are_counted_and_the_first_is_named():
    ev = tl(corner(t=0), ev_(t=2, type="Shot", shot_outcome="Blocked", id="s1"), ev_(t=3, type="Pass", team=20),
            ev_(t=4, type="Shot", shot_outcome="Saved", id="s2"), ev_(t=5, type="Half End"))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.shot_count, r.first_shot_event_id, r.goal, r.goal_event_id) == (2, "s1", False, None)
    assert build_chain(tl(corner(t=0), ev_(t=2, type="Half End")), 0, CAP, RS).first_shot_event_id is None


def test_a_corner_shot_is_a_shot_not_a_new_corner():
    ev = tl(corner(t=0), ev_(t=2, type="Shot", shot_type="Corner", shot_outcome="Off T"), ev_(t=3, type="Half End"))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.shot_count) == ("period_end", 1)


def test_older_possession_id_is_still_a_change():
    ev = tl(corner(t=0, poss=5), ev_(t=2, type="Pass", team=20, poss=3, poss_team=20))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.n_events, r.lost_possession) == ("possession_change", 0, True)


def test_ball_receipt_with_an_older_id_after_a_ball_drop():
    # real data: a Ball Receipt* right after a Referee Ball-Drop can carry an older possession_id
    ev = tl(corner(t=0, poss=5), ev_(t=9, type="Referee Ball-Drop", team=20, poss=6, poss_team=20),
            ev_(t=9, type="Ball Receipt*", team=20, poss=4, poss_team=20))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.restart_type, r.n_events) == ("restart", "Referee Ball-Drop", 0)


def test_null_possession_event_is_included_not_a_change():
    ev = tl(corner(t=0), ev_(t=2, type="Pass", team=20, poss=None, poss_team=None), ev_(t=3, type="Half End"))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.n_events, r.lost_possession, r.outcome_status) == ("period_end", 1, False, "incomplete")


def test_corner_without_possession_id_skips_the_possession_check():
    ev = tl(corner(t=0, poss=None), ev_(t=2, type="Pass", team=10, poss=7), ev_(t=3, type="Half End"))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.end_reason, r.n_events, r.outcome_status) == ("period_end", 1, "incomplete")


def test_lost_possession_needs_the_defending_team():
    ev = tl(corner(t=0), ev_(t=2, type="Pass", team=20, poss=6, poss_team=None))
    assert build_chain(ev, 0, CAP, RS).lost_possession is False


def test_unknown_team_is_neither_an_opponent_nor_a_restart_side():
    ev = tl(corner(t=0), ev_(t=2, type="Pass", team=None), ev_(t=3, type="Pass", pass_type="Throw-in", team=None, poss=6))
    r = build_chain(ev, 0, CAP, RS)
    assert (r.opponent_action_count, r.end_reason, r.restart_type, r.restart_team) == (0, "restart", "Throw-in", None)


def test_penalty_lookup_by_side_and_guards():
    def run(*events):
        return build_chain(tl(corner(t=0), *events, ev_(t=9, type="Half End")), 0, CAP, RS).penalty_awarded

    won = ev_(t=2, type="Foul Won", team=10, details={"foul_won": {"penalty": True}})
    assert run(won) is True
    assert run(ev_(t=2, type="Foul Won", team=20, details={"foul_won": {"penalty": True}})) is False
    assert run(ev_(t=2, type="Foul Committed", team=10, details={"foul_committed": {"penalty": True}})) is False
    assert run(ev_(t=2, type="Foul Committed", team=20, details={"foul_committed": {"penalty": False}})) is False
    assert run(ev_(t=2, type="Foul Committed", team=20, details={"foul_committed": {"card": "Yellow"}})) is False
    assert run(ev_(t=2, type="Foul Committed", team=20, details={"other": 1})) is False
    assert run(ev_(t=2, type="Foul Committed", team=20, details={"foul_committed": None})) is False
    assert run(ev_(t=2, type="Foul Committed", team=20)) is False  # details null


def test_penalty_after_the_chain_ended_is_not_flagged():
    ev = tl(corner(t=0), ev_(t=2, type="Pass", team=20, poss=6, poss_team=20),
            ev_(t=3, type="Foul Committed", team=20, poss=6, details={"foul_committed": {"penalty": True}}))
    assert build_chain(ev, 0, CAP, RS).penalty_awarded is False


def test_walk_starts_at_corner_pos_and_leaves_the_frame_alone():
    ev = tl(ev_(t=0, type="Pass", team=20, poss=4, poss_team=20), ev_(t=1, type="Half End"),
            corner(t=100, id="c1"), ev_(t=102, type="Pass", id="a"), ev_(t=103, type="Half End"))
    before = ev.copy()
    r = build_chain(ev, 2, CAP, RS)
    assert (r.included_event_ids, r.end_reason, r.last_included_event_id) == (["a"], "period_end", "a")
    assert ev.equals(before) and list(ev.index) == list(before.index)


def test_corner_pos_out_of_range():
    ev = tl(corner(t=0))
    for pos in (1, -1):
        with pytest.raises(IndexError):
            build_chain(ev, pos, CAP, RS)


def test_restarts_match_names_the_restart_type():
    ev = tl(ev_(t=0, type="Pass", pass_type="Throw-in"), ev_(t=0, type="Pass", pass_type="Free Kick"),
            ev_(t=0, type="Shot", shot_type="Free Kick"), ev_(t=0, type="Shot", shot_type="Penalty"),
            ev_(t=0, type="Referee Ball-Drop"), ev_(t=0, type="Pass", pass_type="Corner"),
            ev_(t=0, type="Pass", pass_type="Penalty"), ev_(t=0, type="Shot", shot_type="Throw-in"),
            ev_(t=0, type="Shot", shot_type="Open Play"), ev_(t=0, type="Pass"), ev_(t=0, type="Throw-in"))
    assert [RS.match(row) for _, row in ev.iterrows()] == [
        "Throw-in", "Free Kick", "Free Kick", "Penalty", "Referee Ball-Drop", None, None, None, None, None, None,
    ]


def test_restarts_from_config_reads_the_three_params(cfg):
    assert Restarts.from_config(cfg) == RS
    params = {**cfg.params,
              "restart_pass_types": Param(["Kick Off"], "locked", ""),
              "restart_shot_types": Param(["Penalty"], "locked", ""),
              "restart_event_types": Param(["Referee Ball-Drop", "Half Start"], "locked", "")}
    r = Restarts.from_config(dataclasses.replace(cfg, params=params))
    assert (r.pass_types, r.shot_types, r.event_types) == (["Kick Off"], ["Penalty"], ["Referee Ball-Drop", "Half Start"])
    ev = tl(ev_(t=0, type="Pass", pass_type="Throw-in"), ev_(t=0, type="Pass", pass_type="Kick Off"),
            ev_(t=0, type="Shot", shot_type="Free Kick"), ev_(t=0, type="Half Start"))
    assert [r.match(row) for _, row in ev.iterrows()] == [None, "Kick Off", None, "Half Start"]
