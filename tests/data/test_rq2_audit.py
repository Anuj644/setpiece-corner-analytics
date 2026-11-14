import pytest

from setpiece.io import read_table
from setpiece.rq2.chains import build_rq2_outcomes, restarts_from
from setpiece.rq2.diagnostics import opponent_action_within


@pytest.mark.data
def test_audit_opponent_action_844(cfg):
    ev = read_table("events", cfg, [(9, 281), (11, 90), (7, 235)])
    assert int(opponent_action_within(ev, 15).sum()) == 844

@pytest.mark.data
@pytest.mark.parametrize("cap,shots,goals", [(10, 306, 29), (15, 343, 34), (20, 365, 37), (None, 414, 45)])
def test_audit_slice_shot_totals_by_cap(cfg, cap, shots, goals):
    # measured 2026-11-10 by local_docs/verification/2026-11-10/verify2.py, which implements spec §8.1 as amended
    ev = read_table("events", cfg, [(9, 281), (11, 90), (7, 235)])
    out = build_rq2_outcomes(ev, cap, restarts_from(cfg))
    assert (int(out.shot_count.sum()), int(out.goal.sum())) == (shots, goals)
    assert (out.outcome_status == "complete").all()

@pytest.mark.data
def test_audit_slice_end_reasons_at_15s(cfg):
    ev = read_table("events", cfg, [(9, 281), (11, 90), (7, 235)])
    out = build_rq2_outcomes(ev, 15, restarts_from(cfg))
    assert out.end_reason.value_counts().to_dict() == {
        "time_cap": 295, "possession_change": 242, "stoppage": 206, "restart": 154,
        "goal": 34, "new_corner": 22, "period_end": 12}
    assert int(out.lost_possession.sum()) == 239
    r = out[out.end_reason == "restart"]
    assert int(((r.restart_type == "Goal Kick") & (r.restart_team == "defending")).sum()) == 124
