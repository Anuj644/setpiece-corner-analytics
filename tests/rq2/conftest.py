"""Fixtures for the RQ2 tests."""

import dataclasses

import pandas as pd
import pytest
from rq2_helpers import (
    SLICES,
    corner,
    ev_,
    outcome,
    outcomes_df,
    rq1_corner,
    rq1_df,
    tl,
    write_rq1_inputs,
)

from setpiece.io import write_table


def _as_match(frame, match_id, competition_id, season_id):
    """``frame`` (one ``tl`` match) moved to another match and slice, with event ids made unique."""

    def column(value):
        return pd.array([value] * len(frame), dtype="Int64")

    return frame.assign(
        match_id=column(match_id),
        competition_id=column(competition_id),
        season_id=column(season_id),
        event_id=f"m{match_id}-" + frame["event_id"],
    )


@pytest.fixture
def two_match_events():
    """Two matches in two slices, each with two corners; the rows are ordered by match.

    Match 1 (slice 43/106, team 10 against 20): team 10's corner is followed by a shot and an
    opposing clearance out of play (a stoppage, then a goal kick); team 20's corner is the last
    event pair of the match, so its chain runs off the end of the timeline.

    Match 2 (slice 9/281, team 30 against 40): team 30's corner draws an opposing touch and then a
    goal; team 40's corner ends in a turnover and the half.
    """
    match1 = tl(
        corner(t=0, id="c1"),
        ev_(t=2, type="Pass", id="p1"),
        ev_(t=4, type="Shot", shot_outcome="Saved", id="s1"),
        ev_(t=6, type="Clearance", team=20, out=True, id="clr"),
        ev_(t=30, type="Pass", pass_type="Goal Kick", team=20, poss=6, poss_team=20, id="gk"),
        corner(t=60, team=20, poss=7, poss_team=20, id="c2"),
        ev_(t=62, type="Pass", team=20, poss=7, poss_team=20, id="p2"),
    )
    match2 = tl(
        ev_(t=10, type="Pass", team=40, poss=4, poss_team=40, id="lead"),
        corner(t=100, team=30, poss=5, poss_team=30, id="c1"),
        ev_(t=103, type="Pass", team=40, poss=5, poss_team=30, id="opp"),
        ev_(t=106, type="Shot", team=30, poss=5, poss_team=30, shot_outcome="Goal", id="goal"),
        ev_(t=110, type="Pass", pass_type="Kick Off", team=40, poss=6, poss_team=40, id="ko"),
        corner(t=200, team=40, poss=7, poss_team=40, id="c2"),
        ev_(t=204, type="Pass", team=30, poss=8, poss_team=30, id="turn"),
        ev_(t=205, type="Half End", team=30, poss=8, poss_team=30, id="he"),
    )
    return pd.concat(
        [_as_match(match1, 1, 43, 106), _as_match(match2, 2, 9, 281)], ignore_index=True
    )


@pytest.fixture
def rq2_cfg(cfg_tmp, two_match_events):
    """``cfg_tmp`` with the two fixture slices configured, their events written as partitions and
    the ``team_samples`` and ``rq1_corners`` tables ``rq1`` would have left for them."""
    cfg = dataclasses.replace(cfg_tmp, slices=list(SLICES))
    for (comp, season), part in two_match_events.groupby(["competition_id", "season_id"]):
        write_table(part.reset_index(drop=True), "events", cfg, (int(comp), int(season)))
    write_rq1_inputs(cfg)
    return cfg


@pytest.fixture
def samples():
    """``team_samples`` rows of slice 9/281 (a configured slice): team 1 played four matches (one
    without a corner of its own), team 2 three, team 3 three and took no corner at all."""
    rows = [
        (1, "Team A", 4, "incomplete_season (4 of 34)"),
        (2, "Team B", 3, "incomplete_season (3 of 34)"),
        (3, "Team C", 3, "incomplete_season (3 of 34)"),
    ]
    return pd.DataFrame(
        [
            {"competition_id": 9, "season_id": 281, "team_id": team, "team_name": name,
             "matches_in_sample": n, "sample_label": label}
            for team, name, n, label in rows
        ]
    ).astype({"competition_id": "Int64", "season_id": "Int64", "team_id": "Int64",
              "team_name": "string", "matches_in_sample": "Int64", "sample_label": "string"})  # fmt: skip


@pytest.fixture
def outcomes_with_incomplete():
    """Nine chains of slice 9/281; every chain id is ``<team>-<n>``.

    Team 1: five corners in matches 101 (two), 102 (two) and 103 (one, incomplete, with three
    shots that must not count). Its complete chains: a goal, a goal-kick restart by the defending
    team, a stoppage (a shot) before a defending throw-in, a shot and then an open-play turnover.
    Team 2: three complete corners in match 201; one shot, a new corner by the attacking team, a
    period end. Team 3: none.
    """
    return outcomes_df([
        outcome("1-1", 101, 1, end_reason="goal", shot_count=1, goal=True),
        outcome("1-2", 101, 1, restart_type="Goal Kick", restart_team="defending"),
        outcome("1-3", 102, 1, end_reason="stoppage", shot_count=1, restart_type="Throw-in",
                restart_team="defending"),
        outcome("1-4", 102, 1, end_reason="possession_change", shot_count=1, lost_possession=True),
        outcome("1-5", 103, 1, end_reason="timeline_end", shot_count=3, outcome_status="incomplete"),
        outcome("2-1", 201, 2, end_reason="time_cap", shot_count=1),
        outcome("2-2", 201, 2, end_reason="new_corner", restart_type="Corner",
                restart_team="attacking"),
        outcome("2-3", 201, 2, end_reason="period_end"),
    ])  # fmt: skip


@pytest.fixture
def rq1_for_outcomes():
    """``rq1_corners`` rows of the chains of ``outcomes_with_incomplete``, plus one other slice's."""
    shapes = {
        "1-1": ("Inswinging", "near", False), "1-2": ("Unknown", None, True),
        "1-3": ("Inswinging", "near", None), "1-4": ("Outswinging", "central", False),
        "1-5": ("Inswinging", "near", False), "2-1": ("Inswinging", "near", False),
        "2-2": ("Unknown", None, True), "2-3": ("Outswinging", "central", None),
    }  # fmt: skip
    rows = [rq1_corner(eid, *shape, competition_id=9, season_id=281) for eid, shape in shapes.items()]
    rows.append(rq1_corner("elsewhere", "Inswinging", "near", False, competition_id=43, season_id=106))
    return rq1_df(rows)
