"""Fixtures for the RQ2 tests."""

import pandas as pd
import pytest
from rq2_helpers import corner, ev_, tl


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
