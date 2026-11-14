"""Shared fixtures for RQ1 tests (Tasks 13–15)."""

import pandas as pd
import pytest

from tests.helpers import corner_row, events_df


def _build_rq1_small():
    """8 corners, one starting at y=40 (id weird-start), ground passes of length 6, 9, 12, 20."""
    rows = [
        corner_row(event_id="c1", start_y=0.1, end=(110, 35), technique="Inswinging", pass_length=20.0),
        corner_row(event_id="c2", start_y=79.9, end=(110, 45), technique="Outswinging", pass_length=18.0),
        corner_row(event_id="g1", start_y=0.1, end=(114, 30), technique=None, pass_length=6.0),
        corner_row(event_id="g2", start_y=0.1, end=(113, 28), technique=None, pass_length=9.0),
        corner_row(event_id="g3", start_y=0.1, end=(112, 32), technique=None, pass_length=12.0),
        corner_row(event_id="g4", start_y=0.1, end=(110, 38), technique=None, pass_length=20.0),
        corner_row(event_id="weird-start", start_y=40.0, end=(110, 40), technique="Inswinging", pass_length=15.0),
        corner_row(event_id="c3", start_y=79.9, end=(108, 42), technique="Outswinging", pass_length=22.0),
    ]
    return events_df(rows)


@pytest.fixture
def rq1_small():
    """A small rq1 events frame for review export tests."""
    return _build_rq1_small()


@pytest.fixture
def rq1_two_teams():
    """2 teams x 3 matches x 4 corners, one null technique."""
    rows = []
    for m in range(1, 4):
        for t_idx, team_id in enumerate([1, 2]):
            for c in range(4):
                eid = f"m{m}t{team_id}c{c}"
                side_y = 0.1 if c % 2 == 0 else 79.9
                tech = None if (team_id == 1 and c == 0) else "Inswinging"
                rows.append(
                    corner_row(
                        event_id=eid,
                        start_y=side_y,
                        end=(110, 35),
                        technique=tech,
                        pass_length=15.0,
                        match_id=m,
                        team_id=team_id,
                        home_team_id=team_id if t_idx == 0 else (3 - team_id),
                        away_team_id=(3 - team_id) if t_idx == 0 else team_id,
                    )
                )
    return events_df(rows)


@pytest.fixture
def samples():
    """A team_sample_table-like output with one team that has no corners."""
    return pd.DataFrame(
        [
            {"competition_id": 9, "season_id": 281, "team_id": 1, "team_name": "Team A",
             "matches_in_sample": 3, "sample_label": "incomplete_season (3 of 34)"},
            {"competition_id": 9, "season_id": 281, "team_id": 2, "team_name": "Team B",
             "matches_in_sample": 3, "sample_label": "incomplete_season (3 of 34)"},
            {"competition_id": 9, "season_id": 281, "team_id": 3, "team_name": "Team C",
             "matches_in_sample": 3, "sample_label": "incomplete_season (3 of 34)"},
        ]
    )


@pytest.fixture
def summary_one_team():
    """summarise_rq1 output for team 1 (minimal synthetic)."""
    return pd.DataFrame(
        [
            {"competition_id": 9, "season_id": 281, "team_id": 1, "team_name": "Team A",
             "matches_in_sample": 3, "sample_label": "incomplete_season (3 of 34)",
             "attribute": "technique", "category": "Inswinging", "n": 6, "denominator": 8,
             "pct": 75.0, "ci_lo": 50.0, "ci_hi": 100.0},
            {"competition_id": 9, "season_id": 281, "team_id": 1, "team_name": "Team A",
             "matches_in_sample": 3, "sample_label": "incomplete_season (3 of 34)",
             "attribute": "technique", "category": "Outswinging", "n": 2, "denominator": 8,
             "pct": 25.0, "ci_lo": 0.0, "ci_hi": 50.0},
        ]
    )
