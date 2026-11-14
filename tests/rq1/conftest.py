"""Shared fixtures for RQ1 tests (Tasks 13–15)."""

import pandas as pd
import pytest
from helpers import corner_row, events_df


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


@pytest.fixture
def cfg_tmp_with_synthetic_rq1(cfg_tmp):
    """cfg_tmp (slices 9/281 and 2/27 only) whose ``derived/rq1_corners`` is synthetic.

    Slice 9/281: 60 matches played in pairs on the same date (so the tie-to-test rule matters),
    enough for ``forecast_min_matches``. Slice 2/27: only 5 matches, which must be skipped.
    """
    import dataclasses

    from forecast_helpers import synthetic_corners

    from setpiece.config import Slice
    from setpiece.io import write_table

    big = synthetic_corners(60, [1, 2, 3, 4], seed=0, same_date_pairs=True)
    big["competition_id"], big["season_id"] = 9, 281
    small = synthetic_corners(5, [1, 2], seed=1)
    small["competition_id"], small["season_id"] = 2, 27
    small["match_id"] += 1000
    write_table(pd.concat([big, small], ignore_index=True), "derived/rq1_corners", cfg_tmp, None)
    slices = [
        Slice(9, 281, "1. Bundesliga 2023/24", "league", True, expected_matches=34),
        Slice(2, 27, "Premier League 2015/16", "league", False, expected_matches=38),
    ]
    return dataclasses.replace(cfg_tmp, slices=slices)

@pytest.fixture
def cfg_tmp_with_events(cfg_tmp, rq1_two_teams):
    """``cfg_tmp`` whose ``events`` table holds the ``rq1_two_teams`` corners as slice 9/281.

    The other four core slices have no events table partition: a stage must cope with that.
    """
    from setpiece.io import write_table

    write_table(rq1_two_teams, "events", cfg_tmp, (9, 281))
    return cfg_tmp
