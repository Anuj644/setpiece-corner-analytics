"""Tests for sample labels and team sample table (Task 11)."""

from setpiece.config import Slice
from setpiece.coverage import sample_label, team_sample_table
from tests.helpers import events_df


def test_labels():
    assert sample_label(34, "league", 34) == "complete_season"
    assert sample_label(35, "league", 38) == "incomplete_season (35 of 38)"
    assert sample_label(2, "league", 34) == "incomplete_season (2 of 34)"
    assert sample_label(7, "league", None) == "season_length_unknown (7 matches)"
    assert sample_label(3, "tournament", None) == "full_tournament"


def test_complete_season_when_matches_exceed_expected():
    # A team that plays more than expected (e.g. cups don't affect league count)
    assert sample_label(39, "league", 38) == "complete_season"


def test_tournament_ignores_expected_matches():
    # expected_matches is None for tournaments; any value should still work
    assert sample_label(7, "tournament", None) == "full_tournament"


def test_label_uses_config_not_team_count():
    # 19 distinct teams in the data, but the league season is 38 matches per team
    ev = events_df(
        [
            {
                "match_id": m,
                "event_id": f"e{m}",
                "event_index": 1,
                "competition_id": 11,
                "season_id": 90,
                "team_id": 1,
                "home_team_id": 1,
                "away_team_id": 2 + m % 18,
            }
            for m in range(35)
        ]
    )
    t = team_sample_table(
        ev,
        [Slice(11, 90, "La Liga 2020/21", "league", True, expected_matches=38)],
    )
    assert t.set_index("team_id").loc[1, "sample_label"] == "incomplete_season (35 of 38)"


def test_team_sample_table_columns():
    ev = events_df(
        [
            {
                "match_id": 1,
                "event_id": "a",
                "event_index": 1,
                "competition_id": 43,
                "season_id": 106,
                "team_id": 10,
                "home_team_id": 10,
                "away_team_id": 20,
            },
            {
                "match_id": 2,
                "event_id": "b",
                "event_index": 1,
                "competition_id": 43,
                "season_id": 106,
                "team_id": 20,
                "home_team_id": 10,
                "away_team_id": 20,
            },
        ]
    )
    slices = [Slice(43, 106, "FIFA World Cup 2022", "tournament", True)]
    t = team_sample_table(ev, slices)
    expected_cols = {
        "competition_id",
        "season_id",
        "team_id",
        "team_name",
        "matches_in_sample",
        "sample_label",
    }
    assert expected_cols <= set(t.columns)


def test_team_sample_table_matches_in_sample():
    # Team 1 appears in matches 1, 2, 3 (home or away).  Team 2 only in match 1.
    ev = events_df(
        [
            {
                "match_id": 1,
                "event_id": "a",
                "event_index": 1,
                "competition_id": 9,
                "season_id": 281,
                "team_id": 1,
                "home_team_id": 1,
                "away_team_id": 2,
                "team_name": "Team A",
            },
            {
                "match_id": 2,
                "event_id": "b",
                "event_index": 1,
                "competition_id": 9,
                "season_id": 281,
                "team_id": 1,
                "home_team_id": 1,
                "away_team_id": 3,
                "team_name": "Team A",
            },
            {
                "match_id": 3,
                "event_id": "c",
                "event_index": 1,
                "competition_id": 9,
                "season_id": 281,
                "team_id": 1,
                "home_team_id": 4,
                "away_team_id": 1,
                "team_name": "Team A",
            },
        ]
    )
    slices = [Slice(9, 281, "Bundesliga 2023/24", "league", True, expected_matches=34)]
    t = team_sample_table(ev, slices)
    row = t.set_index("team_id").loc[1]
    assert row["matches_in_sample"] == 3
    assert row["sample_label"] == "incomplete_season (3 of 34)"


def test_team_sample_table_tournament_label():
    ev = events_df(
        [
            {
                "match_id": m,
                "event_id": f"e{m}",
                "event_index": 1,
                "competition_id": 43,
                "season_id": 106,
                "team_id": 1,
                "home_team_id": 1,
                "away_team_id": 2,
            }
            for m in range(7)
        ]
    )
    slices = [Slice(43, 106, "FIFA World Cup 2022", "tournament", True)]
    t = team_sample_table(ev, slices)
    assert t.set_index("team_id").loc[1, "sample_label"] == "full_tournament"
