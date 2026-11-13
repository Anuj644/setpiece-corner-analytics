import json

import pandas as pd
import pytest
from helpers import events_df

from setpiece.ingest.events import (
    EVENTS_COLUMNS,
    EVENTS_DTYPES,
    build_events_frame,
    flatten_event,
    match_context,
)

SPEC_COLUMNS = [
    # Match context
    "match_id", "match_date", "competition_id", "competition_name", "season_id",
    "season_name", "home_team_id", "home_team_name", "away_team_id", "away_team_name",
    # Identity and order
    "event_id", "event_index", "period", "timestamp", "minute", "second", "duration",
    "event_type",
    # Actor and possession
    "team_id", "team_name", "player_id", "player_name", "possession_id",
    "possession_team_id", "play_pattern", "related_event_ids",
    # Position and stoppage
    "start_x", "start_y", "out", "foul_advantage",
    # Pass
    "pass_type", "pass_end_x", "pass_end_y", "pass_length", "pass_angle", "pass_height",
    "pass_technique", "pass_body_part", "pass_outcome", "pass_recipient_id",
    # Shot
    "shot_type", "shot_outcome",
    # Remainder
    "event_details",
]  # fmt: skip


def test_columns_and_order(df):
    assert list(df.columns) == EVENTS_COLUMNS


def test_columns_are_the_spec_5_4_columns():
    assert len(SPEC_COLUMNS) == 43
    assert EVENTS_COLUMNS == SPEC_COLUMNS


def test_corner_pass_mapping(df):
    c = df[df.event_id == "corner-1"].iloc[0]
    assert (c.event_type, c.pass_type, c.pass_technique) == ("Pass", "Corner", "Inswinging")
    assert (c.start_x, c.start_y, c.pass_end_x, c.pass_end_y) == (120.0, 0.1, 110.0, 38.0)
    assert c.match_id == 1 and c.competition_id == 43 and c.season_id == 106


def test_missing_technique_stays_null_and_legacy_flag_kept_in_details(df):
    c2 = df[df.event_id == "corner-2"].iloc[0]
    assert pd.isna(c2.pass_technique) and c2.pass_height == "Ground Pass"
    c1 = df[df.event_id == "corner-1"].iloc[0]
    assert json.loads(c1.event_details)["pass"]["inswinging"] is True  # kept, never used to fill


def test_legacy_flag_never_fills_a_missing_technique(raw_match):
    raw = {"id": "x", "index": 1, "type": {"name": "Pass"}, "pass": {"inswinging": True}}
    row = flatten_event(raw, match_context(raw_match))
    assert row["pass_technique"] is None
    assert json.loads(row["event_details"]) == {"pass": {"inswinging": True}}


def test_foul_flags(df):
    f = df[df.event_type == "Foul Committed"].iloc[0]
    assert f.foul_advantage is True or f.foul_advantage == True
    assert json.loads(f.event_details)["foul_committed"]["penalty"] is True


def test_non_pass_has_null_pass_columns(df):
    s = df[df.event_type == "Shot"].iloc[0]
    assert pd.isna(s.pass_type) and s.shot_outcome == "Goal"


def test_related_events_is_list(df):
    assert isinstance(df.iloc[1].related_event_ids, list)


def test_dtypes_follow_the_schema(df):
    assert dict(df.dtypes) == {
        col: pd.api.types.pandas_dtype(t) for col, t in EVENTS_DTYPES.items()
    }
    # "string" must be the NA-backed dtype, not pandas 3's default NaN-backed "str".
    assert df.event_id.dtype.na_value is pd.NA
    assert df.event_details.dtype.na_value is pd.NA


def test_dtype_groups_follow_the_contract():
    ids = [c for c in EVENTS_COLUMNS if c.endswith("_id") and c not in ("event_id",)]
    assert {EVENTS_DTYPES[c] for c in ids} == {"Int64"}
    assert {EVENTS_DTYPES[c] for c in ("period", "minute", "second", "event_index")} == {"Int64"}
    floats = (
        "duration",
        "start_x",
        "start_y",
        "pass_end_x",
        "pass_end_y",
        "pass_length",
        "pass_angle",
    )
    assert {EVENTS_DTYPES[c] for c in floats} == {"Float64"}
    assert {EVENTS_DTYPES[c] for c in ("out", "foul_advantage")} == {"boolean"}
    assert EVENTS_DTYPES["related_event_ids"] == "object"
    strings = ("event_id", "match_date", "timestamp", "event_type", "team_name", "event_details")
    assert {EVENTS_DTYPES[c] for c in strings} == {"string"}


def test_match_context_has_the_ten_context_columns(raw_match):
    assert match_context(raw_match) == {
        "match_id": 1,
        "match_date": "2022-11-20",
        "competition_id": 43,
        "competition_name": "FIFA World Cup",
        "season_id": 106,
        "season_name": "2022",
        "home_team_id": 10,
        "home_team_name": "A",
        "away_team_id": 20,
        "away_team_name": "B",
    }


def test_flatten_event_returns_every_column_in_order(raw_events, raw_match):
    row = flatten_event(raw_events[1], match_context(raw_match))
    assert list(row) == EVENTS_COLUMNS
    assert row["event_id"] == "corner-1" and row["player_name"] == "P"
    assert row["possession_id"] == 2 and row["possession_team_id"] == 10
    assert row["play_pattern"] == "From Corner" and row["pass_recipient_id"] == 8
    assert row["pass_body_part"] == "Right Foot" and row["pass_length"] == 39.9


def test_rows_are_sorted_by_event_index_with_a_fresh_index(df, raw_events, raw_match):
    shuffled = build_events_frame(list(reversed(raw_events)), raw_match)
    pd.testing.assert_frame_equal(shuffled, df)
    assert df.event_index.is_monotonic_increasing
    assert list(df.index) == list(range(len(df)))


def test_absent_related_events_is_null_not_an_empty_list(df):
    starting_xi = df[df.event_type == "Starting XI"].iloc[0]
    assert starting_xi.related_event_ids is None
    corner = df[df.event_id == "corner-1"].iloc[0]
    assert corner.related_event_ids == ["recv-1"]


def test_foul_won_advantage_fills_foul_advantage_and_other_events_stay_null(df):
    won = df[df.event_type == "Foul Won"].iloc[0]
    assert won.foul_advantage is True or won.foul_advantage == True
    others = df[~df.event_type.isin(["Foul Committed", "Foul Won"])]
    assert others.foul_advantage.isna().all()


def test_out_flag_is_null_unless_the_source_supplies_it(df):
    assert df[df.event_type == "Clearance"].iloc[0].out == True
    assert df[df.event_type != "Clearance"].out.isna().all()


def test_missing_values_stay_null(df):
    half_end = df[df.event_type == "Half End"].iloc[0]
    assert half_end[["start_x", "start_y", "player_id", "player_name"]].isna().all()
    assert pd.isna(df[df.event_id == "corner-2"].iloc[0].pass_outcome)  # not supplied


def test_event_details_keeps_exactly_the_unmapped_source_keys(df):
    def details(event_id):
        return json.loads(df[df.event_id == event_id].iloc[0].event_details)

    assert details("corner-1") == {
        "pass": {"inswinging": True},
        "position": {"id": 23, "name": "Center Forward"},
    }
    assert details("recv-1") == {
        "position": {"id": 17, "name": "Right Wing"},
        "under_pressure": True,
    }
    assert details("carry-1") == {
        "carry": {"end_location": [108.0, 36.5]},
        "position": {"id": 17, "name": "Right Wing"},
    }
    assert details("clearance-1") == {
        "clearance": {"aerial_won": True, "body_part": {"id": 37, "name": "Head"}},
        "position": {"id": 3, "name": "Right Center Back"},
    }
    assert details("shot-1")["shot"] == {
        "statsbomb_xg": 0.12,
        "end_location": [120.0, 38.0, 1.2],
        "technique": {"id": 93, "name": "Normal"},
        "body_part": {"id": 37, "name": "Head"},
        "first_time": True,
    }
    assert details("keeper-1")["goalkeeper"] == {
        "type": {"id": 27, "name": "Goal Conceded"},
        "outcome": {"id": 47, "name": "Failed"},
    }
    assert details("foul-1") == {
        "counterpress": True,
        "foul_committed": {"penalty": True},
        "position": {"id": 5, "name": "Left Center Back"},
    }
    assert set(details("starting-xi-a")) == {"tactics"}


def test_event_details_is_null_when_everything_is_mapped(df):
    for event_id in ("half-end-1", "own-goal-for-1", "foul-won-1"):
        assert pd.isna(df[df.event_id == event_id].iloc[0].event_details), event_id


def test_event_details_is_a_sorted_key_json_dump(df):
    raw = df[df.event_id == "foul-1"].iloc[0].event_details
    assert raw == json.dumps(json.loads(raw), sort_keys=True)


def test_mapped_keys_are_consumed_whole_in_type_specific_dicts(df):
    shot = json.loads(df[df.event_id == "shot-1"].iloc[0].event_details)["shot"]
    assert not {"type", "outcome"} & set(shot)
    corner = json.loads(df[df.event_id == "corner-1"].iloc[0].event_details)["pass"]
    assert set(corner) == {"inswinging"}


def test_build_events_frame_with_no_events_keeps_the_schema(raw_match):
    empty = build_events_frame([], raw_match)
    assert list(empty.columns) == EVENTS_COLUMNS and empty.empty
    assert dict(empty.dtypes) == {c: pd.api.types.pandas_dtype(t) for c, t in EVENTS_DTYPES.items()}


def test_fixture_has_exactly_two_corner_passes(df):
    corners = df[(df.event_type == "Pass") & (df.pass_type == "Corner")]
    assert list(corners.event_id) == ["corner-1", "corner-2"]


def test_events_alias_is_the_same_frame(df, events):
    assert events is df


def test_events_df_fills_missing_columns_with_null_and_casts():
    frame = events_df([{"match_id": 1, "event_id": "x", "event_type": "Pass", "start_x": 5}])
    assert list(frame.columns) == EVENTS_COLUMNS
    assert dict(frame.dtypes) == {c: pd.api.types.pandas_dtype(t) for c, t in EVENTS_DTYPES.items()}
    row = frame.iloc[0]
    assert row.match_id == 1 and row.event_id == "x" and row.start_x == 5.0
    assert row[["player_id", "pass_type", "foul_advantage", "event_details"]].isna().all()
    assert row.related_event_ids is None


def test_events_df_rejects_unknown_columns():
    with pytest.raises(ValueError, match="pass_typo"):
        events_df([{"pass_typo": "Corner"}])
