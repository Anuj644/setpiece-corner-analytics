import pandas as pd
import pytest
from helpers import events_df

from setpiece.ingest.frames import (
    CORNER_360_COLUMNS,
    CORNER_360_DTYPES,
    FrameStats,
    build_corner_360,
)


def corner_row(event_id, event_index, **extra):
    return {
        "match_id": 1,
        "competition_id": 43,
        "season_id": 106,
        "event_id": event_id,
        "event_index": event_index,
        "event_type": "Pass",
        "pass_type": "Corner",
        **extra,
    }


def test_only_corner_passes_kept_with_exact_uuid_join(events, frames):
    df, st = build_corner_360(frames, events)
    assert set(df.event_id) == {"corner-1"}  # corner-2 has no frame
    assert st.n_corners == 2 and st.n_corners_matched == 1


def test_unmatched_frames_counted(events, frames):
    _, st = build_corner_360(frames, events)  # fixture has 1 frame with unknown uuid
    assert st.n_unmatched == 1 and st.n_frames == len(frames)


def test_raw_coordinates_unchanged(events, frames):
    df, _ = build_corner_360(frames, events)
    assert df.iloc[0].players[0] == {
        "x": 108.2,
        "y": 41.0,
        "teammate": True,
        "actor": False,
        "keeper": False,
    }
    assert df.iloc[0].visible_area[:2] == frames[0]["visible_area"][:2]


def test_corners_without_frames_counted(events):
    df, st = build_corner_360(None, events)
    assert df.empty and st.has_file is False and st.n_corners == 2


def test_full_stats_for_fixture_match(events, frames):
    # The matched non-corner frame (shot-1) is neither unmatched nor a corner_360 row.
    _, st = build_corner_360(frames, events)
    assert st == FrameStats(
        match_id=1,
        n_frames=3,
        n_unmatched=1,
        n_corners=2,
        n_corners_matched=1,
        has_file=True,
    )


def test_no_file_gives_empty_table_and_zero_frame_counts(events):
    df, st = build_corner_360(None, events)
    assert list(df.columns) == CORNER_360_COLUMNS
    assert len(df) == 0
    assert st == FrameStats(
        match_id=1,
        n_frames=0,
        n_unmatched=0,
        n_corners=2,
        n_corners_matched=0,
        has_file=False,
    )


def test_empty_file_is_a_file_with_no_frames(events):
    df, st = build_corner_360([], events)
    assert df.empty and list(df.columns) == CORNER_360_COLUMNS
    assert st.has_file is True and st.n_frames == 0 and st.n_corners_matched == 0


def test_columns_dtypes_and_keys(events, frames):
    df, _ = build_corner_360(frames, events)
    assert list(df.columns) == CORNER_360_COLUMNS
    assert {col: str(dtype) for col, dtype in df.dtypes.items()} == CORNER_360_DTYPES
    row = df.iloc[0]
    assert (row.match_id, row.competition_id, row.season_id) == (1, 43, 106)
    assert row.event_id == "corner-1"


def test_dtypes_hold_for_the_empty_table(events):
    df, _ = build_corner_360(None, events)
    assert {col: str(dtype) for col, dtype in df.dtypes.items()} == CORNER_360_DTYPES


def test_players_carry_every_freeze_frame_entry_in_order(events, frames):
    df, _ = build_corner_360(frames, events)
    players = df.iloc[0].players
    assert [(p["x"], p["y"]) for p in players] == [
        (108.2, 41.0),
        (120.0, 0.1),
        (118.4, 39.6),
        (111.7, 35.3),
    ]
    assert [p["actor"] for p in players] == [False, True, False, False]
    assert [p["keeper"] for p in players] == [False, False, True, False]
    assert all(set(p) == {"x", "y", "teammate", "actor", "keeper"} for p in players)
    assert df.iloc[0].visible_area == frames[0]["visible_area"]


def test_non_corner_frame_is_not_in_the_table(events, frames):
    df, _ = build_corner_360(frames, events)
    assert "shot-1" not in set(df.event_id)
    assert "no-such-event" not in set(df.event_id)


def test_rows_follow_corner_event_index_order():
    events = events_df([corner_row("late", 20), corner_row("early", 5)])
    frames = [
        {"event_uuid": "late", "visible_area": [1.0, 2.0], "freeze_frame": []},
        {"event_uuid": "early", "visible_area": [3.0, 4.0], "freeze_frame": []},
    ]
    df, _ = build_corner_360(frames, events)
    assert list(df.event_id) == ["early", "late"]


def test_corner_filter_uses_event_type_and_pass_type():
    events = events_df(
        [
            corner_row("corner", 1),
            corner_row("free-kick", 2, pass_type="Free Kick"),
            corner_row("corner-shot", 3, event_type="Shot", shot_type="Corner"),
            corner_row("no-type", 4, pass_type=None),
        ]
    )
    frames = [
        {"event_uuid": uuid, "visible_area": [], "freeze_frame": []}
        for uuid in ("corner", "free-kick", "corner-shot", "no-type")
    ]
    df, st = build_corner_360(frames, events)
    assert list(df.event_id) == ["corner"]
    assert (st.n_corners, st.n_corners_matched, st.n_unmatched) == (1, 1, 0)


def test_missing_values_are_not_filled():
    events = events_df([corner_row("c", 1)])
    frames = [
        {
            "event_uuid": "c",
            "visible_area": [1.0, 2.0],
            "freeze_frame": [
                {"teammate": True, "actor": False, "keeper": False},  # no location
                {"location": [10.0, 20.0]},  # no flags
            ],
        }
    ]
    df, _ = build_corner_360(frames, events)
    no_location, no_flags = df.iloc[0].players
    assert no_location == {"x": None, "y": None, "teammate": True, "actor": False, "keeper": False}
    assert no_flags == {"x": 10.0, "y": 20.0, "teammate": None, "actor": None, "keeper": None}


def test_missing_freeze_frame_and_visible_area_stay_null():
    events = events_df([corner_row("c", 1)])
    df, _ = build_corner_360([{"event_uuid": "c"}], events)
    assert df.iloc[0].players is None and df.iloc[0].visible_area is None


def test_integer_coordinates_become_floats_without_changing_values():
    events = events_df([corner_row("c", 1)])
    frames = [
        {
            "event_uuid": "c",
            "visible_area": [0, 0, 120, 80],
            "freeze_frame": [
                {"teammate": True, "actor": False, "keeper": False, "location": [110, 40]}
            ],
        }
    ]
    df, _ = build_corner_360(frames, events)
    row = df.iloc[0]
    assert row.visible_area == [0.0, 0.0, 120.0, 80.0]
    assert all(type(v) is float for v in row.visible_area)
    assert (type(row.players[0]["x"]), type(row.players[0]["y"])) == (float, float)


def test_repeated_uuid_of_a_corner_pass_raises(events, frames):
    # Two frames for one corner make the join ambiguous: surface it, never pick one.
    duplicated = [*frames, {**frames[0]}]
    with pytest.raises(ValueError, match="corner-1"):
        build_corner_360(duplicated, events)


def test_repeated_uuids_of_unmatched_and_non_corner_frames_are_counted_not_raised(events, frames):
    # frames: corner-1, shot-1 (exists, not a corner), no-such-event (unknown).
    repeated = [*frames, {**frames[1]}, {**frames[2]}, {**frames[2]}]
    df, st = build_corner_360(repeated, events)
    assert list(df.event_id) == ["corner-1"]
    assert st.n_frames == 6  # records, repeats included
    assert st.n_unmatched == 3  # the unknown uuid, three records
    assert st.n_corners == 2 and st.n_corners_matched == 1


def test_frames_without_uuid_never_join_and_count_as_unmatched():
    events = events_df([corner_row(None, 1), corner_row("c", 2)])  # first corner: null event_id
    frames = [
        {"visible_area": [1.0, 2.0], "freeze_frame": []},  # no event_uuid key
        {"event_uuid": None, "visible_area": [3.0, 4.0], "freeze_frame": []},
        {"event_uuid": "c", "visible_area": [5.0, 6.0], "freeze_frame": []},
    ]
    df, st = build_corner_360(frames, events)
    assert list(df.event_id) == ["c"]
    assert df.iloc[0].visible_area == [5.0, 6.0]
    assert (st.n_frames, st.n_unmatched) == (3, 2)
    assert (st.n_corners, st.n_corners_matched) == (2, 1)


def test_events_of_several_matches_raise():
    events = events_df([corner_row("a", 1), {**corner_row("b", 2), "match_id": 2}])
    with pytest.raises(ValueError, match="one match"):
        build_corner_360([], events)


def test_inputs_are_not_modified(events, frames):
    events_before = events.copy(deep=True)
    frames_before = [{**f, "freeze_frame": [dict(p) for p in f["freeze_frame"]]} for f in frames]
    df, _ = build_corner_360(frames, events)
    df.iloc[0].players[0]["x"] = -1.0
    df.iloc[0].visible_area[0] = -1.0
    pd.testing.assert_frame_equal(events, events_before)
    assert frames == frames_before
