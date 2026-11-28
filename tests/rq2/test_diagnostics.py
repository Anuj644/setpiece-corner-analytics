import pandas as pd
from helpers import events_df

from setpiece.rq2.diagnostics import (
    flag_possession_starts,
    possession_start_summary,
    possession_starts,
)

START_COLUMNS = [
    "match_id",
    "possession_id",
    "first_event_type",
    "first_pass_type",
    "previous_event_type",
    "previous_out",
    "same_team_as_previous",
]


def test_possession_starts():
    ev = events_df([
        {"match_id": 1, "event_id": "a", "event_index": 1, "possession_id": 1, "possession_team_id": 10, "event_type": "Pass"},
        {"match_id": 1, "event_id": "b", "event_index": 2, "possession_id": 1, "possession_team_id": 10, "event_type": "Pass", "out": True},
        {"match_id": 1, "event_id": "c", "event_index": 3, "possession_id": 2, "possession_team_id": 10, "event_type": "Pass", "pass_type": "Throw-in"}])
    s = possession_starts(ev)
    assert s.iloc[-1].first_pass_type == "Throw-in" and bool(s.iloc[-1].same_team_as_previous)
    assert bool(s.iloc[-1].previous_out)


def _row(match, index, pid, team, event_type="Pass", **extra):
    return {
        "match_id": match,
        "event_id": f"{match}-{index}",
        "event_index": index,
        "possession_id": pid,
        "possession_team_id": team,
        "event_type": event_type,
        **extra,
    }


def test_possession_starts_columns_and_one_row_per_new_possession():
    ev = events_df([
        _row(1, 1, 1, 10),
        _row(1, 2, 1, 10),
        _row(1, 3, 2, 20, "Carry"),
        _row(1, 4, 2, 20),
        _row(1, 5, 3, 10, "Pass", pass_type="Goal Kick"),
    ])
    s = possession_starts(ev)
    assert list(s.columns) == START_COLUMNS
    assert s.possession_id.tolist() == [1, 2, 3]
    assert s.first_event_type.tolist() == ["Pass", "Carry", "Pass"]
    assert s.previous_event_type.tolist()[1:] == ["Pass", "Pass"]
    assert s.same_team_as_previous.tolist()[1:] == [False, False]
    assert s.index.tolist() == [0, 1, 2]


def test_first_possession_of_a_match_has_nulls_and_nothing_is_filled_in():
    ev = events_df([_row(1, 1, 1, 10, out=True), _row(1, 2, 2, 10)])
    first = possession_starts(ev).iloc[0]
    assert first.possession_id == 1
    assert pd.isna(first.previous_event_type)
    assert pd.isna(first.previous_out)
    assert pd.isna(first.same_team_as_previous)
    assert pd.isna(first.first_pass_type)


def test_possessions_never_chain_across_matches():
    # Match 1 ends on an out flag; match 2 opens with the possession_id and team match 1 ended on.
    ev = events_df([
        _row(1, 1, 1, 10),
        _row(1, 2, 2, 10),
        _row(1, 3, 2, 10, out=True),
        _row(2, 1, 2, 10),
        _row(2, 2, 3, 20, pass_type="Throw-in"),
    ])
    s = possession_starts(ev)
    assert s.match_id.tolist() == [1, 1, 2, 2]
    assert s.possession_id.tolist() == [1, 2, 2, 3]
    opening = s.iloc[2]  # match 2's first possession: nothing precedes it, match 1 is not "before"
    assert pd.isna(opening.previous_event_type)
    assert pd.isna(opening.previous_out)
    assert pd.isna(opening.same_team_as_previous)
    # Match 2's second possession follows match 2's own first event, not match 1's out flag.
    assert s.iloc[3].previous_event_type == "Pass" and pd.isna(s.iloc[3].previous_out)
    assert not bool(s.iloc[3].same_team_as_previous)


def test_rows_are_sorted_by_event_index_whatever_the_input_order():
    ev = events_df([
        _row(1, 3, 2, 20, pass_type="Throw-in"),
        _row(1, 1, 1, 10),
        _row(1, 2, 1, 10, out=True),
    ])
    s = possession_starts(ev)
    assert s.possession_id.tolist() == [1, 2]
    assert s.iloc[-1].previous_event_type == "Pass" and bool(s.iloc[-1].previous_out)
    assert not bool(s.iloc[-1].same_team_as_previous)


def test_a_possession_id_that_repeats_later_in_the_match_counts_as_a_new_start():
    ev = events_df([_row(1, 1, 1, 10), _row(1, 2, 2, 20), _row(1, 3, 1, 10)])
    assert possession_starts(ev).possession_id.tolist() == [1, 2, 1]


def test_null_team_gives_a_null_same_team_flag():
    ev = events_df([_row(1, 1, 1, 10), _row(1, 2, 2, None)])
    assert pd.isna(possession_starts(ev).iloc[-1].same_team_as_previous)


def test_flag_possession_starts_keeps_every_event_in_order():
    ev = events_df([
        _row(1, 2, 1, 10, "Carry"),
        _row(1, 1, 1, 10),
        _row(1, 3, 2, 20, pass_type="Throw-in"),
        _row(2, 1, 1, 10),
    ])
    flagged = flag_possession_starts(ev)
    assert len(flagged) == 4
    assert flagged.event_index.tolist() == [1, 2, 3, 1]
    assert flagged.starts_possession.tolist() == [True, False, True, True]
    assert flagged.starts_possession.dtype == bool
    assert list(ev.event_index) == [2, 1, 3, 1]  # the input is not reordered or modified


def test_possession_start_summary_on_a_tiny_frame():
    ev = events_df([
        _row(1, 1, 1, 10),
        _row(1, 2, 2, 10, pass_type="Throw-in"),
        _row(1, 3, 3, 20, pass_type="Throw-in"),
        _row(1, 4, 4, 20, pass_type="Throw-in"),
        _row(1, 5, 5, 10, "Carry"),
    ])
    summary = possession_start_summary(ev)
    assert list(summary.columns) == [
        "first_event_type",
        "first_pass_type",
        "same_team_as_previous",
        "n",
    ]
    assert summary.n.sum() == 5
    counts = {
        tuple(None if pd.isna(v) else v for v in key): n
        for *key, n in summary.itertuples(index=False)
    }
    # Null keys survive: the first possession has no previous team, most starts have no pass type.
    assert counts == {
        ("Pass", None, None): 1,
        ("Pass", "Throw-in", True): 2,
        ("Pass", "Throw-in", False): 1,
        ("Carry", None, False): 1,
    }
