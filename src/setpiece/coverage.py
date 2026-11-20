"""Sample labels and team-season sample tables (Task 11).

``sample_label`` produces a human-readable string that describes how much of a
team's season is covered by the corpus slice.  It reads ``expected_matches``
from the slice config — never from counting distinct teams in the data.

``team_sample_table`` builds one row per (team-season) in the events table,
counting each team's distinct matches and attaching the label.
"""

from __future__ import annotations

import pandas as pd

from setpiece.config import Slice


def sample_label(
    matches_in_sample: int,
    kind: str,
    expected_matches: int | None,
) -> str:
    """Human-readable coverage label for one team-season.

    Parameters
    ----------
    matches_in_sample:
        Number of distinct matches the team appears in within the slice.
    kind:
        ``"league"`` or ``"tournament"``.
    expected_matches:
        The number of fixtures a team plays in a full season of this league
        (from the slice config, **not** derived from the data).  ``None`` for
        tournaments and for leagues whose season length hasn't been verified.
    """
    if kind == "tournament":
        return "full_tournament"
    # League
    if expected_matches is None:
        return f"season_length_unknown ({matches_in_sample} matches)"
    if matches_in_sample >= expected_matches:
        return "complete_season"
    return f"incomplete_season ({matches_in_sample} of {expected_matches})"


TEAM_SAMPLE_COLUMNS: list[str] = [
    "competition_id",
    "season_id",
    "team_id",
    "team_name",
    "matches_in_sample",
    "sample_label",
]
TEAM_SAMPLE_DTYPES: dict[str, str] = {
    "competition_id": "Int64",
    "season_id": "Int64",
    "team_id": "Int64",
    "team_name": "string",
    "matches_in_sample": "Int64",
    "sample_label": "string",
}
_SLICE = ["competition_id", "season_id"]


def team_sample_table(
    events: pd.DataFrame,
    slices: list[Slice],
) -> pd.DataFrame:
    """One row per (competition_id, season_id, team_id) in *events*.

    ``matches_in_sample`` is the number of distinct match_ids in which a team appears as home or
    away (``home_team_id`` / ``away_team_id``, the match-context columns of the events table).
    ``team_name`` is null for a team whose name no event carries. A slice of *events* that is not
    in *slices* raises ``ValueError``: its season length is unknown, and guessing "league" would
    publish a wrong label.

    Columns: competition_id, season_id, team_id, team_name, matches_in_sample, sample_label.
    """
    if events.empty:
        return pd.DataFrame(columns=TEAM_SAMPLE_COLUMNS).astype(TEAM_SAMPLE_DTYPES)

    matches = events[["match_id", *_SLICE, "home_team_id", "away_team_id"]].drop_duplicates(
        "match_id"
    )
    sides = [
        matches[["match_id", *_SLICE]].assign(team_id=matches[col])
        for col in ("home_team_id", "away_team_id")
    ]
    long = pd.concat(sides).dropna(subset=["team_id"])
    counts = (
        long.groupby([*_SLICE, "team_id"])["match_id"]
        .nunique()
        .rename("matches_in_sample")
        .reset_index()
    )

    names = events[[*_SLICE, "team_id", "team_name"]]
    names = names.dropna(subset=["team_id", "team_name"]).drop_duplicates([*_SLICE, "team_id"])
    table = counts.merge(names, how="left", on=[*_SLICE, "team_id"])

    slice_map = {(s.competition_id, s.season_id): s for s in slices}
    unknown = {
        (int(c), int(s)) for c, s in zip(table["competition_id"], table["season_id"])
    } - set(slice_map)
    if unknown:
        raise ValueError(f"events from slices that are not in the config: {sorted(unknown)}")
    table["sample_label"] = [
        sample_label(
            int(n),
            slice_map[int(c), int(s)].kind,
            slice_map[int(c), int(s)].expected_matches,
        )
        for c, s, n in zip(table["competition_id"], table["season_id"], table["matches_in_sample"])
    ]
    table = table.sort_values([*_SLICE, "team_id"]).reset_index(drop=True)
    return table[TEAM_SAMPLE_COLUMNS].astype(TEAM_SAMPLE_DTYPES)