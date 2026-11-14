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


def team_sample_table(
    events: pd.DataFrame,
    slices: list[Slice],
) -> pd.DataFrame:
    """One row per (competition_id, season_id, team_id) in *events*.

    ``matches_in_sample`` is the number of **distinct match_ids** in which a
    team appears as home **or** away.  It uses the ``home_team_id`` and
    ``away_team_id`` columns from the events table (match-context columns
    carried through the ingest stage).

    Returns a DataFrame with columns:
        competition_id, season_id, team_id, team_name,
        matches_in_sample, sample_label
    """
    if events.empty:
        return pd.DataFrame(
            columns=[
                "competition_id",
                "season_id",
                "team_id",
                "team_name",
                "matches_in_sample",
                "sample_label",
            ]
        )

    # Build a lookup: (competition_id, season_id) -> Slice
    slice_map: dict[tuple[int, int], Slice] = {
        (s.competition_id, s.season_id): s for s in slices
    }

    # For each match_id we need to know the home and away team.
    # Drop duplicates so each match contributes exactly one row.
    match_cols = ["match_id", "competition_id", "season_id", "home_team_id", "away_team_id"]
    # Only keep columns that actually exist (pre-Task-4 helpers may omit some)
    present = [c for c in match_cols if c in events.columns]
    matches = events[present].drop_duplicates(subset=["match_id"])

    rows = []
    for (comp_id, seas_id), grp in matches.groupby(["competition_id", "season_id"]):
        sl = slice_map.get((int(comp_id), int(seas_id)))

        # Collect (team_id, team_name) pairs from home and away columns.
        # team_name is optional — fall back to empty string when absent.
        teams: dict[int, str] = {}

        if "home_team_id" in grp.columns:
            for _, mrow in grp.iterrows():
                htid = mrow.get("home_team_id", pd.NA)
                if pd.notna(htid):
                    teams.setdefault(int(htid), "")
        if "away_team_id" in grp.columns:
            for _, mrow in grp.iterrows():
                atid = mrow.get("away_team_id", pd.NA)
                if pd.notna(atid):
                    teams.setdefault(int(atid), "")

        # Resolve team_name from the events rows where team_id matches.
        if "team_name" in events.columns and "team_id" in events.columns:
            slice_ev = events[
                (events["competition_id"] == comp_id)
                & (events["season_id"] == seas_id)
            ]
            for tid in list(teams.keys()):
                name_series = slice_ev.loc[
                    slice_ev["team_id"] == tid, "team_name"
                ].dropna()
                if not name_series.empty:
                    teams[tid] = str(name_series.iloc[0])

        # Count distinct matches per team (home or away).
        match_count_per_team: dict[int, set] = {tid: set() for tid in teams}
        for _, mrow in grp.iterrows():
            mid = mrow.get("match_id")
            htid = mrow.get("home_team_id", pd.NA)
            atid = mrow.get("away_team_id", pd.NA)
            if pd.notna(htid) and int(htid) in match_count_per_team:
                match_count_per_team[int(htid)].add(mid)
            if pd.notna(atid) and int(atid) in match_count_per_team:
                match_count_per_team[int(atid)].add(mid)

        for tid, match_set in match_count_per_team.items():
            n = len(match_set)
            lbl = sample_label(
                n,
                sl.kind if sl else "league",
                sl.expected_matches if sl else None,
            )
            rows.append(
                {
                    "competition_id": int(comp_id),
                    "season_id": int(seas_id),
                    "team_id": tid,
                    "team_name": teams[tid],
                    "matches_in_sample": n,
                    "sample_label": lbl,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "competition_id",
                "season_id",
                "team_id",
                "team_name",
                "matches_in_sample",
                "sample_label",
            ]
        )

    return pd.DataFrame(rows)
