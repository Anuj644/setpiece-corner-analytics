"""How StatsBomb's ``possession_id`` behaves, read off the ``events`` table before the chain rule.

A possession starts at the first event of a match, and at every event whose ``possession_id``
differs from that of the event just before it in ``event_index`` order. Matches are handled
separately, so nothing chains across a match boundary. An event with a null ``possession_id``
never starts a possession (the one after it does count as a start, since its id differs).

Nothing is filled in: the first possession of a match has null ``previous_*`` values and a null
``same_team_as_previous``, and a null ``out`` or ``possession_team_id`` stays null.
"""

import pandas as pd

START_COLUMNS = [
    "match_id",
    "possession_id",
    "first_event_type",
    "first_pass_type",
    "previous_event_type",
    "previous_out",
    "same_team_as_previous",
]
SUMMARY_KEYS = ["first_event_type", "first_pass_type", "same_team_as_previous"]


def flag_possession_starts(events: pd.DataFrame) -> pd.DataFrame:
    """Every event, sorted by ``match_id`` and ``event_index``, with four columns added.

    * ``starts_possession`` (plain bool): the event opens a possession, as defined above;
    * ``previous_event_type`` and ``previous_out``: those of the previous event in the same match;
    * ``same_team_as_previous``: whether ``possession_team_id`` equals the previous event's.

    The last three are null for the first event of a match, and ``same_team_as_previous`` is also
    null when either team is null. ``events`` itself is not modified.
    """
    ev = events.sort_values(["match_id", "event_index"], kind="stable").reset_index(drop=True)
    by_match = ev.groupby("match_id", sort=False, dropna=False)
    first_in_match = by_match.cumcount() == 0
    changed = (ev["possession_id"] != by_match["possession_id"].shift()).fillna(True)
    return ev.assign(
        starts_possession=(first_in_match | changed.astype(bool)) & ev["possession_id"].notna(),
        previous_event_type=by_match["event_type"].shift(),
        previous_out=by_match["out"].shift(),
        same_team_as_previous=ev["possession_team_id"] == by_match["possession_team_id"].shift(),
    )


def possession_starts(events: pd.DataFrame) -> pd.DataFrame:
    """One row per new possession within each match, in ``event_index`` order.

    Columns: ``match_id``, ``possession_id``, ``first_event_type``, ``first_pass_type`` (the event
    type and pass type of the possession's first event), ``previous_event_type``, ``previous_out``
    (the event before it, within the match) and ``same_team_as_previous`` (does the possession
    team equal the previous event's). The first possession of a match has nulls in the last three.
    """
    flagged = flag_possession_starts(events)
    starts = flagged[flagged["starts_possession"]].rename(
        columns={"event_type": "first_event_type", "pass_type": "first_pass_type"}
    )
    return starts[START_COLUMNS].reset_index(drop=True)


def possession_start_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Count of new possessions (column ``n``) by first event type, first pass type and team flag.

    Null keys are kept as groups: most starts have no pass type, and each match's first
    possession has no ``same_team_as_previous``.
    """
    starts = possession_starts(events)
    return starts.groupby(SUMMARY_KEYS, dropna=False).size().reset_index(name="n")
