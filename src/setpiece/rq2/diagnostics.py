"""Checks read off the ``events`` table around the RQ2 chain rule.

``flag_possession_starts`` and its two summaries show how StatsBomb's ``possession_id`` behaves.
``opponent_action_within`` reproduces the audit's "opposing action within 15 s" count.

A possession starts at the first event of a match, and at every event whose ``possession_id``
differs from that of the event just before it in ``event_index`` order. Matches are handled
separately, so nothing chains across a match boundary. An event with a null ``possession_id``
never starts a possession (the one after it does count as a start, since its id differs).

Nothing is filled in: the first possession of a match has null ``previous_*`` values and a null
``same_team_as_previous``, and a null ``out`` or ``possession_team_id`` stays null.
"""

import numpy as np
import pandas as pd

from setpiece.rq2.chains import iter_match_corners, parse_timestamp

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


def _floats(column: pd.Series) -> np.ndarray:
    """A nullable numeric column as float64, nulls as NaN (ids and periods are far below 2**53)."""
    return column.to_numpy(dtype="float64", na_value=np.nan)


def opponent_action_within(events: pd.DataFrame, seconds: float) -> pd.Series:
    """Per corner pass: does the other team act in the corner's possession within ``seconds``?

    This is the audit's check (``verify.py`` of 2026-11-10, 844 of the 965 corners of the three audit
    slices at 15 s), not the chain rule: stoppages, restarts, new corners and the possession change
    that would end a chain are all ignored. A corner is flagged when some later event of the same
    match has the same ``period`` and ``possession_id``, a ``team_id`` that is not the corner
    taker's, and ``0 <= t(event) - t(corner) <= seconds``. Both ends of the window count; an event
    stamped before the corner is outside it (the elapsed time is not clamped to 0, unlike the
    chain's). Times are those of ``parse_timestamp``. Any null period, possession, team or
    timestamp on either side means no opposing action.

    The result is a plain ``bool`` Series named ``opponent_action_within`` and indexed by the
    corner's ``event_id`` (name ``corner_event_id``), one entry per corner pass, in ``match_id``
    order and within a match in ``event_index`` order: the row order of ``build_rq2_outcomes``.
    """
    flags: list[bool] = []
    ids: list = []
    for match, positions in iter_match_corners(events):
        if len(positions) == 0:
            continue
        period = _floats(match["period"])
        possession = _floats(match["possession_id"])
        team = _floats(match["team_id"])
        elapsed = np.array(
            [parse_timestamp(t) if isinstance(t, str) else np.nan for t in match["timestamp"]]
        )
        for pos in positions:
            later = slice(pos + 1, None)
            delta = elapsed[later] - elapsed[pos]
            hit = (
                (period[later] == period[pos])
                & (possession[later] == possession[pos])
                & (team[later] != team[pos])
                & ~np.isnan(team[later])
                & (delta >= 0)
                & (delta <= seconds)
            )
            flags.append(bool(hit.any()) and not np.isnan(team[pos]))
        ids.extend(match["event_id"].iloc[positions])
    return pd.Series(
        flags,
        index=pd.Index(ids, name="corner_event_id", dtype=object),
        dtype=bool,
        name="opponent_action_within",
    )
