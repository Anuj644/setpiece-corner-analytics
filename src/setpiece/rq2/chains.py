"""The RQ2 chain rule (spec section 8.1): how the phase after one corner ends.

``build_chain`` walks forward from a corner through the events of one match, in ``event_index``
order, and stops at the first ending. It reads only the rows from the corner onward, a few columns
at a time, so its cost depends on the length of the chain and not on the size of the match.

Values read out of the nullable ``events`` dtypes are normalised once, at the row boundary: any
null (``pd.NA``, NaN, None) becomes None and numpy scalars become Python ones, so the rule is
written against plain values.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from setpiece.config import Config

CORNER = "Corner"
HALF_END = "Half End"

# The columns the walk reads, in the field order of ``_Row``.
_COLUMNS = (
    "event_id", "period", "timestamp", "event_type", "team_id", "possession_id",
    "possession_team_id", "out", "foul_advantage", "pass_type", "shot_type", "shot_outcome",
    "event_details",
)  # fmt: skip


class _Row(NamedTuple):
    event_id: str | None
    period: int | None
    timestamp: str | None
    event_type: str | None
    team_id: int | None
    possession_id: int | None
    possession_team_id: int | None
    out: bool | None
    foul_advantage: bool | None
    pass_type: str | None
    shot_type: str | None
    shot_outcome: str | None
    event_details: str | None


def _scalar(value: Any) -> Any:
    """``value`` as a plain Python scalar, or None for any kind of null."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    return None if pd.isna(value) else value


def parse_timestamp(ts: str) -> float:
    """Seconds in ``"HH:MM:SS.fff"`` form."""
    hours, minutes, seconds = ts.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


@dataclass
class Restarts:
    """The event kinds that restart play after a dead ball, from the ``restart_*`` config params."""

    pass_types: list[str]
    shot_types: list[str]
    event_types: list[str]

    @classmethod
    def from_config(cls, cfg: Config) -> Restarts:
        return cls(
            list(cfg.p("restart_pass_types")),
            list(cfg.p("restart_shot_types")),
            list(cfg.p("restart_event_types")),
        )

    def match(self, row: Any) -> str | None:
        """The restart's type name, or None if ``row`` (a row, or a ``Series``) is not a restart.

        A restart is a Pass whose ``pass_type`` is in ``pass_types``, a Shot whose ``shot_type`` is
        in ``shot_types``, or an event whose ``event_type`` is in ``event_types``.
        """
        event_type = _scalar(row.event_type)
        if event_type == "Pass":
            pass_type = _scalar(row.pass_type)
            if pass_type is not None and pass_type in self.pass_types:
                return pass_type
        elif event_type == "Shot":
            shot_type = _scalar(row.shot_type)
            if shot_type is not None and shot_type in self.shot_types:
                return shot_type
        if event_type is not None and event_type in self.event_types:
            return event_type
        return None


@dataclass
class ChainResult:
    """The ``rq2_outcomes`` fields of spec 8.2 other than ``match_id`` and ``corner_event_id``."""

    last_included_event_id: str
    included_event_ids: list[str]  # the corner itself is not one of them
    n_events: int
    sequence_duration_seconds: float | None  # null when the last included event has no time
    end_reason: str
    restart_type: str | None
    restart_team: str | None  # "attacking" or "defending"
    shot_count: int  # attacking-team shots only
    first_shot_event_id: str | None
    goal: bool
    goal_event_id: str | None
    lost_possession: bool  # an open-play turnover, never a restart
    penalty_awarded: bool
    opponent_action_count: int
    outcome_status: str  # "complete" or "incomplete"


def _rows_from(match_events: pd.DataFrame, start: int) -> Iterator[_Row]:
    """The rows from position ``start`` on, built one at a time (nothing past a stop is read)."""
    if start < 0:
        raise IndexError(f"corner_pos {start} is negative")
    tail = match_events.iloc[start:]
    for values in zip(*(tail[column] for column in _COLUMNS)):
        yield _Row(*map(_scalar, values))


def _is_corner_pass(e: _Row) -> bool:
    return e.event_type == "Pass" and e.pass_type == CORNER


def _is_attacking(e: _Row, attacking: int | None) -> bool:
    return attacking is not None and e.team_id == attacking


def _side(e: _Row, attacking: int | None) -> str | None:
    """``attacking`` or ``defending`` from ``e.team_id``; None if either team is unknown."""
    if e.team_id is None or attacking is None:
        return None
    return "attacking" if e.team_id == attacking else "defending"


def _elapsed(e: _Row, corner_seconds: float | None) -> float | None:
    """Seconds from the corner to ``e``, a step backwards counting as 0; None if a time is null."""
    if corner_seconds is None or e.timestamp is None:
        return None
    return max(0.0, parse_timestamp(e.timestamp) - corner_seconds)


def _penalty_flag(e: _Row, attacking: int | None) -> bool:
    """A defending-team foul with ``foul_committed.penalty``, or an attacking one with ``foul_won``."""
    if e.event_details is None or e.team_id is None or attacking is None:
        return False
    if e.event_type == "Foul Committed" and e.team_id != attacking:
        key = "foul_committed"
    elif e.event_type == "Foul Won" and e.team_id == attacking:
        key = "foul_won"
    else:
        return False
    details = json.loads(e.event_details)
    block = details.get(key) if isinstance(details, dict) else None
    return isinstance(block, dict) and block.get("penalty") is True


def build_chain(
    match_events: pd.DataFrame, corner_pos: int, cap_seconds: float | None, restarts: Restarts
) -> ChainResult:
    """How the phase after the corner at position ``corner_pos`` ends (spec 8.1).

    ``match_events`` is one match sorted by ``event_index`` and ``corner_pos`` the positional index
    of the corner in it. ``cap_seconds=None`` means no time cap. The checks run in the order of the
    spec: period end, new corner, restart, possession change, time cap; then the included event may
    end the chain as a goal or a stoppage.
    """
    rows = _rows_from(match_events, corner_pos)
    corner = next(rows, None)
    if corner is None:
        raise IndexError(f"corner_pos {corner_pos} is past the end of the match")
    attacking = corner.team_id
    corner_seconds = parse_timestamp(corner.timestamp) if corner.timestamp is not None else None
    # Without a possession_id on the corner, no event can be said to change possession; without
    # its team_id the attacking side is unknown, and without its period the end of the period is.
    incomplete = corner.possession_id is None or attacking is None or corner.period is None

    included: list[_Row] = []
    last_elapsed: float | None = 0.0
    end_reason = "timeline_end"
    excluded: _Row | None = None  # the event that ended the chain, if it was left out
    for e in rows:
        if e.period != corner.period or e.event_type == HALF_END:
            end_reason, excluded = "period_end", e
            break
        if _is_corner_pass(e):
            end_reason, excluded = "new_corner", e
            break
        if restarts.match(e) is not None:  # before the possession check: restarts open a possession
            end_reason, excluded = "restart", e
            break
        if (
            corner.possession_id is not None
            and e.possession_id is not None
            and e.possession_id != corner.possession_id
        ):
            end_reason, excluded = "possession_change", e
            break
        elapsed = _elapsed(e, corner_seconds)
        if cap_seconds is not None and elapsed is not None and elapsed > cap_seconds:
            end_reason, excluded = "time_cap", e
            break

        included.append(e)
        last_elapsed = elapsed
        if e.possession_id is None or elapsed is None:
            incomplete = True
        if _is_attacking(e, attacking) and (
            (e.event_type == "Shot" and e.shot_outcome == "Goal") or e.event_type == "Own Goal For"
        ):
            end_reason = "goal"
            break
        if (
            e.out
            or (e.event_type == "Foul Committed" and not e.foul_advantage)
            or e.event_type == "Offside"
        ):
            end_reason = "stoppage"
            break
    if end_reason == "timeline_end":
        incomplete = True

    # How play resumes, for a dead-ball ending only.
    restart_type = restart_team = None
    if end_reason == "restart":
        restart_type, restart_team = restarts.match(excluded), _side(excluded, attacking)
    elif end_reason == "new_corner":
        restart_type, restart_team = CORNER, _side(excluded, attacking)
    elif end_reason == "stoppage":
        for e in rows:  # the events after the stoppage; no time cap, same period
            if e.period != corner.period or e.event_type == HALF_END:
                break
            kind = CORNER if _is_corner_pass(e) else restarts.match(e)
            if kind is not None:
                restart_type, restart_team = kind, _side(e, attacking)
                break

    shots = [e for e in included if e.event_type == "Shot" and _is_attacking(e, attacking)]
    goal = end_reason == "goal"
    return ChainResult(
        last_included_event_id=included[-1].event_id if included else corner.event_id,
        included_event_ids=[e.event_id for e in included],
        n_events=len(included),
        sequence_duration_seconds=last_elapsed if included else 0.0,
        end_reason=end_reason,
        restart_type=restart_type,
        restart_team=restart_team,
        shot_count=len(shots),
        first_shot_event_id=shots[0].event_id if shots else None,
        goal=goal,
        goal_event_id=included[-1].event_id if goal else None,
        lost_possession=(
            end_reason == "possession_change"
            and excluded.possession_team_id is not None
            and attacking is not None
            and excluded.possession_team_id != attacking
        ),
        penalty_awarded=any(_penalty_flag(e, attacking) for e in included),
        opponent_action_count=sum(
            1 for e in included if e.team_id is not None and e.team_id != attacking
        ),
        outcome_status="incomplete" if incomplete else "complete",
    )


def restarts_from(cfg: Config) -> Restarts:
    """The restart kinds named by the ``restart_*`` params of ``cfg`` (see ``Restarts.from_config``)."""
    return Restarts.from_config(cfg)


# --- the rq2_outcomes table (spec 8.2) -----------------------------------------------------------

RQ2_COLUMNS = [
    "match_id",
    "corner_event_id",
    *(field.name for field in dataclasses.fields(ChainResult)),
    "competition_id",
    "season_id",
    "team_id",
]
RQ2_DTYPES: dict[str, str] = {
    "match_id": "Int64",
    "corner_event_id": "string",
    "last_included_event_id": "string",
    "included_event_ids": "object",  # list of event_id strings
    "n_events": "Int64",
    "sequence_duration_seconds": "Float64",
    "end_reason": "string",
    "restart_type": "string",
    "restart_team": "string",
    "shot_count": "Int64",
    "first_shot_event_id": "string",
    "goal": "boolean",
    "goal_event_id": "string",
    "lost_possession": "boolean",
    "penalty_awarded": "boolean",
    "opponent_action_count": "Int64",
    "outcome_status": "string",
    "competition_id": "Int64",
    "season_id": "Int64",
    "team_id": "Int64",  # the corner taker
}
assert list(RQ2_DTYPES) == RQ2_COLUMNS


def iter_match_corners(events: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Each match of ``events`` in ``match_id`` order, with the positions of its corner passes.

    The match is sorted by ``event_index`` (stable, nulls last) and given a fresh ``RangeIndex``, so
    the positions index it directly, as ``build_chain`` expects. A corner pass is a Pass whose
    ``pass_type`` is ``Corner``; a match without one gives an empty position array. A null
    ``match_id`` is a match of its own. ``events`` itself is not modified.
    """
    for _, match in events.groupby("match_id", sort=True, dropna=False):
        match = match.sort_values("event_index", kind="stable").reset_index(drop=True)
        corners = ((match["event_type"] == "Pass") & (match["pass_type"] == CORNER)).fillna(False)
        yield match, np.flatnonzero(corners.to_numpy(dtype=bool))


def build_rq2_outcomes(
    events: pd.DataFrame, cap_seconds: float | None, restarts: Restarts
) -> pd.DataFrame:
    """One row per corner pass of ``events``: how its phase ends (``RQ2_COLUMNS``, ``RQ2_DTYPES``).

    The matches are handled one at a time, in ``match_id`` order and within a match in
    ``event_index`` order, so a chain never reaches into another match. ``competition_id``,
    ``season_id`` and ``team_id`` (the corner taker) are those of the corner row. Nulls stay null.
    ``cap_seconds=None`` means no time cap. No corners gives an empty frame with the columns.
    """
    columns: dict[str, list] = {name: [] for name in RQ2_COLUMNS}
    chain_fields = [field.name for field in dataclasses.fields(ChainResult)]
    for match, positions in iter_match_corners(events):
        corners = match.iloc[positions]
        for pos, *ids in zip(
            positions, corners["match_id"], corners["event_id"], corners["competition_id"],
            corners["season_id"], corners["team_id"],
        ):  # fmt: skip
            match_id, event_id, competition_id, season_id, team_id = map(_scalar, ids)
            result = build_chain(match, int(pos), cap_seconds, restarts)
            columns["match_id"].append(match_id)
            columns["corner_event_id"].append(event_id)
            for name in chain_fields:
                columns[name].append(getattr(result, name))
            columns["competition_id"].append(competition_id)
            columns["season_id"].append(season_id)
            columns["team_id"].append(team_id)
    return pd.DataFrame(
        {name: pd.Series(columns[name], dtype=RQ2_DTYPES[name]) for name in RQ2_COLUMNS}
    )
