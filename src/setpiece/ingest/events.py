"""Flatten raw StatsBomb events into the shared ``events`` table (spec section 5.4).

Null means "not supplied by the source": nothing is filled in, and the legacy ``pass.inswinging``
flag is never used to fill ``pass_technique``.
"""

import json
from typing import Any

import pandas as pd

# Column -> dtype, in the table order of spec section 5.4.
EVENTS_DTYPES: dict[str, str] = {
    # Match context
    "match_id": "Int64",
    "match_date": "string",
    "competition_id": "Int64",
    "competition_name": "string",
    "season_id": "Int64",
    "season_name": "string",
    "home_team_id": "Int64",
    "home_team_name": "string",
    "away_team_id": "Int64",
    "away_team_name": "string",
    # Identity and order
    "event_id": "string",
    "event_index": "Int64",
    "period": "Int64",
    "timestamp": "string",
    "minute": "Int64",
    "second": "Int64",
    "duration": "Float64",
    "event_type": "string",
    # Actor and possession
    "team_id": "Int64",
    "team_name": "string",
    "player_id": "Int64",
    "player_name": "string",
    "possession_id": "Int64",
    "possession_team_id": "Int64",
    "play_pattern": "string",
    "related_event_ids": "object",  # list of event_id strings, or None
    # Position and stoppage
    "start_x": "Float64",
    "start_y": "Float64",
    "out": "boolean",
    "foul_advantage": "boolean",
    # Pass
    "pass_type": "string",
    "pass_end_x": "Float64",
    "pass_end_y": "Float64",
    "pass_length": "Float64",
    "pass_angle": "Float64",
    "pass_height": "string",
    "pass_technique": "string",
    "pass_body_part": "string",
    "pass_outcome": "string",
    "pass_recipient_id": "Int64",
    # Shot
    "shot_type": "string",
    "shot_outcome": "string",
    # Remainder
    "event_details": "string",  # JSON dump of the unmapped source keys, or null
}
EVENTS_COLUMNS: list[str] = list(EVENTS_DTYPES)

# Top-level raw keys that have a column (``type.id`` and the like go with them).
_MAPPED_TOP_LEVEL = frozenset(
    {
        "id", "index", "period", "timestamp", "minute", "second", "duration", "type", "team",
        "player", "possession", "possession_team", "play_pattern", "related_events", "location",
        "out",
    }
)  # fmt: skip

# Type-specific dict -> its keys that have a column. A mapped key is consumed whole.
_MAPPED_SUB_KEYS = {
    "pass": frozenset(
        {"recipient", "length", "angle", "height", "end_location", "type", "technique",
         "body_part", "outcome"}
    ),
    "shot": frozenset({"type", "outcome"}),
    "foul_committed": frozenset({"advantage"}),
    "foul_won": frozenset({"advantage"}),
}  # fmt: skip


def _dig(obj: Any, *path: str) -> Any:
    """Follow ``path`` through nested dicts; None as soon as a step is absent."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _coord(point: Any, i: int) -> Any:
    """``point[i]`` of a coordinate list, None if the list is absent or too short."""
    return point[i] if isinstance(point, (list, tuple)) and len(point) > i else None


def _event_details(raw: dict) -> str | None:
    """JSON dump of every raw key that no column holds, or None if nothing remains."""
    rest = {}
    for key, value in raw.items():
        if key in _MAPPED_TOP_LEVEL:
            continue
        mapped = _MAPPED_SUB_KEYS.get(key)
        if mapped is not None and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in mapped}
            if not value:
                continue
        rest[key] = value
    return json.dumps(rest, sort_keys=True) if rest else None


def match_context(match: dict) -> dict:
    """The 10 match-context columns, taken from one raw match record."""
    return {
        "match_id": match.get("match_id"),
        "match_date": match.get("match_date"),
        "competition_id": _dig(match, "competition", "competition_id"),
        "competition_name": _dig(match, "competition", "competition_name"),
        "season_id": _dig(match, "season", "season_id"),
        "season_name": _dig(match, "season", "season_name"),
        "home_team_id": _dig(match, "home_team", "home_team_id"),
        "home_team_name": _dig(match, "home_team", "home_team_name"),
        "away_team_id": _dig(match, "away_team", "away_team_id"),
        "away_team_name": _dig(match, "away_team", "away_team_name"),
    }


def flatten_event(raw: dict, ctx: dict) -> dict:
    """One raw event plus its match context as a row with every ``EVENTS_COLUMNS`` key."""
    advantage = _dig(raw, "foul_committed", "advantage")
    if advantage is None:
        advantage = _dig(raw, "foul_won", "advantage")
    related = raw.get("related_events")
    return {
        **ctx,
        "event_id": raw.get("id"),
        "event_index": raw.get("index"),
        "period": raw.get("period"),
        "timestamp": raw.get("timestamp"),
        "minute": raw.get("minute"),
        "second": raw.get("second"),
        "duration": raw.get("duration"),
        "event_type": _dig(raw, "type", "name"),
        "team_id": _dig(raw, "team", "id"),
        "team_name": _dig(raw, "team", "name"),
        "player_id": _dig(raw, "player", "id"),
        "player_name": _dig(raw, "player", "name"),
        "possession_id": raw.get("possession"),
        "possession_team_id": _dig(raw, "possession_team", "id"),
        "play_pattern": _dig(raw, "play_pattern", "name"),
        "related_event_ids": list(related) if related is not None else None,
        "start_x": _coord(raw.get("location"), 0),
        "start_y": _coord(raw.get("location"), 1),
        "out": raw.get("out"),
        "foul_advantage": advantage,
        "pass_type": _dig(raw, "pass", "type", "name"),
        "pass_end_x": _coord(_dig(raw, "pass", "end_location"), 0),
        "pass_end_y": _coord(_dig(raw, "pass", "end_location"), 1),
        "pass_length": _dig(raw, "pass", "length"),
        "pass_angle": _dig(raw, "pass", "angle"),
        "pass_height": _dig(raw, "pass", "height", "name"),
        "pass_technique": _dig(raw, "pass", "technique", "name"),
        "pass_body_part": _dig(raw, "pass", "body_part", "name"),
        "pass_outcome": _dig(raw, "pass", "outcome", "name"),
        "pass_recipient_id": _dig(raw, "pass", "recipient", "id"),
        "shot_type": _dig(raw, "shot", "type", "name"),
        "shot_outcome": _dig(raw, "shot", "outcome", "name"),
        "event_details": _event_details(raw),
    }


def build_events_frame(raw_events: list[dict], match: dict) -> pd.DataFrame:
    """The ``events`` table for one match: ``EVENTS_COLUMNS``, sorted by ``event_index``."""
    ctx = match_context(match)
    rows = [flatten_event(raw, ctx) for raw in raw_events]
    frame = pd.DataFrame(rows, columns=EVENTS_COLUMNS).astype(EVENTS_DTYPES)
    return frame.sort_values("event_index", kind="stable").reset_index(drop=True)
