"""Attach each corner kick's 360 freeze-frame to the corner (spec section 5.4, ``corner_360``).

The join is exact: a frame's ``event_uuid`` must equal the corner's ``event_id``. Nothing is ever
matched on time. Coordinates stay raw (never mirrored) and a missing value stays null.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

# Column -> dtype, in the table order of spec section 5.4.
CORNER_360_DTYPES: dict[str, str] = {
    "match_id": "Int64",
    "event_id": "string",
    "competition_id": "Int64",
    "season_id": "Int64",
    "visible_area": "object",  # flat list of floats (x1, y1, x2, y2, ...), or None
    "players": "object",  # list of {x, y, teammate, actor, keeper} dicts, or None
}
CORNER_360_COLUMNS: list[str] = list(CORNER_360_DTYPES)

_KEY_COLUMNS = ["match_id", "event_id", "competition_id", "season_id"]


@dataclass(frozen=True)
class FrameStats:
    """What one match's 360 file contributed; Task 7 records it as a QA side output."""

    match_id: int
    n_frames: int  # records in the 360 file
    n_unmatched: int  # records whose event_uuid is not among the match's event ids
    n_corners: int  # corner passes in the match's events
    n_corners_matched: int  # corner passes that got a frame (rows of corner_360)
    has_file: bool  # False when the match has no 360 file


def _as_float(value: Any) -> float | None:
    """``value`` as a float (same number, raw units), None if it is absent."""
    return None if value is None else float(value)


def _coord(point: Any, i: int) -> float | None:
    """``point[i]`` of a coordinate list as a float, None if the list is absent or too short."""
    return _as_float(point[i]) if isinstance(point, (list, tuple)) and len(point) > i else None


def _player(entry: dict) -> dict:
    location = entry.get("location")
    return {
        "x": _coord(location, 0),
        "y": _coord(location, 1),
        "teammate": entry.get("teammate"),
        "actor": entry.get("actor"),
        "keeper": entry.get("keeper"),
    }


def _visible_area(frame: dict) -> list[float] | None:
    area = frame.get("visible_area")
    return None if area is None else [float(v) for v in area]


def _players(frame: dict) -> list[dict] | None:
    freeze_frame = frame.get("freeze_frame")
    return None if freeze_frame is None else [_player(entry) for entry in freeze_frame]


def _frames_by_corner_uuid(frames: list[dict], corner_ids: set[str]) -> dict[str, dict]:
    """The frames whose ``event_uuid`` is a corner's ``event_id``, keyed by that uuid.

    Two frames for one corner make the join ambiguous, so that raises rather than picking one.
    Frames with no ``event_uuid``, or whose uuid is not a corner's, never join, and repeats among
    them are of no consequence.
    """
    by_uuid: dict[str, dict] = {}
    for frame in frames:
        uuid = frame.get("event_uuid")
        if uuid is None or uuid not in corner_ids:
            continue
        if uuid in by_uuid:
            raise ValueError(f"360 file has more than one frame for corner event_uuid {uuid!r}")
        by_uuid[uuid] = frame
    return by_uuid


def build_corner_360(
    frames: list[dict] | None, events: pd.DataFrame
) -> tuple[pd.DataFrame, FrameStats]:
    """The ``corner_360`` table of one match, with the stats of the join.

    ``frames`` is the match's raw 360 file (None if the match has none) and ``events`` is that
    match's ``events`` table. A row is a corner pass (``event_type == "Pass"`` and
    ``pass_type == "Corner"``) with a frame whose ``event_uuid`` equals its ``event_id``; rows
    follow ``event_index``. A corner without a frame has no row. More than one frame for the same
    corner raises ``ValueError``; repeated uuids of any other frame are just counted.

    ``n_frames`` and ``n_unmatched`` count records: a frame whose ``event_uuid`` is not among the
    match's event ids, or that has none, is unmatched, and so is each repeat of such a frame.
    """
    match_ids = events["match_id"].dropna().unique()
    if len(match_ids) != 1:
        raise ValueError(f"events must hold exactly one match, got match_ids {list(match_ids)}")

    is_corner = (events["event_type"] == "Pass") & (events["pass_type"] == "Corner")
    corners = events[is_corner.to_numpy(dtype=bool, na_value=False)]
    corners = corners.sort_values("event_index", kind="stable")

    all_frames = frames or []
    by_uuid = _frames_by_corner_uuid(all_frames, set(corners["event_id"].dropna()))
    event_ids = set(events["event_id"].dropna())
    matched = corners[corners["event_id"].isin(list(by_uuid))]

    table = matched[_KEY_COLUMNS].reset_index(drop=True)
    matched_frames = [by_uuid[event_id] for event_id in table["event_id"]]
    table["visible_area"] = pd.Series([_visible_area(f) for f in matched_frames], dtype=object)
    table["players"] = pd.Series([_players(f) for f in matched_frames], dtype=object)

    stats = FrameStats(
        match_id=int(match_ids[0]),
        n_frames=len(all_frames),
        n_unmatched=sum(1 for f in all_frames if f.get("event_uuid") not in event_ids),
        n_corners=len(corners),
        n_corners_matched=len(table),
        has_file=frames is not None,
    )
    return table.astype(CORNER_360_DTYPES), stats
