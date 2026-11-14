"""Test helpers shared across the suite; import with ``from helpers import events_df``."""

import pandas as pd

from setpiece.ingest.events import EVENTS_COLUMNS, EVENTS_DTYPES

_CORNER_DEFAULTS: dict = {
    "match_id": 1,
    "competition_id": 9,
    "season_id": 281,
    "match_date": "2023-10-01",
    "event_index": 1,
    "period": 1,
    "timestamp": "00:00:01.000",
    "event_type": "Pass",
    "pass_type": "Corner",
    "pass_height": "High Pass",
    "team_id": 1,
    "team_name": "Team A",
    "home_team_id": 1,
    "away_team_id": 2,
    "possession_id": 1,
    "start_x": 120.0,
    "start_y": 0.1,
    "pass_end_x": 110.0,
    "pass_end_y": 35.0,
    "pass_length": 15.0,
}


def corner_row(
    *,
    event_id: str = "corner-1",
    start_y: float | None = 0.1,
    end: tuple = (110.0, 35.0),
    technique: str | None = "Inswinging",
    **kw,
) -> dict:
    """Return a minimal Pass/Corner event dict with sensible defaults.

    Parameters
    ----------
    event_id:   unique identifier for this corner.
    start_y:    y-coordinate of the corner origin (determines side).
    end:        ``(pass_end_x, pass_end_y)`` tuple.
    technique:  ``pass_technique`` value; ``None`` → Ground Pass style (null technique).
    kw:         any other ``EVENTS_COLUMNS`` overrides.
    """
    row = dict(_CORNER_DEFAULTS)
    row["event_id"] = event_id
    if start_y is None:
        row["start_y"] = None
        row["start_x"] = None
    else:
        row["start_y"] = start_y
        row["start_x"] = 120.0 if start_y < 40 else 0.0
    row["pass_end_x"] = end[0]
    row["pass_end_y"] = end[1]
    if technique is None:
        row["pass_technique"] = None
        row["pass_height"] = "Ground Pass"
    else:
        row["pass_technique"] = technique
        row["pass_height"] = "High Pass"
    row.update(kw)
    return row


def events_df(rows: list[dict]) -> pd.DataFrame:
    """Build an ``events`` frame from partial rows.

    Every ``EVENTS_COLUMNS`` column a row does not give is null, and the frame has the same
    column order and dtypes as ``build_events_frame``. Unknown keys raise, so a typo in a test
    row cannot silently vanish.
    """
    unknown = {key for row in rows for key in row} - set(EVENTS_COLUMNS)
    if unknown:
        raise ValueError(f"not events columns: {sorted(unknown)}")
    full = [{col: row.get(col) for col in EVENTS_COLUMNS} for row in rows]
    return pd.DataFrame(full, columns=EVENTS_COLUMNS).astype(EVENTS_DTYPES)
