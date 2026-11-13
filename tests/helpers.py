"""Test helpers shared across the suite; import with ``from helpers import events_df``."""

import pandas as pd

from setpiece.ingest.events import EVENTS_COLUMNS, EVENTS_DTYPES


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
