"""Synthetic 360 frames and their corresponding RQ1 corner rows."""

import pandas as pd

from setpiece.ingest.frames import CORNER_360_DTYPES
from setpiece.rq1.corners import RQ1_CORNERS_DTYPES
from setpiece.rq3.features import CORNER_COLUMNS

FULL_CAMERA = [100, 25, 120, 25, 120, 55, 100, 55]
FAR_CAMERA = [100, 40, 120, 40, 120, 55, 100, 55]


def player(x, y, attacking=True, keeper=False):
    return {"x": x, "y": y, "teammate": attacking, "actor": False, "keeper": keeper}


def corner(match_id, event_id, *, side="left", short=False, camera=None, players=None):
    key = {"competition_id": 43, "season_id": 106, "match_id": match_id, "event_id": event_id}
    frame = {
        **key,
        "visible_area": FULL_CAMERA if camera is None else camera,
        "players": [] if players is None else players,
    }
    metadata = {
        **key,
        "team_id": match_id % 2 + 1,
        "team_name": f"Team {match_id % 2 + 1}",
        "corner_side": side,
        "short_corner_proxy": short,
        "pass_height": "Ground Pass" if short else "High Pass",
    }
    return frame, metadata


def tables(pairs):
    frames, corners = zip(*pairs) if pairs else ([], [])
    return (
        pd.DataFrame(frames, columns=CORNER_360_DTYPES).astype(CORNER_360_DTYPES),
        pd.DataFrame(corners, columns=CORNER_COLUMNS).astype(
            {name: RQ1_CORNERS_DTYPES[name] for name in CORNER_COLUMNS}
        ),
    )


def two_setups():
    """Every match contains one corner of each of two distinct arrangements."""
    return tables(
        [
            corner(
                match,
                f"{match}-{setup}",
                players=[
                    player(117 if setup == "a" else 109, 36 if setup == "a" else 44),
                    player(116 if setup == "a" else 107, 37 if setup == "a" else 45),
                    player(118, 38, attacking=False, keeper=True),
                ],
            )
            for match in range(1, 9)
            for setup in ("a", "b")
        ]
    )
