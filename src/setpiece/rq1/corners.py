"""Derive ``rq1_corners`` from the shared ``events`` table (Task 12, spec §6.1–6.2).

All geometry follows the StatsBomb coordinate system (x: 0–120, y: 0–80).
Mirroring normalises right-side corners to the left-side reference frame so
zone labels are independent of which end the corner was taken from.

Boundary rules (§6.2):
  box lines count as inside (≥ / ≤, never strict).
  Width bands: near  18 ≤ y′ < 30
               central 30 ≤ y′ ≤ 50
               far    50 < y′ ≤ 62
  Six-yard:    x ≥ six_yard_x_min  (114)
  Edge of box: edge_of_box_x_min ≤ x < box_x_min  (90 ≤ x < 102)  AND  18 ≤ y′ ≤ 62
"""

from __future__ import annotations

import pandas as pd

from setpiece.config import Config

RQ1_CORNERS_COLUMNS: list[str] = [
    "match_id",
    "event_id",
    "competition_id",
    "season_id",
    "match_date",
    "team_id",
    "team_name",
    "opponent_team_id",
    "period",
    "timestamp",
    "possession_id",
    "start_x",
    "start_y",
    "pass_end_x",
    "pass_end_y",
    "end_y_mirrored",
    "corner_side",
    "ends_in_box",
    "fine_zone",
    "coarse_zone",
    "short_corner_proxy",
    "technique",
    "pass_height",
    "pass_body_part",
    "pass_outcome",
    "pass_length",
]

_COARSE_MAP: dict[str, str] = {
    "near_six": "near",
    "near_deep": "near",
    "central_six": "central",
    "central_deep": "central",
    "far_six": "far",
    "far_deep": "far",
    "edge_of_box": "outside_box",
    "other_outside": "outside_box",
}


def corner_side(start_y: float | None, split: float) -> str | None:
    """``"left"`` if ``start_y < split``, ``"right"`` otherwise; ``None`` if null."""
    if start_y is None or (isinstance(start_y, float) and pd.isna(start_y)):
        return None
    return "left" if start_y < split else "right"


def mirror_y(y: float | None, side: str | None) -> float | None:
    """``80 - y`` for right-side corners; unchanged for left; ``None`` if null."""
    if y is None or (isinstance(y, float) and pd.isna(y)):
        return None
    if side == "right":
        return 80.0 - y
    return float(y)


def ends_in_box(x: float | None, y_m: float | None, box: dict) -> bool | None:
    """``True`` iff ``(x, y_m)`` is inside or on the penalty box boundary."""
    if x is None or y_m is None:
        return None
    if pd.isna(x) or pd.isna(y_m):
        return None
    return bool(
        x >= box["x_min"] and box["y_min"] <= y_m <= box["y_max"]
    )


def fine_zone(x: float | None, y_m: float | None, cfg: Config) -> str | None:
    """One of the eight zone labels, or ``None`` if coordinates are null."""
    if x is None or y_m is None:
        return None
    if pd.isna(x) or pd.isna(y_m):
        return None

    box = cfg.p("box")
    six_x = cfg.p("six_yard_x_min")
    edge_x = cfg.p("edge_of_box_x_min")
    bands = cfg.p("zone_width_bands")
    near_lo, near_hi = bands["near"]
    central_lo, central_hi = bands["central"]
    far_lo, far_hi = bands["far"]

    # Only classify inside the penalty area or at edge-of-box.
    in_box = x >= box["x_min"] and box["y_min"] <= y_m <= box["y_max"]
    at_edge = edge_x <= x < box["x_min"] and box["y_min"] <= y_m <= box["y_max"]

    if not in_box and not at_edge:
        return "other_outside"

    if at_edge:
        return "edge_of_box"

    # Inside box: determine depth (six-yard vs deep).
    is_six = x >= six_x

    # Width band.
    if near_lo <= y_m < near_hi:
        return "near_six" if is_six else "near_deep"
    if central_lo <= y_m <= central_hi:
        return "central_six" if is_six else "central_deep"
    if far_lo < y_m <= far_hi:
        return "far_six" if is_six else "far_deep"

    return "other_outside"


def coarse_zone(fine: str | None) -> str | None:
    """Map fine zone to coarse zone, or ``None`` for null."""
    if fine is None:
        return None
    return _COARSE_MAP.get(fine)


def short_corner_proxy(
    height: str | None,
    length: float | None,
    max_len: float,
) -> bool | None:
    """``True`` iff the pass is a Ground Pass with length ≤ ``max_len``."""
    if height is None:
        return None
    if height != "Ground Pass":
        return False
    if length is None or (isinstance(length, float) and pd.isna(length)):
        return None
    return bool(length <= max_len)


def build_rq1_corners(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Filter and enrich corner-kick passes from the ``events`` table.

    Returns a DataFrame with exactly ``RQ1_CORNERS_COLUMNS`` columns.
    ``technique`` maps null → ``"Unknown"``; all geometry fields propagate
    null when coordinates are absent.
    """
    corners = events[
        (events["event_type"] == "Pass") & (events["pass_type"] == "Corner")
    ].copy()

    split_y = cfg.p("corner_side_split_y")
    box = cfg.p("box")
    max_len = cfg.p("short_max_length")

    # Side and mirrored endpoint.
    corners["corner_side"] = corners["start_y"].apply(
        lambda y: corner_side(None if pd.isna(y) else float(y), split_y)
    )

    # opponent_team_id: home/away depending on which team takes the corner.
    if "home_team_id" in corners.columns and "away_team_id" in corners.columns:
        corners["opponent_team_id"] = corners.apply(
            lambda r: r["away_team_id"] if r["team_id"] == r["home_team_id"] else r["home_team_id"],
            axis=1,
        )
    elif "opponent_team_id" not in corners.columns:
        corners["opponent_team_id"] = pd.NA

    corners["end_y_mirrored"] = corners.apply(
        lambda r: mirror_y(
            None if pd.isna(r["pass_end_y"]) else float(r["pass_end_y"]),
            r["corner_side"],
        ),
        axis=1,
    )

    # Zone geometry on the mirrored endpoint.
    corners["ends_in_box"] = corners.apply(
        lambda r: ends_in_box(
            None if pd.isna(r["pass_end_x"]) else float(r["pass_end_x"]),
            None if pd.isna(r["end_y_mirrored"]) else float(r["end_y_mirrored"]),
            box,
        ),
        axis=1,
    )
    corners["fine_zone"] = corners.apply(
        lambda r: fine_zone(
            None if pd.isna(r["pass_end_x"]) else float(r["pass_end_x"]),
            None if pd.isna(r["end_y_mirrored"]) else float(r["end_y_mirrored"]),
            cfg,
        ),
        axis=1,
    )
    corners["coarse_zone"] = corners["fine_zone"].apply(coarse_zone)

    corners["short_corner_proxy"] = corners.apply(
        lambda r: short_corner_proxy(
            None if pd.isna(r["pass_height"]) else r["pass_height"],
            None if pd.isna(r["pass_length"]) else float(r["pass_length"]),
            max_len,
        ),
        axis=1,
    )

    # technique: null → "Unknown".
    corners["technique"] = corners["pass_technique"].fillna("Unknown")

    # Return only the contract columns in order.
    result = corners.reindex(columns=RQ1_CORNERS_COLUMNS)
    return result.reset_index(drop=True)
