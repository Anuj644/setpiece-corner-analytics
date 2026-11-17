"""Screen StatsBomb 360 frames and derive mirrored central-box counts."""

import numpy as np
import pandas as pd

from setpiece.config import Config
from setpiece.ingest.frames import CORNER_360_DTYPES
from setpiece.io import list_partitions, read_table
from setpiece.plots.pitch import PITCH_LENGTH, PITCH_WIDTH
from setpiece.rq3.geometry import (
    mirror_xy,
    points_in_polygon,
    polygon_area,
    polygon_from_flat,
    zone_grid,
)

KEY = ["competition_id", "season_id", "match_id", "event_id"]
TEAM_KEY = ["competition_id", "season_id", "team_id"]
CORNER_COLUMNS = [*KEY, "team_id", "team_name", "corner_side", "short_corner_proxy", "pass_height"]
ZONE_NAMES = ("near_central_six", "near_central_deep", "far_central_six", "far_central_deep")
FEATURES = [f"{side}_{zone}" for side in ("att", "def") for zone in ZONE_NAMES]
COVERAGE_COLUMNS = [f"cov_{zone}" for zone in ZONE_NAMES]
PROFILES = ("main", "audit", "tight", "ge16", "poly80", "poly100", "no_keeper")

_FEATURE_DTYPES = {
    **dict.fromkeys(KEY[:-1] + ["team_id"], "Int64"),
    **dict.fromkeys(
        ["event_id", "team_name", "corner_side", "pass_height", "keeper_zone"], "string"
    ),
    **dict.fromkeys(["visible_total", "visible_attackers", "visible_defenders"], "Int64"),
    **dict.fromkeys(
        [
            "finite_ok",
            "flags_ok",
            "polygon_ok",
            "n_visible_ge16",
            "screened",
            "zone_eligible",
            "short_corner_proxy",
        ],
        "boolean",
    ),
    **dict.fromkeys([*FEATURES, *COVERAGE_COLUMNS, "polygon_area", "consistency"], "float64"),
}


def read_rq3_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the prepared frames and corner metadata for the configured slices."""
    all_slices = [(sl.competition_id, sl.season_id) for sl in cfg.slices]
    if not all_slices:
        raise ValueError("RQ3 needs at least one configured slice")
    slices_360 = [(sl.competition_id, sl.season_id) for sl in cfg.slices if sl.has_360]
    if missing := sorted(set(slices_360) - set(list_partitions("corner_360", cfg))):
        raise FileNotFoundError(f"no corner_360 partition for {missing}; run ingest first")
    try:
        corners = read_table("derived/rq1_corners", cfg, all_slices)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; run rq1 first") from exc
    frames = (
        read_table("corner_360", cfg, slices_360)
        if slices_360
        else pd.DataFrame(columns=CORNER_360_DTYPES).astype(CORNER_360_DTYPES)
    )
    return frames, corners


def zones(cfg: Config) -> dict[str, tuple[float, float, float, float]]:
    """Return the four configured boxes as (x0, x1, y0, y1)."""
    bounds = cfg.p("rq3_zones")
    return {
        f"{width}_{depth}": (*bounds[depth], *bounds[width])
        for width in ("near_central", "far_central")
        for depth in ("six", "deep")
    }


def profile_parameters(cfg: Config, profile: str) -> dict:
    """Satisfy the primary screen plus any one profile's documented change."""
    if profile not in PROFILES:
        raise ValueError(f"unknown RQ3 profile {profile!r}")
    consistency = cfg.p("polygon_consistency_min")
    if profile in ("poly80", "poly100"):
        low, high = cfg.p("polygon_consistency_sensitivity")
        consistency = low if profile == "poly80" else high
    return {
        "consistency_min": consistency,
        "min_players": cfg.p("min_visible_players_sensitivity")
        if profile in ("ge16", "audit")
        else 0,
        "buffer": cfg.p("tight_screen_buffer") if profile == "tight" else 0,
        "include_keepers": cfg.p("keepers_in_counts") and profile != "no_keeper",
    }


def join_snapshots(frames: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """Match each frame to exactly one RQ1 corner and reject incomplete or duplicate keys."""
    if frames[KEY].isna().any().any() or corners[KEY].isna().any().any():
        raise ValueError("RQ3 snapshot and corner keys must be complete")
    event_key = ["match_id", "event_id"]
    if frames.duplicated(event_key).any() or corners.duplicated(event_key).any():
        raise ValueError("RQ3 inputs must have unique (match_id, event_id) keys")
    joined = frames.merge(corners[CORNER_COLUMNS], on=KEY, how="left", indicator=True)
    missing = joined["_merge"] != "both"
    if missing.any():
        row = joined.loc[missing].iloc[0]
        raise ValueError(
            f"RQ3 found {int(missing.sum())} frames without a matching RQ1 corner, "
            f"starting with match {row.match_id}, event {row.event_id}; rerun ingest and rq1"
        )
    return joined.drop(columns="_merge").sort_values(KEY, ignore_index=True)


def visible_polygon(flat, side: str = "left") -> np.ndarray | None:
    """Parse and optionally mirror a nondegenerate visible-area polygon."""
    if pd.isna(side) or side not in ("left", "right"):
        return None
    try:
        polygon = mirror_xy(polygon_from_flat(flat), side)
    except (TypeError, ValueError):
        return None
    return polygon if polygon_area(polygon) > 0 else None


def _coordinates(players) -> np.ndarray | None:
    if not isinstance(players, (tuple, list, np.ndarray)) or not len(players):
        return None
    if any(not isinstance(player, dict) for player in players):
        return None
    try:
        xy = np.asarray([[p.get("x"), p.get("y")] for p in players], dtype=float)
    except (TypeError, ValueError):
        return None
    return xy if np.isfinite(xy).all() else None


def _flags_ok(players) -> bool:
    return bool(
        isinstance(players, (tuple, list, np.ndarray))
        and len(players)
        and all(
            isinstance(p, dict)
            and all(isinstance(p.get(flag), (bool, np.bool_)) for flag in ("teammate", "keeper"))
            for p in players
        )
    )


def snapshot_players(players, side: str):
    """Mirror observed coordinates and return the attacker and outfield masks."""
    xy = _coordinates(players)
    if xy is None or not _flags_ok(players) or pd.isna(side) or side not in ("left", "right"):
        return None
    return (
        mirror_xy(xy, side),
        np.asarray([p["teammate"] for p in players], dtype=bool),
        ~np.asarray([p["keeper"] for p in players], dtype=bool),
    )


def screen(row, profile_params: dict) -> dict:
    """Test frame integrity in the source coordinates before any zone calculation."""
    players = row["players"]
    n_visible = len(players) if isinstance(players, (tuple, list, np.ndarray)) else None
    xy = _coordinates(players)
    polygon = visible_polygon(row["visible_area"])
    flags_ok = _flags_ok(players)
    consistency = (
        float(points_in_polygon(xy, polygon).mean())
        if xy is not None and polygon is not None
        else np.nan
    )
    return {
        "finite_ok": xy is not None,
        "flags_ok": flags_ok,
        "polygon_ok": polygon is not None,
        "consistency": consistency,
        "n_visible": n_visible,
        "screened": flags_ok
        and consistency >= profile_params["consistency_min"]
        and n_visible >= profile_params["min_players"],
    }


def zone_visibility(poly, boxes: dict, step: float, buffer: float = 0) -> dict[str, float]:
    """Evaluate visibility on inclusive grids clipped to the pitch."""
    return {
        zone: float(
            points_in_polygon(
                zone_grid(
                    max(0, x0 - buffer),
                    min(PITCH_LENGTH, x1 + buffer),
                    max(0, y0 - buffer),
                    min(PITCH_WIDTH, y1 + buffer),
                    step,
                ),
                poly,
            ).mean()
        )
        for zone, (x0, x1, y0, y1) in boxes.items()
    }


def _in_zone(x: float, y: float, zone: str, box: tuple) -> bool:
    x0, x1, y0, y1 = box
    x_inside = x0 <= x <= x1 if zone.endswith("_six") else x0 <= x < x1
    y_inside = y0 <= y <= y1 if zone.startswith("far_") else y0 <= y < y1
    return x_inside and y_inside


def zone_counts(players_m: list[dict], boxes: dict, include_keepers: bool) -> dict[str, int]:
    """Count each observed player at most once using the specified shared-edge rules."""
    counts = dict.fromkeys([f"{side}_{zone}" for side in ("att", "def") for zone in boxes], 0)
    for p in players_m:
        if not include_keepers and p["keeper"]:
            continue
        for zone, box in boxes.items():
            if _in_zone(p["x"], p["y"], zone, box):
                team = "att" if p["teammate"] else "def"
                counts[f"{team}_{zone}"] += 1
                break
    return counts


def _frame_features(row, boxes, step, min_visibility, qa_min_players, params):
    qa = screen(row, params)
    n_visible = qa.pop("n_visible")
    side_ok = not pd.isna(row["corner_side"]) and row["corner_side"] in ("left", "right")
    qa["screened"] = qa["screened"] and side_ok
    qa["zone_eligible"] = False
    qa["visible_total"] = n_visible
    qa["n_visible_ge16"] = n_visible >= qa_min_players if n_visible is not None else pd.NA
    if qa["flags_ok"]:
        qa["visible_attackers"] = sum(p["teammate"] for p in row["players"])
        qa["visible_defenders"] = n_visible - qa["visible_attackers"]
    polygon = visible_polygon(row["visible_area"], row["corner_side"])
    if polygon is None:
        return qa

    qa["polygon_area"] = polygon_area(polygon)
    coverage = zone_visibility(polygon, boxes, step, params["buffer"])
    qa.update({f"cov_{zone}": share for zone, share in coverage.items()})
    if not qa["screened"]:
        return qa

    xy, _, _ = snapshot_players(row["players"], row["corner_side"])
    players = [dict(p, x=x, y=y) for p, (x, y) in zip(row["players"], xy)]
    visible_zones = {name: b for name, b in boxes.items() if coverage[name] >= min_visibility}
    qa["zone_eligible"] = len(visible_zones) == len(boxes)
    qa.update(zone_counts(players, visible_zones, params["include_keepers"]))
    defending_keepers = [p for p in players if p["keeper"] and not p["teammate"]]
    if len(defending_keepers) == 1:
        keeper = defending_keepers[0]
        qa["keeper_zone"] = next(
            (name for name, b in boxes.items() if _in_zone(keeper["x"], keeper["y"], name, b)),
            "outside_central",
        )
    return qa


def build_rq3_features(
    corner_360: pd.DataFrame, rq1: pd.DataFrame, cfg: Config, profile: str = "main"
) -> pd.DataFrame:
    """Return one row per matched frame with null counts for unobservable zones."""
    joined = join_snapshots(corner_360, rq1)
    boxes = zones(cfg)
    step = cfg.p("zone_grid_step")
    minimum = cfg.p("zone_visibility_min")
    qa_min = cfg.p("min_visible_players_sensitivity")
    params = profile_parameters(cfg, profile)
    rows = [
        _frame_features(row, boxes, step, minimum, qa_min, params)
        for row in joined.to_dict("records")
    ]
    return (
        pd.concat([joined[CORNER_COLUMNS], pd.DataFrame(rows, index=joined.index)], axis=1)
        .reindex(
            columns=[*CORNER_COLUMNS, *[c for c in _FEATURE_DTYPES if c not in CORNER_COLUMNS]]
        )
        .astype(_FEATURE_DTYPES)
    )


def funnel(rq1: pd.DataFrame, corner_360: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Reconcile all corners, matched frames, screened frames, and zone-eligible frames."""
    matched = corner_360[KEY].assign(matched=True)
    flag_columns = ["screened", "zone_eligible"]
    joined = (
        rq1[[*KEY, "team_id", "team_name", "pass_height"]]
        .merge(matched, how="left", on=KEY, validate="one_to_one")
        .merge(features[[*KEY, *flag_columns]], how="left", on=KEY, validate="one_to_one")
    )
    for col in ["matched", *flag_columns]:
        joined[col] = joined[col].astype("boolean").fillna(False)
    joined["ground"] = joined["pass_height"].eq("Ground Pass").fillna(False)
    joined["ground_eligible"] = joined["ground"] & joined["zone_eligible"]
    return (
        joined.groupby(TEAM_KEY, dropna=False)
        .agg(
            team_name=("team_name", "first"),
            corners=("event_id", "size"),
            matched=("matched", "sum"),
            screened=("screened", "sum"),
            zone_eligible=("zone_eligible", "sum"),
            ground_all=("ground", "sum"),
            ground_eligible=("ground_eligible", "sum"),
        )
        .reset_index()
        .assign(
            ground_share_all=lambda r: r["ground_all"] / r["corners"].replace(0, np.nan),
            ground_share_eligible=lambda r: (
                r["ground_eligible"] / r["zone_eligible"].replace(0, np.nan)
            ),
        )
    )
