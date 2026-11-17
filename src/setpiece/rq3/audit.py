"""Audit the planned 360 coverage profiles against the available evidence."""

import pandas as pd

from setpiece.config import Config
from setpiece.rq3.features import COVERAGE_COLUMNS, KEY, visible_polygon
from setpiece.rq3.geometry import points_in_polygon, zone_grid

AUDIT_SLICES = [(9, 281), (11, 90), (7, 235)]
AUDIT_CLUBS = ["Bayer Leverkusen", "Barcelona", "Paris Saint-Germain"]
# Planning decision D-41 verifies these counts for the documented visibility grid.
AUDIT_REFERENCES = (366, 211, 48)
_DELIVERIES = (
    "all",
    "short_proxy",
    "aerial_cross_proxy",
    "other_ground",
    "ground_length_unknown",
    "unknown",
)


def _delivery_type(corners: pd.DataFrame) -> pd.Series:
    height = corners["pass_height"]
    ground = height.eq("Ground Pass").fillna(False)
    short = corners["short_corner_proxy"]
    delivery = pd.Series("unknown", index=corners.index, dtype="string")
    delivery.loc[height.isin(["High Pass", "Low Pass"])] = "aerial_cross_proxy"
    delivery.loc[ground] = "ground_length_unknown"
    delivery.loc[ground & short.eq(False)] = "other_ground"
    delivery.loc[short.eq(True)] = "short_proxy"
    return delivery


def _counts(group, cfg, thresholds):
    n_corners = len(group)
    framed = group[group["has_frame"]]
    n_frames = len(framed)
    eligible = int(framed["zone_eligible"].sum())
    screened = int(framed["screened"].sum())
    failed = {
        f"{threshold:g}": n_frames - int(framed["consistency"].ge(threshold).sum())
        for threshold in thresholds
    }
    return {
        "corners": n_corners,
        "frames": n_frames,
        "screened": screened,
        "zone_eligible": eligible,
        "frames_meeting_player_count": int(framed["n_visible_ge16"].sum()),
        "frames_showing_all_zones": int(
            framed[COVERAGE_COLUMNS].ge(cfg.p("zone_visibility_min")).all(axis=1).sum()
        ),
        "frame_share_of_corners": n_frames / n_corners if n_corners else None,
        "eligible_share_of_corners": eligible / n_corners if n_corners else None,
        "eligible_share_of_frames": eligible / n_frames if n_frames else None,
        **{f"fails_polygon_consistency_{key}": count for key, count in failed.items()},
        **{
            f"fail_share_polygon_consistency_{key}": count / n_frames if n_frames else None
            for key, count in failed.items()
        },
    }


def coverage_audit(corners: pd.DataFrame, features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep an all-corner denominator for each slice and delivery type."""
    columns = [
        *KEY,
        "screened",
        "n_visible_ge16",
        "zone_eligible",
        "consistency",
        *COVERAGE_COLUMNS,
    ]
    merged = (
        corners[KEY]
        .assign(delivery=_delivery_type(corners))
        .merge(features[columns], on=KEY, how="left", indicator=True, validate="one_to_one")
    )
    merged["has_frame"] = merged["_merge"].eq("both")
    thresholds = sorted(
        {cfg.p("polygon_consistency_min"), *cfg.p("polygon_consistency_sensitivity")}
    )
    rows = []
    for sl in cfg.slices:
        group = merged[
            merged["competition_id"].eq(sl.competition_id) & merged["season_id"].eq(sl.season_id)
        ]
        for delivery in _DELIVERIES:
            subset = group if delivery == "all" else group[group["delivery"].eq(delivery)]
            rows.append(
                {
                    "competition_id": sl.competition_id,
                    "season_id": sl.season_id,
                    "slice": sl.name,
                    "delivery": delivery,
                    **_counts(subset, cfg, thresholds),
                }
            )
    return pd.DataFrame(rows)


def audit_club_coverage(
    frames: pd.DataFrame, audit: pd.DataFrame, main: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Compare the two profiles for the original audit clubs, including full-box coverage."""
    focal = audit.loc[
        audit["team_name"].isin(AUDIT_CLUBS)
        & pd.MultiIndex.from_frame(audit[KEY[:2]]).isin(AUDIT_SLICES)
    ].copy()
    screened = focal.loc[focal["screened"], [*KEY, "corner_side"]].merge(
        frames[[*KEY, "visible_area"]], on=KEY, validate="one_to_one"
    )
    box = cfg.p("box")
    # The union of the six full-box zones is the penalty area. Their grid boundaries
    # are on the default unit lattice.
    x1 = cfg.p("rq3_zones")["six"][1]
    grid = zone_grid(box["x_min"], x1, box["y_min"], box["y_max"], cfg.p("zone_grid_step"))
    screened["audit_full_box_visible"] = [
        points_in_polygon(grid, visible_polygon(row.visible_area, row.corner_side)).mean()
        >= cfg.p("zone_visibility_min")
        for row in screened.itertuples(index=False)
    ]
    focal = focal.merge(screened[[*KEY, "audit_full_box_visible"]], on=KEY, how="left")
    focal = focal.merge(
        main[[*KEY, "screened", "zone_eligible"]],
        on=KEY,
        validate="one_to_one",
        suffixes=("_audit", "_main"),
    )
    return (
        focal.groupby("team_name")
        .agg(
            matched=("event_id", "size"),
            main_screened=("screened_main", "sum"),
            main_zone_eligible=("zone_eligible_main", "sum"),
            audit_screened=("screened_audit", "sum"),
            audit_zone_eligible=("zone_eligible_audit", "sum"),
            audit_full_box_visible=("audit_full_box_visible", "sum"),
        )
        .reset_index()
    )
