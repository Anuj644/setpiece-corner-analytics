import numpy as np
import pandas as pd
import pytest
from rq3_helpers import FAR_CAMERA, FULL_CAMERA, corner, player, tables

from setpiece.rq3.features import FEATURES, build_rq3_features, funnel


def test_side_normalization_and_edges_preserve_counts(cfg):
    original = [
        player(114, 40),
        player(115, 39),
        player(112, 32),
        player(113, 50, False),
        player(120, 40, False, True),
        player(117, 29, False),
    ]
    right = [dict(p, y=80 - p["y"]) for p in original]
    right_polygon = [80 - v if i % 2 else v for i, v in enumerate(FULL_CAMERA)]
    frames, corners = tables(
        [
            corner(1, "left", players=original),
            corner(1, "right", side="right", camera=right_polygon, players=right),
        ]
    )
    rows = build_rq3_features(frames, corners, cfg)
    assert rows["zone_eligible"].all() and not rows["n_visible_ge16"].any()
    assert rows["consistency"].eq(1).all()
    np.testing.assert_array_equal(rows.iloc[0][FEATURES], rows.iloc[1][FEATURES])
    row = rows.iloc[0]
    assert row.att_far_central_six == row.att_near_central_deep == 1
    assert row.att_near_central_six == row.def_far_central_deep == 1
    assert row.def_far_central_six == 1
    assert row.def_near_central_deep == 0
    assert row[FEATURES].sum() == 5


def test_hidden_zone_is_nan_and_integrity_checks_precede_counting(cfg):
    players = [player(115 + i * 0.1, 45) for i in range(16)]
    frames, corners = tables(
        [
            corner(1, "cropped", camera=FAR_CAMERA, players=players),
            corner(1, "bad-polygon", camera=[0, 0, 1, 1, 2, 2], players=players),
            corner(1, "bad-players", players=[player(None, 45)]),
            corner(1, "missing-keeper-flag", players=[player(116, 45, keeper=None)]),
        ]
    )
    features = build_rq3_features(frames, corners, cfg).set_index("event_id")
    cropped = features.loc["cropped"]
    assert cropped.n_visible_ge16 and cropped.screened and not cropped.zone_eligible
    assert np.isnan(cropped.att_near_central_six)
    assert cropped.att_far_central_six == 16
    assert not features.loc["bad-polygon", "polygon_ok"]
    assert not features.loc["bad-players", "finite_ok"]
    assert not features.loc["missing-keeper-flag", "flags_ok"]
    assert not features.loc[["bad-polygon", "bad-players", "missing-keeper-flag"], "screened"].any()


def test_profiles_only_change_the_specified_condition(cfg):
    frames, corners = tables(
        [
            corner(1, "ten", players=[player(116, 37, False, True), *[player(108, 35)] * 9]),
            corner(1, "cropped", camera=FAR_CAMERA, players=[player(116, 45)] * 16),
        ]
    )
    main = build_rq3_features(frames, corners, cfg).set_index("event_id")
    no_keeper = build_rq3_features(frames, corners, cfg, "no_keeper").set_index("event_id")
    ge16 = build_rq3_features(frames, corners, cfg, "ge16").set_index("event_id")
    assert main.loc["ten", "zone_eligible"]
    assert main.loc["ten", "def_near_central_six"] == 1
    assert main.loc["ten", "keeper_zone"] == "near_central_six"
    assert no_keeper.loc["ten", "def_near_central_six"] == 0
    assert not ge16.loc["ten", "screened"]
    assert ge16.loc["cropped", "screened"] and not ge16.loc["cropped", "zone_eligible"]


def test_join_rejects_duplicates_and_orphans(cfg):
    frames, corners = tables([corner(1, "x", players=[player(116, 45)])])
    with pytest.raises(ValueError, match="unique"):
        build_rq3_features(pd.concat([frames, frames]), corners, cfg)
    with pytest.raises(ValueError, match="without a matching"):
        build_rq3_features(frames, corners.iloc[0:0], cfg)


def test_funnel_keeps_unmatched_corners(cfg):
    frames, corners = tables(
        [
            corner(1, "x", players=[player(116, 45)]),
            corner(1, "short", short=True, players=[player(116, 45)]),
            corner(1, "short-without-frame", short=True),
            corner(2, "unscreened", camera=[0, 0, 1, 1, 2, 2], players=[player(116, 45)]),
        ]
    )
    corners.loc[corners["event_id"].eq("short-without-frame"), "team_name"] = "Team 2 alias"
    frames = frames.drop(index=2)
    features = build_rq3_features(frames, corners, cfg)
    per_team = funnel(corners, frames, features)
    assert per_team["team_id"].is_unique
    assert per_team[["corners", "matched", "screened", "zone_eligible"]].sum().tolist() == [
        4,
        3,
        2,
        2,
    ]
    counts = per_team[["corners", "matched", "screened", "zone_eligible"]].to_numpy()
    assert (np.diff(counts, axis=1) <= 0).all()
