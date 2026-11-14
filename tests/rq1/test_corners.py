"""Tests for rq1_corners derivation (Task 12)."""

import pandas as pd
import pytest

from setpiece.rq1.corners import (
    RQ1_CORNERS_COLUMNS,
    build_rq1_corners,
    coarse_zone,
    corner_side,
    ends_in_box,
    fine_zone,
    mirror_y,
    short_corner_proxy,
)
from tests.helpers import corner_row, events_df


@pytest.mark.parametrize(
    "x,y,zone",
    [
        (102, 18, "near_deep"),
        (113.99, 29.99, "near_deep"),
        (114, 30, "central_six"),
        (110, 50, "central_deep"),
        (110, 50.01, "far_deep"),
        (120, 62, "far_six"),
        (101.99, 40, "edge_of_box"),
        (90, 18, "edge_of_box"),
        (89.99, 40, "other_outside"),
        (105, 62.01, "other_outside"),
        (118, 5, "other_outside"),
    ],
)
def test_fine_zone_boundaries(cfg, x, y, zone):
    assert fine_zone(x, y, cfg) == zone


def test_side_and_mirror():
    assert corner_side(0.1, 40) == "left" and corner_side(40, 40) == "right"
    assert mirror_y(79.9, "right") == pytest.approx(0.1) and mirror_y(30, "left") == 30


def test_ends_in_box_inclusive(cfg):
    assert ends_in_box(102, 18, cfg.p("box")) is True
    assert ends_in_box(101.99, 40, cfg.p("box")) is False


def test_short_proxy():
    assert short_corner_proxy("Ground Pass", 10.0, 10) is True
    assert short_corner_proxy("Ground Pass", 10.01, 10) is False
    assert short_corner_proxy("High Pass", 5.0, 10) is False
    assert short_corner_proxy(None, 5.0, 10) is None


def test_coarse_zone_mapping():
    assert coarse_zone("near_six") == "near"
    assert coarse_zone("central_deep") == "central"
    assert coarse_zone("far_six") == "far"
    assert coarse_zone("edge_of_box") == "outside_box"
    assert coarse_zone("other_outside") == "outside_box"
    assert coarse_zone(None) is None


def test_build_keeps_unknown_technique_and_mirrors_right(cfg):
    ev = events_df(
        [
            corner_row(event_id="L", start_y=0.1, end=(110, 35), technique=None),
            corner_row(event_id="R", start_y=79.9, end=(110, 45), technique="Outswinging"),
        ]
    )
    df = build_rq1_corners(ev, cfg).set_index("event_id")
    assert df.loc["L", "technique"] == "Unknown"
    assert df.loc["R", "end_y_mirrored"] == pytest.approx(35)
    assert df.loc["L", "fine_zone"] == df.loc["R", "fine_zone"] == "central_deep"
    assert list(build_rq1_corners(ev, cfg).columns) == RQ1_CORNERS_COLUMNS


def test_null_coordinates_propagate_as_null(cfg):
    ev = events_df(
        [corner_row(event_id="N", start_y=None, end=(None, None), technique=None)]
    )
    r = build_rq1_corners(ev, cfg).iloc[0]
    assert pd.isna(r.corner_side) and pd.isna(r.fine_zone) and pd.isna(r.ends_in_box)


def test_build_rq1_corners_filters_only_corners(cfg):
    # A non-corner Pass event should not appear.
    ev = events_df(
        [
            corner_row(event_id="c1", start_y=0.1, end=(110, 35)),
            {
                "match_id": 1,
                "event_id": "p1",
                "event_index": 2,
                "event_type": "Pass",
                "pass_type": "Throw-in",
                "team_id": 1,
                "home_team_id": 1,
                "away_team_id": 2,
                "possession_id": 1,
                "start_x": 60.0,
                "start_y": 40.0,
            },
        ]
    )
    df = build_rq1_corners(ev, cfg)
    assert set(df.event_id) == {"c1"}


def test_build_rq1_corners_short_proxy_ground_pass(cfg):
    ev = events_df(
        [corner_row(event_id="g1", start_y=0.1, end=(110, 35), technique=None)]
    )
    # corner_row with technique=None sets pass_height=Ground Pass
    r = build_rq1_corners(ev, cfg).iloc[0]
    assert r.pass_height == "Ground Pass"
    # pass_length default is 15.0 which is > 10, so short_proxy should be False
    assert r.short_corner_proxy is False or bool(r.short_corner_proxy) is False
