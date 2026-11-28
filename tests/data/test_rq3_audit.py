"""Data gates for the documented StatsBomb counts and original focal-club audit."""

import pandas as pd
import pytest

from setpiece.rq3.audit import AUDIT_REFERENCES, audit_club_coverage, coverage_audit
from setpiece.rq3.features import build_rq3_features, read_rq3_inputs


@pytest.fixture(scope="module")
def rq3_audit_data():
    from pathlib import Path

    from setpiece.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
    frames, corners = read_rq3_inputs(cfg)
    main = build_rq3_features(frames, corners, cfg)
    audit = build_rq3_features(frames, corners, cfg, profile="audit")
    return cfg, frames, corners, main, audit


@pytest.mark.data
def test_world_cup_snapshot_and_visibility_counts(rq3_audit_data):
    cfg, _, corners, main, _ = rq3_audit_data
    table = coverage_audit(corners, main, cfg).query("competition_id == 43").set_index("delivery")
    columns = ["corners", "frames", "frames_meeting_player_count", "frames_showing_all_zones"]
    assert tuple(table.loc["all", columns]) == (569, 300, 286, 140)
    assert tuple(table.loc["short_proxy", ["corners", "frames"]]) == (41, 13)
    assert tuple(table.loc["aerial_cross_proxy", ["corners", "frames"]]) == (460, 266)


@pytest.mark.data
def test_club_audit_profile_reproduces_screened_count(rq3_audit_data):
    cfg, frames, _, main, audit = rq3_audit_data
    table = audit_club_coverage(frames, audit, main, cfg)
    assert table["audit_screened"].sum() == AUDIT_REFERENCES[0]


@pytest.mark.data
def test_club_audit_profile_reproduces_zone_counts(rq3_audit_data):
    cfg, frames, _, main, audit = rq3_audit_data
    table = audit_club_coverage(frames, audit, main, cfg)
    totals = table[["audit_zone_eligible", "audit_full_box_visible"]].sum()
    pd.testing.assert_series_equal(
        totals.astype(int), pd.Series(AUDIT_REFERENCES[1:], index=totals.index)
    )
