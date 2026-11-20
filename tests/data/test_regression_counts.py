import pytest

from setpiece.io import read_table

pytestmark = pytest.mark.data

AUDIT = [(9, 281), (11, 90), (7, 235)]


def test_world_cup_2022(cfg):
    ev = read_table("events", cfg, [(43, 106)])
    corners = ev[(ev.event_type == "Pass") & (ev.pass_type == "Corner")]
    assert (ev.match_id.nunique(), len(ev), len(corners)) == (64, 234_637, 569)
    c360 = read_table("corner_360", cfg, [(43, 106)])
    assert len(c360) == 300
    assert int((c360.players.map(len) >= 16).sum()) == 286
    assert corners.pass_technique.isna().sum() == 109

def test_audit_slices(cfg):
    ev = read_table("events", cfg, AUDIT)                      # [(9,281),(11,90),(7,235)]
    corners = ev[(ev.event_type == "Pass") & (ev.pass_type == "Corner")]
    assert (ev.match_id.nunique(), len(ev), len(corners)) == (101, 406_285, 965)
    assert len(read_table("corner_360", cfg, AUDIT)) == 640
    fs = read_table("frame_stats", cfg, AUDIT)
    assert int(fs.n_unmatched.sum()) == 13_761

def test_technique_is_null_exactly_for_ground_passes(cfg):
    ev = read_table("events", cfg, AUDIT + [(43, 106)])
    c = ev[(ev.event_type == "Pass") & (ev.pass_type == "Corner")]
    assert (c.pass_technique.isna() == (c.pass_height == "Ground Pass")).all()

@pytest.mark.parametrize("name,n,matched,missing_tech", [("Bayer Leverkusen", 236, 148, 76),
                                                          ("Barcelona", 202, 147, 108),
                                                          ("Paris Saint-Germain", 170, 82, 59)])
def test_focal_clubs(cfg, name, n, matched, missing_tech):
    ev = read_table("events", cfg, AUDIT)
    c = ev[(ev.event_type == "Pass") & (ev.pass_type == "Corner") & (ev.team_name == name)]
    c360 = read_table("corner_360", cfg, AUDIT)
    assert len(c) == n
    assert c.event_id.isin(c360.event_id).sum() == matched
    assert c.pass_technique.isna().sum() == missing_tech

def test_missing_technique_total(cfg):
    ev = read_table("events", cfg, AUDIT)
    c = ev[(ev.event_type == "Pass") & (ev.pass_type == "Corner")]
    assert c.pass_technique.isna().sum() == 307
