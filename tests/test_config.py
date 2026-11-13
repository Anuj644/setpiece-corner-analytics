import dataclasses
from pathlib import Path

import pytest

from setpiece.config import ConfigError, load_config

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "default.yaml"

def test_default_config_values():
    cfg = load_config(CFG)
    assert cfg.sha == "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
    assert [(s.competition_id, s.season_id) for s in cfg.slices] == \
        [(43, 106), (2, 27), (9, 281), (11, 90), (7, 235)]
    assert [s.has_360 for s in cfg.slices] == [True, False, True, True, True]
    assert [s.expected_matches for s in cfg.slices] == [None, 38, 34, 38, 38]
    assert cfg.p("cap_sensitivity") == [10, 20, None]
    assert cfg.p("short_max_length") == 10
    assert cfg.p("cap_seconds") == 15
    assert cfg.p("keepers_in_counts") is True
    assert cfg.params["penalty_handling"].status == "locked"

def test_paths_resolve_against_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(CFG)
    assert cfg.root == REPO
    assert cfg.raw_dir == REPO / "data" / "raw"

def test_invalid_status_rejected(tmp_path):
    bad = CFG.read_text().replace("status: locked", "status: final", 1)
    (tmp_path / "configs").mkdir()
    p = tmp_path / "configs" / "bad.yaml"; p.write_text(bad)
    with pytest.raises(ConfigError, match="status"):
        load_config(p)

def test_config_hash_stable_and_sensitive(tmp_path):
    a = load_config(CFG).config_hash()
    assert a == load_config(CFG).config_hash() and len(a) == 12
    (tmp_path / "configs").mkdir()
    p = tmp_path / "configs" / "x.yaml"
    p.write_text(CFG.read_text().replace("value: 15", "value: 20", 1))
    assert load_config(p).config_hash() != a


def test_config_outside_configs_dir_rejected(tmp_path):
    (tmp_path / "proj").mkdir()
    p = tmp_path / "proj" / "variant.yaml"
    p.write_text(CFG.read_text())
    with pytest.raises(ConfigError, match="configs"):
        load_config(p)


def test_invalid_slice_kind_rejected(tmp_path):
    (tmp_path / "configs").mkdir()
    p = tmp_path / "configs" / "bad.yaml"
    p.write_text(CFG.read_text().replace("kind: tournament", "kind: Tournament", 1))
    with pytest.raises(ConfigError, match="kind"):
        load_config(p)


def test_duplicate_yaml_key_rejected(tmp_path):
    (tmp_path / "configs").mkdir()
    p = tmp_path / "configs" / "dup.yaml"
    # params is the last top-level block, so this repeats a key inside it
    dup = '  cap_seconds:\n    value: 99\n    status: locked\n    note: "repeat"\n'
    p.write_text(CFG.read_text() + dup)
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(p)


def test_replace_keeps_hash(tmp_path):
    cfg = load_config(CFG)
    assert dataclasses.replace(cfg, raw_dir=tmp_path).config_hash() == cfg.config_hash()
