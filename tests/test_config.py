import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from setpiece.config import KINDS, ConfigError, expand_slices, load_config

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "default.yaml"

def test_default_config_values():
    cfg = load_config(CFG)
    assert cfg.sha == "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
    core = cfg.core_slices  # default.yaml's own slices, with or without the expansion on top
    assert [(s.competition_id, s.season_id) for s in core] == \
        [(43, 106), (2, 27), (9, 281), (11, 90), (7, 235)]
    assert [s.has_360 for s in core] == [True, False, True, True, True]
    assert [s.expected_matches for s in core] == [None, 38, 34, 38, 38]
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

def test_config_hash_stable_and_sensitive(config_copy):
    path = config_copy(enabled=False)
    a = load_config(path).config_hash()
    assert a == load_config(path).config_hash() and len(a) == 12
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("value: 15", "value: 20", 1), encoding="utf-8")
    assert load_config(path).config_hash() != a


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


def test_replace_keeps_hash(config_copy, tmp_path):
    cfg = load_config(config_copy(enabled=False))  # off: the hash is that of the YAML alone
    assert dataclasses.replace(cfg, raw_dir=tmp_path).config_hash() == cfg.config_hash()


# --- corpus expansion ----------------------------------------------------------------------------

CORE_KEYS = [(43, 106), (2, 27), (9, 281), (11, 90), (7, 235)]
DISCOVERED_KEYS = [(12, 27), (55, 43)]  # the catalogue slices of cfg_expanded that are not core
REAL_OVERRIDES = REPO / "configs" / "catalogue_overrides.yaml"


def keys_of(cfg):
    return [(s.competition_id, s.season_id) for s in cfg.slices]


def test_expansion_keeps_core_first_and_does_not_duplicate(cfg_expanded):
    keys = [(s.competition_id, s.season_id) for s in cfg_expanded.slices]
    assert keys[:5] == [(43, 106), (2, 27), (9, 281), (11, 90), (7, 235)]
    assert len(keys) == len(set(keys))
    assert [s.core for s in cfg_expanded.slices[:5]] == [True] * 5


def test_overrides_set_kind_and_expected_matches(cfg_expanded):
    s = {(x.competition_id, x.season_id): x for x in cfg_expanded.slices}
    assert s[(55, 43)].kind == "tournament" and s[(12, 27)].expected_matches == 38


def test_discovered_slices_follow_the_core_ones_with_core_false(cfg_expanded):
    assert keys_of(cfg_expanded)[5:] == DISCOVERED_KEYS
    assert [s.core for s in cfg_expanded.slices[5:]] == [False, False]
    serie_a = cfg_expanded.slices[5]
    assert serie_a.name == "Serie A 2015/2016" and serie_a.has_360 is False
    assert cfg_expanded.catalogue_pending is False


def test_a_catalogue_entry_claiming_core_true_is_still_not_core(expansion_cfg, catalogue_entries):
    catalogue_entries[1]["core"] = True  # Serie A: a hand-edited catalogue.json must not promote it
    cfg = expand_slices(expansion_cfg(catalogue=catalogue_entries))
    assert [s.core for s in cfg.slices[5:]] == [False, False]
    assert len(cfg.core_slices) == 5


def test_core_slices_are_only_the_core_ones(cfg_expanded):
    assert cfg_expanded.core_slices == cfg_expanded.slices[:5]
    assert [(s.competition_id, s.season_id) for s in cfg_expanded.core_slices] == CORE_KEYS


def test_the_core_definition_in_default_yaml_wins_over_the_catalogue_and_the_overrides(
    cfg_expanded, cfg_tmp
):
    # the overrides hold a "kind: league" entry for 43/106 and the catalogue lists it as well
    assert cfg_expanded.slices[0] == cfg_tmp.slices[0]
    assert cfg_expanded.slices[0].kind == "tournament"


def test_expanding_twice_gives_the_same_config(cfg_expanded):
    assert expand_slices(cfg_expanded) == cfg_expanded


def test_a_disabled_expansion_leaves_the_config_alone(cfg_tmp):
    assert expand_slices(cfg_tmp) is cfg_tmp
    assert cfg_tmp.catalogue_pending is False
    assert cfg_tmp.core_slices == cfg_tmp.slices


def test_overrides_are_applied_at_load_time_so_editing_them_needs_no_rediscovery(
    expansion_cfg, catalogue_entries
):
    cfg = expansion_cfg(catalogue=catalogue_entries)
    assert {s.name: s.expected_matches for s in expand_slices(cfg).slices}["Serie A 2015/2016"] == 38
    Path(cfg.p("corpus_expansion")["overrides"]).write_text(
        '"12/27": {kind: league, expected_matches: 40}\n', encoding="utf-8"
    )
    again = {s.name: s for s in expand_slices(cfg).slices}
    assert again["Serie A 2015/2016"].expected_matches == 40
    assert again["UEFA Euro 2020"].kind == "league"  # no entry any more: back to league
    stored = json.loads((cfg.raw_dir / cfg.sha / "catalogue.json").read_text(encoding="utf-8"))
    assert stored == catalogue_entries  # the file the overrides apply to is untouched


def test_an_override_without_a_matching_catalogue_slice_is_ignored(expansion_cfg, catalogue_entries):
    cfg = expand_slices(
        expansion_cfg({"99/1": {"kind": "league", "expected_matches": 1}}, catalogue=catalogue_entries)
    )
    assert keys_of(cfg) == CORE_KEYS + DISCOVERED_KEYS


# --- first run: expansion on, catalogue.json not built yet ---------------------------------------


@pytest.fixture
def enabled_config_path(config_copy, overrides_doc):
    """A copy of default.yaml with expansion switched on, in a ``configs/`` directory of its own."""
    path = config_copy(enabled=True)
    (path.parent / "catalogue_overrides.yaml").write_text(
        yaml.safe_dump(overrides_doc), encoding="utf-8"
    )
    return path


def write_catalogue(config_path, entries):
    cache = config_path.parent.parent / "data" / "raw" / "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "catalogue.json").write_text(json.dumps(entries), encoding="utf-8")


def test_load_config_without_a_catalogue_returns_the_core_slices_and_is_pending(
    enabled_config_path,
):
    cfg = load_config(enabled_config_path)
    assert cfg.catalogue_pending is True
    assert keys_of(cfg) == CORE_KEYS
    assert cfg.core_slices == cfg.slices


def test_load_config_reads_the_catalogue_from_the_cache_of_the_pinned_revision(
    enabled_config_path, catalogue_entries
):
    write_catalogue(enabled_config_path, catalogue_entries)
    cfg = load_config(enabled_config_path)
    assert cfg.catalogue_pending is False
    assert keys_of(cfg) == CORE_KEYS + DISCOVERED_KEYS
    assert {s.name: s.kind for s in cfg.slices}["UEFA Euro 2020"] == "tournament"


def test_a_disabled_expansion_needs_neither_the_overrides_file_nor_the_catalogue(config_copy):
    path = config_copy(enabled=False)  # no overrides file next to it, no data/raw
    cfg = load_config(path)
    assert cfg.catalogue_pending is False and len(cfg.slices) == 5


def test_a_catalogue_in_the_cache_is_ignored_while_expansion_is_disabled(
    config_copy, catalogue_entries
):
    path = config_copy(enabled=False)
    write_catalogue(path, catalogue_entries)
    assert len(load_config(path).slices) == 5


def test_an_enabled_expansion_without_an_overrides_file_is_an_error(enabled_config_path):
    (enabled_config_path.parent / "catalogue_overrides.yaml").unlink()
    with pytest.raises(ConfigError, match="catalogue_overrides.yaml"):
        load_config(enabled_config_path)


def test_an_absolute_overrides_path_stays_absolute(enabled_config_path, tmp_path):
    elsewhere = tmp_path / "elsewhere.yaml"
    elsewhere.write_text(yaml.safe_dump({"12/27": {"kind": "league", "expected_matches": 38}}))
    (enabled_config_path.parent / "catalogue_overrides.yaml").unlink()
    enabled_config_path.write_text(
        enabled_config_path.read_text().replace(
            'overrides: "configs/catalogue_overrides.yaml"', f"overrides: '{elsewhere.as_posix()}'"
        )
    )
    assert load_config(enabled_config_path).catalogue_pending is True  # found: no ConfigError


# --- bad overrides and a bad catalogue -----------------------------------------------------------


@pytest.mark.parametrize(
    "text, message",
    [
        pytest.param('"12/27": {kind: Tournament, expected_matches: null}\n', "kind", id="bad-kind"),
        pytest.param('"12/27": {expected_matches: 38}\n', "kind", id="missing-kind"),
        pytest.param(
            '"99/1": {kind: cup, expected_matches: null}\n', "kind", id="bad-kind-not-in-catalogue"
        ),
        pytest.param('"12-27": {kind: league}\n', "12-27", id="key-without-slash"),
        pytest.param('"a/b": {kind: league}\n', "a/b", id="key-not-ints"),
        pytest.param('"12/27/1": {kind: league}\n', "12/27/1", id="key-with-three-parts"),
        pytest.param('"12/": {kind: league}\n', "12/", id="key-with-half"),
        pytest.param("12: {kind: league}\n", "12", id="key-not-a-string"),
        pytest.param('"12/27": league\n', "12/27", id="entry-not-a-mapping"),
        pytest.param(
            '"12/27": {kind: league, expected_matches: "38"}\n', "expected_matches", id="text-count"
        ),
        pytest.param(
            '"12/27": {kind: league, expected_matches: 0}\n', "expected_matches", id="zero-count"
        ),
        pytest.param(
            '"12/27": {kind: league, expected_matches: true}\n', "expected_matches", id="bool-count"
        ),
        pytest.param('"12/27": {kind: league, extra: 1}\n', "extra", id="unknown-field"),
        pytest.param(
            '"12/27": {kind: league}\n"012/27": {kind: league}\n', "012/27", id="same-slice-twice"
        ),
        pytest.param(
            '"12/27": {kind: league}\n"12/27": {kind: league}\n', "duplicate", id="key-twice"
        ),
        pytest.param("- 12/27\n", "mapping", id="not-a-mapping"),
        pytest.param("{unclosed\n", "catalogue_overrides", id="not-yaml"),
    ],
)
def test_bad_overrides_are_a_config_error(expansion_cfg, catalogue_entries, text, message):
    cfg = expansion_cfg(overrides_text=text, catalogue=catalogue_entries)
    with pytest.raises(ConfigError, match=message):
        expand_slices(cfg)


def test_bad_overrides_are_an_error_even_before_the_catalogue_is_built(expansion_cfg):
    with pytest.raises(ConfigError, match="kind"):
        expand_slices(expansion_cfg(overrides_text='"12/27": {kind: cup}\n'))


def test_an_empty_overrides_file_means_no_overrides(expansion_cfg, catalogue_entries):
    cfg = expand_slices(expansion_cfg(overrides_text="", catalogue=catalogue_entries))
    assert {s.name: s.kind for s in cfg.slices}["UEFA Euro 2020"] == "league"


@pytest.mark.parametrize(
    "content", ['{"not": "a list"}', "[1]", '[{"competition_id": 1}]', "not json"]
)
def test_a_malformed_catalogue_is_a_config_error(expansion_cfg, content):
    cfg = expansion_cfg()
    cache = cfg.raw_dir / cfg.sha
    cache.mkdir(parents=True)
    (cache / "catalogue.json").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="catalogue.json"):
        expand_slices(cfg)


# --- config_hash -------------------------------------------------------------------------------


def hash_of(document):
    return hashlib.sha256(yaml.safe_dump(document, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def test_hash_with_expansion_disabled_is_that_of_the_parsed_yaml_alone(cfg_tmp, catalogue_entries):
    assert cfg_tmp.config_hash() == hash_of(cfg_tmp._doc)
    # a catalogue lying in the cache does not enter a disabled config's hash
    cache = cfg_tmp.raw_dir / cfg_tmp.sha
    cache.mkdir(parents=True)
    (cache / "catalogue.json").write_text(json.dumps(catalogue_entries), encoding="utf-8")
    assert cfg_tmp.config_hash() == hash_of(cfg_tmp._doc)


def test_hash_with_expansion_enabled_folds_in_the_overrides_and_the_catalogue(
    cfg_expanded, overrides_doc, catalogue_entries
):
    assert cfg_expanded.config_hash() == hash_of(
        {"config": cfg_expanded._doc, "overrides": overrides_doc, "catalogue": catalogue_entries}
    )
    assert cfg_expanded.config_hash() != hash_of(cfg_expanded._doc)


def test_hash_with_expansion_enabled_and_no_catalogue_yet_holds_none(expansion_cfg, overrides_doc):
    pending = expand_slices(expansion_cfg())
    assert pending.catalogue_pending is True
    assert pending.config_hash() == hash_of(
        {"config": pending._doc, "overrides": overrides_doc, "catalogue": None}
    )


def test_different_overrides_never_share_a_hash(expansion_cfg, catalogue_entries, overrides_doc):
    a = expansion_cfg(catalogue=catalogue_entries).config_hash()
    other = {**overrides_doc, "12/27": {"kind": "league", "expected_matches": 30}}
    assert expansion_cfg(other, catalogue=catalogue_entries).config_hash() != a


def test_different_catalogues_never_share_a_hash(expansion_cfg, catalogue_entries):
    a = expansion_cfg(catalogue=catalogue_entries).config_hash()
    assert expansion_cfg(catalogue=catalogue_entries[:2]).config_hash() != a


# --- the real overrides file -------------------------------------------------------------------


def test_the_real_overrides_file_has_80_well_formed_entries():
    # Structure only: the kinds and the match counts are reviewed by hand (plan Step 5) and may
    # change without a test change.
    raw = REAL_OVERRIDES.read_bytes()
    raw.decode("ascii")  # raises if any byte is not ASCII
    entries = yaml.safe_load(raw)
    assert len(entries) == 80
    for key, entry in entries.items():
        assert isinstance(key, str) and re.fullmatch(r"[0-9]+/[0-9]+", key), key
        assert set(entry) == {"kind", "expected_matches"}, key
        assert entry["kind"] in KINDS, key
        expected = entry["expected_matches"]
        assert expected is None or (type(expected) is int and expected > 0), key


def test_the_real_overrides_file_passes_the_loader_validation(expansion_cfg):
    cfg = expansion_cfg(overrides_text=REAL_OVERRIDES.read_text(encoding="ascii"))
    assert expand_slices(cfg).catalogue_pending is True  # parsed and validated, no catalogue yet
