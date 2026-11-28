import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest
import requests
import yaml
from helpers import read_fixture

from setpiece.config import Param, Slice, load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    """The real repo config; with ``corpus_expansion`` on it reads the real overrides and cache."""
    return load_config(REPO / "configs" / "default.yaml")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test reaches the real network through requests."""

    def refuse(self, method, url, **kwargs):
        raise AssertionError(f"test tried to use the network: {method} {url}")

    monkeypatch.setattr(requests.Session, "request", refuse)


@pytest.fixture
def fake_cfg(cfg, tmp_path):
    """The default config, core slices only, with every output directory redirected to ``tmp_path``.

    The ``corpus_expansion`` param is off and the config is not pending, so a test built on it
    gets the same slices whatever the repo's flag says and whether or not a ``catalogue.json`` is
    cached, and it reads neither the overrides file nor the catalogue (only ``cfg``, which this
    starts from, does when the flag is on). ``expansion_cfg`` turns the param on again.
    """
    expansion = cfg.params["corpus_expansion"]
    off = Param({**expansion.value, "enabled": False}, expansion.status, expansion.note)
    return dataclasses.replace(
        cfg,
        slices=cfg.core_slices,
        params={**cfg.params, "corpus_expansion": off},
        catalogue_pending=False,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
        paper_dir=tmp_path / "paper",
    )


@pytest.fixture
def cfg_tmp(fake_cfg):
    return fake_cfg


# The ``enabled`` flag of the ``corpus_expansion`` param, in the form default.yaml writes it.
EXPANSION_FLAG = re.compile(r"(corpus_expansion:\s*\n\s*value:\s*\{enabled:\s*)(?:true|false)")


@pytest.fixture
def config_copy(tmp_path):
    """Factory: a copy of the real ``default.yaml`` with ``corpus_expansion.enabled`` forced.

    ``make(enabled=False)`` writes ``tmp_path/configs/default.yaml`` and returns its path, so a
    test does not depend on the flag the repo ships with. No overrides file sits beside the copy
    and there is no cache: a test that turns expansion on adds what it needs. Fails loudly if
    the flag is no longer in the expected form.
    """

    def make(*, enabled=False):
        text = (REPO / "configs" / "default.yaml").read_text(encoding="utf-8")
        text, found = EXPANSION_FLAG.subn(lambda m: m.group(1) + str(enabled).lower(), text)
        assert found == 1, "the corpus_expansion flag of default.yaml is not in the expected form"
        path = tmp_path / "configs" / "default.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return make


def catalogue_entry(competition_id, season_id, name, has_360):
    """One slice as ``catalogue.json`` stores it: discovered, so no overrides and ``core`` false."""
    return {"competition_id": competition_id, "season_id": season_id, "name": name,
            "kind": "league", "has_360": has_360, "expected_matches": None, "core": False}  # fmt: skip


# Three discovered slices: the World Cup 2022 is already a core slice, so expansion must skip it.
CATALOGUE_ENTRIES = [
    catalogue_entry(43, 106, "FIFA World Cup 2022", True),
    catalogue_entry(12, 27, "Serie A 2015/2016", False),
    catalogue_entry(55, 43, "UEFA Euro 2020", True),
]

# The 43/106 entry is informational: the core slice's definition in default.yaml wins.
OVERRIDES = {
    "43/106": {"kind": "league", "expected_matches": None},
    "55/43": {"kind": "tournament", "expected_matches": None},
    "12/27": {"kind": "league", "expected_matches": 38},
}


@pytest.fixture
def catalogue_entries():
    """The three discovered slices of ``cfg_expanded``'s ``catalogue.json`` (a fresh copy)."""
    return copy.deepcopy(CATALOGUE_ENTRIES)


@pytest.fixture
def overrides_doc():
    """The overrides of ``cfg_expanded``, as the parsed YAML mapping (a fresh copy)."""
    return copy.deepcopy(OVERRIDES)


@pytest.fixture
def expansion_cfg(cfg_tmp, tmp_path):
    """Factory: ``cfg_tmp`` with ``corpus_expansion`` switched on, not yet expanded.

    ``make(overrides=OVERRIDES, catalogue=None, overrides_text=None)`` writes the overrides file
    into ``tmp_path`` (dumped from ``overrides``, or ``overrides_text`` verbatim) and, if
    ``catalogue`` is given, ``catalogue.json`` into the cfg's cache directory. The Param is swapped
    for a new one, so no test reads the real overrides file.
    """

    def make(overrides=OVERRIDES, catalogue=None, overrides_text=None):
        path = tmp_path / "catalogue_overrides.yaml"
        path.write_text(
            overrides_text if overrides_text is not None else yaml.safe_dump(overrides),
            encoding="utf-8",
        )
        if catalogue is not None:
            cache = cfg_tmp.raw_dir / cfg_tmp.sha
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "catalogue.json").write_text(json.dumps(catalogue), encoding="utf-8")
        old = cfg_tmp.params["corpus_expansion"]
        new = Param({**old.value, "enabled": True, "overrides": str(path)}, old.status, old.note)
        return dataclasses.replace(cfg_tmp, params={**cfg_tmp.params, "corpus_expansion": new})

    return make


@pytest.fixture
def cfg_expanded(expansion_cfg):
    """``cfg_tmp`` expanded: the 5 core slices, then Serie A 2015/16 and Euro 2020."""
    from setpiece.config import expand_slices

    return expand_slices(expansion_cfg(catalogue=CATALOGUE_ENTRIES))


class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    """Stands in for ``requests.Session``: serves ``rel_path -> JSON bytes`` and counts calls."""

    def __init__(self):
        self.files = {}
        self.calls = 0
        self.urls = []
        self._failures = []
        self.serve("competitions.json", [{"competition_id": 43}])

    def serve(self, rel_path, obj):
        self.files[rel_path] = json.dumps(obj).encode("utf-8")

    def add_404(self, rel_path):
        self.files.pop(rel_path, None)

    def fail_next(self, times, *, status=503, exc=None):
        """Make the next ``times`` requests fail with ``exc`` if given, else HTTP ``status``."""
        self._failures.extend([exc if exc is not None else status] * times)

    def get(self, url, **kwargs):
        self.calls += 1
        self.urls.append(url)
        if self._failures:
            failure = self._failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return FakeResponse(failure)
        rel_path = url.split("/data/", 1)[1]
        if rel_path not in self.files:
            return FakeResponse(404)
        return FakeResponse(200, self.files[rel_path])


@pytest.fixture
def fake_session():
    return FakeSession()


@pytest.fixture
def slice_with_one_match_and_360(fake_session):
    fake_session.serve("matches/43/106.json", [{"match_id": 1}])
    fake_session.serve("events/1.json", [{"id": "e1", "type": {"name": "Pass"}}])
    return Slice(43, 106, "t", "tournament", True)


@pytest.fixture
def cfg_tmp_one_slice(cfg_tmp):
    """``cfg_tmp`` with only the World Cup 2022 slice, which the fixture match belongs to."""
    world_cup = Slice(43, 106, "FIFA World Cup 2022", "tournament", True)
    return dataclasses.replace(cfg_tmp, slices=[world_cup])


@pytest.fixture
def fake_session_with_fixtures(fake_session):
    """Serves the one-match fixture slice: match 1 of 43/106 with its events and 360 frames."""
    fake_session.serve("matches/43/106.json", read_fixture("raw_matches_min.json"))
    fake_session.serve("events/1.json", read_fixture("raw_events_min.json"))
    fake_session.serve("three-sixty/1.json", read_fixture("raw_360_min.json"))
    return fake_session
