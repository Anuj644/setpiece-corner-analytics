import dataclasses
import json
from pathlib import Path

import pytest
import requests

from setpiece.config import Slice, load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(REPO / "configs" / "default.yaml")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test reaches the real network through requests."""

    def refuse(self, method, url, **kwargs):
        raise AssertionError(f"test tried to use the network: {method} {url}")

    monkeypatch.setattr(requests.Session, "request", refuse)


@pytest.fixture
def fake_cfg(cfg, tmp_path):
    """The default config with every output directory redirected into ``tmp_path``."""
    return dataclasses.replace(
        cfg,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
        paper_dir=tmp_path / "paper",
    )


@pytest.fixture
def cfg_tmp(fake_cfg):
    return fake_cfg


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
