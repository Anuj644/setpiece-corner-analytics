import json
from pathlib import Path

import pytest

from setpiece.ingest.events import build_events_frame

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def raw_match():
    return json.loads((FIXTURES / "raw_matches_min.json").read_text(encoding="utf-8"))[0]


@pytest.fixture
def raw_events():
    return json.loads((FIXTURES / "raw_events_min.json").read_text(encoding="utf-8"))


@pytest.fixture
def df(raw_events, raw_match):
    """The Task 4 fixture match flattened into the ``events`` table."""
    return build_events_frame(raw_events, raw_match)


@pytest.fixture
def events(df):
    """Alias of ``df`` under the name later tasks use."""
    return df


@pytest.fixture
def frames():
    """Raw 360 frames: corner-1, a non-corner event that exists, and one unknown uuid."""
    return json.loads((FIXTURES / "raw_360_min.json").read_text(encoding="utf-8"))
