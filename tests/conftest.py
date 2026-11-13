from pathlib import Path

import pytest

from setpiece.config import load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(REPO / "configs" / "default.yaml")
