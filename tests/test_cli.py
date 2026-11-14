import shutil
from pathlib import Path

import pytest

from setpiece import cli
from setpiece.cli import main
from setpiece.config import load_config
from setpiece.stages.ingest import run_fetch, run_ingest

REPO = Path(__file__).resolve().parents[1]


def test_help_lists_all_stages(capsys):
    try:
        main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    for stage in ["fetch", "ingest", "qa", "rq1", "forecast", "rq2", "rq3",
                  "coverage", "dossier", "paper-assets", "all", "review-export"]:
        assert stage in out

def test_unimplemented_stage_returns_2(capsys):
    assert main(["coverage"]) == 2
    assert "not implemented yet" in capsys.readouterr().err


# --- fetch and ingest wiring ---------------------------------------------------------------


@pytest.fixture
def config_path(tmp_path):
    """A copy of the real config in a ``configs/`` directory of its own."""
    path = tmp_path / "configs" / "default.yaml"
    path.parent.mkdir()
    shutil.copy(REPO / "configs" / "default.yaml", path)
    return path


@pytest.fixture
def calls(monkeypatch):
    """Replace the fetch and ingest runners with recorders, so no network is touched."""
    seen = []
    for stage in ("fetch", "ingest"):
        monkeypatch.setitem(cli.RUNNERS, stage, lambda cfg, stage=stage: seen.append((stage, cfg)))
    return seen


@pytest.mark.parametrize("stage", ["fetch", "ingest"])
def test_implemented_stage_runs_with_the_loaded_config_and_returns_0(stage, config_path, calls):
    assert main([stage, "--config", str(config_path)]) == 0
    [(ran, cfg)] = calls
    assert ran == stage
    assert cfg == load_config(config_path)
    assert cfg.root == config_path.parent.parent


def test_default_config_is_the_repos_when_none_is_given(calls):
    assert main(["ingest"]) == 0
    assert calls[0][1] == load_config(REPO / "configs" / "default.yaml")


def test_config_error_prints_to_stderr_and_returns_1_without_running(config_path, calls, capsys):
    config_path.write_text("source: {}\n", encoding="utf-8")
    assert main(["ingest", "--config", str(config_path)]) == 1
    assert "missing required top-level key" in capsys.readouterr().err
    assert calls == []


def test_missing_config_file_returns_1_without_running(tmp_path, calls, capsys):
    assert main(["fetch", "--config", str(tmp_path / "configs" / "nope.yaml")]) == 1
    assert "nope.yaml" in capsys.readouterr().err
    assert calls == []


def test_unimplemented_stage_does_not_read_the_config(tmp_path, calls, capsys):
    broken = tmp_path / "not-a-configs-dir" / "default.yaml"
    assert main(["coverage", "--config", str(broken)]) == 2
    assert "stage 'coverage' not implemented yet" in capsys.readouterr().err
    assert calls == []


def test_fetch_and_ingest_are_the_implemented_stages():
    assert set(cli.RUNNERS) == {"fetch", "ingest"}
    assert set(cli.RUNNERS) <= set(cli.STAGES)


def test_runners_are_the_stage_functions():
    assert cli.RUNNERS == {"fetch": run_fetch, "ingest": run_ingest}
