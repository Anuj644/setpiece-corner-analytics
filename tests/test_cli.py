import shutil
from pathlib import Path

import pytest

from setpiece import cli
from setpiece.cli import main
from setpiece.config import load_config
from setpiece.qa import QAResult, run_qa
from setpiece.stages.ingest import run_fetch, run_ingest
from setpiece.stages.rq2 import run_rq2

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
    stage = next(s for s in cli.STAGES if s not in cli.RUNNERS)
    assert main([stage]) == 2
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
    """Replace the fetch, ingest and rq2 runners with recorders, so nothing real runs."""
    seen = []
    for stage in ("fetch", "ingest", "rq2"):
        monkeypatch.setitem(cli.RUNNERS, stage, lambda cfg, stage=stage: seen.append((stage, cfg)))
    return seen


@pytest.mark.parametrize("stage", ["fetch", "ingest", "rq2"])
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
    stage = next(s for s in cli.STAGES if s not in cli.RUNNERS)
    broken = tmp_path / "not-a-configs-dir" / "default.yaml"
    assert main([stage, "--config", str(broken)]) == 2
    assert f"stage '{stage}' not implemented yet" in capsys.readouterr().err
    assert calls == []


def test_the_implemented_stages_are_real_stages_and_qa_is_one_of_them():
    assert set(cli.RUNNERS) <= set(cli.STAGES)
    assert {"fetch", "ingest", "qa"} <= set(cli.RUNNERS)


def test_runners_are_the_stage_functions():
    assert cli.RUNNERS["fetch"] is run_fetch
    assert cli.RUNNERS["ingest"] is run_ingest
    assert cli.RUNNERS["qa"] is run_qa
    assert cli.RUNNERS["rq2"] is run_rq2


# --- qa wiring -----------------------------------------------------------------------------------


def qa_result(errors=()):
    warnings = {"coords_out_of_range": 3, "timestamp_decreases": 18}
    return QAResult(list(errors), warnings, {"start_x": 0.25}, 7)


@pytest.fixture
def qa_runs(monkeypatch):
    """Stub the qa runner: it returns ``state["errors"]`` as errors and records the config."""
    state = {"errors": [], "configs": []}

    def stub(cfg):
        state["configs"].append(cfg)
        return qa_result(state["errors"])

    monkeypatch.setitem(cli.RUNNERS, "qa", stub)
    return state


def test_qa_with_no_errors_runs_with_the_loaded_config_and_returns_0(config_path, qa_runs, capsys):
    assert main(["qa", "--config", str(config_path)]) == 0
    assert qa_runs["configs"] == [load_config(config_path)]
    out = capsys.readouterr()
    assert "coords_out_of_range=3" in out.out and "timestamp_decreases=18" in out.out
    assert "shot corners: 7" in out.out
    assert out.err == ""


def test_qa_with_errors_prints_them_and_returns_1(config_path, qa_runs, capsys):
    qa_runs["errors"] = [
        "43/106: duplicate (match_id, event_id): 2 rows",
        "11/90: null period: 1 row",
    ]
    assert main(["qa", "--config", str(config_path)]) == 1
    err = capsys.readouterr().err
    assert "43/106: duplicate (match_id, event_id): 2 rows" in err
    assert "11/90: null period: 1 row" in err


def test_qa_config_error_returns_1_without_running(config_path, qa_runs, capsys):
    config_path.write_text("source: {}\n", encoding="utf-8")
    assert main(["qa", "--config", str(config_path)]) == 1
    assert "missing required top-level key" in capsys.readouterr().err
    assert qa_runs["configs"] == []


def test_the_other_stages_still_return_0_whatever_their_runner_returns(config_path, monkeypatch):
    monkeypatch.setitem(cli.RUNNERS, "ingest", lambda cfg: {"43/106": {"matches": 1}})
    assert main(["ingest", "--config", str(config_path)]) == 0


# --- rq1, forecast and review-export wiring -----------------------------------------------------


def test_rq1_and_forecast_are_registered_runners():
    from setpiece.stages.forecast import run_forecast
    from setpiece.stages.rq1 import run_rq1

    assert cli.RUNNERS["rq1"] is run_rq1
    assert cli.RUNNERS["forecast"] is run_forecast


def test_review_export_rq1_runs_with_the_loaded_config(config_path, monkeypatch):
    seen = []
    monkeypatch.setitem(cli.REVIEW_RUNNERS, "rq1", lambda cfg: seen.append(cfg))
    assert main(["review-export", "rq1", "--config", str(config_path)]) == 0
    assert seen == [load_config(config_path)]


@pytest.mark.parametrize("target", ["rq2", "rq3"])
def test_review_export_of_a_question_not_built_yet_returns_2(target, config_path, capsys):
    assert main(["review-export", target, "--config", str(config_path)]) == 2
    assert f"stage 'review-export {target}' not implemented yet" in capsys.readouterr().err


def test_review_export_needs_a_target(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["review-export"])
    assert exc.value.code == 2
    assert "target" in capsys.readouterr().err
