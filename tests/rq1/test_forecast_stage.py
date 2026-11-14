"""Tests for the forecast stage (Task 20)."""

import pandas as pd

from setpiece.io import derived_dir
from setpiece.stages.forecast import run_forecast


def test_stage_outputs_on_synthetic(cfg_tmp_with_synthetic_rq1):
    run_forecast(cfg_tmp_with_synthetic_rq1)
    res = pd.read_parquet(derived_dir(cfg_tmp_with_synthetic_rq1) / "forecast_results.parquet")
    assert set(res.method) == {"model", "league_side", "team_side_shrunk"}
    assert res.groupby(["competition_id", "season_id"]).chosen.sum().eq(1).all()


def test_supported_threshold(cfg_tmp_with_synthetic_rq1):
    run_forecast(cfg_tmp_with_synthetic_rq1)
    t = pd.read_parquet(derived_dir(cfg_tmp_with_synthetic_rq1) / "forecast_team.parquet")
    assert (t.supported == (t.n_test_corners >= 20)).all()


def test_small_slice_is_skipped_not_fitted(cfg_tmp_with_synthetic_rq1):
    run_forecast(cfg_tmp_with_synthetic_rq1)
    sk = pd.read_parquet(derived_dir(cfg_tmp_with_synthetic_rq1) / "forecast_skipped.parquet")
    res = pd.read_parquet(derived_dir(cfg_tmp_with_synthetic_rq1) / "forecast_results.parquet")
    assert len(sk) == 1 and sk.n_matches.iloc[0] == 5
    assert sk.season_id.iloc[0] not in set(res.season_id)
