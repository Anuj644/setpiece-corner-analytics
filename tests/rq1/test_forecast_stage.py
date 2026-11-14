"""Tests for the forecast stage (Task 20)."""

import json

import pandas as pd
from forecast_helpers import synthetic_corners

from setpiece.io import read_table, write_table
from setpiece.rq1.forecast.split import prepare, time_split
from setpiece.stages.forecast import RESULTS_DTYPES, SKIPPED_DTYPES, TEAM_DTYPES, run_forecast

BIG = (9, 281)


def tables(cfg):
    return (
        read_table("derived/forecast_results", cfg),
        read_table("derived/forecast_team", cfg),
        read_table("derived/forecast_skipped", cfg),
    )


def test_stage_outputs_on_synthetic(cfg_tmp_with_synthetic_rq1):
    run_forecast(cfg_tmp_with_synthetic_rq1)
    res, _, _ = tables(cfg_tmp_with_synthetic_rq1)
    assert list(res.columns) == list(RESULTS_DTYPES)
    assert set(res.method) == {"model", "league_side", "team_side_shrunk"}
    assert res.groupby(["competition_id", "season_id"]).chosen.sum().eq(1).all()
    assert res.groupby("method").n_test_corners.nunique().eq(1).all()


def test_supported_threshold(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    corners = read_table("derived/rq1_corners", cfg)
    big = corners.competition_id == BIG[0]
    last = corners[big].match_id.max()  # team 4 now appears only in the final match: few corners
    corners.loc[big & (corners.team_id == 4) & (corners.match_id < last), "team_id"] = 3
    write_table(corners, "derived/rq1_corners", cfg, None)
    run_forecast(cfg)
    _, t, _ = tables(cfg)
    assert list(t.columns) == list(TEAM_DTYPES)
    minimum = cfg.p("supported_backtest_min_test")
    assert (t.supported == (t.n_test_corners >= minimum)).all()
    by_team = t.set_index("team_id")
    assert by_team.loc[4, "n_test_corners"] < minimum and not by_team.loc[4, "supported"]
    assert by_team.loc[3, "n_test_corners"] >= minimum and by_team.loc[3, "supported"]


def test_small_slice_is_skipped_not_fitted(cfg_tmp_with_synthetic_rq1):
    run_forecast(cfg_tmp_with_synthetic_rq1)
    res, team, sk = tables(cfg_tmp_with_synthetic_rq1)
    assert list(sk.columns) == list(SKIPPED_DTYPES)
    assert len(sk) == 1 and sk.n_matches.iloc[0] == 5 and sk.season_id.iloc[0] == 27
    assert 27 not in set(res.season_id) and 27 not in set(team.season_id)


def test_team_probabilities_are_the_chosen_models_for_every_team_and_side(
    cfg_tmp_with_synthetic_rq1,
):
    """Every team seen in training or testing gets both probabilities, test corners or not."""
    cfg = cfg_tmp_with_synthetic_rq1
    # team 9 plays only in the first two matches, which are all training: no test corner at all
    corners = read_table("derived/rq1_corners", cfg, [BIG])
    corners.loc[corners.match_id <= 2, "team_id"] = 9
    write_table(
        pd.concat([corners, read_table("derived/rq1_corners", cfg, [(2, 27)])]),
        "derived/rq1_corners", cfg, None,
    )  # fmt: skip
    run_forecast(cfg)
    _, team, _ = tables(cfg)
    train, test = time_split(prepare(corners)[0], cfg.p("forecast_test_share"))
    assert 9 in set(train.team_id) and 9 not in set(test.team_id)
    assert set(team.team_id) == set(train.team_id) | set(test.team_id)
    assert team[["p_left", "p_right"]].notna().all().all()
    assert team[["p_left", "p_right"]].gt(0).all().all() and team[["p_left", "p_right"]].lt(1).all().all()
    train_only = team[team.team_id == 9].iloc[0]
    assert train_only.n_test_corners == 0 and not train_only.supported


def test_as_of_date_is_after_every_training_date_even_with_ties(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    run_forecast(cfg)
    res, _, _ = tables(cfg)
    corners = read_table("derived/rq1_corners", cfg, [BIG])
    assert corners.groupby("match_date").match_id.nunique().eq(2).all()  # matches come in pairs
    train, test = time_split(prepare(corners)[0], cfg.p("forecast_test_share"))
    as_of = res.as_of_date.unique()
    assert len(as_of) == 1 and as_of[0] == test.match_date.min()
    assert as_of[0] > train.match_date.max()
    assert res.n_test_matches.iloc[0] == test.match_id.nunique()


def test_excluded_rows_are_counted(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    corners = read_table("derived/rq1_corners", cfg)
    corners.loc[corners.index[:7], "ends_in_box"] = pd.NA
    write_table(corners, "derived/rq1_corners", cfg, None)
    run_forecast(cfg)
    res, _, _ = tables(cfg)
    assert res.n_excluded.eq(7).all()


def test_writes_a_manifest_and_one_reliability_figure_per_fitted_slice(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    counts = run_forecast(cfg)
    manifest = json.loads((cfg.processed_dir / "manifests" / "forecast.json").read_text())
    assert manifest["stage"] == "forecast" and manifest["counts"] == counts
    assert counts["slices_fitted"] == 1 and counts["slices_skipped"] == 1
    figures = [p.name for p in (cfg.reports_dir / "figures" / "forecast").glob("*.svg")]
    assert figures == ["9_281_reliability.svg"]


def test_all_slices_skipped_still_writes_typed_empty_tables(cfg_tmp_with_synthetic_rq1):
    """Nothing to fit is not an error: the tables exist with their columns and dtypes."""
    import dataclasses

    cfg = dataclasses.replace(cfg_tmp_with_synthetic_rq1, slices=cfg_tmp_with_synthetic_rq1.slices[1:])
    run_forecast(cfg)
    res, team, sk = tables(cfg)
    assert res.empty and team.empty and len(sk) == 1
    assert list(res.columns) == list(RESULTS_DTYPES) and list(team.columns) == list(TEAM_DTYPES)
    assert res.competition_id.dtype == "Int64" and team.supported.dtype == "boolean"
    assert json.loads((cfg.processed_dir / "manifests" / "forecast.json").read_text())["counts"][
        "slices_fitted"
    ] == 0


def test_a_slice_with_no_corners_is_skipped_with_zero_matches(cfg_tmp_with_synthetic_rq1):
    import dataclasses

    from setpiece.config import Slice

    cfg = dataclasses.replace(
        cfg_tmp_with_synthetic_rq1, slices=[Slice(43, 106, "WC", "tournament", True)]
    )
    run_forecast(cfg)
    _, _, sk = tables(cfg)
    assert sk.n_matches.tolist() == [0]


def test_single_class_training_labels_are_skipped(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    corners = read_table("derived/rq1_corners", cfg)
    corners["ends_in_box"] = True
    write_table(corners, "derived/rq1_corners", cfg, None)
    run_forecast(cfg)
    res, _, sk = tables(cfg)
    assert res.empty and (sk.reason == "single class in training labels").sum() == 1


def test_run_is_repeatable(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    run_forecast(cfg)
    first = tables(cfg)
    run_forecast(cfg)
    for a, b in zip(first, tables(cfg)):
        pd.testing.assert_frame_equal(a, b)


def test_helper_is_the_documented_one():
    assert synthetic_corners(2, [1, 2], seed=0).match_id.nunique() == 2
