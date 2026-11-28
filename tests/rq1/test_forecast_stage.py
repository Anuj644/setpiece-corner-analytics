"""Tests for the forecast stage (Task 20)."""

import dataclasses
import json

import pandas as pd
from forecast_helpers import synthetic_corners

from setpiece.config import Param
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


def fig_dir(cfg):
    return cfg.reports_dir / "figures" / "forecast"


def test_writes_a_manifest_and_one_reliability_figure_per_fitted_slice(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    counts = run_forecast(cfg)
    manifest = json.loads((cfg.processed_dir / "manifests" / "forecast.json").read_text())
    assert manifest["stage"] == "forecast" and manifest["counts"] == counts
    assert counts["slices_fitted"] == 1 and counts["slices_skipped"] == 1
    assert [p.name for p in fig_dir(cfg).glob("*.svg")] == ["9_281_reliability.svg"]


def test_manifest_counts_each_slice_and_the_rows_of_the_three_tables(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    corners = read_table("derived/rq1_corners", cfg)
    corners.loc[corners.index[:7], "ends_in_box"] = pd.NA  # 7 rows excluded, all in slice 9/281
    write_table(corners, "derived/rq1_corners", cfg, None)
    counts = run_forecast(cfg)
    res, team, sk = tables(cfg)

    big = counts["slices"]["9/281"]
    assert big["corners"] == int((corners.competition_id == 9).sum())
    assert big["n_excluded"] == 7
    assert big["test_corners"] == int(res.n_test_corners.iloc[0])
    assert big["test_matches"] == int(res.n_test_matches.iloc[0])
    assert counts["slices"]["2/27"] == {
        "corners": int((corners.competition_id == 2).sum()),
        "skipped": "below forecast_min_matches",
    }
    assert counts["tables"] == {
        "forecast_results": len(res),
        "forecast_team": len(team),
        "forecast_skipped": len(sk),
    }
    assert counts["tables"]["forecast_results"] == 3 and counts["tables"]["forecast_skipped"] == 1


def test_the_stage_deletes_the_figures_of_an_earlier_run(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    fig_dir(cfg).mkdir(parents=True)
    stale = fig_dir(cfg) / "2_27_reliability.svg"  # the slice that is skipped: no figure may stay
    stale.write_text("<svg/>")
    run_forecast(cfg)
    assert not stale.exists() and (fig_dir(cfg) / "9_281_reliability.svg").exists()


def test_a_slice_that_becomes_skipped_loses_its_old_figure(cfg_tmp_with_synthetic_rq1):
    cfg = cfg_tmp_with_synthetic_rq1
    run_forecast(cfg)
    assert (fig_dir(cfg) / "9_281_reliability.svg").exists()
    param = cfg.params["forecast_min_matches"]
    stricter = dataclasses.replace(
        cfg, params={**cfg.params, "forecast_min_matches": Param(1000, param.status, "")}
    )
    run_forecast(stricter)
    assert list(fig_dir(cfg).glob("*.svg")) == []
    assert tables(cfg)[0].empty


def test_the_reliability_figure_overlays_the_three_methods_and_dates_the_test_window(
    cfg_tmp_with_synthetic_rq1,
):
    cfg = cfg_tmp_with_synthetic_rq1
    run_forecast(cfg)
    svg = (fig_dir(cfg) / "9_281_reliability.svg").read_text(encoding="utf-8")
    corners = read_table("derived/rq1_corners", cfg, [BIG])
    _, test = time_split(prepare(corners)[0], cfg.p("forecast_test_share"))
    for part in (
        "1. Bundesliga 2023/24",
        f"test window {test.match_date.min()}",
        str(test.match_date.max()),
        f"n={len(test)} corners",
        "model", "league_side", "team_side_shrunk", "(chosen)",
    ):  # fmt: skip
        assert part in svg, f"{part!r} missing from the reliability figure"


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
