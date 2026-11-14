"""``forecast`` stage: time split, baselines, logreg, decision rule, outputs (Task 20)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from setpiece.config import Config
from setpiece.io import derived_dir
from setpiece.rq1.forecast.baselines import (
    ShrunkTeamSide,
    choose_m,
    league_side_rates,
)
from setpiece.rq1.forecast.evaluate import brier, brier_diff_ci, decide
from setpiece.rq1.forecast.model import fit_logreg
from setpiece.rq1.forecast.split import prepare, time_split


def run_forecast(cfg: Config) -> dict:
    """Run the forecast stage.

    Reads ``derived/rq1_corners.parquet``, applies the time split, fits
    baselines and the L2 logistic regression, evaluates with the decision
    rule, and writes three derived tables plus a reliability figure.
    """
    dd = derived_dir(cfg)
    dd.mkdir(parents=True, exist_ok=True)

    corners = pd.read_parquet(dd / "rq1_corners.parquet")

    min_matches = cfg.p("forecast_min_matches")
    test_share = cfg.p("forecast_test_share")
    c_grid = cfg.p("forecast_C_grid")
    m_grid = cfg.p("forecast_m_grid")
    n_folds = cfg.p("forecast_cv_folds")
    seed = cfg.seed
    supported_min = cfg.p("supported_backtest_min_test")

    skipped_rows = []
    results_rows = []
    team_rows = []

    # Per-slice processing.
    slice_keys = ["competition_id", "season_id"]
    for (comp_id, seas_id), grp in corners.groupby(slice_keys):
        n_matches = grp["match_id"].nunique()

        # Prepare: drop null labels.
        clean, n_excluded = prepare(grp)

        if n_matches < min_matches:
            skipped_rows.append(
                {
                    "competition_id": comp_id,
                    "season_id": seas_id,
                    "n_matches": n_matches,
                    "reason": "below forecast_min_matches",
                }
            )
            continue

        # Time split.
        train, test = time_split(clean, test_share)

        # Check for single-class training labels.
        if train["ends_in_box"].nunique() < 2:
            skipped_rows.append(
                {
                    "competition_id": comp_id,
                    "season_id": seas_id,
                    "n_matches": n_matches,
                    "reason": "single class in training labels",
                }
            )
            continue

        y_test = test["ends_in_box"].fillna(False).astype(float).to_numpy()
        n_test_corners = len(test)
        n_test_matches = test["match_id"].nunique()

        # As-of date = first test-match date.
        as_of = test["match_date"].min()

        # --- Fit baselines ---
        league_rates = league_side_rates(train)

        # League side baseline.
        p_league = np.array([league_rates.get(s, 0.5) for s in test["corner_side"]])
        brier_league = brier(y_test, p_league)

        # Shrunk team-side baseline.
        best_m = choose_m(train, m_grid, n_folds, seed)
        shrunk_model = ShrunkTeamSide.fit(train, m=best_m)
        p_shrunk = shrunk_model.predict(test)
        brier_shrunk = brier(y_test, p_shrunk)

        # --- Fit logreg ---
        logreg_model = fit_logreg(train, c_grid, n_folds, seed)
        p_model = logreg_model.predict(test)
        brier_model = brier(y_test, p_model)

        # --- CIs on the difference ---
        diff_league, lo_l, hi_l = brier_diff_ci(test, p_model, p_league, cfg.p("bootstrap_reps"), seed)
        diff_shrunk, lo_s, hi_s = brier_diff_ci(test, p_model, p_shrunk, cfg.p("bootstrap_reps"), seed)

        # --- Decision rule ---
        base_results = {
            "league_side": (brier_league, diff_league, lo_l, hi_l),
            "team_side_shrunk": (brier_shrunk, diff_shrunk, lo_s, hi_s),
        }
        chosen = decide(brier_model, base_results)

        results_rows.append(
            {
                "competition_id": comp_id,
                "season_id": seas_id,
                "method": "model",
                "brier": brier_model,
                "diff_vs_model": 0.0,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "n_test_corners": n_test_corners,
                "n_test_matches": n_test_matches,
                "n_excluded": n_excluded,
                "chosen": chosen == "model",
                "C": logreg_model.C,
                "m": np.nan,
                "as_of_date": as_of,
            }
        )
        results_rows.append(
            {
                "competition_id": comp_id,
                "season_id": seas_id,
                "method": "league_side",
                "brier": brier_league,
                "diff_vs_model": diff_league,
                "ci_lo": lo_l,
                "ci_hi": hi_l,
                "n_test_corners": n_test_corners,
                "n_test_matches": n_test_matches,
                "n_excluded": n_excluded,
                "chosen": chosen == "league_side",
                "C": np.nan,
                "m": np.nan,
                "as_of_date": as_of,
            }
        )
        results_rows.append(
            {
                "competition_id": comp_id,
                "season_id": seas_id,
                "method": "team_side_shrunk",
                "brier": brier_shrunk,
                "diff_vs_model": diff_shrunk,
                "ci_lo": lo_s,
                "ci_hi": hi_s,
                "n_test_corners": n_test_corners,
                "n_test_matches": n_test_matches,
                "n_excluded": n_excluded,
                "chosen": chosen == "team_side_shrunk",
                "C": np.nan,
                "m": best_m,
                "as_of_date": as_of,
            }
        )

        # --- Team-level table ---
        # Use the chosen method for p_left / p_right, fitted on train.
        chosen_model = None
        chosen_rates: dict[str, float] | None = None

        if chosen == "model":
            chosen_model = fit_logreg(train, c_grid, n_folds, seed)
        elif chosen == "team_side_shrunk":
            chosen_model = ShrunkTeamSide.fit(train, m=best_m)
        else:
            chosen_rates = dict(league_rates)

        for team_id in test["team_id"].unique():
            team_test = test[test["team_id"] == team_id]
            n_tc = len(team_test)
            if n_tc == 0:
                continue

            if chosen_model is not None:
                p_pred = chosen_model.predict(team_test)
            else:
                p_pred = np.array(
                    [chosen_rates.get(s, 0.5) for s in team_test["corner_side"]]
                )

            # p_left, p_right: average prediction per side.
            left_mask = team_test["corner_side"] == "left"
            right_mask = team_test["corner_side"] == "right"
            p_left = float(p_pred[left_mask].mean()) if left_mask.any() else np.nan
            p_right = float(p_pred[right_mask].mean()) if right_mask.any() else np.nan

            team_rows.append(
                {
                    "competition_id": comp_id,
                    "season_id": seas_id,
                    "team_id": team_id,
                    "n_test_corners": n_tc,
                    "supported": n_tc >= supported_min,
                    "p_left": p_left,
                    "p_right": p_right,
                }
            )

        # --- Reliability figure ---
        fig_dir = cfg.reports_dir / "figures" / "forecast"
        fig_dir.mkdir(parents=True, exist_ok=True)

        from setpiece.rq1.forecast.figures import plot_reliability

        fig = plot_reliability(y_test, p_model, f"{comp_id}/{seas_id} reliability")
        fig.savefig(fig_dir / f"{comp_id}_{seas_id}_reliability.svg")
        import matplotlib.pyplot as plt

        plt.close(fig)

    # Write outputs.
    pd.DataFrame(
        skipped_rows,
        columns=["competition_id", "season_id", "n_matches", "reason"],
    ).to_parquet(dd / "forecast_skipped.parquet", index=False)

    res_df = pd.DataFrame(results_rows)
    res_df.to_parquet(dd / "forecast_results.parquet", index=False)

    team_df = pd.DataFrame(team_rows)
    team_df.to_parquet(dd / "forecast_team.parquet", index=False)

    return {"skipped": len(skipped_rows), "results": len(res_df), "teams": len(team_df)}
