"""``forecast`` stage: time split, two baselines and a logistic regression, decision rule (Task 20).

For every configured slice the first delivery's ``ends_in_box`` is forecast from team and side,
trained on the earlier matches and scored on the later ones. Corners are read one slice at a time,
and the three tables are written (empty, with their columns, if nothing was fitted) through
``write_table`` as ``derived/forecast_*``, with a manifest.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from setpiece.config import Config, Slice
from setpiece.io import read_table, write_manifest, write_table
from setpiece.rq1.forecast.baselines import LeagueSide, ShrunkTeamSide, choose_m
from setpiece.rq1.forecast.evaluate import brier, brier_diff_ci, decide
from setpiece.rq1.forecast.figures import plot_reliability
from setpiece.rq1.forecast.model import fit_logreg
from setpiece.rq1.forecast.split import SIDES, labels, prepare, time_split

BASELINES = ("league_side", "team_side_shrunk")  # in the order of the decision rule's preference

SKIPPED_DTYPES = {
    "competition_id": "Int64",
    "season_id": "Int64",
    "n_matches": "Int64",
    "reason": "string",
}
RESULTS_DTYPES = {
    "competition_id": "Int64",
    "season_id": "Int64",
    "method": "string",
    "brier": "Float64",
    "diff_vs_model": "Float64",
    "ci_lo": "Float64",
    "ci_hi": "Float64",
    "n_test_corners": "Int64",
    "n_test_matches": "Int64",
    "n_excluded": "Int64",
    "chosen": "boolean",
    "C": "Float64",
    "m": "Float64",
    "as_of_date": "string",
}
TEAM_DTYPES = {
    "competition_id": "Int64",
    "season_id": "Int64",
    "team_id": "Int64",
    "n_test_corners": "Int64",
    "supported": "boolean",
    "p_left": "Float64",
    "p_right": "Float64",
}


def _table(rows: list[dict], dtypes: dict[str, str]) -> pd.DataFrame:
    """``rows`` as a table with exactly the columns and dtypes of ``dtypes``, even with no rows."""
    return pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)


def _team_forecasts(model, train: pd.DataFrame, test: pd.DataFrame, min_test: int) -> list[dict]:
    """What the chosen method predicts for each team and side, whatever the test set holds.

    Every team seen in training or testing gets a row. ``p_left`` and ``p_right`` are the model's
    probabilities for a corner of that team on that side (not an average over the corners the team
    happened to take in the test period), and ``supported`` says whether the backtest holds enough
    corners of the team to judge the forecast.
    """
    teams = sorted(set(train["team_id"]) | set(test["team_id"]))
    grid = pd.DataFrame({"team_id": np.repeat(teams, len(SIDES)), "corner_side": SIDES * len(teams)})
    grid["p"] = model.predict(grid)
    p = grid.pivot(index="team_id", columns="corner_side", values="p")
    n_test = test.groupby("team_id").size()
    return [
        {
            "team_id": team,
            "n_test_corners": int(n_test.get(team, 0)),
            "supported": int(n_test.get(team, 0)) >= min_test,
            "p_left": float(p.loc[team, "left"]),
            "p_right": float(p.loc[team, "right"]),
        }
        for team in teams
    ]


def _forecast_slice(sl: Slice, corners: pd.DataFrame, cfg: Config):
    """Fit and score one slice: ``(results rows, team rows, skip row, figure)``.

    Exactly one of the skip row and the other three is not None.
    """
    key = {"competition_id": sl.competition_id, "season_id": sl.season_id}
    n_matches = corners["match_id"].nunique()

    def skip(reason: str):
        return None, None, {**key, "n_matches": n_matches, "reason": reason}, None

    if n_matches < cfg.p("forecast_min_matches"):
        return skip("below forecast_min_matches")
    clean, n_excluded = prepare(corners)
    train, test = time_split(clean, cfg.p("forecast_test_share"))
    if train["ends_in_box"].nunique() < 2:
        return skip("single class in training labels")

    seed, folds = cfg.seed, cfg.p("forecast_cv_folds")
    m = choose_m(train, cfg.p("forecast_m_grid"), folds, seed)
    models = {
        "model": fit_logreg(train, cfg.p("forecast_C_grid"), folds, seed),
        "league_side": LeagueSide.fit(train),
        "team_side_shrunk": ShrunkTeamSide.fit(train, m),
    }
    y = labels(test)
    pred = {name: model.predict(test) for name, model in models.items()}
    score = {name: brier(y, p) for name, p in pred.items()}
    diff = {
        name: brier_diff_ci(test, pred["model"], pred[name], cfg.p("bootstrap_reps"), seed)
        for name in BASELINES
    }
    chosen = decide(score["model"], {name: (score[name], *diff[name]) for name in BASELINES})

    common = {
        **key,
        "n_test_corners": len(test),
        "n_test_matches": test["match_id"].nunique(),
        "n_excluded": n_excluded,
        "as_of_date": test["match_date"].min(),  # the first test match: training ends before it
    }
    extra = {"model": {"C": models["model"].C}, "team_side_shrunk": {"m": m}}
    results = []
    for name in ("model", *BASELINES):
        d, lo, hi = diff.get(name, (0.0, np.nan, np.nan))
        results.append(
            {
                **common, "method": name, "brier": score[name], "diff_vs_model": d,
                "ci_lo": lo, "ci_hi": hi, "chosen": name == chosen, **extra.get(name, {}),
            }
        )  # fmt: skip
    teams = [
        {**key, **row}
        for row in _team_forecasts(
            models[chosen], train, test, cfg.p("supported_backtest_min_test")
        )
    ]
    figure = plot_reliability(y, pred[chosen], f"{sl.name} — {chosen} (n={len(test)} test corners)")
    return results, teams, None, figure


def run_forecast(cfg: Config) -> dict:
    """Run the forecast stage and return its counts (also written to ``manifests/forecast.json``).

    Reads ``derived/rq1_corners`` and writes ``derived/forecast_results``, ``forecast_team`` and
    ``forecast_skipped``, plus ``reports/figures/forecast/<comp>_<season>_reliability.svg`` of the
    chosen method for each fitted slice. A slice with too few matches, none at all, or one class
    in its training labels is skipped, with the reason recorded; the stage still succeeds.
    """
    fig_dir = cfg.reports_dir / "figures" / "forecast"
    results, teams, skipped = [], [], []
    chosen_by_slice = {}

    for sl in cfg.slices:
        corners = read_table("derived/rq1_corners", cfg, [(sl.competition_id, sl.season_id)])
        slice_results, slice_teams, skip_row, figure = _forecast_slice(sl, corners, cfg)
        if skip_row is not None:
            skipped.append(skip_row)
            continue
        results += slice_results
        teams += slice_teams
        chosen_by_slice[f"{sl.competition_id}/{sl.season_id}"] = next(
            r["method"] for r in slice_results if r["chosen"]
        )
        fig_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(fig_dir / f"{sl.competition_id}_{sl.season_id}_reliability.svg")
        plt.close(figure)

    write_table(_table(results, RESULTS_DTYPES), "derived/forecast_results", cfg, None)
    write_table(_table(teams, TEAM_DTYPES), "derived/forecast_team", cfg, None)
    write_table(_table(skipped, SKIPPED_DTYPES), "derived/forecast_skipped", cfg, None)

    counts = {
        "slices_fitted": len(chosen_by_slice),
        "slices_skipped": len(skipped),
        "chosen": chosen_by_slice,
        "team_rows": len(teams),
    }
    write_manifest("forecast", cfg, counts)
    return counts
