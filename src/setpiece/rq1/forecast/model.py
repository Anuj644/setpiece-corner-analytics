"""Forecast lane: L2 logistic regression on team and side (Task 18).

Pipeline: ``OneHotEncoder(handle_unknown="ignore")`` on ``[team_id, corner_side]`` followed by
``LogisticRegression`` (L2 is the default penalty; scikit-learn 1.8 deprecated spelling it out).
``C``, the inverse regularisation strength, is the value with the lowest mean cross-validated Brier
score over the grouped folds of ``baselines.grouped_folds``; a tie goes to the smaller ``C``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from setpiece.rq1.forecast.baselines import grouped_folds
from setpiece.rq1.forecast.evaluate import brier
from setpiece.rq1.forecast.split import labels

FEATURES = ["team_id", "corner_side"]


def _pipeline(C: float) -> Pipeline:
    return Pipeline(
        [
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("clf", LogisticRegression(C=C, solver="lbfgs", max_iter=1000)),
        ]
    )


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """The model inputs as text, so team ids and sides are categories, not numbers."""
    return df[FEATURES].astype(str)


@dataclass
class FittedLogReg:
    """A logistic regression fitted on all of the training data, with the ``C`` it used."""

    pipeline: Pipeline
    C: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Probability that each corner of ``df`` ends in the box; unseen teams are tolerated."""
        return self.pipeline.predict_proba(_features(df))[:, 1]


def fit_logreg(
    train: pd.DataFrame,
    C_grid: list[float],
    n_folds: int,
    seed: int,
) -> FittedLogReg:
    """Fit the model, choosing ``C`` from ``C_grid`` by grouped cross-validated Brier score.

    ``train`` needs ``team_id``, ``corner_side``, ``match_id`` and a null-free ``ends_in_box``.
    """
    folds = grouped_folds(train, n_folds, seed)
    best_C, best_brier = None, float("inf")
    for C in sorted(C_grid):
        scores = []
        for fit_idx, val_idx in folds:
            fit, val = train.iloc[fit_idx], train.iloc[val_idx]
            pipe = _pipeline(C).fit(_features(fit), labels(fit).astype(int))
            scores.append(brier(labels(val), pipe.predict_proba(_features(val))[:, 1]))
        mean_score = float(np.mean(scores))
        if mean_score < best_brier:  # strict: on a tie the earlier, smaller C stays
            best_C, best_brier = C, mean_score

    final = _pipeline(best_C).fit(_features(train), labels(train).astype(int))
    return FittedLogReg(final, best_C)
