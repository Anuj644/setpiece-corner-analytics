"""Forecast lane — L2 logistic regression model (Task 18).

Pipeline: ``OneHotEncoder(handle_unknown="ignore")`` on ``[team_id, corner_side]``
→ ``LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000)``.

``C`` (inverse regularisation strength) is chosen by minimising mean CV Brier
score across the grouped folds from Task 17.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from setpiece.rq1.forecast.baselines import _brier_score, grouped_folds

FEATURES = ["team_id", "corner_side"]


def _build_pipeline(C: float) -> Pipeline:
    return Pipeline(
        [
            (
                "enc",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
            (
                "clf",
                # penalty="l2" is the default; l1_ratio=0 is equivalent in
                # scikit-learn ≥1.8 which deprecated the penalty kwarg.
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    )


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[FEATURES].astype(str)
    y = df["ends_in_box"].fillna(False).astype(int).to_numpy()
    return X, y


@dataclass
class FittedLogReg:
    """A fitted logistic regression model with its chosen ``C``."""

    _pipeline: Pipeline
    C: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted ``ends_in_box`` probabilities (class 1)."""
        X = df[FEATURES].astype(str)
        return self._pipeline.predict_proba(X)[:, 1]


def fit_logreg(
    train: pd.DataFrame,
    C_grid: list[float],
    n_folds: int,
    seed: int,
) -> FittedLogReg:
    """Fit an L2 logistic regression with CV-selected ``C``.

    Parameters
    ----------
    train:
        Training corners with ``team_id``, ``corner_side``, ``ends_in_box``
        and ``match_id`` columns.
    C_grid:
        Candidate ``C`` values to try.
    n_folds:
        Number of CV folds (grouped by ``match_id``).
    seed:
        Seed passed to ``grouped_folds`` for reproducibility.

    Returns
    -------
    FittedLogReg
        Model fitted on the full ``train`` set with the best ``C``.
    """
    folds = grouped_folds(train, n_folds, seed)

    best_C = C_grid[0]
    best_brier = float("inf")

    for C in C_grid:
        fold_briers: list[float] = []
        for tr_idx, val_idx in folds:
            tr = train.iloc[tr_idx]
            val = train.iloc[val_idx]
            pipe = _build_pipeline(C)
            X_tr, y_tr = _xy(tr)
            X_val, y_val = _xy(val)
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict_proba(X_val)[:, 1]
            fold_briers.append(_brier_score(y_val.astype(float), preds))
        mean_brier = float(np.mean(fold_briers))
        if mean_brier < best_brier:
            best_brier = mean_brier
            best_C = C

    # Refit on full training set with the best C.
    final_pipe = _build_pipeline(best_C)
    X_all, y_all = _xy(train)
    final_pipe.fit(X_all, y_all)

    return FittedLogReg(_pipeline=final_pipe, C=best_C)
