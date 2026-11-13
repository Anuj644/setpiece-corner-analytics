"""Forecast lane — shrinkage baselines (Task 17).

Two baselines are provided:

1. **League-side rate** — a single probability per side (left/right) pooled
   across all teams.  The simplest possible model.

2. **Shrunk team-side rate** — per-(team, side) rate shrunk toward the league
   side rate using an additive pseudo-count ``m``.  ``m`` is chosen by
   minimising mean CV Brier score over ``grouped_folds``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def league_side_rates(train: pd.DataFrame) -> dict[str, float]:
    """Empirical ``ends_in_box`` rate by corner side across all teams."""
    result: dict[str, float] = {}
    for side in ("left", "right"):
        sub = train[train["corner_side"] == side]["ends_in_box"].dropna()
        result[side] = float(sub.mean()) if len(sub) > 0 else 0.5
    return result


def shrunk_rate(k: int, n: int, m: float, p_league: float) -> float:
    """James-Stein-style shrinkage toward the league rate.

    Returns ``(k + m * p_league) / (n + m)``.
    When ``n + m == 0`` (edge case with m=0 and n=0) returns ``p_league``.
    """
    denom = n + m
    if denom == 0:
        return p_league
    return (k + m * p_league) / denom


# ---------------------------------------------------------------------------
# Grouped CV folds
# ---------------------------------------------------------------------------

def grouped_folds(
    train: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``GroupKFold`` splits by ``match_id``.

    Returns a list of ``(train_idx, val_idx)`` arrays into ``train``.
    The ``seed`` parameter is accepted for API consistency but ``GroupKFold``
    is deterministic given the group ordering.
    """
    gkf = GroupKFold(n_splits=n_folds)
    groups = train["match_id"].to_numpy()
    # Dummy X (not used by GroupKFold split logic).
    X = np.zeros((len(train), 1))
    return [(tr_idx, val_idx) for tr_idx, val_idx in gkf.split(X, groups=groups)]


# ---------------------------------------------------------------------------
# ShrunkTeamSide model
# ---------------------------------------------------------------------------

@dataclass
class ShrunkTeamSide:
    """Shrunk per-(team, side) empirical rate baseline.

    Attributes
    ----------
    _rates : dict mapping (team_id, side) -> shrunk probability
    _league_rates : dict mapping side -> league-wide probability
    """

    _rates: dict[tuple[int, str], float] = field(default_factory=dict)
    _league_rates: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, train: pd.DataFrame, m: float) -> ShrunkTeamSide:
        """Fit shrunk rates from ``train``.

        Parameters
        ----------
        train:
            DataFrame with ``team_id``, ``corner_side`` and ``ends_in_box``.
        m:
            Additive pseudo-count (shrinkage strength).
        """
        league_rates = league_side_rates(train)
        rates: dict[tuple[int, str], float] = {}

        clean = train.dropna(subset=["ends_in_box", "corner_side"])
        for (team, side), grp in clean.groupby(["team_id", "corner_side"]):
            k = int(grp["ends_in_box"].sum())
            n = len(grp)
            p_lg = league_rates.get(str(side), 0.5)
            rates[(int(team), str(side))] = shrunk_rate(k, n, m, p_lg)

        obj = cls(_rates=rates, _league_rates=league_rates)
        return obj

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted ``ends_in_box`` probability for each row of ``df``.

        Unseen team or unseen side falls back to the league side rate.
        """
        preds = np.empty(len(df), dtype=float)
        for i, row in enumerate(df.itertuples(index=False)):
            key = (int(row.team_id), str(row.corner_side))
            if key in self._rates:
                preds[i] = self._rates[key]
            else:
                # Unseen team/side — use league side rate for that side.
                preds[i] = self._league_rates.get(str(row.corner_side), 0.5)
        return preds


# ---------------------------------------------------------------------------
# CV-based m selection
# ---------------------------------------------------------------------------

def _brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between labels and probabilities."""
    return float(np.mean((y_true.astype(float) - y_pred) ** 2))


def choose_m(
    train: pd.DataFrame,
    m_grid: list[float],
    n_folds: int,
    seed: int,
) -> float:
    """Choose the shrinkage ``m`` that minimises mean CV Brier score.

    Ties are broken by choosing the **smaller** ``m``.
    """
    folds = grouped_folds(train, n_folds, seed)
    best_m = m_grid[0]
    best_brier = float("inf")

    for m in m_grid:
        fold_briers: list[float] = []
        for tr_idx, val_idx in folds:
            tr = train.iloc[tr_idx]
            val = train.iloc[val_idx]
            model = ShrunkTeamSide.fit(tr, m)
            preds = model.predict(val)
            y = val["ends_in_box"].fillna(False).astype(float).to_numpy()
            fold_briers.append(_brier_score(y, preds))
        mean_brier = float(np.mean(fold_briers))
        # Strict < so ties go to the first (smallest) m encountered.
        if mean_brier < best_brier:
            best_brier = mean_brier
            best_m = m

    return best_m
