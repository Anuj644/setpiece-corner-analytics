"""Forecast lane: the two baselines and the grouped cross-validation they share (Task 17).

1. ``LeagueSide``: one probability per side (left/right) pooled over all teams, the simplest model.
2. ``ShrunkTeamSide``: a per-(team, side) rate shrunk toward the league side rate with the
   pseudo-count ``m``, chosen by mean cross-validated Brier score (``choose_m``).

Every model here has ``fit(train, ...)`` and ``predict(df) -> probabilities``, like the logistic
regression in ``model.py``, so the stage can treat the three alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from setpiece.rq1.forecast.evaluate import brier
from setpiece.rq1.forecast.split import SIDES, labels


def league_side_rates(train: pd.DataFrame) -> dict[str, float]:
    """Empirical ``ends_in_box`` rate per side over all teams.

    A side with no training corner gets the rate over both sides; with no corner at all there is
    nothing to estimate and ``ValueError`` is raised.
    """
    y = labels(train)
    if len(y) == 0:
        raise ValueError("cannot estimate league rates from no corners")
    side = train["corner_side"].to_numpy()
    pooled = float(y.mean())
    return {s: float(y[side == s].mean()) if (side == s).any() else pooled for s in SIDES}


def shrunk_rate(k: int, n: int, m: float, p_league: float) -> float:
    """``(k + m * p_league) / (n + m)``: the team's rate pulled toward the league rate.

    With ``n + m == 0`` (no corners and no shrinkage) the league rate is all there is.
    """
    denom = n + m
    if denom == 0:
        return p_league
    return (k + m * p_league) / denom


def grouped_folds(
    train: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``GroupKFold`` splits by ``match_id``: the corners of a match are never split.

    The groups are shuffled with ``seed``, so the same seed gives the same folds. Returns a list of
    ``(fit_idx, validation_idx)`` position arrays into ``train``.
    """
    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    groups = train["match_id"].to_numpy()
    return list(splitter.split(np.zeros((len(train), 1)), groups=groups))


@dataclass
class LeagueSide:
    """Predict the league rate of the corner's side, whoever takes it."""

    rates: dict[str, float]

    @classmethod
    def fit(cls, train: pd.DataFrame) -> LeagueSide:
        return cls(league_side_rates(train))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.rates[str(s)] for s in df["corner_side"]], dtype=float)


@dataclass
class ShrunkTeamSide:
    """Shrunk per-(team, side) rate; an unseen team (or a team's unseen side) gets the league rate.

    ``rates`` maps (team_id, side) to the shrunk probability, ``league`` side to league rate.
    """

    rates: dict[tuple[int, str], float] = field(default_factory=dict)
    league: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, train: pd.DataFrame, m: float) -> ShrunkTeamSide:
        """Fit the shrunk rates from ``train`` (``team_id``, ``corner_side``, ``ends_in_box``)."""
        league = league_side_rates(train)
        y = pd.Series(labels(train), index=train.index)
        per_team_side = y.groupby([train["team_id"], train["corner_side"]]).agg(["sum", "size"])
        rates = {
            (int(team), str(side)): shrunk_rate(int(k), int(n), m, league[str(side)])
            for (team, side), k, n in zip(
                per_team_side.index, per_team_side["sum"], per_team_side["size"]
            )
        }
        return cls(rates, league)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Probability per row of ``df`` (``team_id``, ``corner_side``)."""
        return np.array(
            [
                self.rates.get((int(t), str(s)), self.league[str(s)])
                for t, s in zip(df["team_id"], df["corner_side"])
            ],
            dtype=float,
        )


def choose_m(
    train: pd.DataFrame,
    m_grid: list[float],
    n_folds: int,
    seed: int,
) -> float:
    """The shrinkage ``m`` with the lowest mean cross-validated Brier score.

    A tie goes to the smaller ``m``, whatever the order of ``m_grid``.
    """
    folds = grouped_folds(train, n_folds, seed)
    best_m, best_brier = None, float("inf")
    for m in sorted(m_grid):
        scores = []
        for fit_idx, val_idx in folds:
            val = train.iloc[val_idx]
            model = ShrunkTeamSide.fit(train.iloc[fit_idx], m)
            scores.append(brier(labels(val), model.predict(val)))
        mean_score = float(np.mean(scores))
        if mean_score < best_brier:  # strict: on a tie the earlier, smaller m stays
            best_m, best_brier = m, mean_score
    return best_m
