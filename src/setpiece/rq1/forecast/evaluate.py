"""Forecast lane — evaluation metrics and decision rule (Task 19).

``brier``             — mean squared error between binary labels and probs.
``brier_diff_ci``     — bootstrap CI on ``brier(model) - brier(baseline)``
                        using match-level resampling (Task 10).
``decide``            — select the best method using the §6.4 decision rule.
``reliability_table`` — calibration table for reliability diagrams.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from setpiece.rq1.forecast.split import labels
from setpiece.stats import match_bootstrap_ratio

# Priority order for the simplicity tie-break rule (§6.4).
_SIMPLICITY_ORDER = ["league_side", "team_side_shrunk"]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean Brier score (mean squared error) between labels ``y`` and probs ``p``."""
    return float(np.mean((y.astype(float) - p) ** 2))


def brier_diff_ci(
    test: pd.DataFrame,
    p_model: np.ndarray,
    p_base: np.ndarray,
    n_reps: int,
    seed: int,
) -> tuple[float, float, float]:
    """Match-level bootstrap CI on ``brier(model) - brier(baseline)``.

    The difference of the two mean squared errors is a ratio of per-match sums (the sum over a
    match's corners of ``err_model - err_base``, divided by the number of corners), so the CI comes
    from ``match_bootstrap_ratio``: whole matches are resampled, never single corners.

    Parameters
    ----------
    test:
        Test-set corners with ``match_id`` and ``ends_in_box`` (no nulls) columns.
    p_model, p_base:
        Predicted probabilities, aligned with the rows of ``test``.
    n_reps, seed:
        Bootstrap replicates and the seed of its random generator.

    Returns
    -------
    (diff, lo, hi)
        The point estimate and the 2.5th and 97.5th percentiles of the bootstrap.
    """
    y = labels(test)
    err_diff = (y - p_model) ** 2 - (y - p_base) ** 2
    per_match = (
        pd.DataFrame({"match_id": test["match_id"].to_numpy(), "err_diff": err_diff})
        .groupby("match_id")["err_diff"]
        .agg(["sum", "size"])
    )
    lo, hi = match_bootstrap_ratio(
        per_match["sum"].to_numpy(), per_match["size"].to_numpy(), n_reps, seed
    )
    return (float(err_diff.mean()), lo, hi)

def decide(
    model_brier: float,
    base_results: dict[str, tuple[float, float, float, float]],
) -> str:
    """Select the best forecasting method using the §6.4 decision rule.

    Parameters
    ----------
    model_brier:
        Brier score of the logistic regression model on the test set.
    base_results:
        Mapping from baseline name to ``(base_brier, diff, lo, hi)`` where
        ``diff = brier(model) - brier(base)`` and ``[lo, hi]`` is its CI.

    Decision rule
    -------------
    Choose the **simplest** baseline (in the order ``league_side``,
    ``team_side_shrunk``) for which **either**:
      * ``base_brier ≤ model_brier``  (baseline is at least as good), **or**
      * the CI on the difference includes 0  (improvement is not significant).

    If no baseline qualifies, return ``"model"``.
    """
    for name in _SIMPLICITY_ORDER:
        if name not in base_results:
            continue
        base_brier, _diff, lo, hi = base_results[name]
        ci_includes_zero = lo <= 0.0 <= hi
        if base_brier <= model_brier or ci_includes_zero:
            return name
    return "model"


def reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calibration reliability table.

    Splits the probability range ``[0, 1]`` into ``n_bins`` equal-width bins
    and computes the mean predicted probability and observed positive fraction
    in each bin.

    Parameters
    ----------
    y:
        Binary labels (0 / 1).
    p:
        Predicted probabilities in ``[0, 1]``.
    n_bins:
        Number of bins.

    Returns
    -------
    DataFrame with columns ``bin, mean_p, frac_pos, n``.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo_e = edges[i]
        hi_e = edges[i + 1]
        # Last bin is closed on both sides.
        if i < n_bins - 1:
            mask = (p >= lo_e) & (p < hi_e)
        else:
            mask = (p >= lo_e) & (p <= hi_e)
        n = int(mask.sum())
        if n == 0:
            mean_p = float((lo_e + hi_e) / 2)
            frac_pos = float("nan")
        else:
            mean_p = float(p[mask].mean())
            frac_pos = float(y[mask].astype(float).mean())
        rows.append(
            {
                "bin": i,
                "mean_p": mean_p,
                "frac_pos": frac_pos,
                "n": n,
            }
        )
    return pd.DataFrame(rows)
