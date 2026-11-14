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

from setpiece.stats import match_bootstrap

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
    """Bootstrap CI on ``brier(model) − brier(baseline)`` at the match level.

    Parameters
    ----------
    test:
        Test-set corners DataFrame; must contain ``match_id`` and
        ``ends_in_box`` columns.
    p_model, p_base:
        Model and baseline predicted probabilities (aligned with ``test``).
    n_reps:
        Number of bootstrap replicates.
    seed:
        RNG seed.

    Returns
    -------
    (diff, lo, hi)
        ``diff`` is the point estimate ``brier(model) - brier(baseline)``.
        ``lo`` and ``hi`` are the 2.5th and 97.5th percentile bootstrap CIs.
    """
    y = test["ends_in_box"].fillna(False).astype(float).to_numpy()
    diff_point = brier(y, p_model) - brier(y, p_base)

    # Attach predictions and labels to the frame for the bootstrap stat.
    aug = test[["match_id"]].copy()
    aug = aug.assign(
        y=y,
        p_model=p_model,
        p_base=p_base,
    )

    def _stat(df: pd.DataFrame) -> float:
        yi = df["y"].to_numpy()
        return brier(yi, df["p_model"].to_numpy()) - brier(yi, df["p_base"].to_numpy())

    lo, hi = match_bootstrap(aug, _stat, n_reps, seed)
    return (diff_point, lo, hi)


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
