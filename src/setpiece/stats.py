"""Match-level bootstrap helpers used by every RQ summary table.

All randomness is seeded through ``numpy.random.default_rng``.

Two bootstrap variants are provided:

* ``match_bootstrap`` – generic: rebuilds a DataFrame per replicate from a
  resampled group index.  Kept for statistics that cannot be expressed as a
  ratio of per-match sums (e.g. the Brier-score difference in Task 19).

* ``match_bootstrap_ratio`` – fast path for shares and per-corner rates.
  It draws a single ``(n_reps, n_matches)`` integer matrix and vectorises the
  numerator / denominator sums over all replicates at once.  Replicates whose
  denominator sum is zero are silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def rate(num: int, den: int) -> float:
    """Return ``num / den``, or ``nan`` when ``den == 0``."""
    if den == 0:
        return float("nan")
    return num / den


def match_bootstrap(
    df: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    n_reps: int,
    seed: int,
    group: str = "match_id",
) -> tuple[float, float]:
    """2.5th and 97.5th percentiles of ``stat`` under match-level resampling.

    Whole groups (identified by ``group``) are resampled with replacement.
    Returns ``(nan, nan)`` for an empty DataFrame.
    """
    groups = df[group].unique()
    if len(groups) == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_reps):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        # Build the resampled frame by concatenating each group's rows.
        parts = [df[df[group] == g] for g in sampled_groups]
        resampled = pd.concat(parts, ignore_index=True)
        stats.append(stat(resampled))

    arr = np.array(stats, dtype=float)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))


def match_bootstrap_ratio(
    num: np.ndarray,
    den: np.ndarray,
    n_reps: int,
    seed: int,
) -> tuple[float, float]:
    """2.5th and 97.5th percentiles of ``sum(num) / sum(den)`` under match-level resampling.

    Parameters
    ----------
    num:
        Per-match numerator sums (one element per match).
    den:
        Per-match denominator sums (one element per match).
    n_reps:
        Number of bootstrap replicates.
    seed:
        Seed for ``numpy.random.default_rng``.

    Returns ``(nan, nan)`` when ``num`` / ``den`` are empty.
    Replicates where the total denominator is zero are discarded.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    n_matches = len(num)
    if n_matches == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    # Draw all replicate indices at once: shape (n_reps, n_matches)
    idx = rng.integers(0, n_matches, size=(n_reps, n_matches))

    num_sums = num[idx].sum(axis=1)   # shape (n_reps,)
    den_sums = den[idx].sum(axis=1)   # shape (n_reps,)

    valid = den_sums > 0
    ratios = num_sums[valid] / den_sums[valid]

    if len(ratios) == 0:
        return (float("nan"), float("nan"))

    return (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))
