"""Pooled k-means, match-level stability, and the descriptive fallback."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from setpiece.config import Config
from setpiece.rq3.features import FEATURES

_K_DTYPES = {"k": "Int64", "silhouette": "float64", "mean_ari": "float64", "jaccard": "object"}


def fit_kmeans(X: np.ndarray, k: int, seed: int, n_init: int) -> KMeans:
    """Fit unscaled counts; each feature has the same unit."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def subsample_matches(df: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    """Choose distinct matches without replacement and keep every row for each match."""
    if not 0 < frac < 1:
        raise ValueError("stability fraction must be between zero and one")
    matches = df["match_id"].unique()
    n = round(frac * len(matches))
    chosen = rng.choice(matches, n, replace=False)
    return df.loc[df["match_id"].isin(chosen)]


def cluster_jaccard(reference: np.ndarray, refit: np.ndarray, k: int) -> np.ndarray:
    """Best Jaccard overlap for each reference cluster, regardless of refit label numbers."""
    shared = np.bincount(reference * k + refit, minlength=k * k).reshape(k, k)
    sizes = shared.sum(axis=1)
    union = sizes[:, None] + shared.sum(axis=0) - shared
    scores = np.divide(shared, union, out=np.zeros_like(shared, dtype=float), where=union > 0)
    return np.where(sizes > 0, scores.max(axis=1), np.nan)


def _stability(df, model: KMeans, n_reps: int, frac: float, seed: int, n_init: int):
    if n_reps < 1:
        raise ValueError("stability requires at least one refit")
    rng = np.random.default_rng(seed)
    k = model.n_clusters
    rows = df.reset_index(drop=True)
    ari, jaccard = [], []
    for _ in range(n_reps):
        sample = subsample_matches(rows, frac, rng)
        matrix = sample[FEATURES].to_numpy(dtype=float)
        reference = model.labels_[sample.index]
        refit_seed = int(rng.integers(2**32))
        if len(np.unique(matrix, axis=0)) >= k:
            refit = fit_kmeans(matrix, k, refit_seed, n_init).labels_
            if len(np.unique(refit)) == k:
                ari.append(adjusted_rand_score(reference, refit))
                jaccard.append(cluster_jaccard(reference, refit, k))
                continue
        # A draw that cannot sustain k clusters counts against overall stability.
        # An absent reference cluster cannot be assessed in that draw.
        ari.append(0.0)
        jaccard.append(np.where(np.bincount(reference, minlength=k) > 0, 0.0, np.nan))

    overlaps = np.asarray(jaccard)
    observed = np.isfinite(overlaps).sum(axis=0)
    mean_jaccard = np.divide(
        np.nansum(overlaps, axis=0), observed, out=np.zeros(k), where=observed > 0
    )
    return float(np.mean(ari)), mean_jaccard


def subsample_stability(
    df: pd.DataFrame, k: int, n_reps: int, frac: float, seed: int, n_init: int
) -> tuple[float, np.ndarray]:
    """Compare refits with the full fit on the snapshots shared by each draw."""
    model = fit_kmeans(df[FEATURES].to_numpy(dtype=float), k, seed, n_init)
    return _stability(df, model, n_reps, frac, seed, n_init)


def evaluate_k_range(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return the silhouette, mean ARI and clusterwise Jaccard for feasible values of k."""
    lo, hi = cfg.p("k_range")
    matrix = df[FEATURES].to_numpy(dtype=float)
    unique = len(np.unique(matrix, axis=0))
    rows = []
    for k in range(lo, hi + 1):
        if k >= len(df) or k > unique:
            rows.append({"k": k, "silhouette": np.nan, "mean_ari": np.nan, "jaccard": []})
            continue
        model = fit_kmeans(matrix, k, cfg.seed, cfg.p("kmeans_n_init"))
        if len(np.unique(model.labels_)) < k:
            rows.append({"k": k, "silhouette": np.nan, "mean_ari": np.nan, "jaccard": []})
            continue
        ari, jaccard = _stability(
            df,
            model,
            cfg.p("stability_reps"),
            cfg.p("stability_frac"),
            cfg.seed,
            cfg.p("kmeans_n_init"),
        )
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(matrix, model.labels_),
                "mean_ari": ari,
                "jaccard": jaccard.tolist(),
            }
        )
    return pd.DataFrame(rows, columns=_K_DTYPES).astype(_K_DTYPES)


def choose_k(table: pd.DataFrame, ari_min: float) -> tuple[int, bool]:
    """Maximize silhouette among stable candidates, with a descriptive k=2 fallback."""
    eligible = table[table["mean_ari"].ge(ari_min) & table["silhouette"].notna()]
    if eligible.empty:
        return 2, True
    ranked = eligible.sort_values(["silhouette", "k"], ascending=[False, True])
    return int(ranked.iloc[0]["k"]), False


def supported_clusters(jaccard: np.ndarray, threshold: float) -> list[int]:
    return np.flatnonzero(np.asarray(jaccard) >= threshold).tolist()


def empty_k_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_K_DTYPES).astype(_K_DTYPES)
