"""Tables and review images for fitted central-box positions."""

from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from setpiece.config import Config
from setpiece.rq3.features import COVERAGE_COLUMNS, FEATURES, TEAM_KEY, ZONE_NAMES
from setpiece.stats import match_bootstrap_ratio, rate

EVENT_KEY = ["match_id", "event_id"]
REPRESENTATIVES_PER_CLUSTER = 3
_MIN_SHARED_SNAPSHOTS = 10
_ZONE_LABELS = dict(
    zip(
        ZONE_NAMES,
        (
            "near-central six-yard",
            "near-central deep",
            "far-central six-yard",
            "far-central deep",
        ),
    )
)
_SHARE_DTYPES = {
    **dict.fromkeys(TEAM_KEY + ["n", "denominator", "excluded_corners"], "Int64"),
    "team_name": "string",
    "cluster": "string",
    **dict.fromkeys(["share", "ci_lo", "ci_hi"], "float64"),
}


def describe_centroid(centroid: dict[str, float]) -> str:
    """Describe a center solely with zone names and player counts."""
    terms = []
    for zone, label in _ZONE_LABELS.items():
        att = centroid[f"att_{zone}"]
        deff = centroid[f"def_{zone}"]
        if max(att, deff) >= 0.5:
            terms.append(f"{label}: {att:.1f} att / {deff:.1f} def")
    return "; ".join(terms) if terms else "sparse central box"


def compare_fits(main_labels: pd.DataFrame, alt_labels: pd.DataFrame) -> float:
    common = main_labels.merge(
        alt_labels, on=EVENT_KEY, suffixes=("_main", "_alt"), validate="one_to_one"
    )
    if len(common) < _MIN_SHARED_SNAPSHOTS:
        return np.nan
    return float(adjusted_rand_score(common["label_main"], common["label_alt"]))


def centroid_comparison(main: np.ndarray, alt: np.ndarray) -> pd.DataFrame:
    """Align centers one-to-one by minimum total squared Euclidean distance."""
    squared = ((main[:, None] - alt[None, :]) ** 2).sum(axis=2)
    main_labels = np.arange(len(main))
    pairing = min(
        permutations(range(len(alt))), key=lambda p: squared[main_labels, np.asarray(p)].sum()
    )
    alt_labels = np.asarray(pairing)
    return pd.DataFrame(
        {
            "main_label": main_labels,
            "alt_label": alt_labels,
            "centroid_distance": np.sqrt(squared[main_labels, alt_labels]),
            **{f"delta_{name}": alt[alt_labels, i] - main[:, i] for i, name in enumerate(FEATURES)},
        }
    )


def visibility_by_cluster(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    visible = ["polygon_area", "visible_total", *COVERAGE_COLUMNS]
    joined = df[[*EVENT_KEY, *visible]].merge(labels, on=EVENT_KEY, validate="one_to_one")
    return (
        joined.groupby("label")
        .agg(n=("event_id", "size"), **{c: (c, "mean") for c in visible})
        .reset_index()
        .astype(dict.fromkeys(visible, "float64"))
    )


def team_cluster_shares(
    df: pd.DataFrame, labels: pd.DataFrame, supported, samples: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Report every team-season and pool all unsupported labels into one category."""
    eligible = df.loc[df["zone_eligible"], [*TEAM_KEY, *EVENT_KEY]].merge(
        labels, on=EVENT_KEY, how="left", validate="one_to_one"
    )
    eligible["cluster"] = (
        eligible["label"].astype("string").where(eligible["label"].isin(supported), "unsupported")
    )
    categories = [str(label) for label in supported]
    if not categories or eligible["cluster"].eq("unsupported").any():
        categories.append("unsupported")
    groups = dict(tuple(eligible.groupby(TEAM_KEY, sort=False)))
    rows = []
    for sample in samples.to_dict("records"):
        key = tuple(sample[col] for col in TEAM_KEY)
        team = groups.get(key, eligible.iloc[:0])
        complete = team["label"].notna().all()
        denominator = len(team)
        per_match_den = team.groupby("match_id").size()
        for category in categories:
            yes = team["cluster"].eq(category)
            n = int(yes.sum()) if complete else 0
            lo, hi = (np.nan, np.nan)
            if complete and denominator:
                per_match_num = yes.groupby(team["match_id"]).sum().reindex(per_match_den.index)
                lo, hi = match_bootstrap_ratio(
                    per_match_num.to_numpy(),
                    per_match_den.to_numpy(),
                    cfg.p("bootstrap_reps"),
                    cfg.seed,
                )
            total = sample.get("corners")
            rows.append(
                {
                    **{name: sample[name] for name in TEAM_KEY},
                    "team_name": sample.get("team_name"),
                    "cluster": category,
                    "n": n,
                    "denominator": denominator,
                    "excluded_corners": total - denominator if total is not None else pd.NA,
                    "share": rate(n, denominator) if complete else np.nan,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return pd.DataFrame(rows, columns=_SHARE_DTYPES).astype(_SHARE_DTYPES)


def representative_rows(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    centers: pd.DataFrame,
    n: int = REPRESENTATIVES_PER_CLUSTER,
    team_id: int | None = None,
) -> pd.DataFrame:
    """Find the closest assigned snapshots to each centroid using persisted fit outputs."""
    assigned = df.merge(labels, on=EVENT_KEY, validate="one_to_one")
    if team_id is not None:
        assigned = assigned[assigned["team_id"].eq(team_id)]
    if assigned.empty:
        return assigned.assign(rank=pd.Series(dtype="Int64"), distance=pd.Series(dtype=float))
    center_matrix = (
        centers.set_index("label").loc[assigned["label"], FEATURES].to_numpy(dtype=float)
    )
    squared = ((assigned[FEATURES].to_numpy(dtype=float) - center_matrix) ** 2).sum(axis=1)
    ranked = assigned.assign(distance=np.sqrt(squared)).sort_values(
        ["label", "distance", *EVENT_KEY]
    )
    best = ranked.groupby("label", sort=False).head(n).copy()
    best["rank"] = best.groupby("label", sort=False).cumcount() + 1
    return best


def representatives(
    df: pd.DataFrame, model, n: int = REPRESENTATIVES_PER_CLUSTER, team_id: int | None = None
) -> pd.DataFrame:
    """Find examples for every group, with optional filtering to one team."""
    usable = df.dropna(subset=FEATURES)
    centers = (
        pd.DataFrame(model.cluster_centers_, columns=FEATURES).rename_axis("label").reset_index()
    )
    labels = usable[EVENT_KEY].copy()
    labels["label"] = model.predict(usable[FEATURES].to_numpy(dtype=float)) if len(usable) else []
    return representative_rows(usable, labels, centers, n, team_id)
