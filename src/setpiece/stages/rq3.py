"""Build RQ3 setup features, fit the main pool, and write the planned reports."""

import numpy as np
import pandas as pd

from setpiece.config import Config
from setpiece.io import read_table, write_manifest, write_table
from setpiece.rq3.audit import AUDIT_REFERENCES, audit_club_coverage, coverage_audit
from setpiece.rq3.clustering import (
    choose_k,
    empty_k_table,
    evaluate_k_range,
    fit_kmeans,
    supported_clusters,
)
from setpiece.rq3.features import (
    FEATURES,
    KEY,
    TEAM_KEY,
    build_rq3_features,
    funnel,
    read_rq3_inputs,
)
from setpiece.rq3.figures import save_cluster_profiles, save_snapshot
from setpiece.rq3.report import (
    EVENT_KEY,
    centroid_comparison,
    compare_fits,
    describe_centroid,
    representative_rows,
    team_cluster_shares,
    visibility_by_cluster,
)

FEATURES_TABLE = "derived/rq3_features"
K_TABLE = "derived/rq3_k_table"
LABELS_TABLE = "derived/rq3_labels"
CENTROIDS_TABLE = "derived/rq3_centroids"
SENSITIVITY_TABLE = "derived/rq3_sensitivity"
VISIBILITY_TABLE = "derived/rq3_visibility"
TEAM_TABLE = "derived/rq3_team_shares"
FUNNEL_TABLE = "derived/rq3_funnel"

_VARIANTS = ("tight", "no_keeper", "ge16", "poly80", "poly100")
_CENTROID_DTYPES = {
    "label": "Int64",
    "n": "Int64",
    "matches": "Int64",
    "jaccard": "float64",
    "supported": "boolean",
    "description": "string",
    **dict.fromkeys(FEATURES, "float64"),
}
_SENSITIVITY_DTYPES = {
    "profile": "string",
    "status": "string",
    **dict.fromkeys(
        ["k", "eligible_snapshots", "shared_snapshots", "main_label", "alt_label"], "Int64"
    ),
    **dict.fromkeys(["ari", "centroid_distance", *(f"delta_{c}" for c in FEATURES)], "float64"),
}
_FUNNEL_COUNTS = [
    "corners",
    "matched",
    "screened",
    "zone_eligible",
    "ground_all",
    "ground_eligible",
]


def rq3_pool(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The primary model sees the configured core 360 slices or the all-360 pool."""
    mode = cfg.p("rq3_pool")
    if mode == "all_360":
        return features
    if mode != "core_360":
        raise ValueError(f"unknown rq3_pool {mode!r}")
    core = [(sl.competition_id, sl.season_id) for sl in cfg.slices if sl.core and sl.has_360]
    return features[pd.MultiIndex.from_frame(features[KEY[:2]]).isin(core)]


def _fit_primary(eligible, cfg):
    if len(eligible) < cfg.p("rq3_min_snapshots"):
        return empty_k_table(), None, False, "insufficient_snapshots"
    if eligible["match_id"].nunique() < 2:
        return empty_k_table(), None, False, "insufficient_matches"
    k_table = evaluate_k_range(eligible, cfg)
    k, weak = choose_k(k_table, cfg.p("ari_min"))
    selected = k_table.loc[k_table["k"].eq(k)]
    if selected.empty or selected["silhouette"].isna().all():
        return k_table, None, True, "indistinguishable_snapshots"
    model = fit_kmeans(
        eligible[FEATURES].to_numpy(dtype=float), k, cfg.seed, cfg.p("kmeans_n_init")
    )
    return k_table, model, weak, "weak" if weak else "ok"


def _labels(eligible, model):
    assigned = (
        eligible[[*KEY, "team_id"]].copy()
        if model is not None
        else eligible.iloc[:0][[*KEY, "team_id"]].copy()
    )
    assigned["label"] = pd.array(model.labels_ if model is not None else [], dtype="Int64")
    return assigned


def _centroids(eligible, model, jaccard, support):
    rows = []
    if model is not None:
        for label, values in enumerate(model.cluster_centers_):
            group = eligible.iloc[np.flatnonzero(model.labels_ == label)]
            center = dict(zip(FEATURES, values))
            rows.append(
                {
                    "label": label,
                    "n": len(group),
                    "matches": group["match_id"].nunique(),
                    "jaccard": jaccard[label],
                    "supported": label in support,
                    "description": describe_centroid(center),
                    **center,
                }
            )
    return pd.DataFrame(rows, columns=_CENTROID_DTYPES).astype(_CENTROID_DTYPES)


def _fixed_k_sensitivity(name, eligible, main_model, main_labels, cfg):
    comparison = {"profile": name, "eligible_snapshots": len(eligible), "shared_snapshots": 0}
    if main_model is None:
        return [{**comparison, "status": "no_main_fit"}]
    k = main_model.n_clusters
    comparison["k"] = k
    matrix = eligible[FEATURES].to_numpy(dtype=float)
    if len(matrix) < cfg.p("rq3_min_snapshots") or len(np.unique(matrix, axis=0)) < k:
        return [{**comparison, "status": "insufficient_snapshots"}]
    model = fit_kmeans(matrix, k, cfg.seed, cfg.p("kmeans_n_init"))
    if len(np.unique(model.labels_)) < k:
        return [{**comparison, "status": "indistinguishable_snapshots"}]
    alt_labels = _labels(eligible, model)[[*EVENT_KEY, "label"]]
    comparison |= {
        "status": "ok",
        "shared_snapshots": len(main_labels.merge(alt_labels, on=EVENT_KEY, validate="one_to_one")),
        "ari": compare_fits(main_labels, alt_labels),
    }
    centers = centroid_comparison(main_model.cluster_centers_, model.cluster_centers_)
    return [{**comparison, **match} for match in centers.to_dict("records")]


def _complete_funnel(corners, frames, features, samples):
    team_counts = funnel(corners, frames, features).drop(columns="team_name")
    table = samples[[*TEAM_KEY, "team_name"]].merge(
        team_counts, on=TEAM_KEY, how="left", validate="one_to_one"
    )
    table[_FUNNEL_COUNTS] = table[_FUNNEL_COUNTS].fillna(0).astype("Int64")
    return table.assign(excluded=table["corners"] - table["zone_eligible"])


def _save_figures(cfg, examples, centroids, frames):
    root = cfg.reports_dir / "figures" / "rq3"
    root.mkdir(parents=True, exist_ok=True)
    for old in [*root.glob("representative_*.svg"), root / "cluster_profiles.svg"]:
        old.unlink(missing_ok=True)
    if centroids.empty:
        return
    save_cluster_profiles(centroids, root / "cluster_profiles.svg")
    rows = examples.merge(
        frames[[*EVENT_KEY, "visible_area", "players"]], on=EVENT_KEY, validate="one_to_one"
    )
    for row in rows.to_dict("records"):
        path = root / f"representative_{row['label']}_{row['rank']}.svg"
        save_snapshot(row, row, cfg, path)


def run_rq3(cfg: Config) -> dict:
    frames, corners = read_rq3_inputs(cfg)
    slices = [(sl.competition_id, sl.season_id) for sl in cfg.slices]
    try:
        samples = read_table("derived/team_samples", cfg, slices)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; run rq1 first") from exc
    features = build_rq3_features(frames, corners, cfg)
    primary = rq3_pool(features, cfg)
    eligible = primary[primary["zone_eligible"]]
    k_table, model, weak, status = _fit_primary(eligible, cfg)
    jaccard = (
        [] if model is None else k_table.loc[k_table["k"].eq(model.n_clusters), "jaccard"].item()
    )
    support = [] if weak else supported_clusters(jaccard, cfg.p("jaccard_supported"))
    labels = _labels(eligible, model)
    assignments = labels[[*EVENT_KEY, "label"]]
    centroids = _centroids(eligible, model, jaccard, support)
    k_table["selected"] = k_table["k"].eq(model.n_clusters).astype("boolean") if model else False

    alternatives = {
        name: rq3_pool(build_rq3_features(frames, corners, cfg, profile=name), cfg)
        for name in _VARIANTS
    }
    sensitivity = []
    for name, alt in alternatives.items():
        sensitivity.extend(
            _fixed_k_sensitivity(name, alt[alt["zone_eligible"]], model, assignments, cfg)
        )
    if cfg.p("rq3_pool") == "core_360" and len(features) > len(primary):
        sensitivity.extend(
            _fixed_k_sensitivity(
                "all_360", features[features["zone_eligible"]], model, assignments, cfg
            )
        )
    visibility = visibility_by_cluster(features, assignments)
    buffered = (
        alternatives["tight"][[*EVENT_KEY, "zone_eligible"]]
        .merge(assignments, on=EVENT_KEY, validate="one_to_one")
        .groupby("label")["zone_eligible"]
        .mean()
        .rename("buffered_visible_share")
    )
    visibility = visibility.merge(buffered, on="label", how="left")
    funnel_table = _complete_funnel(corners, frames, features, samples)
    shares = team_cluster_shares(primary, assignments, support, funnel_table, cfg)
    audit = build_rq3_features(frames, corners, cfg, profile="audit")
    audit_table = audit_club_coverage(frames, audit, features, cfg)
    audit_counts = [
        int(audit_table[c].sum())
        for c in ("audit_screened", "audit_zone_eligible", "audit_full_box_visible")
    ]
    examples = representative_rows(features, assignments, centroids)
    sensitivity = pd.DataFrame(sensitivity, columns=_SENSITIVITY_DTYPES).astype(_SENSITIVITY_DTYPES)
    tables = {
        FEATURES_TABLE: features,
        K_TABLE: k_table,
        LABELS_TABLE: labels,
        CENTROIDS_TABLE: centroids,
        SENSITIVITY_TABLE: sensitivity,
        VISIBILITY_TABLE: visibility,
        TEAM_TABLE: shares,
        FUNNEL_TABLE: funnel_table,
    }
    for name, table in tables.items():
        write_table(table, name, cfg, None)
    reports = cfg.reports_dir / "rq3"
    reports.mkdir(parents=True, exist_ok=True)
    coverage_audit(corners, features, cfg).to_csv(reports / "coverage.csv", index=False)
    audit_table.to_csv(reports / "audit_reproduction.csv", index=False)
    _save_figures(cfg, examples, centroids, frames)
    counts = {
        "corners": len(corners),
        "frames": len(frames),
        "zone_eligible": len(eligible),
        "assigned": len(labels),
        "pool": cfg.p("rq3_pool"),
        "status": status,
        "weak": weak,
        "chosen_k": model.n_clusters if model is not None else None,
        "supported_clusters": support,
        "representatives": len(examples),
        "audit_clubs": {
            "screened": audit_counts[0],
            "central_visible": audit_counts[1],
            "full_box_visible": audit_counts[2],
            "matches_reference": tuple(audit_counts) == AUDIT_REFERENCES,
        },
    }
    write_manifest("rq3", cfg, counts)
    return counts
