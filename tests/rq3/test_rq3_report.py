import numpy as np
import pandas as pd
from rq3_helpers import two_setups

from setpiece.rq3.clustering import fit_kmeans
from setpiece.rq3.features import FEATURES, build_rq3_features
from setpiece.rq3.report import (
    EVENT_KEY,
    centroid_comparison,
    compare_fits,
    describe_centroid,
    representatives,
    team_cluster_shares,
)


def test_description_uses_only_zone_vocabulary_and_counts():
    centroid = dict.fromkeys(FEATURES, 0.0)
    assert describe_centroid(centroid) == "sparse central box"
    centroid.update(att_near_central_six=2.1, def_near_central_six=1.4)
    assert describe_centroid(centroid) == "near-central six-yard: 2.1 att / 1.4 def"


def test_centroid_comparison_finds_one_to_one_label_permutation():
    matrix = np.array([[0] * len(FEATURES), [3] * len(FEATURES)])
    compared = centroid_comparison(matrix, matrix[::-1])
    assert compared["alt_label"].tolist() == [1, 0]
    assert compared["centroid_distance"].eq(0).all()


def test_comparison_requires_ten_shared_snapshots():
    labels = pd.DataFrame({"match_id": range(12), "event_id": "id", "label": [0] * 6 + [1] * 6})
    assert np.isnan(compare_fits(labels, labels.iloc[:9]))
    assert compare_fits(labels, labels.assign(label=1 - labels["label"])) == 1


def test_team_filter_and_unsupported_cluster_shares(cfg):
    frames, corners = two_setups()
    features = build_rq3_features(frames, corners, cfg)
    model = fit_kmeans(features[FEATURES].to_numpy(), 2, cfg.seed, 2)
    examples = representatives(features, model, team_id=1)
    assert examples["team_id"].eq(1).all() and len(examples) == 6
    labels = features[EVENT_KEY].assign(label=model.labels_)
    samples = features[["competition_id", "season_id", "team_id", "team_name"]].drop_duplicates()
    shares = team_cluster_shares(features, labels, [0], samples.assign(corners=10), cfg)
    assert set(shares["cluster"]) == {"0", "unsupported"}
    assert shares["denominator"].eq(8).all() and shares["excluded_corners"].eq(2).all()
    assert shares.groupby("team_id")["share"].sum().eq(1).all()
    assert shares["ci_lo"].notna().all() and shares["ci_hi"].notna().all()
