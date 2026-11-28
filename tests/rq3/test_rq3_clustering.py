from dataclasses import replace

import numpy as np
import pandas as pd

from setpiece.rq3.clustering import (
    FEATURES,
    choose_k,
    cluster_jaccard,
    evaluate_k_range,
    subsample_matches,
    subsample_stability,
    supported_clusters,
)


def _quick_config(cfg):
    values = {"kmeans_n_init": 2, "stability_reps": 8}
    params = cfg.params | {
        name: replace(cfg.params[name], value=value) for name, value in values.items()
    }
    return replace(cfg, params=params)


def test_match_subsample_rounds_and_keeps_all_corners():
    matches = np.repeat(np.arange(7), [1, 3, 2, 4, 2, 3, 2])
    df = pd.DataFrame({"match_id": matches})
    first = subsample_matches(df, 0.8, np.random.default_rng(23))
    second = subsample_matches(df, 0.8, np.random.default_rng(23))
    pd.testing.assert_frame_equal(first, second)
    assert first["match_id"].nunique() == 6
    pd.testing.assert_frame_equal(first, df[df["match_id"].isin(first["match_id"])])


def test_overlap_is_invariant_to_label_names():
    reference = np.array([0, 0, 1, 1, 2, 2])
    refit = np.array([2, 2, 0, 0, 1, 1])
    np.testing.assert_array_equal(cluster_jaccard(reference, refit, 3), [1, 1, 1])


def test_three_integer_setups_are_chosen_and_stable(cfg):
    cfg = _quick_config(cfg)
    centers = np.array(
        [[3, 0, 0, 0, 2, 0, 0, 0], [0, 3, 0, 0, 0, 2, 0, 0], [0, 0, 3, 0, 0, 0, 2, 0]]
    )
    df = pd.DataFrame(np.tile(centers, (12, 1)), columns=FEATURES)
    df["match_id"] = np.repeat(np.arange(12), 3)
    candidates = evaluate_k_range(df, cfg)
    assert choose_k(candidates, cfg.p("ari_min")) == (3, False)
    ari, jaccard = subsample_stability(df, 3, 8, 0.8, cfg.seed, 2)
    assert ari == 1 and supported_clusters(jaccard, 0.6) == [0, 1, 2]
    assert choose_k(candidates.assign(mean_ari=0), cfg.p("ari_min")) == (2, True)
