import pandas as pd
import pytest
import yaml
from rq3_helpers import tables, two_setups

from setpiece.cli import main
from setpiece.config import load_config
from setpiece.io import read_table, write_table
from setpiece.stages.rq3 import (
    CENTROIDS_TABLE,
    FEATURES_TABLE,
    FUNNEL_TABLE,
    K_TABLE,
    LABELS_TABLE,
    SENSITIVITY_TABLE,
    TEAM_TABLE,
    VISIBILITY_TABLE,
)


@pytest.fixture
def rq3_config(cfg, tmp_path):
    path = tmp_path / "configs" / "default.yaml"
    path.parent.mkdir()
    doc = yaml.safe_load((cfg.root / "configs" / "default.yaml").read_text(encoding="utf-8"))
    doc["slices"] = [doc["slices"][0]]
    for param, value in {
        "rq3_min_snapshots": 8,
        "kmeans_n_init": 2,
        "stability_reps": 4,
        "rq3_review_sample": 3,
    }.items():
        doc["params"][param]["value"] = value
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _write_inputs(config_path, inputs):
    cfg = load_config(config_path)
    frames, corners = inputs
    write_table(frames, "corner_360", cfg, (43, 106))
    write_table(corners, "derived/rq1_corners", cfg, None)
    write_table(
        pd.DataFrame(
            {
                "competition_id": [43, 43, 43],
                "season_id": [106, 106, 106],
                "team_id": [1, 2, 9],
                "team_name": ["Team 1", "Team 2", "Team 9"],
            }
        ),
        "derived/team_samples",
        cfg,
        None,
    )
    return cfg


@pytest.mark.parametrize("has_frames", [True, False])
def test_rq3_stage_and_g4_export_via_cli(rq3_config, has_frames):
    cfg = _write_inputs(rq3_config, two_setups() if has_frames else tables([]))
    assert main(["rq3", "--config", str(rq3_config)]) == 0
    assert main(["review-export", "rq3", "--config", str(rq3_config)]) == 0
    features, choices, labels, centers, sensitivity, visibility, shares, funnel = [
        read_table(name, cfg)
        for name in [
            FEATURES_TABLE,
            K_TABLE,
            LABELS_TABLE,
            CENTROIDS_TABLE,
            SENSITIVITY_TABLE,
            VISIBILITY_TABLE,
            TEAM_TABLE,
            FUNNEL_TABLE,
        ]
    ]
    review = cfg.reports_dir / "review" / "rq3"
    figures = cfg.reports_dir / "figures" / "rq3"
    sheet = pd.read_csv(review / "review.csv")
    assert sheet.columns.tolist() == [
        "image",
        "polygon_matches_players",
        "orientation_ok",
        "counts_ok",
        "note",
    ]
    assert len(sheet) == (9 if has_frames else 0)
    assert sorted(p.name for p in review.glob("*.png")) == sorted(sheet["image"])
    assert sheet["note"].isna().all()
    assert len(features) == len(labels) == (16 if has_frames else 0)
    assert funnel["corners"].sum() == len(features)
    assert shares.query("team_id == 9")["denominator"].eq(0).all()
    assert shares.query("team_id == 9")["share"].isna().all()
    if has_frames:
        assert choices.loc[choices["selected"], "k"].item() == 2
        assert len(centers) == len(visibility) == 2 and centers["supported"].all()
        assert centers["n"].sum() == len(labels)
        assert shares.query("team_id != 9").groupby("team_id")["share"].sum().eq(1).all()
        assert sensitivity.query("status == 'ok'")["k"].eq(2).all()
        assert len(list(figures.glob("representative_*.svg"))) == 6
    else:
        assert choices.empty and centers.empty and visibility.empty
        assert sensitivity["status"].eq("no_main_fit").all()
        assert not list(figures.glob("*.svg"))


def test_missing_ingestion_or_rq3_names_the_prerequisite(rq3_config):
    with pytest.raises(FileNotFoundError, match="run ingest first"):
        main(["rq3", "--config", str(rq3_config)])
    with pytest.raises(FileNotFoundError, match="setpiece rq3"):
        main(["review-export", "rq3", "--config", str(rq3_config)])
