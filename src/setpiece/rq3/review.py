"""Export G4 images and a blank sheet from the completed RQ3 run."""

import json

from setpiece.config import Config
from setpiece.io import read_table
from setpiece.rq3.features import read_rq3_inputs
from setpiece.rq3.report import EVENT_KEY, representative_rows, write_snapshot_review
from setpiece.stages.rq3 import (
    CENTROIDS_TABLE,
    FEATURES_TABLE,
    LABELS_TABLE,
    rq3_pool,
)


def run_review_export_rq3(cfg: Config) -> dict:
    """Export three representative images per cluster plus the seeded random sample."""
    manifest = cfg.processed_dir / "manifests" / "rq3.json"
    if not manifest.is_file():
        raise FileNotFoundError("RQ3 manifest is missing; run `python -m setpiece rq3` first")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    if recorded["config_hash"] != cfg.config_hash():
        raise ValueError("RQ3 config differs from the last run; rerun `python -m setpiece rq3`")

    features = rq3_pool(read_table(FEATURES_TABLE, cfg), cfg)
    eligible = features[features["zone_eligible"]]
    labels = read_table(LABELS_TABLE, cfg)[[*EVENT_KEY, "label"]]
    centroids = read_table(CENTROIDS_TABLE, cfg)
    examples = representative_rows(eligible, labels, centroids)
    frames, _ = read_rq3_inputs(cfg)
    return write_snapshot_review(
        eligible, examples, frames, cfg, cfg.reports_dir / "review" / "rq3"
    )
