"""``rq1`` stage: derive corners, summarise, write derived tables and figures (Task 15).

Events are read and processed one slice at a time, so memory stays at one slice however large the
corpus is. Every output goes through ``write_table`` as ``derived/<name>`` and is written even when
empty, with its columns and dtypes.
"""

from __future__ import annotations

import pandas as pd

from setpiece.config import Config
from setpiece.coverage import TEAM_SAMPLE_COLUMNS, TEAM_SAMPLE_DTYPES, team_sample_table
from setpiece.io import list_partitions, read_table, write_manifest, write_table
from setpiece.rq1.corners import RQ1_CORNERS_COLUMNS, RQ1_CORNERS_DTYPES, build_rq1_corners
from setpiece.rq1.figures import save_team_figures
from setpiece.rq1.summaries import SUMMARY_COLUMNS, SUMMARY_DTYPES, summarise_rq1


def _stack(parts: list[pd.DataFrame], columns: list[str], dtypes: dict[str, str]) -> pd.DataFrame:
    """The non-empty ``parts`` one under the other; a typed empty table when there are none."""
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=columns).astype(dtypes)
    return pd.concat(parts, ignore_index=True)[columns].astype(dtypes)


def _slices_without_events(cfg: Config) -> list[tuple[int, int]]:
    """The configured slices that have no ``events`` partition (an empty partition counts)."""
    have = set(list_partitions("events", cfg))
    wanted = [(s.competition_id, s.season_id) for s in cfg.slices]
    return [key for key in wanted if key not in have]


def run_rq1(cfg: Config) -> dict:
    """Run the RQ1 stage and return its counts (also written to ``manifests/rq1.json``).

    Writes ``derived/rq1_corners``, ``derived/rq1_summary`` and ``derived/team_samples``, and the
    figures ``reports/figures/rq1/<comp>_<season>_<team_id>_<attribute>.svg`` (plus
    ``_endpoints.svg``) for every team-season with corners. The stage is a full rebuild: it deletes
    the old figures first, so a slice that left the config leaves none behind.

    ``FileNotFoundError`` ("run ingest first") when a configured slice has no ``events``
    partition, rather than writing zero corners for it. A partition with no rows is fine.
    """
    missing = _slices_without_events(cfg)
    if missing:
        raise FileNotFoundError(
            f"no events partition for slice(s) {missing}: run `python -m setpiece ingest` first"
        )

    fig_dir = cfg.reports_dir / "figures" / "rq1"
    for old in fig_dir.glob("*.svg"):
        old.unlink()

    corners_parts, samples_parts, summary_parts = [], [], []
    per_slice: dict[str, dict[str, int]] = {}

    for sl in cfg.slices:
        events = read_table("events", cfg, [(sl.competition_id, sl.season_id)])
        n_events = len(events)
        rq1 = build_rq1_corners(events, cfg)
        samples = team_sample_table(events, [sl])
        del events  # the next slice is read into the freed memory: peak is one slice, not two
        summary = summarise_rq1(rq1, samples, cfg)

        for team_id, team_corners in rq1.groupby("team_id"):
            team_summary = summary[summary["team_id"] == team_id]
            save_team_figures(team_corners, team_summary, sl.name, fig_dir, cfg)

        corners_parts.append(rq1)
        samples_parts.append(samples)
        summary_parts.append(summary)
        per_slice[f"{sl.competition_id}/{sl.season_id}"] = {
            "events": n_events,
            "corners": len(rq1),
            "teams_with_corners": int(rq1["team_id"].nunique()),
        }

    corners = _stack(corners_parts, RQ1_CORNERS_COLUMNS, RQ1_CORNERS_DTYPES)
    samples = _stack(samples_parts, TEAM_SAMPLE_COLUMNS, TEAM_SAMPLE_DTYPES)
    summary = _stack(summary_parts, SUMMARY_COLUMNS, SUMMARY_DTYPES)
    write_table(corners, "derived/rq1_corners", cfg, None)
    write_table(samples, "derived/team_samples", cfg, None)
    write_table(summary, "derived/rq1_summary", cfg, None)

    counts = {
        "events": sum(s["events"] for s in per_slice.values()),
        "rq1_corners": len(corners),
        "team_seasons": len(samples),  # every team that played, with or without a corner
        "teams_with_corners": sum(s["teams_with_corners"] for s in per_slice.values()),
        "summary_rows": len(summary),
        "slices": per_slice,
    }
    write_manifest("rq1", cfg, counts)
    return counts