"""The ``fetch`` and ``ingest`` stages: raw StatsBomb JSON -> gzip cache -> shared tables.

Both stages walk the configured slices one match at a time: the slice's match list, then each
match's events file, then (only for a slice with 360 data) its three-sixty file. A match's raw JSON
is dropped as soon as its frames are built, because a big league season is several GB once parsed.

A missing file (``NotFound``) is handled like this:

* events file: the run aborts, since the pinned revision should never lack one;
* three-sixty file of a 360 slice: the match has no frames (``build_corner_360`` gets ``None``);
* a slice without 360 data never asks for three-sixty files.

Nothing else is treated as missing: ``FetchError`` (a download that kept failing) and
``json.JSONDecodeError`` (a bad body) propagate.
"""

from dataclasses import asdict

import pandas as pd

from setpiece.config import Config, Slice
from setpiece.ingest.events import EVENTS_DTYPES, build_events_frame
from setpiece.ingest.frames import CORNER_360_DTYPES, build_corner_360
from setpiece.io import (
    delete_partition,
    list_partitions,
    write_manifest,
    write_table,
)
from setpiece.sources import Fetcher, NotFound

FRAME_STATS_DTYPES: dict[str, str] = {
    "match_id": "Int64",
    "n_frames": "Int64",
    "n_unmatched": "Int64",
    "n_corners": "Int64",
    "n_corners_matched": "Int64",
    "has_file": "boolean",
    "competition_id": "Int64",
    "season_id": "Int64",
}

_PARTITIONED_TABLES = ("events", "corner_360")


def _key(slice: Slice) -> str:
    return f"{slice.competition_id}/{slice.season_id}"


def _load_match(
    fetcher: Fetcher, slice: Slice, match_id: int
) -> tuple[list[dict], list[dict] | None]:
    """The raw events of a match and its raw 360 frames (None: the slice has none, or no file)."""
    try:
        raw_events = fetcher.get_json(f"events/{match_id}.json")
    except NotFound as exc:
        raise NotFound(
            f"match {match_id} of slice {_key(slice)} ({slice.name}) has no events file: {exc}"
        ) from exc
    if not slice.has_360:
        return raw_events, None
    try:
        return raw_events, fetcher.get_json(f"three-sixty/{match_id}.json")
    except NotFound:
        return raw_events, None


def _cache_match(fetcher: Fetcher, slice: Slice, match_id: int) -> bool | None:
    """Fill the cache for one match; True/False: its 360 file was/was not there, None: not asked."""
    _, frames = _load_match(fetcher, slice, match_id)
    return None if not slice.has_360 else frames is not None


def _require_slices(cfg: Config) -> None:
    """An empty slice list is a config mistake; ingest would prune every partition for it."""
    if not cfg.slices:
        raise ValueError("no slices configured")


def run_fetch(cfg: Config, fetcher: Fetcher | None = None) -> dict:
    """Download every configured slice into the cache and save the download manifest.

    Returns ``{"c/s": {matches, events_files, three_sixty_files, three_sixty_404}}``.
    ``ValueError`` if the config has no slices.
    """
    _require_slices(cfg)
    fetcher = fetcher if fetcher is not None else Fetcher(cfg)
    counts: dict[str, dict[str, int]] = {}
    for slice in cfg.slices:
        matches = fetcher.get_json(f"matches/{slice.competition_id}/{slice.season_id}.json")
        found_360 = [_cache_match(fetcher, slice, match["match_id"]) for match in matches]
        counts[_key(slice)] = {
            "matches": len(matches),
            "events_files": len(matches),
            "three_sixty_files": found_360.count(True),
            "three_sixty_404": found_360.count(False),
        }
    fetcher.save_manifest()
    write_manifest("fetch", cfg, counts)
    return counts


def _ingest_match(fetcher: Fetcher, slice: Slice, match: dict):
    """One match's events table, corner_360 table and frame stats; its raw JSON dies on return."""
    raw_events, raw_frames = _load_match(fetcher, slice, match["match_id"])
    events = build_events_frame(raw_events, match)
    corner_360, stats = build_corner_360(raw_frames, events)
    return events, corner_360, stats


def _concat(frames: list[pd.DataFrame], dtypes: dict[str, str]) -> pd.DataFrame:
    """The frames stacked with a fresh index; no frames at all give an empty frame of ``dtypes``."""
    if not frames:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in dtypes.items()})
    return pd.concat(frames, ignore_index=True)


def _ingest_slice(cfg: Config, fetcher: Fetcher, slice: Slice) -> tuple[dict[str, int], list[dict]]:
    """Write the slice's ``events`` and ``corner_360`` partitions; return its counts and stats rows."""
    matches = fetcher.get_json(f"matches/{slice.competition_id}/{slice.season_id}.json")
    event_frames, corner_frames, stats_rows = [], [], []
    for match in matches:
        events, corner_360, stats = _ingest_match(fetcher, slice, match)
        event_frames.append(events)
        corner_frames.append(corner_360)
        stats_rows.append(
            asdict(stats) | {"competition_id": slice.competition_id, "season_id": slice.season_id}
        )
    events = _concat(event_frames, EVENTS_DTYPES)
    corner_360 = _concat(corner_frames, CORNER_360_DTYPES)
    del event_frames, corner_frames  # the per-match pieces are not needed while writing
    partition = (slice.competition_id, slice.season_id)
    write_table(events, "events", cfg, partition)
    write_table(corner_360, "corner_360", cfg, partition)  # a partition even for a no-360 slice
    counts = {
        "matches": len(matches),
        "events": len(events),
        "corners": sum(row["n_corners"] for row in stats_rows),
        "corners_matched": sum(row["n_corners_matched"] for row in stats_rows),
        "frames_unmatched": sum(row["n_unmatched"] for row in stats_rows),
    }
    return counts, stats_rows


def _prune(cfg: Config) -> list[str]:
    """Delete ``events``/``corner_360`` partitions of slices no longer in the config."""
    wanted = {(slice.competition_id, slice.season_id) for slice in cfg.slices}
    pruned = set()
    for table in _PARTITIONED_TABLES:
        for partition in list_partitions(table, cfg):
            if partition not in wanted:
                delete_partition(table, cfg, partition)
                pruned.add(partition)
    return [f"{comp}/{season}" for comp, season in sorted(pruned)]


def run_ingest(cfg: Config, fetcher: Fetcher | None = None) -> dict:
    """Build ``events``, ``corner_360`` (one partition per slice) and ``frame_stats`` from the cache.

    Slices removed from the config are pruned from the partitioned tables. Returns
    ``{"c/s": {matches, events, corners, corners_matched, frames_unmatched}}``; the ``ingest``
    manifest holds the same counts plus ``"pruned"``, the ``"c/s"`` of the slices it pruned. Every
    file read is recorded in the download manifest, so a cold-cache ingest leaves it complete.
    ``ValueError`` if the config has no slices (nothing is written or pruned).
    """
    _require_slices(cfg)
    fetcher = fetcher if fetcher is not None else Fetcher(cfg)
    counts: dict[str, dict[str, int]] = {}
    all_stats: list[dict] = []
    for slice in cfg.slices:
        counts[_key(slice)], stats_rows = _ingest_slice(cfg, fetcher, slice)
        all_stats.extend(stats_rows)
    fetcher.save_manifest()

    frame_stats = pd.DataFrame(all_stats, columns=list(FRAME_STATS_DTYPES))
    write_table(frame_stats.astype(FRAME_STATS_DTYPES), "frame_stats", cfg, None)
    pruned = _prune(cfg)
    write_manifest("ingest", cfg, {**counts, "pruned": pruned})
    return counts
