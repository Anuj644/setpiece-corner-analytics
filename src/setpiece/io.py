"""Partitioned Parquet tables and run manifests: the storage layer every pipeline stage shares.

Layout under ``cfg.processed_dir``:

* a partitioned table ``<table>/competition_id=<c>/season_id=<s>/part.parquet``, one partition per
  slice, replaced whole when the slice is written again;
* an unpartitioned table ``<table>.parquet``;
* a run manifest ``manifests/<stage>.json``.

Tables are read back with ``read_table`` only. A partition's columns, ``competition_id`` and
``season_id`` among them, live inside its file with their pandas dtypes, so a round trip returns the
columns in the order written. The directory names only say which files to open; nothing relies on
Arrow's hive-partition inference (``pyarrow.dataset`` is not needed).
"""

import json
import numbers
import os
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from setpiece.config import Config

PARTITION_COLUMNS = ("competition_id", "season_id")
_PART_FILE = "part.parquet"


def derived_dir(cfg: Config) -> Path:
    """Where the analysis lanes put the tables they derive from ``events`` and ``corner_360``."""
    return cfg.processed_dir / "derived"


def _write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Have ``write`` fill a temp file next to ``path``, then move it into place in one step."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_table(
    df: pd.DataFrame, table: str, cfg: Config, partition: tuple[int, int] | None
) -> Path:
    """Write ``df`` as ``table``: one partition of a partitioned table, or the whole table.

    With ``partition=(competition_id, season_id)`` the partition directory is deleted first, so a
    rewrite replaces the slice and never leaves a stale file; other partitions are not touched. An
    empty ``df`` clears the partition and stores a file with no rows, which ``read_table`` skips
    over. The partition columns of the stored frame are always those of the partition, as ``Int64``:
    a column the frame lacks is added at the end, nulls are filled in, and a value that disagrees
    with ``partition`` raises ``ValueError`` before anything is deleted.

    With ``partition=None`` the table is one file, replaced atomically. The index of ``df`` is not
    stored. Returns the path of the file written.
    """
    partitioned = cfg.processed_dir / table
    single = cfg.processed_dir / f"{table}.parquet"
    if partition is None:
        if partitioned.exists():
            raise ValueError(f"table {table!r} is partitioned; refusing to write it unpartitioned")
        path = single
    else:
        if single.exists():
            raise ValueError(f"table {table!r} is unpartitioned; refusing to write a partition")
        values = (int(partition[0]), int(partition[1]))
        df = _with_partition_columns(df, values)
        path = partitioned.joinpath(
            *(f"{name}={value}" for name, value in zip(PARTITION_COLUMNS, values)), _PART_FILE
        )

    # Convert before deleting anything: a frame Arrow cannot store must not cost the old partition.
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    if partition is None:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if path.parent.exists():
            shutil.rmtree(path.parent)
        path.parent.mkdir(parents=True)
    _write_atomically(path, lambda tmp: pq.write_table(arrow_table, tmp))
    return path


def _with_partition_columns(df: pd.DataFrame, values: tuple[int, int]) -> pd.DataFrame:
    """``df`` with the partition columns set to ``values`` as Int64 (a copy)."""
    for name, value in zip(PARTITION_COLUMNS, values):
        if name in df.columns and (df[name].dropna() != value).any():
            raise ValueError(f"{name} in the frame disagrees with the partition value {value}")
    return df.assign(
        **{
            name: pd.Series(value, index=df.index, dtype="Int64")
            for name, value in zip(PARTITION_COLUMNS, values)
        }
    )


def _read_file(path: Path) -> pa.Table:
    """The whole Parquet file as an Arrow table; the file is closed again before returning."""
    with pq.ParquetFile(path) as parquet_file:
        return parquet_file.read()


def _partition_files(root: Path) -> list[tuple[tuple[int, int], Path]]:
    """``((competition_id, season_id), file)`` of every partition, in numeric order."""
    found = []
    for path in root.glob(f"{PARTITION_COLUMNS[0]}=*/{PARTITION_COLUMNS[1]}=*/{_PART_FILE}"):
        comp, season = (int(d.name.partition("=")[2]) for d in (path.parent.parent, path.parent))
        found.append(((comp, season), path))
    return sorted(found)


def _only_slices(table: pa.Table, slices: list[tuple[int, int]]) -> pa.Table:
    """The rows of ``table`` whose (competition_id, season_id) is one of ``slices``."""
    missing = [name for name in PARTITION_COLUMNS if name not in table.column_names]
    if missing:
        raise ValueError(f"cannot select slices: the table has no {', '.join(missing)} column")
    keep = None
    for comp, season in slices:
        hit = pc.and_kleene(
            pc.equal(table["competition_id"], comp), pc.equal(table["season_id"], season)
        )
        keep = hit if keep is None else pc.or_kleene(keep, hit)
    return table.slice(0, 0) if keep is None else table.filter(keep)


def read_table(
    table: str, cfg: Config, slices: list[tuple[int, int]] | None = None
) -> pd.DataFrame:
    """Read ``table``: its partitions in (competition_id, season_id) order, else its one file.

    ``slices`` keeps only the rows of those (competition_id, season_id) pairs, for both layouts;
    ``None`` keeps everything. The columns come back in the order written with the dtypes written,
    and the result has a fresh ``RangeIndex``. A frame that no row of ``slices`` matches comes back
    empty but with the table's columns. Lists come back as numpy arrays.

    Partitions whose list columns hold no element (all None, or all empty) are combined with ones
    that do. ``FileNotFoundError`` names a table that does not exist.
    """
    partitioned = cfg.processed_dir / table
    single = cfg.processed_dir / f"{table}.parquet"
    if partitioned.is_dir():
        partitions = _partition_files(partitioned)
        if not partitions:
            raise FileNotFoundError(f"table {table!r} has no partition files in {partitioned}")
        wanted = {tuple(s) for s in slices} if slices is not None else None
        tables = [_read_file(p) for key, p in partitions if wanted is None or key in wanted]
        if not tables:  # nothing selected: keep the columns of the table
            tables = [pq.read_schema(partitions[0][1]).empty_table()]
    elif single.is_file():
        tables = [_read_file(single)]
        if slices is not None:
            tables = [_only_slices(tables[0], slices)]
    else:
        raise FileNotFoundError(f"table {table!r} not found under {cfg.processed_dir}")

    # An empty partition says nothing about dtypes, so the pandas metadata comes from a full one.
    metadata = next((t.schema.metadata for t in tables if t.num_rows), tables[0].schema.metadata)
    combined = pa.concat_tables(tables, promote_options="permissive")
    return combined.replace_schema_metadata(metadata).to_pandas()


def _json_default(obj):
    """Accept numpy (and any other) integer scalars as ints; everything else is an error."""
    if isinstance(obj, numbers.Integral):
        return int(obj)
    raise TypeError(f"manifest counts must be JSON-serialisable, got {type(obj).__name__}")


def write_manifest(stage: str, cfg: Config, counts: dict) -> Path:
    """Write ``manifests/<stage>.json`` atomically and return its path.

    It records the stage, the pinned data ``sha``, the ``config_hash``, when it was written
    (``created_utc``, ISO 8601 in UTC), the ``package_version`` and ``counts``. Integer scalars in
    ``counts`` may be numpy integers; anything else JSON cannot hold raises ``TypeError``.
    """
    manifest = {
        "stage": stage,
        "sha": cfg.sha,
        "config_hash": cfg.config_hash(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": version("setpiece"),
        "counts": counts,
    }
    text = json.dumps(manifest, indent=2, default=_json_default) + "\n"  # fails before any file
    path = cfg.processed_dir / "manifests" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    return path
