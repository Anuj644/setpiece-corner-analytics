"""The ``qa`` stage: structural checks, diagnostic warnings and null rates for ``events`` (5.5).

``check_events`` judges one events frame and never reads the disk. ``run_qa`` applies it to each
configured slice, one partition at a time (the full catalogue is millions of rows), adds the check
that every match the slice lists is in ``events``, and writes ``processed_dir/qa_report.json``.

* **Errors** are structural and fail the run: a repeated ``(match_id, event_id)``, an
  ``event_index`` that does not strictly increase within a match, a null in a key column, a listed
  match with no events.
* **Warnings** are counts that do not fail anything. Their five names are always present, 0 when
  nothing was found: ``coords_out_of_range``, ``corner_start_mid_pitch``,
  ``technique_height_mismatch``, ``legacy_flag_without_technique`` and ``timestamp_decreases``.
  They exist to show where a slice behaves differently from the slices already audited.
* **Null rates** cover every column. Shot events tagged Corner are counted apart
  (``shot_corner_count``) and never mixed into the corner passes.

A corner is a Pass whose ``pass_type`` is Corner; a ground pass has ``pass_height`` Ground Pass.
A null never counts as a warning (a missing coordinate is not out of range), except that a corner
of unknown height counts as not ground.
"""

import json
from dataclasses import dataclass

import pandas as pd

from setpiece.config import Config, Slice
from setpiece.io import _write_atomically, read_table, write_manifest
from setpiece.sources import Fetcher

# Facts of the StatsBomb data format, used for diagnostics only; they are not method parameters.
PITCH_BOUNDS = (120, 80)  # (x, y) of the far corner; the pitch runs from (0, 0)
CORNER_MID_PITCH_BAND = (2, 78)  # a corner starting strictly between these y values is odd

KEY_COLUMNS = ("match_id", "event_id", "event_index", "period", "team_id", "possession_id")
WARNING_KEYS = (
    "coords_out_of_range",
    "corner_start_mid_pitch",
    "technique_height_mismatch",
    "legacy_flag_without_technique",
    "timestamp_decreases",
)
LEGACY_FLAGS = ("inswinging", "outswinging", "straight")  # keys of ``pass`` in ``event_details``
MAX_MISSING_SHOWN = 10  # ids named in a missing-matches error, the rest are only counted


@dataclass
class QAResult:
    errors: list[str]
    warnings: dict[str, int]
    null_rates: dict[str, float]
    shot_corner_count: int


def _true(mask: pd.Series) -> pd.Series:
    """``mask`` as plain booleans, a null (the comparison could not be made) counting as False."""
    return mask.fillna(False).astype(bool)


def _is(column: pd.Series, value: str) -> pd.Series:
    return _true(column == value)


def _rows(count: int) -> str:
    return f"{count} row" if count == 1 else f"{count} rows"


def _outside(column: pd.Series, high: float) -> pd.Series:
    return _true((column < 0) | (column > high))


def _has_legacy_flag(details: str) -> bool:
    """Whether the ``pass`` of an ``event_details`` JSON string sets a legacy swing flag."""
    try:
        parsed = json.loads(details)
    except ValueError:
        return False
    pass_details = parsed.get("pass") if isinstance(parsed, dict) else None
    return isinstance(pass_details, dict) and any(pass_details.get(f) for f in LEGACY_FLAGS)


def _timestamp_decreases(events: pd.DataFrame) -> int:
    """Events earlier than the previous one, by ``event_index``, of the same match and period.

    ``HH:MM:SS.fff`` timestamps; an equal timestamp is not a decrease, and an event without a
    parseable timestamp is skipped (the previous event is then the last one that had one).
    """
    columns = ["match_id", "period", "event_index", "timestamp"]
    ordered = events[columns].dropna(subset=columns[:3]).sort_values("event_index", kind="stable")
    clock = pd.to_timedelta(ordered["timestamp"].astype(object), errors="coerce")
    timed = ordered[["match_id", "period"]].assign(clock=clock).dropna(subset=["clock"])
    previous = timed.groupby(["match_id", "period"], sort=False)["clock"].shift()
    return int(_true(timed["clock"] < previous).sum())


def _structural_errors(events: pd.DataFrame, null_counts: dict[str, int]) -> list[str]:
    errors = [
        f"null {column}: {_rows(null_counts[column])}"
        for column in KEY_COLUMNS
        if null_counts[column]
    ]
    # Rows with a null key are reported as nulls above, not again as repeats or misordered.
    keyed = events[["match_id", "event_id"]].dropna()
    repeated = int(keyed.duplicated(keep=False).sum())
    if repeated:
        errors.append(f"duplicate (match_id, event_id): {_rows(repeated)}")

    indexed = events[["match_id", "event_index"]].dropna()
    before = indexed.groupby("match_id", sort=False)["event_index"].shift()
    misordered = int(_true(indexed["event_index"] <= before).sum())
    if misordered:
        errors.append(f"event_index not strictly increasing within a match: {_rows(misordered)}")
    return errors


def _warning_counts(events: pd.DataFrame) -> dict[str, int]:
    x_max, y_max = PITCH_BOUNDS
    out_of_range = (
        _outside(events["start_x"], x_max)
        | _outside(events["pass_end_x"], x_max)
        | _outside(events["start_y"], y_max)
        | _outside(events["pass_end_y"], y_max)
    )

    corner = _is(events["event_type"], "Pass") & _is(events["pass_type"], "Corner")
    low, high = CORNER_MID_PITCH_BAND
    mid_pitch = corner & _true((events["start_y"] > low) & (events["start_y"] < high))

    ground = _is(events["pass_height"], "Ground Pass")
    no_technique = events["pass_technique"].isna()
    mismatch = corner & ((~ground & no_technique) | (ground & ~no_technique))

    details = events.loc[corner & no_technique, "event_details"].dropna()
    legacy = sum(_has_legacy_flag(text) for text in details)

    counts = (
        int(out_of_range.sum()),
        int(mid_pitch.sum()),
        int(mismatch.sum()),
        legacy,
        _timestamp_decreases(events),
    )
    return dict(zip(WARNING_KEYS, counts))


def _null_rates(null_counts: dict[str, int], rows: int) -> dict[str, float]:
    """Null share per column; every rate of an empty table is 0.0, never NaN."""
    return {column: count / rows if rows else 0.0 for column, count in null_counts.items()}


def _check(events: pd.DataFrame) -> tuple[QAResult, dict[str, int]]:
    """``check_events`` and the null counts it derived its rates from."""
    events = events.reset_index(drop=True)  # masks line up by position, whatever the caller's index
    null_counts = {column: int(events[column].isna().sum()) for column in events.columns}
    shot_corners = _is(events["event_type"], "Shot") & _is(events["shot_type"], "Corner")
    result = QAResult(
        errors=_structural_errors(events, null_counts),
        warnings=_warning_counts(events),
        null_rates=_null_rates(null_counts, len(events)),
        shot_corner_count=int(shot_corners.sum()),
    )
    return result, null_counts


def check_events(events: pd.DataFrame) -> QAResult:
    """The errors, warning counts, null rates and shot-corner count of an ``events`` frame.

    Works on any frame with the ``events`` columns, an empty one included (no errors, every
    warning 0, every null rate 0.0). ``event_index`` is checked in row order within each match.
    """
    return _check(events)[0]


def _key(slice: Slice) -> str:
    return f"{slice.competition_id}/{slice.season_id}"


def _missing_matches_error(missing: list[int]) -> str:
    shown = ", ".join(str(match_id) for match_id in missing[:MAX_MISSING_SHOWN])
    if len(missing) > MAX_MISSING_SHOWN:
        shown += f", ... and {len(missing) - MAX_MISSING_SHOWN} more"
    return f"listed matches missing from events (n={len(missing)}): {shown}"


def _qa_slice(cfg: Config, fetcher: Fetcher, slice: Slice) -> tuple[QAResult, dict[str, int], int]:
    """The slice's result (listed-match errors included), null counts and number of events."""
    events = read_table("events", cfg, [(slice.competition_id, slice.season_id)])
    matches = fetcher.get_json(f"matches/{slice.competition_id}/{slice.season_id}.json")
    result, null_counts = _check(events)
    present = set(events["match_id"].dropna().unique().tolist())
    missing = sorted({match["match_id"] for match in matches} - present)
    if missing:
        result.errors.append(_missing_matches_error(missing))
    return result, null_counts, len(events)


def _section(result: QAResult, rows: int) -> dict:
    return {
        "events": rows,
        "errors": result.errors,
        "warnings": result.warnings,
        "null_rates": result.null_rates,
        "shot_corner_count": result.shot_corner_count,
    }


def run_qa(cfg: Config, fetcher: Fetcher | None = None) -> QAResult:
    """Check the ``events`` of every configured slice and write ``processed_dir/qa_report.json``.

    Each slice's match list comes through the ``Fetcher`` (the cache first); the raw events are
    never read again. The report has one section per slice, keyed ``"c/s"``, and an ``"overall"``
    one, each with ``events`` (rows), ``errors``, ``warnings``, ``null_rates`` and
    ``shot_corner_count``. The overall errors carry their slice as a ``"c/s: "`` prefix, the
    warnings and shot corners are summed, and the null rates are over all events. The ``qa``
    manifest holds the number of overall errors and each warning count.

    Returns the overall result; the report is written whether or not there are errors.
    ``ValueError`` if the config has no slices.
    """
    if not cfg.slices:
        raise ValueError("no slices configured")
    fetcher = fetcher if fetcher is not None else Fetcher(cfg)
    report: dict[str, dict] = {}
    errors: list[str] = []
    warnings = dict.fromkeys(WARNING_KEYS, 0)
    null_counts: dict[str, int] = {}
    rows = shot_corners = 0
    for slice in cfg.slices:
        result, slice_nulls, slice_rows = _qa_slice(cfg, fetcher, slice)
        report[_key(slice)] = _section(result, slice_rows)
        errors += [f"{_key(slice)}: {error}" for error in result.errors]
        for name, count in result.warnings.items():
            warnings[name] += count
        for column, count in slice_nulls.items():
            null_counts[column] = null_counts.get(column, 0) + count
        rows += slice_rows
        shot_corners += result.shot_corner_count
    fetcher.save_manifest()

    overall = QAResult(errors, warnings, _null_rates(null_counts, rows), shot_corners)
    report["overall"] = _section(overall, rows)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"  # fails before any file
    path = cfg.processed_dir / "qa_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    write_manifest("qa", cfg, {"errors": len(errors), **warnings})
    return overall
