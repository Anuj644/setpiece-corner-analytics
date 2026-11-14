"""The ``rq2`` stage: ``events`` -> ``rq2_outcomes``, one row per corner, at four time caps, and
the summaries of the main table.

The main table uses ``cap_seconds`` and has the name ``rq2_outcomes``; each ``cap_sensitivity``
entry gives one more table (``rq2_outcomes_cap10``, ``rq2_outcomes_cap20``, ``rq2_outcomes_nocap``
for the default config, ``null`` meaning no cap). All of them are written under ``derived/``.

A chain never crosses a match, so ``events`` is read one configured slice at a time and the outcomes
of the slices are stacked afterwards. Every table is rebuilt whole from the configured slices.

The outcome tables are written first. The summaries then need ``derived/team_samples`` and
``derived/rq1_corners`` from the ``rq1`` stage (team-seasons and matches played, delivery types and
dates): ``rq2_summary``, ``rq2_sensitivity``, ``rq2_by_delivery`` and ``rq2_end_reasons``, with the
figures in ``reports/figures/rq2/``, which is emptied when the stage starts.
"""

import shutil

import pandas as pd

from setpiece.config import Config
from setpiece.io import list_partitions, read_table, write_manifest, write_table
from setpiece.rq2.chains import build_rq2_outcomes, restarts_from
from setpiece.rq2.figures import save_rq2_figures
from setpiece.rq2.summaries import (
    end_reason_table,
    rates_by_delivery,
    sensitivity_flags,
    summarise_rq2,
)

MAIN_TABLE = "rq2_outcomes"


def _variants(cfg: Config) -> dict[str, float | None]:
    """The table name of every cap variant, main one first, mapped to its ``cap_seconds``."""
    variants: dict[str, float | None] = {MAIN_TABLE: cfg.p("cap_seconds")}
    for cap in cfg.p("cap_sensitivity"):
        suffix = "nocap" if cap is None else f"cap{cap:g}"
        variants[f"{MAIN_TABLE}_{suffix}"] = cap
    return variants


def _counts(outcomes: pd.DataFrame) -> dict[str, int]:
    """Corners, complete chains, attacking shots and goals of one table, as plain ints."""
    return {
        "corners": len(outcomes),
        "complete": int((outcomes["outcome_status"] == "complete").sum()),
        "shots": int(outcomes["shot_count"].sum()),
        "goals": int(outcomes["goal"].sum()),
    }


def _read_rq1_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``derived/team_samples`` and ``derived/rq1_corners``; a missing one says to run rq1 first."""
    tables = []
    for name in ("team_samples", "rq1_corners"):
        try:
            tables.append(read_table(f"derived/{name}", cfg))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"derived/{name} is missing: run rq1 first") from exc
    return tables[0], tables[1]


def run_rq2(cfg: Config) -> dict[str, dict[str, int]]:
    """Write ``derived/rq2_outcomes`` and its cap variants, then the summaries of the main table.

    Returns ``{table: {corners, complete, shots, goals}}`` of the outcome tables. The ``rq2``
    manifest holds those counts, under ``slice_corners`` the number of corners of each slice
    (``"<competition>/<season>"``) and under ``summary_rows`` the rows of ``rq2_summary``,
    ``rq2_sensitivity``, ``rq2_by_delivery`` and ``rq2_end_reasons``.
    ``ValueError`` if the config has no slices, ``FileNotFoundError`` naming the slices whose
    ``events`` partition is not on disk (a read would give an empty frame, so a missing slice would
    pass for one without corners); in both cases nothing is written. Without ``team_samples`` or
    ``rq1_corners`` (``rq1`` not run) the outcome tables are written and the stage then raises
    ``FileNotFoundError`` saying to run rq1 first; no summary and no manifest is written.
    """
    if not cfg.slices:
        raise ValueError("no slices configured")
    on_disk = set(list_partitions("events", cfg))
    missing = [s for s in cfg.slices if (s.competition_id, s.season_id) not in on_disk]
    if missing:
        names = ", ".join(f"{s.competition_id}/{s.season_id} ({s.name})" for s in missing)
        raise FileNotFoundError(f"no 'events' partition for the configured slice(s) {names}")
    fig_dir = cfg.reports_dir / "figures" / "rq2"
    shutil.rmtree(fig_dir, ignore_errors=True)  # no figure of an earlier run outlives this one
    restarts = restarts_from(cfg)
    variants = _variants(cfg)
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}
    slice_corners: dict[str, int] = {}
    for slice in cfg.slices:
        events = read_table("events", cfg, [(slice.competition_id, slice.season_id)])
        for name, cap in variants.items():
            pieces[name].append(build_rq2_outcomes(events, cap, restarts))
        # one row per corner pass whatever the cap, so any variant gives the slice's corners
        slice_corners[f"{slice.competition_id}/{slice.season_id}"] = len(pieces[MAIN_TABLE][-1])
        del events  # a big league season is gigabytes; only the slice at hand is kept
    counts = {}
    outcomes_by_name = {}
    for name, frames in pieces.items():
        outcomes = pd.concat(frames, ignore_index=True)
        write_table(outcomes, f"derived/{name}", cfg, None)
        counts[name] = _counts(outcomes)
        outcomes_by_name[name] = outcomes
    del pieces

    samples, rq1 = _read_rq1_inputs(cfg)
    summaries = {name: summarise_rq2(o, samples, cfg) for name, o in outcomes_by_name.items()}
    summary = summaries[MAIN_TABLE]
    alternatives = {
        name.removeprefix(f"{MAIN_TABLE}_"): s
        for name, s in summaries.items()
        if name != MAIN_TABLE
    }
    main = outcomes_by_name[MAIN_TABLE]
    tables = {
        "rq2_summary": summary,
        "rq2_sensitivity": sensitivity_flags(summary, alternatives),
        "rq2_by_delivery": rates_by_delivery(main, rq1, cfg),
        "rq2_end_reasons": end_reason_table(main),
    }
    for name, table in tables.items():
        write_table(table, f"derived/{name}", cfg, None)
    save_rq2_figures(summary, tables["rq2_end_reasons"], rq1, cfg, fig_dir)

    summary_rows = {name: len(table) for name, table in tables.items()}
    write_manifest(
        "rq2", cfg, {**counts, "slice_corners": slice_corners, "summary_rows": summary_rows}
    )
    return counts
