"""The ``rq2`` stage: ``events`` -> ``rq2_outcomes``, one row per corner, at four time caps.

The main table uses ``cap_seconds`` and has the name ``rq2_outcomes``; each ``cap_sensitivity``
entry gives one more table (``rq2_outcomes_cap10``, ``rq2_outcomes_cap20``, ``rq2_outcomes_nocap``
for the default config, ``null`` meaning no cap). All of them are written under ``derived/``.

A chain never crosses a match, so ``events`` is read one configured slice at a time and the outcomes
of the slices are stacked afterwards. Every table is rebuilt whole from the configured slices.
"""

import pandas as pd

from setpiece.config import Config
from setpiece.io import read_table, write_manifest, write_table
from setpiece.rq2.chains import build_rq2_outcomes, restarts_from

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


def run_rq2(cfg: Config) -> dict[str, dict[str, int]]:
    """Write ``derived/rq2_outcomes`` and its cap variants from the configured slices.

    Returns ``{table: {corners, complete, shots, goals}}``, the counts the ``rq2`` manifest holds.
    ``ValueError`` if the config has no slices (nothing is written).
    """
    if not cfg.slices:
        raise ValueError("no slices configured")
    restarts = restarts_from(cfg)
    variants = _variants(cfg)
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in variants}
    for slice in cfg.slices:
        events = read_table("events", cfg, [(slice.competition_id, slice.season_id)])
        for name, cap in variants.items():
            pieces[name].append(build_rq2_outcomes(events, cap, restarts))
        del events  # a big league season is gigabytes; only the slice at hand is kept
    counts = {}
    for name, frames in pieces.items():
        outcomes = pd.concat(frames, ignore_index=True)
        write_table(outcomes, f"derived/{name}", cfg, None)
        counts[name] = _counts(outcomes)
    write_manifest("rq2", cfg, counts)
    return counts
