"""RQ2 summaries: per-team-season rates with match-bootstrap intervals, cap sensitivity, end
reasons and shot rates by delivery type (Task 26, spec sections 6.3, 7 and 8.5).

Denominators are the **complete** chains (``outcome_status == "complete"``); incomplete chains are
counted in ``n_corners`` and ``n_incomplete`` only. A null ``shot_count``, ``goal`` or
``lost_possession`` in a complete chain raises ``ValueError``: it is never filled.

Intervals resample the matches of the sample (spec 6.3): the match-level numerators and
denominators of a team-season, padded with ``(0, 0)`` for its matches without a corner of its own
(``matches_in_sample`` of them in all), go to ``match_bootstrap_ratio``. With fewer than two
matches that have a complete chain the interval is NaN, because a single match would give a
zero-width interval that looks precise; every row says how many matches it rests on
(``n_matches``). The delivery-type rates resample the matches of the slice that have a corner.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from setpiece.config import Config
from setpiece.stats import match_bootstrap_ratio, rate

MEASURES: list[str] = [
    "shots_per_corner",
    "pct_corners_with_shot",
    "goals_per_corner",
    "losses_per_corner",
    "defending_restarts_per_corner",
]
# The measures of the delivery-type table (spec 8.5: shot rates only).
DELIVERY_MEASURES: list[str] = MEASURES[:2]
DELIVERY_ATTRIBUTES: list[str] = ["technique", "coarse_zone", "short_corner_proxy"]
# The sensitivity flag is about this measure only (spec 8.5).
SENSITIVITY_MEASURE = "shots_per_corner"

_TEAM_SEASON = ["competition_id", "season_id", "team_id"]

SUMMARY_COLUMNS: list[str] = [
    "competition_id",
    "season_id",
    "team_id",
    "team_name",
    "matches_in_sample",
    "sample_label",
    "measure",
    "numerator",
    "denominator",
    "n_incomplete",
    "n_corners",
    "n_matches",
    "value",
    "ci_lo",
    "ci_hi",
]
SUMMARY_DTYPES: dict[str, str] = {
    "competition_id": "Int64",
    "season_id": "Int64",
    "team_id": "Int64",
    "team_name": "string",
    "matches_in_sample": "Int64",
    "sample_label": "string",
    "measure": "string",
    "numerator": "Int64",
    "denominator": "Int64",
    "n_incomplete": "Int64",
    "n_corners": "Int64",
    "n_matches": "Int64",
    "value": "Float64",
    "ci_lo": "Float64",
    "ci_hi": "Float64",
}
assert list(SUMMARY_DTYPES) == SUMMARY_COLUMNS

DELIVERY_COLUMNS: list[str] = [
    "competition_id",
    "season_id",
    "slice_name",
    "attribute",
    "category",
    "measure",
    "numerator",
    "denominator",
    "n_incomplete",
    "n_corners",
    "n_matches",
    "value",
    "ci_lo",
    "ci_hi",
]
DELIVERY_DTYPES: dict[str, str] = {
    "competition_id": "Int64",
    "season_id": "Int64",
    "slice_name": "string",
    "attribute": "string",
    "category": "string",
    "measure": "string",
    "numerator": "Int64",
    "denominator": "Int64",
    "n_incomplete": "Int64",
    "n_corners": "Int64",
    "n_matches": "Int64",
    "value": "Float64",
    "ci_lo": "Float64",
    "ci_hi": "Float64",
}
assert list(DELIVERY_DTYPES) == DELIVERY_COLUMNS

END_REASON_COLUMNS: list[str] = [
    "competition_id",
    "season_id",
    "team_id",
    "end_reason",
    "restart_type",
    "restart_team",
    "outcome_status",
    "n",
]
END_REASON_DTYPES: dict[str, str] = {
    "competition_id": "Int64",
    "season_id": "Int64",
    "team_id": "Int64",
    "end_reason": "string",
    "restart_type": "string",
    "restart_team": "string",
    "outcome_status": "string",
    "n": "Int64",
}
assert list(END_REASON_DTYPES) == END_REASON_COLUMNS

# The columns of ``rq2_outcomes`` the summaries read, with the dtype each is cast to first (an
# empty frame built without dtypes is then as good as a stored one).
_OUTCOME_DTYPES: dict[str, str] = {
    "match_id": "Int64",
    "corner_event_id": "string",
    "end_reason": "string",
    "restart_type": "string",
    "restart_team": "string",
    "shot_count": "Int64",
    "goal": "boolean",
    "lost_possession": "boolean",
    "outcome_status": "string",
    "competition_id": "Int64",
    "season_id": "Int64",
    "team_id": "Int64",
}


def _key(*parts) -> str:
    """``"9/281/1"``: the slice and team of a team-season, as the errors name it."""
    return "/".join(str(p) for p in parts)


def _typed(outcomes: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """The ``columns`` of ``outcomes`` cast to their ``rq2_outcomes`` dtypes (a copy)."""
    columns = list(columns)
    return outcomes[columns].astype({c: _OUTCOME_DTYPES[c] for c in columns})


def _slice_keys(cfg: Config) -> set[tuple[int, int]]:
    return {(s.competition_id, s.season_id) for s in cfg.slices}


def _in_configured_slices(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The rows of ``frame`` whose slice is configured; ``frame`` has the slice id columns."""
    keys = _slice_keys(cfg)
    mask = [
        (c, s) in keys
        for c, s in zip(frame["competition_id"].astype(object), frame["season_id"].astype(object))
    ]
    return frame[np.array(mask, dtype=bool)]


def _prepare(outcomes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The chains of the configured slices with the columns the summaries read, validated.

    Adds ``complete`` (bool) and one 0/1-or-count column per measure, which is 0 for an incomplete
    chain: the numerators are those of the complete chains only. ``ValueError`` for a null id, or
    a null ``shot_count``, ``goal`` or ``lost_possession`` in a complete chain.
    """
    frame = _in_configured_slices(_typed(outcomes, _OUTCOME_DTYPES).reset_index(drop=True), cfg)
    for column in ("match_id", "team_id"):
        if frame[column].isna().any():
            raise ValueError(
                f"rq2_outcomes has {int(frame[column].isna().sum())} chain(s) with a null {column}"
            )
    complete = (frame["outcome_status"] == "complete").fillna(False).to_numpy(dtype=bool)
    for column in ("shot_count", "goal", "lost_possession"):
        missing = frame[column].isna().to_numpy() & complete
        if missing.any():
            raise ValueError(
                f"{int(missing.sum())} complete chain(s) have a null {column} "
                f"(first: {frame.loc[missing, 'corner_event_id'].iloc[0]}); it is never filled"
            )
    frame["complete"] = complete
    # Complete chains only: the (possibly null) values of an incomplete chain never enter.
    shots = frame["shot_count"].where(complete, 0).astype("int64")
    frame["shots_per_corner"] = shots
    frame["pct_corners_with_shot"] = (shots > 0).astype("int64")
    frame["goals_per_corner"] = frame["goal"].where(complete, False).astype("int64")
    frame["losses_per_corner"] = frame["lost_possession"].where(complete, False).astype("int64")
    # a null restart_team is a chain that was not followed by a restart, not a missing value
    frame["defending_restarts_per_corner"] = (
        frame["restart_team"].isin(["defending"]).to_numpy() & complete
    ).astype("int64")
    return frame


def _per_match(frame: pd.DataFrame, by: list[str], measures: list[str]) -> pd.DataFrame:
    """Per ``by`` group and match: ``n_chains``, ``den`` (complete chains) and the measure sums."""
    work = frame.assign(n_chains=1, den=frame["complete"].astype("int64"))
    return (
        work.groupby([*by, "match_id"], sort=True)[["n_chains", "den", *measures]]
        .sum()
        .reset_index()
    )


def _interval(num: np.ndarray, den: np.ndarray, cfg: Config) -> tuple[float, float, int]:
    """``(ci_lo, ci_hi, n_matches)``; the interval is NaN with fewer than 2 informative matches."""
    n_matches = int((den > 0).sum())
    if n_matches < 2:
        return float("nan"), float("nan"), n_matches
    lo, hi = match_bootstrap_ratio(num, den, cfg.p("bootstrap_reps"), cfg.seed)
    return lo, hi, n_matches


def _padded(values: pd.Series | np.ndarray, size: int) -> np.ndarray:
    """``values`` followed by zeros, ``size`` long (the matches without a corner)."""
    out = np.zeros(size, dtype=float)
    out[: len(values)] = np.asarray(values, dtype=float)
    return out


def summarise_rq2(outcomes: pd.DataFrame, samples: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per team-season of ``samples`` (in the configured slices) and measure.

    ``samples`` is ``derived/team_samples``. The rows are in ``samples`` order, each with the five
    ``MEASURES`` in order. ``numerator`` and ``denominator`` count the complete chains (the
    numerator of ``pct_corners_with_shot`` is the chains with a shot); ``n_corners`` counts all
    chains and ``n_incomplete`` those left out; ``n_matches`` is the number of matches with a
    complete chain. ``value`` is ``numerator / denominator``, NaN when there is none. The interval
    resamples all ``matches_in_sample`` matches of the team-season and is NaN for fewer than two
    matches with a complete chain.

    ``ValueError`` if a team-season has chains but no row in ``samples``, if it has corners in more
    matches than ``matches_in_sample``, or for the null checks of ``_prepare``.
    """
    frame = _prepare(outcomes, cfg)
    samples = _in_configured_slices(samples.reset_index(drop=True), cfg)
    sampled = {_key(*k) for k in zip(*(samples[c].astype(object) for c in _TEAM_SEASON))}
    seen = frame[_TEAM_SEASON].drop_duplicates()
    absent = [_key(*k) for k in zip(*(seen[c].astype(object) for c in _TEAM_SEASON))]
    absent = sorted(k for k in absent if k not in sampled)
    if absent:
        raise ValueError(
            f"chains of team-seasons that are not in team_samples: {', '.join(absent)}"
        )

    per_match = _per_match(frame, _TEAM_SEASON, MEASURES)
    groups = {k: g for k, g in per_match.groupby(_TEAM_SEASON, sort=False)}
    records = []
    for s in samples.itertuples(index=False):
        key = (s.competition_id, s.season_id, s.team_id)
        name = _key(*key)
        if pd.isna(s.matches_in_sample):
            raise ValueError(f"matches_in_sample is null for team-season {name}")
        size = int(s.matches_in_sample)
        g = groups.get(key)
        if g is None:
            g = per_match.iloc[:0]
        if len(g) > size:
            raise ValueError(
                f"team-season {name} has corners in {len(g)} matches "
                f"but matches_in_sample is {size}"
            )
        n_corners, denominator = int(g["n_chains"].sum()), int(g["den"].sum())
        den = _padded(g["den"], size)
        for measure in MEASURES:
            numerator = int(g[measure].sum())
            lo, hi, n_matches = _interval(_padded(g[measure], size), den, cfg)
            records.append({
                "competition_id": s.competition_id, "season_id": s.season_id, "team_id": s.team_id,
                "team_name": s.team_name, "matches_in_sample": s.matches_in_sample,
                "sample_label": s.sample_label, "measure": measure, "numerator": numerator,
                "denominator": denominator, "n_incomplete": n_corners - denominator,
                "n_corners": n_corners, "n_matches": n_matches,
                "value": rate(numerator, denominator), "ci_lo": lo, "ci_hi": hi,
            })  # fmt: skip
    return pd.DataFrame(records, columns=SUMMARY_COLUMNS).astype(SUMMARY_DTYPES)


def end_reason_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Chains per team-season by ``end_reason``, ``restart_type``, ``restart_team`` and status.

    Every chain counts, incomplete ones included (``outcome_status`` is kept as a column), and a
    null ``restart_type`` or ``restart_team`` is a group of its own, not dropped. Columns are
    ``END_REASON_COLUMNS``, ``n`` the number of chains; an empty typed table for no chains.
    """
    keys = [c for c in END_REASON_COLUMNS if c != "n"]
    typed = outcomes[keys].astype({c: END_REASON_DTYPES[c] for c in keys})
    counts = typed.groupby(keys, dropna=False, sort=True).size().rename("n").reset_index()
    return counts[END_REASON_COLUMNS].astype(END_REASON_DTYPES)


def _sensitivity_dtype(column: str) -> str:
    if column in SUMMARY_DTYPES and column != "value":
        return SUMMARY_DTYPES[column]
    return "boolean" if column == "sensitive" else "Float64"


def sensitivity_flags(main: pd.DataFrame, alt: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Is the shots per corner of a team-season sensitive to the time cap (spec 8.5)?

    ``main`` and every ``alt`` frame (``{variant: summary}``) are ``summarise_rq2`` tables of the
    same team-seasons, only their ``shots_per_corner`` rows being read. A team-season is
    ``sensitive`` if, for any variant, ``abs(value_variant - value_main) > (ci_hi - ci_lo) / 2``.
    The result has one row per team-season (keys: those of ``competition_id``, ``season_id``,
    ``team_id`` and ``measure`` that ``main`` has, then the labels ``team_name``,
    ``matches_in_sample`` and ``sample_label`` if it has them), ``value_main``, ``ci_lo``,
    ``ci_hi``, one ``value_<variant>`` per variant and ``sensitive`` (nullable boolean).

    ``sensitive`` is NA when it cannot be told: the main interval or value is NaN, or a variant
    has no value and no other variant moved by more than the half-width.
    """
    if not alt:
        raise ValueError("sensitivity_flags needs at least one alternative")
    keys = [c for c in (*_TEAM_SEASON, "measure") if c in main.columns]
    if not keys:
        raise ValueError(
            "main has none of the key columns competition_id, season_id, team_id, measure"
        )
    labels = [c for c in ("team_name", "matches_in_sample", "sample_label") if c in main.columns]

    def shots_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if "measure" in frame.columns:
            return frame[frame["measure"] == SENSITIVITY_MEASURE]
        return frame

    out = shots_rows(main)[[*keys, *labels, "value", "ci_lo", "ci_hi"]].rename(
        columns={"value": "value_main"}
    )
    for variant, frame in alt.items():
        values = shots_rows(frame)[[*keys, "value"]].rename(columns={"value": f"value_{variant}"})
        out = out.merge(values, on=keys, how="left", validate="one_to_one")
    value_columns = ["value_main", "ci_lo", "ci_hi", *(f"value_{v}" for v in alt)]
    out = out.astype({c: "Float64" for c in value_columns}).reset_index(drop=True)

    half_width = (out["ci_hi"] - out["ci_lo"]) / 2
    sensitive = None
    for variant in alt:
        moved = (out[f"value_{variant}"] - out["value_main"]).abs() > half_width  # NA if unknown
        sensitive = moved if sensitive is None else sensitive | moved  # Kleene: True beats NA
    out["sensitive"] = sensitive
    return out.astype({c: _sensitivity_dtype(c) for c in out.columns})


def rates_by_delivery(outcomes: pd.DataFrame, rq1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Shot rates by delivery type, per slice (spec 8.5, descriptive only).

    For each configured slice, each attribute of ``DELIVERY_ATTRIBUTES`` (``technique``,
    ``coarse_zone``, ``short_corner_proxy``, read from ``rq1`` through
    ``corner_event_id == event_id``) and each category it takes, the two ``DELIVERY_MEASURES``
    over the complete chains of that category. Categories are text; a null (a ground pass without
    swing recorded as ``Unknown`` already, a corner without an end location, a proxy that cannot be
    computed) is the category ``"Unknown"``, and the proxy's are ``"True"`` and ``"False"``.
    Intervals resample the slice's matches that have a corner, both teams together.

    ``ValueError`` for a chain without an ``rq1`` row (``rq1_corners`` is stale) or a duplicated
    id, and for the null checks of ``_prepare``.
    """
    frame = _prepare(outcomes, cfg)
    attributes = rq1[["event_id", *DELIVERY_ATTRIBUTES]]
    joined = frame.merge(
        attributes, how="left", left_on="corner_event_id", right_on="event_id",
        validate="one_to_one", indicator=True,
    )  # fmt: skip
    unmatched = joined["_merge"] == "left_only"
    if unmatched.any():
        ids = ", ".join(joined.loc[unmatched, "corner_event_id"].astype(str).head(5))
        raise ValueError(
            f"{int(unmatched.sum())} chain(s) have no row in rq1_corners (first: {ids}); "
            "run rq1 again"
        )
    records = []
    for sl in cfg.slices:
        chains = joined[
            (joined["competition_id"] == sl.competition_id) & (joined["season_id"] == sl.season_id)
        ]
        if chains.empty:
            continue
        matches = np.sort(chains["match_id"].unique().astype("int64"))
        for attribute in DELIVERY_ATTRIBUTES:
            category = chains[attribute].astype("string").fillna("Unknown")
            per_match = _per_match(
                chains.assign(_category=category.to_numpy()), ["_category"], DELIVERY_MEASURES
            ).set_index(["_category", "match_id"])
            for cat in sorted(category.unique()):
                g = per_match.loc[cat].reindex(matches, fill_value=0)
                n_corners, denominator = int(g["n_chains"].sum()), int(g["den"].sum())
                for measure in DELIVERY_MEASURES:
                    numerator = int(g[measure].sum())
                    lo, hi, n_matches = _interval(
                        g[measure].to_numpy(dtype=float), g["den"].to_numpy(dtype=float), cfg
                    )
                    records.append({
                        "competition_id": sl.competition_id, "season_id": sl.season_id,
                        "slice_name": sl.name, "attribute": attribute, "category": cat,
                        "measure": measure, "numerator": numerator, "denominator": denominator,
                        "n_incomplete": n_corners - denominator, "n_corners": n_corners,
                        "n_matches": n_matches, "value": rate(numerator, denominator),
                        "ci_lo": lo, "ci_hi": hi,
                    })  # fmt: skip
    return pd.DataFrame(records, columns=DELIVERY_COLUMNS).astype(DELIVERY_DTYPES)
