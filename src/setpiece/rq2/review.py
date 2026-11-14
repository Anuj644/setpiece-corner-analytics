"""RQ2 review for gate G3 (Task 24): a stratified sample of corner chains, blind material to label
them from, and the agreement between labellers and with the chain rule.

``run_review_export_rq2`` draws the sample and writes ``reports/review/rq2/``: ``sheet.csv`` (the
columns a labeller fills in) and one page per chain, ``chains/<review_id>.html``. Neither says which
stratum a chain was drawn from or what the rule decided; the stratum of each ``review_id`` goes to
``reports/review/rq2_key.csv``, outside the folder the labellers see. ``agreement`` compares two
filled sheets, ``rule_labels`` turns the rule's own answers into a sheet to compare against.
"""

from __future__ import annotations

from pathlib import Path

import jinja2
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from setpiece.config import Config
from setpiece.io import read_table

# Draw order and priority: a corner belongs to the first stratum it qualifies for.
STRATA = ["short", "shot", "clearance_first", "repeat_other"]

# The corner and the events after it that a page shows.
EVENTS_SHOWN_AFTER_CORNER = 40

SHEET_COLUMNS = [
    "review_id", "match_id", "corner_event_id", "last_event_id", "shot_count", "goal",
    "lost_possession",
]  # fmt: skip
LABEL_COLUMNS = ["last_event_id", "shot_count", "goal", "lost_possession"]
PAGE_COLUMNS = [
    "event_id", "event_index", "period", "timestamp", "team_name", "player_name", "type",
    "outcome", "possession_id", "out",
]  # fmt: skip
KEY_COLUMNS = [
    "review_id", "corner_event_id", "match_id", "competition_id", "season_id", "team_id", "stratum",
]  # fmt: skip


# --- strata and the sample -------------------------------------------------------------------


def _next_event_of_the_match(events: pd.DataFrame, matches: pd.Series) -> pd.DataFrame:
    """Each event of ``matches`` with the type and team of the event after it, by ``event_index``.

    The columns are ``match_id``, ``event_id``, ``team_id``, ``next_type`` and ``next_team_id``;
    the last event of a match has nulls for the two ``next_`` columns.
    """
    columns = ["match_id", "event_id", "event_index", "event_type", "team_id"]
    wanted = events.loc[events["match_id"].isin(matches), columns]
    wanted = wanted.sort_values(["match_id", "event_index"], kind="stable")
    by_match = wanted.groupby("match_id", sort=False, dropna=False)
    wanted["next_type"] = by_match["event_type"].shift(-1)
    wanted["next_team_id"] = by_match["team_id"].shift(-1)
    return wanted.drop(columns=["event_index", "event_type"])


def assign_stratum(outcomes: pd.DataFrame, rq1: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """The stratum of every corner of ``outcomes`` (same index), by priority:

    * ``short``: ``short_corner_proxy`` of its ``rq1`` row is True (a null proxy is not short);
    * ``shot``: ``shot_count > 0``;
    * ``clearance_first``: the next event of the same match by ``event_index`` is a Clearance by
      a team other than the corner taker's (an event without a team, or no next event, is not one);
    * ``repeat_other``: everything else.

    Those nulls are turned into False on purpose and only here: each test asks "is there positive
    evidence for this stratum", and a missing value is none. ``ValueError`` if a corner has no row
    in ``rq1`` (joined ``corner_event_id`` to ``event_id``, one to one) or its corner event is not
    in ``events``.
    """
    if outcomes["corner_event_id"].isna().any():
        raise ValueError("outcomes has a null corner_event_id")
    proxy = outcomes[["corner_event_id"]].merge(
        rq1[["event_id", "short_corner_proxy"]],
        how="left",
        left_on="corner_event_id",
        right_on="event_id",
        validate="one_to_one",
        indicator=True,
    )
    no_rq1 = proxy["_merge"] == "left_only"
    if no_rq1.any():
        missing = proxy.loc[no_rq1, "corner_event_id"].tolist()
        raise ValueError(f"no rq1 row for {len(missing)} corner(s), e.g. {missing[:3]}")
    short = (proxy["short_corner_proxy"] == True).fillna(False).to_numpy(dtype=bool)

    following = _next_event_of_the_match(events, outcomes["match_id"])
    corners = outcomes[["match_id", "corner_event_id"]].merge(
        following,
        how="left",
        left_on=["match_id", "corner_event_id"],
        right_on=["match_id", "event_id"],
        validate="many_to_one",
        indicator=True,
    )
    no_events = corners["_merge"] == "left_only"
    if no_events.any():
        missing = corners.loc[no_events, "corner_event_id"].tolist()
        raise ValueError(f"{len(missing)} corner(s) not in events, e.g. {missing[:3]}")
    by_other_team = corners["next_team_id"] != corners["team_id"]
    cleared = (corners["next_type"] == "Clearance") & by_other_team
    clearance_first = cleared.fillna(False).to_numpy(dtype=bool)

    shot = (outcomes["shot_count"] > 0).fillna(False).to_numpy(dtype=bool)
    stratum = np.select([short, shot, clearance_first], STRATA[:3], default=STRATA[3])
    return pd.Series(stratum, index=outcomes.index, name="stratum")


def draw_sample(pool: pd.DataFrame, per_stratum: int, seed: int) -> pd.DataFrame:
    """``min(per_stratum, available)`` chains of each stratum of ``pool``, numbered by a shuffle.

    ``pool`` is outcome rows with a ``stratum`` column (``assign_stratum``). The pool is sorted by
    ``corner_event_id`` first, so the draw does not depend on the order of its rows; each stratum,
    in the order of ``STRATA``, is drawn without replacement with ``np.random.default_rng(seed)``.
    The drawn rows are then shuffled with the same generator and numbered ``review_id`` 1..N in
    that order, so the numbering says nothing about the stratum. Returns the drawn rows with
    ``review_id`` as the last column, in ``review_id`` order.
    """
    if per_stratum < 0:
        raise ValueError(f"per_stratum must not be negative, got {per_stratum}")
    if not pool["corner_event_id"].is_unique:
        raise ValueError("the pool has a corner_event_id more than once")
    pool = pool.sort_values("corner_event_id", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    drawn: list[np.ndarray] = []
    for name in STRATA:
        members = np.flatnonzero((pool["stratum"] == name).to_numpy(dtype=bool))
        take = min(per_stratum, len(members))
        if take:
            drawn.append(rng.choice(members, size=take, replace=False))
    positions = np.concatenate(drawn) if drawn else np.array([], dtype=int)
    positions = positions[rng.permutation(len(positions))]
    sample = pool.iloc[positions].reset_index(drop=True)
    sample["review_id"] = np.arange(1, len(sample) + 1)
    return sample


def sample_chains(
    outcomes: pd.DataFrame,
    rq1: pd.DataFrame,
    events: pd.DataFrame,
    per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    """The review sample: ``min(per_stratum, available)`` chains per stratum, across all slices.

    The result is the sampled rows of ``outcomes`` with ``stratum`` and ``review_id`` added. The
    same inputs and seed give the same sample whatever the order of the input rows.
    """
    pool = outcomes.assign(stratum=assign_stratum(outcomes, rq1, events))
    return draw_sample(pool, per_stratum, seed)


# --- the export ------------------------------------------------------------------------------


def _cell(value) -> str:
    """A table cell as text; any kind of null is the empty string."""
    if value is None or (not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value)):
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _event_row(e) -> list[str]:
    """The page cells of one event (a row tuple of ``events``), in ``PAGE_COLUMNS`` order."""
    kind, detail = _cell(e.event_type), _cell(e.pass_type) or _cell(e.shot_type)
    type_text = " ".join(part for part in (kind, f"({detail})" if detail else "") if part)
    return [
        _cell(e.event_id), _cell(e.event_index), _cell(e.period), _cell(e.timestamp),
        _cell(e.team_name), _cell(e.player_name), type_text,
        _cell(e.pass_outcome) or _cell(e.shot_outcome), _cell(e.possession_id), _cell(e.out),
    ]  # fmt: skip


def _environment() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.PackageLoader("setpiece.rq2", "templates"),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
    )


def _render_pages(sample: pd.DataFrame, events: pd.DataFrame) -> dict[int, str]:
    """The HTML page of every sampled chain, by ``review_id``.

    A page is built from ``events`` and the ``review_id`` alone: nothing else of the sample row
    reaches the template, so the rule's output cannot leak into it.
    """
    template = _environment().get_template("chain.html.j2")
    wanted = events[events["match_id"].isin(sample["match_id"])]
    matches = {
        match_id: frame.sort_values("event_index", kind="stable").reset_index(drop=True)
        for match_id, frame in wanted.groupby("match_id", sort=False)
    }
    pages: dict[int, str] = {}
    for review_id, match_id, corner_id in zip(
        sample["review_id"], sample["match_id"], sample["corner_event_id"]
    ):
        match = matches.get(match_id)
        found = np.array([], dtype=int)
        if match is not None:
            is_corner = (match["event_id"] == corner_id).to_numpy(dtype=bool, na_value=False)
            found = np.flatnonzero(is_corner)
        if len(found) != 1:
            raise ValueError(
                f"corner {corner_id!r} of match {match_id}: found {len(found)} times in events"
            )
        start = int(found[0])
        shown = match.iloc[start : start + 1 + EVENTS_SHOWN_AFTER_CORNER]
        corner = shown.iloc[0]
        pages[int(review_id)] = template.render(
            review_id=int(review_id),
            home_team=_cell(corner["home_team_name"]),
            away_team=_cell(corner["away_team_name"]),
            match_date=_cell(corner["match_date"]),
            attacking_team=_cell(corner["team_name"]),
            corner_period=_cell(corner["period"]),
            corner_timestamp=_cell(corner["timestamp"]),
            columns=PAGE_COLUMNS,
            rows=[_event_row(e) for e in shown.itertuples(index=False)],
        )
    return pages


def _blank_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    """The sheet of ``sample``: its identifying columns and empty label columns.

    The label columns are new and empty; the rule's ``shot_count``, ``goal`` and
    ``lost_possession`` in ``sample`` have the same names and are deliberately not carried over.
    """
    sheet = sample[["review_id", "match_id", "corner_event_id"]].copy()
    for column in LABEL_COLUMNS:
        sheet[column] = pd.Series(pd.NA, index=sheet.index, dtype="object")
    return sheet[SHEET_COLUMNS]


def export_review(sample: pd.DataFrame, events: pd.DataFrame, out_dir: str | Path) -> None:
    """Write ``sheet.csv`` and ``chains/<review_id>.html`` for ``sample`` into ``out_dir``.

    ``sample`` needs ``review_id``, ``match_id`` and ``corner_event_id`` (the other columns are
    ignored); ``events`` must hold the whole match of every sampled corner. The sheet has the
    columns ``review_id, match_id, corner_event_id, last_event_id, shot_count, goal,
    lost_possession``, the last four empty. A page lists the corner and the 40 events after it
    (fewer at the end of a match) and shows no rule output. Everything is built before anything is
    written, and the pages and sheet of an earlier export are then replaced, so ``ValueError`` for
    a corner that is not in ``events`` leaves ``out_dir`` as it was. Files are UTF-8.
    """
    out = Path(out_dir)
    pages = _render_pages(sample, events)
    sheet = _blank_sheet(sample)

    chains = out / "chains"
    chains.mkdir(parents=True, exist_ok=True)
    for stale in chains.glob("*.html"):
        stale.unlink()
    for review_id, html in pages.items():
        (chains / f"{review_id}.html").write_text(html, encoding="utf-8")
    sheet.to_csv(out / "sheet.csv", index=False, encoding="utf-8", lineterminator="\n")


# --- agreement -------------------------------------------------------------------------------


def _labels(frame: pd.DataFrame, who: str) -> pd.DataFrame:
    """The four label columns of ``frame``, checked, indexed by ``review_id`` (as int)."""
    missing = [c for c in ("review_id", *LABEL_COLUMNS) if c not in frame.columns]
    if missing:
        raise ValueError(f"{who} has no column(s) {missing}")
    labels = frame[["review_id", *LABEL_COLUMNS]]
    unlabelled = labels.columns[labels.isna().any()].tolist()
    if unlabelled:
        raise ValueError(f"{who} has empty value(s) in {unlabelled}")
    if labels["review_id"].duplicated().any():
        raise ValueError(f"{who} has a review_id more than once")
    labels = labels.assign(review_id=labels["review_id"].astype("int64")).set_index("review_id")
    return labels


def _codes(series: pd.Series, column: str, who: str) -> np.ndarray:
    """``series`` as integer codes: bools and 0/1 alike; ``ValueError`` for anything else."""
    try:
        numbers = pd.to_numeric(series)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{who} has a non-numeric value in {column}") from exc
    if pd.api.types.is_bool_dtype(numbers):
        numbers = numbers.astype("int64")
    if not (numbers == numbers.round()).all():
        raise ValueError(f"{who} has a non-integer value in {column}")
    codes = numbers.astype("int64").to_numpy()
    if column == "shot_count" and (codes < 0).any():
        raise ValueError(f"{who} has a negative shot_count")
    if column != "shot_count" and not np.isin(codes, [0, 1]).all():
        raise ValueError(f"{who} has a value other than 0/1 (or False/True) in {column}")
    return codes


def _kappa(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's kappa of two label arrays; NaN when it is undefined (one shared single category)."""
    if len(np.union1d(x, y)) < 2:
        return float("nan")  # sklearn warns and returns nan here; the answer is "not defined"
    return float(cohen_kappa_score(x, y))


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """How far two labellings of the same chains agree.

    ``a`` and ``b`` have ``review_id``, ``last_event_id``, ``shot_count``, ``goal`` and
    ``lost_possession`` (extra columns are ignored); rows are matched by ``review_id``. Returns

    * ``n``: the number of chains;
    * ``boundary_exact``: the share with the same ``last_event_id``, compared as text with the
      whitespace around it ignored (ids pasted from a page often carry some);
    * ``kappa_shot_count``, ``kappa_goal``, ``kappa_lost_possession``: Cohen's kappa
      (``sklearn.metrics.cohen_kappa_score``, unweighted) over the integer codes, bool or 0/1; NaN
      if both labellers used one and the same single category, where kappa is not defined;
    * ``exact_shot_count``, ``exact_goal``, ``exact_lost_possession``: the raw share that agrees.

    ``ValueError`` if the two have different ``review_id`` sets, a ``review_id`` twice, no rows, an
    empty label, a ``goal`` or ``lost_possession`` that is not 0/1 or a ``shot_count`` that is not
    a non-negative integer.
    """
    first, second = _labels(a, "the first labelling"), _labels(b, "the second labelling")
    if set(first.index) != set(second.index):
        only_a = sorted(set(first.index) - set(second.index))
        only_b = sorted(set(second.index) - set(first.index))
        raise ValueError(
            f"the review_id sets differ: only in the first {only_a[:5]}, only in the second "
            f"{only_b[:5]}"
        )
    if first.empty:
        raise ValueError("no rows to compare")
    second = second.loc[first.index]

    ids_a = first["last_event_id"].astype("string").str.strip()
    ids_b = second["last_event_id"].astype("string").str.strip()
    result: dict = {"n": len(first), "boundary_exact": float((ids_a == ids_b).mean())}
    exact = {}
    for column in ("shot_count", "goal", "lost_possession"):
        x = _codes(first[column], column, "the first labelling")
        y = _codes(second[column], column, "the second labelling")
        result[f"kappa_{column}"] = _kappa(x, y)
        exact[f"exact_{column}"] = float((x == y).mean())
    return {**result, **exact}


def rule_labels(sample: pd.DataFrame) -> pd.DataFrame:
    """The chain rule's answers for ``sample`` in sheet format, to compare with labellers.

    Columns ``review_id``, ``last_event_id`` (the rule's ``last_included_event_id``, which is the
    corner itself when no event follows it in the chain), ``shot_count`` and, as 0/1 integers,
    ``goal`` and ``lost_possession``; sorted by ``review_id``. ``ValueError`` for a null rule value.
    """
    rule = sample[["review_id", "last_included_event_id", "shot_count", "goal", "lost_possession"]]
    unset = rule.columns[rule.isna().any()].tolist()
    if unset:
        raise ValueError(f"the sample has null rule value(s) in {unset}")
    labels = pd.DataFrame(
        {
            "review_id": rule["review_id"].astype("int64"),
            "last_event_id": rule["last_included_event_id"].astype("string"),
            "shot_count": rule["shot_count"].astype("int64"),
            "goal": rule["goal"].astype("int64"),
            "lost_possession": rule["lost_possession"].astype("int64"),
        }
    )
    return labels.sort_values("review_id").reset_index(drop=True)


def rule_vs_consensus(rule: pd.DataFrame, consensus: pd.DataFrame) -> dict:
    """``agreement`` of the rule's labels (``rule_labels``) with the labellers' consensus sheet."""
    return agreement(rule, consensus)


# --- the runner ------------------------------------------------------------------------------


def _read_derived(table: str, cfg: Config, stage: str) -> pd.DataFrame:
    try:
        return read_table(table, cfg)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc}; run `python -m setpiece {stage}` first") from exc


def _in_slice(frame: pd.DataFrame, competition_id: int, season_id: int) -> pd.Series:
    keep = (frame["competition_id"] == competition_id) & (frame["season_id"] == season_id)
    return keep.fillna(False).astype(bool)


def run_review_export_rq2(cfg: Config) -> dict:
    """Write the G3 review material for RQ2 to ``reports/review/rq2/`` and return its counts.

    The pool is the complete chains of the main variant of ``derived/rq2_outcomes`` (written by the
    ``rq2`` stage) in the core slices. ``rq2_review_per_stratum`` chains are drawn from each
    stratum with ``cfg.seed``. ``events`` is read one core slice at a time (a league season is
    gigabytes), first to assign the strata, then again for the matches of the drawn chains.

    Files: ``reports/review/rq2/sheet.csv`` and ``chains/<review_id>.html`` (those of an earlier
    run are removed), and, outside that folder, ``reports/review/rq2_key.csv`` with ``review_id``,
    ``corner_event_id``, ``match_id``, ``competition_id``, ``season_id``, ``team_id`` and
    ``stratum``. Returns ``{"pool", "available", "drawn", "total"}``, the two middle ones per
    stratum.
    """
    outcomes = _read_derived("derived/rq2_outcomes", cfg, "rq2")
    rq1 = _read_derived("derived/rq1_corners", cfg, "rq1")
    core = [(s.competition_id, s.season_id) for s in cfg.slices if s.core]
    if not core:
        raise ValueError("no core slices configured")
    complete = outcomes[outcomes["outcome_status"] == "complete"]

    parts = []
    for competition_id, season_id in core:
        part = complete[_in_slice(complete, competition_id, season_id)]
        if part.empty:
            continue
        events = read_table("events", cfg, [(competition_id, season_id)])
        parts.append(part.assign(stratum=assign_stratum(part, rq1, events)))
        del events
    if not parts:
        raise ValueError("no complete chains in the core slices")
    pool = pd.concat(parts, ignore_index=True)
    sample = draw_sample(pool, cfg.p("rq2_review_per_stratum"), cfg.seed)

    drawn_events = []
    for competition_id, season_id in core:
        chosen = sample[_in_slice(sample, competition_id, season_id)]
        if chosen.empty:
            continue
        events = read_table("events", cfg, [(competition_id, season_id)])
        drawn_events.append(events[events["match_id"].isin(chosen["match_id"])])
        del events
    events = pd.concat(drawn_events, ignore_index=True)

    review = cfg.reports_dir / "review"
    export_review(sample, events, review / "rq2")
    sample[KEY_COLUMNS].to_csv(review / "rq2_key.csv", index=False, lineterminator="\n")

    def per_stratum(frame: pd.DataFrame) -> dict[str, int]:
        counts = frame["stratum"].value_counts()
        return {name: int(counts.get(name, 0)) for name in STRATA}

    return {
        "pool": len(pool),
        "available": per_stratum(pool),
        "drawn": per_stratum(sample),
        "total": len(sample),
    }
