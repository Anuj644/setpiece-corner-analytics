"""Task 21: how does ``possession_id`` behave around restarts, own goals and ``out`` flags?

Run before the RQ2 chain rule (spec section 8.1) is locked::

    .venv/Scripts/python scripts/rq2_possession_check.py [path/to/configs/default.yaml]

It reads the ``events`` table one core slice at a time and answers four questions, for every core
slice, for the four slices the 2026-11-10 pre-check covered (everything except Premier League
2015/16) and for all core slices together:

(a) Does a restart pass (Throw-in, Goal Kick, Free Kick, Kick Off, Corner) always start a new
    ``possession_id``, also when the same team takes it?
(b) Does an own goal change the possession, and what follows it?
(c) After an event with ``out == True``, which event starts the next possession, and how many new
    possessions follow an ``out`` flag?
(d) Do a free-kick shot and a penalty shot start a new possession?

"Starts a new possession" means: the event's ``possession_id`` differs from the previous event's
in the same match, in ``event_index`` order (``setpiece.rq2.diagnostics``). Only counts are kept
across slices, so the totals are sums, never averages of shares. Output is ASCII.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from setpiece.config import Slice, load_config
from setpiece.io import read_table
from setpiece.rq2.diagnostics import (
    flag_possession_starts,
    possession_start_summary,
    possession_starts,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
RESTART_PASS_TYPES = ("Throw-in", "Goal Kick", "Free Kick", "Kick Off", "Corner")
OWN_GOALS = ("Own Goal For", "Own Goal Against")
HEADLINE_SHOTS = ("Free Kick", "Penalty")
# Covered by the 2026-11-10 pre-check (165 matches); Premier League 2015/16 is the first look.
PRE_CHECKED = {(43, 106), (9, 281), (11, 90), (7, 235)}
NEEDED = [
    "match_id", "event_index", "period", "minute", "possession_id", "possession_team_id",
    "event_type", "pass_type", "shot_type", "out",
]  # fmt: skip
LISTING_CAP = 25  # rows listed for any anomaly bucket


# --------------------------------------------------------------------------------- collecting


def _labels(event_type: pd.Series, pass_type: pd.Series) -> np.ndarray:
    """``Pass / Throw-in``-style labels; a pass without a pass_type is ``Pass / (none)``."""
    et = event_type.fillna("(null)").to_numpy(dtype=object)
    pt = pass_type.fillna("(none)").to_numpy(dtype=object)
    return np.where(et == "Pass", "Pass / " + pt, et)


def _team_bucket(same_team: pd.Series) -> np.ndarray:
    """``same`` / ``diff`` / ``unknown`` (first event of a match, or a null team)."""
    value = same_team.fillna(False).to_numpy(dtype=bool)
    return np.where(same_team.isna().to_numpy(), "unknown", np.where(value, "same", "diff"))


def _prior_context(mid, pid, etype, pos: int, cap: int = 6) -> tuple[int, str]:
    """Events before ``pos`` inside its own possession: their number and the last few types."""
    types, count, j = [], 0, pos - 1
    while j >= 0 and mid[j] == mid[pos] and pid[j] == pid[pos]:
        if len(types) < cap:
            types.append(etype[j])
        count += 1
        j -= 1
    return count, " <- ".join(types) if types else "(it opens its match)"


def collect(cfg, sl: Slice) -> dict:
    """Read one slice and reduce it to small count frames; the events are dropped afterwards."""
    events = read_table("events", cfg, [(sl.competition_id, sl.season_id)])[NEEDED]
    starts = possession_starts(events)
    summary = possession_start_summary(events)
    flagged = flag_possession_starts(events)
    del events

    n = len(flagged)
    mid = flagged["match_id"].to_numpy(dtype="int64")
    eidx = flagged["event_index"].to_numpy(dtype="int64")
    pid = flagged["possession_id"].to_numpy(dtype="float64", na_value=np.nan)
    per = flagged["period"].to_numpy(dtype="float64", na_value=np.nan)
    minute = flagged["minute"].to_numpy(dtype="float64", na_value=np.nan)
    et = flagged["event_type"].fillna("(null)").to_numpy(dtype=object)
    pt = flagged["pass_type"].fillna("(none)").to_numpy(dtype=object)
    shot = flagged["shot_type"].fillna("(none)").to_numpy(dtype=object)
    prev_et = flagged["previous_event_type"].fillna("(none)").to_numpy(dtype=object)
    prev_out = flagged["previous_out"].fillna(False).to_numpy(dtype=bool)
    out_flag = flagged["out"].fillna(False).to_numpy(dtype=bool)
    st_mask = flagged["starts_possession"].to_numpy(dtype=bool)
    team = _team_bucket(flagged["same_team_as_previous"])
    label = _labels(flagged["event_type"], flagged["pass_type"])
    assert st_mask.sum() == len(starts)
    assert (pid[st_mask] == starts["possession_id"].to_numpy(dtype="float64")).all()

    same_match_next = np.zeros(n, dtype=bool)  # event i + 1 exists and is in the same match
    same_match_next[:-1] = mid[1:] == mid[:-1]

    def pick(mask: np.ndarray, key: np.ndarray) -> pd.DataFrame:
        idx = np.flatnonzero(mask)
        prior = [
            _prior_context(mid, pid, et, i) if not st_mask[i] else (0, "") for i in idx
        ]  # only events that continue a possession need an explanation
        return pd.DataFrame(
            {
                "slice": sl.name,
                "key": key[idx],
                "match_id": mid[idx],
                "event_index": eidx[idx],
                "period": per[idx],
                "minute": minute[idx],
                "starts_possession": st_mask[idx],
                "team": team[idx],
                "previous_event_type": prev_et[idx],
                "previous_out": prev_out[idx],
                "n_prior_in_possession": [p[0] for p in prior],
                "prior_types": [p[1] for p in prior],
            }
        )

    # (a) restart passes and (d) non-open-play shots, at event level
    restarts = pick((et == "Pass") & np.isin(pt, RESTART_PASS_TYPES), pt)
    shots = pick((et == "Shot") & (shot != "(none)") & (shot != "Open Play"), shot)
    drops = pick(et == "Referee Ball-Drop", et)  # extra: a restart in spec 8.1, but not a pass

    # (b) own goals: skip the sibling event of the pair to find what comes before and after
    own_rows = []
    for i in np.flatnonzero(np.isin(et, OWN_GOALS)):
        before = i - 1
        while before >= 0 and mid[before] == mid[i] and et[before] in OWN_GOALS:
            before -= 1
        after = i + 1
        while after < n and mid[after] == mid[i] and et[after] in OWN_GOALS:
            after += 1
        has_before = before >= 0 and mid[before] == mid[i]
        has_after = after < n and mid[after] == mid[i]
        skipped, k = [], after
        while k < n and mid[k] == mid[i] and et[k] != "Pass":
            skipped.append(et[k])
            k += 1
        has_pass = k < n and mid[k] == mid[i]
        own_rows.append(
            {
                "slice": sl.name,
                "own_goal_type": et[i],
                "match_id": mid[i],
                "event_index": eidx[i],
                "period": per[i],
                "same_possession_as_event_before": bool(has_before and pid[before] == pid[i]),
                "next_event": label[after] if has_after else "(end of match)",
                "next_event_new_possession": bool(has_after and pid[after] != pid[i]),
                "next_pass": label[k] if has_pass else "(no later pass)",
                "next_pass_new_possession": bool(has_pass and pid[k] != pid[i]),
                "events_skipped_before_pass": len(skipped) if has_pass else -1,
                "skipped_types": ", ".join(skipped) if skipped else "-",
            }
        )
    own_goals = pd.DataFrame(own_rows)

    # (c) out flags: the next event, and the first event of the next possession in the period
    out_pos = np.flatnonzero(out_flag)
    start_pos = np.flatnonzero(st_mask)
    j = np.searchsorted(start_pos, out_pos, side="right")  # first start strictly after the event
    found = j < len(start_pos)
    nxt = start_pos[np.minimum(j, len(start_pos) - 1)]
    found &= (mid[nxt] == mid[out_pos]) & (per[nxt] == per[out_pos])
    immediate = found & (nxt == out_pos + 1)
    following = np.minimum(out_pos + 1, n - 1)
    has_following = same_match_next[out_pos]
    outs = pd.DataFrame(
        {
            "slice": sl.name,
            "next_event": np.where(has_following, label[following], "(end of match)"),
            "next_event_starts_possession": immediate,
            "next_start": np.where(found, label[nxt], "(no new possession in the period)"),
        }
    )

    # (c) new possessions, and whether the previous possession held an out flag anywhere
    has_out = (
        pd.Series(out_flag)
        .groupby([mid, pid], sort=False, dropna=False)
        .transform("any")
        .to_numpy(dtype=bool)[st_mask]
    )
    start_match = starts["match_id"].to_numpy(dtype="int64")
    has_previous = np.zeros(len(starts), dtype=bool)
    has_previous[1:] = start_match[1:] == start_match[:-1]
    previous_held_out = np.zeros(len(starts), dtype=bool)
    previous_held_out[1:] = has_out[:-1]
    start_frame = pd.DataFrame(
        {
            "slice": sl.name,
            "first_event": _labels(starts["first_event_type"], starts["first_pass_type"]),
            "has_previous_event": starts["previous_event_type"].notna().to_numpy(),
            "previous_out": starts["previous_out"].fillna(False).to_numpy(dtype=bool),
            "has_previous_possession": has_previous,
            "previous_possession_held_out": previous_held_out & has_previous,
        }
    )

    dropped = np.diff(pid)  # NaN compares false: a null possession_id is not counted here
    decreasing = (mid[1:] == mid[:-1]) & (dropped < 0)
    lower = np.zeros(n, dtype=bool)  # events whose id is lower than the previous event's
    lower[1:] = decreasing
    lower_pos = np.flatnonzero(lower)
    decreases = pd.DataFrame(
        {
            "slice": sl.name,
            "match_id": mid[lower_pos],
            "event_index": eidx[lower_pos],
            "period": per[lower_pos],
            "event": label[lower_pos],
            "previous_event": prev_et[lower_pos],
            "possession_id": pid[lower_pos],
            "previous_possession_id": pid[lower_pos - 1],
            "id_started_earlier": [
                bool(((mid[:i] == mid[i]) & (pid[:i] == pid[i])).any()) for i in lower_pos
            ],
        }
    )
    overview = {
        "matches": int(flagged["match_id"].nunique()),
        "events": n,
        "null_possession_id": int(np.isnan(pid).sum()),
        "matches_with_decreasing_possession_id": len(np.unique(mid[1:][decreasing])),
        "new_possessions": len(starts),
        "possession_ids_started_twice": int(starts.duplicated(["match_id", "possession_id"]).sum()),
    }
    return {
        "overview": overview,
        "summary": summary.assign(slice=sl.name),
        "restarts": restarts,
        "own_goals": own_goals,
        "outs": outs,
        "starts": start_frame,
        "shots": shots,
        "drops": drops,
        "decreases": decreases,
    }


# -------------------------------------------------------------------------------- formatting


def pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


def show(df: pd.DataFrame, indent: int = 2) -> None:
    pad = " " * indent
    print("\n".join(pad + line for line in df.to_string().splitlines()))


def split_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per key: how many start a new possession_id and how many do not, split by team."""
    rows = rows.assign(new=np.where(rows["starts_possession"], "new", "cont"))
    counts = rows.groupby(["key", "new", "team"]).size().unstack(["new", "team"], fill_value=0)
    columns = [(a, b) for a in ("new", "cont") for b in ("same", "diff", "unknown")]
    counts = counts.reindex(columns=pd.MultiIndex.from_tuples(columns), fill_value=0)
    counts.columns = [f"{a}_{b}" for a, b in columns]
    counts.insert(
        0, "cont", counts[[c for c in counts.columns if c.startswith("cont_")]].sum(axis=1)
    )
    counts.insert(0, "new", counts[[c for c in counts.columns if c.startswith("new_")]].sum(axis=1))
    counts.insert(0, "n", counts["new"] + counts["cont"])
    counts.loc["ALL"] = counts.sum()
    counts.index.name = None
    return counts


def exceptions(rows: pd.DataFrame) -> None:
    """The events that continue the running possession, with what precedes them inside it."""
    cont = rows[~rows["starts_possession"]]
    if cont.empty:
        print("  Events that do NOT start a new possession: none.")
        return
    print(f"  Events that do NOT start a new possession: {len(cont)}")
    print("    by previous event type:")
    show(cont["previous_event_type"].value_counts().rename("n").to_frame(), 6)
    print("    by number of earlier events in their own possession (0 = it opens that id):")
    show(
        cont["n_prior_in_possession"]
        .clip(upper=5)
        .value_counts()
        .sort_index()
        .rename("n")
        .to_frame(),
        6,
    )
    print(f"    first {min(LISTING_CAP, len(cont))} (types read most recent first):")
    columns = ["slice", "key", "match_id", "event_index", "period", "minute", "team",
               "previous_event_type", "n_prior_in_possession", "prior_types"]  # fmt: skip
    show(cont[columns].head(LISTING_CAP).reset_index(drop=True), 6)


def report_overview(parts: list[dict]) -> None:
    total = {key: sum(p["overview"][key] for p in parts) for key in parts[0]["overview"]}
    for key, value in total.items():
        print(f"  {key}: {value:,}")
    lower = pd.concat([p["decreases"] for p in parts], ignore_index=True)
    if lower.empty:
        return
    print(f"  events whose possession_id is lower than the previous event's: {len(lower)}")
    pairs = lower.groupby(["event", "previous_event"]).size().rename("n").to_frame()
    show(pairs, 4)
    print(f"    id already started earlier in the match: {int(lower['id_started_earlier'].sum())}")
    print(f"    first {min(LISTING_CAP, len(lower))}:")
    show(lower.head(LISTING_CAP), 6)


def report_summary(parts: list[dict]) -> None:
    keys = ["first_event_type", "first_pass_type", "same_team_as_previous"]
    summary = pd.concat([p["summary"] for p in parts], ignore_index=True)
    summary = summary.assign(
        **{k: summary[k].astype(object).where(summary[k].notna(), "(none)") for k in keys}
    )
    grouped = summary.groupby(keys)["n"].sum().sort_values(ascending=False)
    print(f"  possession_start_summary: {len(grouped)} groups, {int(grouped.sum()):,} possessions;")
    print(f"  top {min(15, len(grouped))} by count (same_team_as_previous: True = same team):")
    show(grouped.head(15).rename("n").to_frame(), 4)


def report_a(parts: list[dict]) -> None:
    rows = pd.concat([p["restarts"] for p in parts], ignore_index=True)
    print("(a) Restart passes: does each start a new possession_id?")
    print("    new/cont = starts a new id / keeps the id of the event before it;")
    print("    same/diff/unknown = possession_team_id equals / differs from the event before it")
    print("    (unknown = first event of its match).")
    show(split_table(rows))
    exceptions(rows)
    drops = pd.concat([p["drops"] for p in parts], ignore_index=True)
    print("  extra: Referee Ball-Drop events (a restart in spec 8.1, though not a pass):")
    if drops.empty:
        print("    none")
    else:
        show(split_table(drops), 4)
        exceptions(drops)


def report_b(parts: list[dict]) -> None:
    goals = pd.concat([p["own_goals"] for p in parts if len(p["own_goals"])], ignore_index=True)
    print(
        "(b) Own goals (the two events of a pair are looked at separately; the sibling is skipped"
    )
    print("    when finding the event before and after):")
    if goals.empty:
        print("  none")
        return
    for kind in OWN_GOALS:
        g = goals[goals["own_goal_type"] == kind]
        stays = int(g["same_possession_as_event_before"].sum())
        print(f"  {kind}: {len(g)}")
        print(
            f"    same possession_id as the event before the pair: {stays}; new id: {len(g) - stays}"
        )
        print("    next event after the pair:")
        show(g["next_event"].value_counts().rename("n").to_frame(), 6)
        print(
            f"    next event has a different possession_id: {int(g['next_event_new_possession'].sum())}"
        )
        passes = g[g["events_skipped_before_pass"] >= 0]
        kick_off = passes[passes["next_pass"] == "Pass / Kick Off"]
        print(
            f"    next Pass after the pair: {len(passes)} of {len(g)}; a Kick Off: {len(kick_off)};"
            f" that Kick Off has a new possession_id: {int(kick_off['next_pass_new_possession'].sum())}"
        )
        print("    events between the pair and that next Pass (0 = it follows directly):")
        skipped = passes["events_skipped_before_pass"].clip(upper=3).value_counts().sort_index()
        show(skipped.rename("n").to_frame(), 6)
    odd = goals[
        goals["same_possession_as_event_before"].eq(False)
        | goals["next_pass"].ne("Pass / Kick Off")
        | goals["events_skipped_before_pass"].gt(0)
    ]
    if len(odd):
        print(
            f"  Own goals that do not look like 'same id, next Pass is a Kick Off' (first {LISTING_CAP}):"
        )
        columns = ["slice", "own_goal_type", "match_id", "event_index", "period",
                   "same_possession_as_event_before", "next_event", "next_pass",
                   "events_skipped_before_pass", "skipped_types"]  # fmt: skip
        show(odd[columns].head(LISTING_CAP).reset_index(drop=True), 4)


def report_c(parts: list[dict]) -> None:
    starts = pd.concat([p["starts"] for p in parts], ignore_index=True)
    outs = pd.concat([p["outs"] for p in parts], ignore_index=True)
    n_start = len(starts)
    with_previous = int(starts["has_previous_event"].sum())
    after_out = starts[starts["previous_out"]]
    with_previous_possession = int(starts["has_previous_possession"].sum())
    held_out = int(starts["previous_possession_held_out"].sum())
    print("(c) After out == True")
    print(f"  new possessions: {n_start:,} ({n_start - with_previous:,} open a match)")
    print(
        f"  reading 1, the event just before the new possession has out == True: {len(after_out):,}"
        f" = {pct(len(after_out), n_start)} of all new possessions,"
        f" {pct(len(after_out), with_previous)} of those with a previous event"
    )
    print(
        f"  reading 2, the previous possession holds an out == True event anywhere: {held_out:,}"
        f" = {pct(held_out, with_previous_possession)} of new possessions with a previous possession"
    )
    print("  first event of the new possessions that follow an out flag (reading 1):")
    table = after_out["first_event"].value_counts().rename("n").to_frame()
    table["pct"] = (table["n"] / len(after_out) * 100).round(1) if len(after_out) else 0.0
    show(table, 4)
    n_out = len(outs)
    direct = int(outs["next_event_starts_possession"].sum())
    print(
        f"  events with out == True: {n_out:,}; the next event starts a new possession: {direct:,}"
        f" = {pct(direct, n_out)}; it does not: {n_out - direct:,}"
    )
    rest = outs[~outs["next_event_starts_possession"]]
    if len(rest):
        print("  next event when it does NOT start a new possession (top 12):")
        show(rest["next_event"].value_counts().head(12).rename("n").to_frame(), 4)
    restarts = pd.concat([p["restarts"] for p in parts], ignore_index=True)
    dead = restarts[restarts["key"].isin(["Throw-in", "Goal Kick", "Corner"])]
    print(
        "  extra: share of throw-ins, goal kicks and corners whose previous event has out == True:"
    )
    cover = (
        dead.groupby("key")["previous_out"]
        .agg(["size", "sum"])
        .rename(columns={"size": "n", "sum": "after_out"})
    )
    cover.loc["ALL"] = cover.sum()
    cover["pct"] = (cover["after_out"] / cover["n"] * 100).round(1)
    show(cover, 4)
    print("  first event of the next new possession after each out == True event, same period:")
    table = outs["next_start"].value_counts().rename("n").to_frame()
    table["pct"] = (table["n"] / n_out * 100).round(1) if n_out else 0.0
    show(table.head(15), 4)


def report_d(parts: list[dict]) -> None:
    shots = pd.concat([p["shots"] for p in parts], ignore_index=True)
    print("(d) Non-open-play shots: do they start a new possession_id?")
    print(
        "    (Free Kick and Penalty are the ones the restart definition uses; other types are extras.)"
    )
    table = split_table(shots)
    headline = [k for k in HEADLINE_SHOTS if k in table.index]
    extras = [k for k in table.index if k not in HEADLINE_SHOTS and k != "ALL"]
    show(table.reindex(headline + extras + ["ALL"]).dropna(), 4)
    for kind in HEADLINE_SHOTS:
        g = shots[shots["key"] == kind]
        shootout = g[g["period"] == 5]
        print(
            f"  {kind}: {len(g)} shots; {int(g['starts_possession'].sum())} start a new possession;"
            f" in period 5 (shootout): {len(shootout)},"
            f" of which {int(shootout['starts_possession'].sum())} start one"
        )
    exceptions(shots[shots["key"].isin(HEADLINE_SHOTS)])


def report(title: str, parts: list[dict]) -> None:
    bar = "=" * 100
    print(f"\n{bar}\n{title}\n{bar}")
    print("Overview")
    report_overview(parts)
    print()
    report_summary(parts)
    for section in (report_a, report_b, report_c, report_d):
        print()
        section(parts)


def main(argv: list[str]) -> int:
    cfg = load_config(Path(argv[1]) if len(argv) > 1 else DEFAULT_CONFIG)
    core = [s for s in cfg.slices if s.core]
    print(f"possession check on events pinned at {cfg.sha}")
    print(f"core slices: {', '.join(s.name for s in core)}")
    collected = {}
    for sl in core:
        collected[(sl.competition_id, sl.season_id)] = collect(cfg, sl)
        print(f"read {sl.name}", file=sys.stderr)
    for sl in core:
        report(
            f"SLICE: {sl.name} ({sl.competition_id}/{sl.season_id})",
            [collected[(sl.competition_id, sl.season_id)]],
        )
    pre = [v for k, v in collected.items() if k in PRE_CHECKED]
    if pre and len(pre) < len(collected):
        report("PRE-CHECKED SLICES TOGETHER (all core slices except Premier League 2015/16)", pre)
    report("ALL CORE SLICES TOGETHER", list(collected.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
