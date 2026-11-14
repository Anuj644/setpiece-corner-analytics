"""Timeline builders for the RQ2 chain tests; import with ``from rq2_helpers import tl, corner, ev_``.

Not named ``helpers.py``: ``tests/helpers.py`` would shadow it. Every spec is a short dict:

* ``t`` seconds into the period, written as ``HH:MM:SS.fff`` (``None`` leaves the timestamp null);
* ``type``, ``team``, ``poss``, ``poss_team``, ``id`` and ``details`` stand for ``event_type``,
  ``team_id``, ``possession_id``, ``possession_team_id``, ``event_id`` and ``event_details``;
* any real ``EVENTS_COLUMNS`` name (``out``, ``pass_type``, ``shot_outcome``, ``period``, ...) is
  passed through, and an unknown key raises.

An explicit ``None`` (``poss=None``) stays null; only an absent key takes the default.
"""

import json

import pandas as pd
from helpers import events_df

from setpiece.config import Slice
from setpiece.coverage import TEAM_SAMPLE_COLUMNS, TEAM_SAMPLE_DTYPES
from setpiece.io import read_table, write_table
from setpiece.rq1.corners import RQ1_CORNERS_COLUMNS, RQ1_CORNERS_DTYPES
from setpiece.rq2.chains import RQ2_COLUMNS, RQ2_DTYPES

ATTACKING = 10  # the corner taker's team
DEFENDING = 20
POSSESSION = 5  # the corner's possession_id unless a spec says otherwise

_ALIASES = {
    "type": "event_type",
    "team": "team_id",
    "poss": "possession_id",
    "poss_team": "possession_team_id",
    "id": "event_id",
}


def ev_(t, type, team=ATTACKING, poss=POSSESSION, poss_team=ATTACKING, **fields):
    """A spec for one event; by default the attacking team's, inside the corner's possession."""
    return {"t": t, "type": type, "team": team, "poss": poss, "poss_team": poss_team, **fields}


def corner(t, **fields):
    """A spec for a corner pass; ``fields`` are those of ``ev_`` (``poss``, ``id``, ...)."""
    return ev_(t, "Pass", pass_type="Corner", **fields)


def _timestamp(seconds):
    if seconds is None:
        return None
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def tl(*specs):
    """The ``events`` frame of one match: ``match_id`` 1, ``period`` 1, sequential ``event_index``."""
    rows = []
    for i, spec in enumerate(specs):
        row = {"match_id": 1, "event_index": i, "period": 1, "event_id": f"e{i}"}
        for key, value in spec.items():
            if key == "t":
                row["timestamp"] = _timestamp(value)
            elif key == "details":  # events_df would store the dict's Python repr
                row["event_details"] = json.dumps(value) if isinstance(value, dict) else value
            else:
                row[_ALIASES.get(key, key)] = value
        rows.append(row)
    return events_df(rows)


# --- tables the RQ2 summaries read ----------------------------------------------------------

SLICES = [
    Slice(43, 106, "FIFA World Cup 2022", "tournament", True),
    Slice(9, 281, "1. Bundesliga 2023/24", "league", True, 34),
]


def outcome(corner_event_id, match_id, team_id, *, end_reason="restart", restart_type=None,
            restart_team=None, shot_count=0, goal=False, lost_possession=False,
            outcome_status="complete", competition_id=9, season_id=281):  # fmt: skip
    """One ``rq2_outcomes`` row as a dict; every field a test does not set is a plain value."""
    return {
        "match_id": match_id, "corner_event_id": corner_event_id, "last_included_event_id": None,
        "included_event_ids": [], "n_events": 0, "sequence_duration_seconds": 1.0,
        "end_reason": end_reason, "restart_type": restart_type, "restart_team": restart_team,
        "shot_count": shot_count, "first_shot_event_id": None, "goal": goal,
        "goal_event_id": None, "lost_possession": lost_possession, "penalty_awarded": False,
        "opponent_action_count": 0, "outcome_status": outcome_status,
        "competition_id": competition_id, "season_id": season_id, "team_id": team_id,
    }  # fmt: skip


def outcomes_df(rows):
    """The ``rq2_outcomes`` frame (``RQ2_COLUMNS``, ``RQ2_DTYPES``) of the ``outcome`` dicts."""
    return pd.DataFrame(rows, columns=RQ2_COLUMNS).astype(RQ2_DTYPES)


def rq1_corner(event_id, technique, coarse_zone, short_corner_proxy, **fields):
    """One ``rq1_corners`` row as a dict: the delivery attributes and ids, the rest null."""
    row = {name: None for name in RQ1_CORNERS_COLUMNS}
    return {**row, "event_id": event_id, "technique": technique, "coarse_zone": coarse_zone,
            "short_corner_proxy": short_corner_proxy, **fields}  # fmt: skip


def rq1_df(rows):
    """The ``rq1_corners`` frame (``RQ1_CORNERS_COLUMNS``, ``RQ1_CORNERS_DTYPES``) of the dicts."""
    return pd.DataFrame(rows, columns=RQ1_CORNERS_COLUMNS).astype(RQ1_CORNERS_DTYPES)


# (technique, coarse_zone, short_corner_proxy) of the n-th corner: every shape the real table has
_DELIVERY_PATTERN = [
    ("Inswinging", "near", False),
    ("Unknown", None, True),  # a ground pass without an end location
    ("Outswinging", "central", None),  # the proxy cannot be computed
    ("Unknown", "far", False),
]


def write_rq1_inputs(cfg):
    """Write ``derived/team_samples`` and ``derived/rq1_corners`` for the events of ``cfg``'s slices.

    One ``team_samples`` row per team with an event (its matches counted from the events) and one
    ``rq1_corners`` row per corner pass, its delivery attributes cycling through the shapes of
    ``_DELIVERY_PATTERN``; an empty, typed ``rq1_corners`` when there is no corner.
    """
    events = pd.concat(
        [read_table("events", cfg, [(s.competition_id, s.season_id)]) for s in cfg.slices],
        ignore_index=True,
    )
    teams = (
        events.dropna(subset=["team_id"])
        .groupby(["competition_id", "season_id", "team_id"])["match_id"]
        .nunique()
        .rename("matches_in_sample")
        .reset_index()
    )
    teams["team_name"] = "Team " + teams["team_id"].astype(str)
    teams["sample_label"] = "test_sample"
    samples = teams[TEAM_SAMPLE_COLUMNS].astype(TEAM_SAMPLE_DTYPES)
    write_table(samples, "derived/team_samples", cfg, None)

    corners = events[(events["event_type"] == "Pass") & (events["pass_type"] == "Corner")]
    rows = []
    for i, (_, c) in enumerate(corners.iterrows()):
        technique, zone, proxy = _DELIVERY_PATTERN[i % len(_DELIVERY_PATTERN)]
        rows.append(rq1_corner(
            c["event_id"], technique, zone, proxy, match_id=c["match_id"],
            competition_id=c["competition_id"], season_id=c["season_id"], team_id=c["team_id"],
            team_name=f"Team {c['team_id']}",
            match_date="2022-11-21" if c["competition_id"] == 43 else "2023-08-18",
        ))  # fmt: skip
    write_table(rq1_df(rows), "derived/rq1_corners", cfg, None)
