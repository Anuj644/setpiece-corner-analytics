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

from helpers import events_df

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
