"""Synthetic corner data for forecast-lane tests (Tasks 16–20).

``synthetic_corners`` produces a minimal ``rq1_corners``-contract DataFrame
with columns:
    match_id, match_date, team_id, corner_side, ends_in_box,
    competition_id, season_id

Generation rules (all seeded via ``numpy.random.default_rng(seed)``):
* One match per calendar date starting 2015-08-01 (or two matches per date
  when ``same_date_pairs=True``).
* Each match has between 4 and 8 corner rows drawn from two of the supplied
  teams (alternating: home team vs. away team, cycling through ``teams``).
* ``corner_side`` alternates "left" / "right" per row.
* ``ends_in_box ~ Bernoulli(p)`` where ``p`` defaults to 0.65.
  ``p_by_team_side`` overrides the default for specific ``(team_id, side)``
  combinations.
* ``competition_id = 1``, ``season_id = 1`` for every row (single slice).
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


def synthetic_corners(
    n_matches: int,
    teams: list[int],
    seed: int,
    p_by_team_side: dict[tuple[int, str], float] | None = None,
    same_date_pairs: bool = False,
) -> pd.DataFrame:
    """Return a synthetic rq1_corners DataFrame.

    Parameters
    ----------
    n_matches:
        Total number of matches to generate.
    teams:
        Pool of team IDs to draw from.  Must have at least 2 entries.
    seed:
        Seed for ``numpy.random.default_rng``.
    p_by_team_side:
        Override ``ends_in_box`` probability per ``(team_id, corner_side)``
        pair.  Missing pairs use the default of 0.65.
    same_date_pairs:
        When True, generate two matches per calendar date so some dates appear
        more than once (used to test that the same-date boundary rule works).
    """
    if len(teams) < 2:
        raise ValueError("teams must have at least 2 entries")

    rng = np.random.default_rng(seed)
    p_by_team_side = p_by_team_side or {}
    default_p = 0.65
    sides = ["left", "right"]

    rows: list[dict] = []
    base_date = datetime.date(2015, 8, 1)

    match_id = 1
    date_offset = 0  # days from base_date

    for i in range(n_matches):
        if same_date_pairs:
            # Two matches share the same date; advance date every other match.
            if i % 2 == 0:
                current_date = base_date + datetime.timedelta(days=date_offset)
            else:
                # Second match of the pair — same date, then advance.
                date_offset += 1
        else:
            current_date = base_date + datetime.timedelta(days=i)

        # Pick a home/away pair from the teams pool (cycling).
        home_team = teams[i % len(teams)]
        away_team = teams[(i + 1) % len(teams)]

        n_corners = int(rng.integers(4, 9))  # 4..8 inclusive

        for j in range(n_corners):
            # Alternate which team takes the corner.
            team = home_team if j % 2 == 0 else away_team
            side = sides[j % 2]
            p = p_by_team_side.get((team, side), default_p)
            ends = bool(rng.random() < p)
            rows.append(
                {
                    "match_id": match_id,
                    "match_date": str(current_date),
                    "team_id": team,
                    "corner_side": side,
                    "ends_in_box": ends,
                    "competition_id": 1,
                    "season_id": 1,
                }
            )

        match_id += 1

        if not same_date_pairs:
            pass  # date already set per-match above
        elif i % 2 != 0:
            pass  # date_offset already advanced inside the pair branch

    df = pd.DataFrame(rows)
    df["ends_in_box"] = df["ends_in_box"].astype(pd.BooleanDtype())
    return df
