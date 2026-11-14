"""Tests for the RQ2 review sample, its export and the agreement metrics (Task 24)."""

import dataclasses
import re
import tomllib
import warnings
from html.parser import HTMLParser
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rq2_helpers import corner, ev_, tl

from setpiece.config import Param, Slice
from setpiece.ingest.events import EVENTS_DTYPES
from setpiece.io import write_table
from setpiece.rq2.chains import build_rq2_outcomes, restarts_from
from setpiece.rq2.review import (
    STRATA,
    agreement,
    assign_stratum,
    export_review,
    rule_labels,
    rule_vs_consensus,
    run_review_export_rq2,
    sample_chains,
)

REPO = Path(__file__).resolve().parents[2]

TEAM_NAMES = {10: "Home FC", 20: "Away FC", 30: "Rovers", 40: "United"}
SHEET_COLUMNS = [
    "review_id", "match_id", "corner_event_id", "last_event_id", "shot_count", "goal",
    "lost_possession",
]  # fmt: skip
LABEL_COLUMNS = ["last_event_id", "shot_count", "goal", "lost_possession"]

# Tokens that would tell a labeller what the rule or the sampling decided. Lowercase and exact: the
# capitalised event type "Shot" is raw data and must stay visible, and the sheet's header names
# shot_count and lost_possession by design, so those two are only searched for in the data rows.
LEAKS = [
    "end_reason", "time_cap", "possession_change", "lost_possession", "outcome_status", "stratum",
    "clearance_first", "repeat_other", "shot", "short", "last_included_event_id",
]  # fmt: skip


# --- the fixture: one corner of every kind of chain, across two slices -----------------------


def _block(kind, tag, t0, poss, att, de):
    """The specs of one corner (id ``tag``) and what follows it, ending in a restart or turnover."""
    mine = {"team": att, "poss": poss, "poss_team": att}
    theirs_same = {"team": de, "poss": poss, "poss_team": att}  # inside the corner's possession
    theirs_new = {"team": de, "poss": poss + 1, "poss_team": de}
    first = corner(t0, id=tag, **mine)
    if kind == "shot":
        return [
            first,
            ev_(t0 + 3, "Shot", shot_outcome="Saved", id=f"{tag}-1", **mine),
            ev_(t0 + 9, "Pass", pass_type="Goal Kick", id=f"{tag}-2", **theirs_new),
        ]
    if kind == "shot_after_clearance":  # a defending clearance first, but an attacking shot too
        return [
            first,
            ev_(t0 + 2, "Clearance", id=f"{tag}-1", **theirs_same),
            ev_(t0 + 4, "Shot", shot_outcome="Blocked", id=f"{tag}-2", **mine),
            ev_(t0 + 10, "Pass", pass_type="Goal Kick", id=f"{tag}-3", **theirs_new),
        ]
    if kind == "goal":
        return [
            first,
            ev_(t0 + 3, "Shot", shot_outcome="Goal", id=f"{tag}-1", **mine),
            ev_(t0 + 9, "Pass", pass_type="Kick Off", id=f"{tag}-2", **theirs_new),
        ]
    if kind == "cleared":
        return [
            first,
            ev_(t0 + 2, "Clearance", out=True, id=f"{tag}-1", **theirs_same),
            ev_(t0 + 8, "Pass", pass_type="Goal Kick", id=f"{tag}-2", **theirs_new),
        ]
    if kind == "own_clearance":  # a clearance by the corner taker's own team is not a defence
        return [
            first,
            ev_(t0 + 2, "Clearance", id=f"{tag}-1", **mine),
            ev_(t0 + 4, "Pass", id=f"{tag}-2", **theirs_new),
        ]
    if kind == "turnover":
        return [
            first,
            ev_(t0 + 2, "Pass", id=f"{tag}-1", **mine),
            ev_(t0 + 4, "Pass", id=f"{tag}-2", **theirs_new),
        ]
    if kind == "last":  # the last event of the match: nothing follows the corner
        return [first]
    raise ValueError(kind)


def _match_events(match_id, competition_id, season_id, att, de, blocks):
    """One match (``events`` frame) of ``(kind, tag)`` blocks, 100 seconds apart."""
    specs = []
    for i, (kind, tag) in enumerate(blocks):
        specs += _block(kind, tag, 100 * i, 10 + 2 * i, att, de)
    frame = tl(*specs)

    def column(name, value):
        return pd.Series([value] * len(frame), index=frame.index, dtype=EVENTS_DTYPES[name])

    teams = {"home": att, "away": de}
    return frame.assign(
        match_id=column("match_id", match_id),
        competition_id=column("competition_id", competition_id),
        season_id=column("season_id", season_id),
        match_date=column("match_date", "2023-10-01"),
        home_team_id=column("home_team_id", teams["home"]),
        home_team_name=column("home_team_name", TEAM_NAMES[teams["home"]]),
        away_team_id=column("away_team_id", teams["away"]),
        away_team_name=column("away_team_name", TEAM_NAMES[teams["away"]]),
        team_name=frame["team_id"].map(TEAM_NAMES).astype("string"),
        player_name=("Player " + frame["team_id"].astype("string")).astype("string"),
    )


MATCH_1 = [
    ("shot", "s1"), ("shot", "s2"), ("shot_after_clearance", "s3"), ("cleared", "c1"),
    ("cleared", "c2"), ("own_clearance", "r1"), ("turnover", "r2"), ("shot", "k1"),
    ("turnover", "k2"), ("shot", "n1"), ("last", "z1"),
]  # fmt: skip
MATCH_2 = [("cleared", "c3"), ("turnover", "r3"), ("shot", "s4"), ("shot", "k3"), ("goal", "g1")]
MATCH_3 = [("shot", "x1"), ("cleared", "x2")]  # in a slice that is not a core one

SHORT = {"k1": True, "k2": True, "k3": True, "n1": pd.NA}  # every other corner: False
EXPECTED = {
    "s1": "shot", "s2": "shot", "s3": "shot", "s4": "shot", "g1": "shot",
    "c1": "clearance_first", "c2": "clearance_first", "c3": "clearance_first",
    "r1": "repeat_other", "r2": "repeat_other", "r3": "repeat_other", "z1": "repeat_other",
    "k1": "short", "k2": "short", "k3": "short",
    "n1": "shot",  # its short proxy is null: not short, so the shot decides
}  # fmt: skip


def _rq1(outcomes):
    """A small ``rq1_corners`` frame for ``outcomes``: the key, the proxy and the slice columns."""
    proxy = [SHORT.get(tag, False) for tag in outcomes["corner_event_id"]]
    return pd.DataFrame(
        {
            "match_id": outcomes["match_id"],
            "event_id": outcomes["corner_event_id"],
            "competition_id": outcomes["competition_id"],
            "season_id": outcomes["season_id"],
            "team_id": outcomes["team_id"],
            "short_corner_proxy": pd.array(proxy, dtype="boolean"),
        }
    )


@pytest.fixture
def review_fixture(cfg):
    """``(outcomes, rq1, events)``: 16 corners in the 9/281 and 43/106 slices, every stratum."""
    restarts, cap = restarts_from(cfg), cfg.p("cap_seconds")
    events = pd.concat(
        [
            _match_events(1, 43, 106, 10, 20, MATCH_1),
            _match_events(2, 9, 281, 30, 40, MATCH_2),
        ],
        ignore_index=True,
    )
    outcomes = build_rq2_outcomes(events, cap, restarts)
    return outcomes, _rq1(outcomes), events


def _strata(sample):
    return dict(zip(sample["corner_event_id"], sample["stratum"]))


def _shuffled(frame, seed):
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


# --- assign_stratum --------------------------------------------------------------------------


def test_fixture_chains_are_what_the_strata_expect(review_fixture):
    """The fixture is only a good test if its chains do what its names promise."""
    outcomes = review_fixture[0].set_index("corner_event_id")
    assert outcomes.loc["s3", "shot_count"] == 1  # a clearance, then an attacking shot
    assert outcomes.loc["c1", "shot_count"] == 0
    assert outcomes.loc["z1", "n_events"] == 0
    assert outcomes.loc["z1", "outcome_status"] == "incomplete"
    assert (outcomes.drop(index="z1")["outcome_status"] == "complete").all()


def test_assign_stratum_priority(review_fixture):
    outcomes, rq1, events = review_fixture
    stratum = assign_stratum(outcomes, rq1, events)
    assert stratum.index.equals(outcomes.index) and len(stratum) == len(outcomes)
    assert dict(zip(outcomes["corner_event_id"], stratum)) == EXPECTED
    assert set(EXPECTED.values()) == set(STRATA)


def test_assign_stratum_does_not_depend_on_row_order(review_fixture):
    outcomes, rq1, events = review_fixture
    shuffled = _shuffled(outcomes, 3)
    stratum = assign_stratum(shuffled, _shuffled(rq1, 4), _shuffled(events, 5))
    assert dict(zip(shuffled["corner_event_id"], stratum)) == EXPECTED


def test_a_null_short_proxy_is_not_short(review_fixture):
    outcomes, rq1, events = review_fixture
    assert rq1.set_index("event_id")["short_corner_proxy"].isna().sum() == 1
    stratum = assign_stratum(outcomes, rq1, events)
    assert stratum[outcomes["corner_event_id"] == "n1"].item() == "shot"


def test_clearance_first_looks_at_the_next_event_of_the_match_not_the_chain(review_fixture):
    outcomes, rq1, events = review_fixture
    # c1's next event is the defending clearance whether or not the chain keeps it
    stripped = outcomes.assign(
        included_event_ids=[np.array([], dtype=object)] * len(outcomes),
        last_included_event_id=outcomes["corner_event_id"],
    )
    assert assign_stratum(stripped, rq1, events).equals(assign_stratum(outcomes, rq1, events))


def test_a_null_team_on_the_next_event_is_not_a_defending_clearance(review_fixture):
    outcomes, rq1, events = review_fixture
    events = events.copy()
    events.loc[events["event_id"] == "c1-1", "team_id"] = pd.NA
    stratum = assign_stratum(outcomes, rq1, events)
    assert stratum[outcomes["corner_event_id"] == "c1"].item() == "repeat_other"


def test_assign_stratum_raises_for_a_corner_without_an_rq1_row(review_fixture):
    outcomes, rq1, events = review_fixture
    with pytest.raises(ValueError, match="rq1"):
        assign_stratum(outcomes, rq1[rq1["event_id"] != "s1"], events)


def test_assign_stratum_raises_for_a_corner_missing_from_events(review_fixture):
    outcomes, rq1, events = review_fixture
    with pytest.raises(ValueError, match="events"):
        assign_stratum(outcomes, rq1, events[events["event_id"] != "s1"])


def test_assign_stratum_raises_for_a_duplicated_rq1_row(review_fixture):
    outcomes, rq1, events = review_fixture
    with pytest.raises(ValueError):
        assign_stratum(outcomes, pd.concat([rq1, rq1.iloc[:1]]), events)


# --- sample_chains ---------------------------------------------------------------------------


def test_strata_sizes_and_priority(review_fixture):
    s = sample_chains(*review_fixture, per_stratum=2, seed=0)
    assert s.groupby("stratum").size().le(2).all()
    assert set(s.stratum) <= {"short", "shot", "clearance_first", "repeat_other"}


def test_each_stratum_gives_min_of_per_stratum_and_available(review_fixture):
    # available: short 3, shot 6, clearance_first 3, repeat_other 4
    for per_stratum, expected in [
        (2, {"short": 2, "shot": 2, "clearance_first": 2, "repeat_other": 2}),
        (4, {"short": 3, "shot": 4, "clearance_first": 3, "repeat_other": 4}),
        (50, {"short": 3, "shot": 6, "clearance_first": 3, "repeat_other": 4}),
        (0, {}),
    ]:
        s = sample_chains(*review_fixture, per_stratum=per_stratum, seed=1)
        assert s.groupby("stratum").size().to_dict() == expected
        assert s["corner_event_id"].is_unique  # without replacement


def test_sample_rows_are_the_outcome_rows_with_stratum_and_review_id(review_fixture):
    outcomes = review_fixture[0]
    s = sample_chains(*review_fixture, per_stratum=2, seed=0)
    assert list(s.columns) == [*outcomes.columns, "stratum", "review_id"]
    assert s["review_id"].tolist() == list(range(1, len(s) + 1))
    assert s["stratum"].tolist() == [EXPECTED[c] for c in s["corner_event_id"]]
    original = outcomes.set_index("corner_event_id")
    for row in s.itertuples():
        assert original.loc[row.corner_event_id, "last_included_event_id"] == (
            row.last_included_event_id
        )


def test_review_ids_follow_a_shuffle_not_the_strata(review_fixture):
    s = sample_chains(*review_fixture, per_stratum=2, seed=0)
    drawn_in_order = sorted(s["stratum"], key=STRATA.index)  # what an unshuffled draw looks like
    assert s["stratum"].tolist() != drawn_in_order
    assert s["stratum"].tolist() != sorted(s["stratum"])


def test_same_inputs_and_seed_give_the_same_sample(review_fixture):
    a = sample_chains(*review_fixture, per_stratum=2, seed=7)
    b = sample_chains(*review_fixture, per_stratum=2, seed=7)
    assert a["corner_event_id"].tolist() == b["corner_event_id"].tolist()
    assert a["review_id"].tolist() == b["review_id"].tolist()
    assert a["stratum"].tolist() == b["stratum"].tolist()


def test_sample_depends_on_the_seed(review_fixture):
    drawn = {
        tuple(sample_chains(*review_fixture, per_stratum=2, seed=seed)["corner_event_id"])
        for seed in range(6)
    }
    assert len(drawn) > 1


def test_sample_does_not_depend_on_the_row_order_of_the_inputs(review_fixture):
    outcomes, rq1, events = review_fixture
    a = sample_chains(outcomes, rq1, events, per_stratum=2, seed=11)
    b = sample_chains(
        _shuffled(outcomes, 1), _shuffled(rq1, 2), _shuffled(events, 3), per_stratum=2, seed=11
    )
    scalar = [c for c in a.columns if c != "included_event_ids"]  # array cells defeat .equals
    assert a[scalar].reset_index(drop=True).equals(b[scalar].reset_index(drop=True))


def test_sample_chains_does_not_modify_its_inputs(review_fixture):
    outcomes, rq1, events = review_fixture
    before = outcomes.columns.tolist(), len(outcomes), rq1.columns.tolist(), events.columns.tolist()
    sample_chains(outcomes, rq1, events, per_stratum=2, seed=0)
    assert before == (outcomes.columns.tolist(), len(outcomes), rq1.columns.tolist(),
                      events.columns.tolist())  # fmt: skip


def test_negative_per_stratum_raises(review_fixture):
    with pytest.raises(ValueError):
        sample_chains(*review_fixture, per_stratum=-1, seed=0)


# --- export_review ---------------------------------------------------------------------------


class _Table(HTMLParser):
    """The rows of the one ``<table>`` of a chain page: (is_corner, [cell texts]); plus headers."""

    def __init__(self):
        super().__init__()
        self.headers, self.rows = [], []
        self._cell = None
        self._row = None
        self._in_head = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self._row = [("corner" in (attrs.get("class") or "").split()), []]
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_head = False
        elif tag in ("td", "th"):
            text = "".join(self._cell).strip()
            self._cell = None
            if self._in_head:
                self.headers.append(text)
            else:
                self._row[1].append(text)
        elif tag == "tr" and not self._in_head:
            self.rows.append(tuple(self._row))
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _parse(html):
    table = _Table()
    table.feed(html)
    return table


def _page(out_dir, review_id):
    return (out_dir / "chains" / f"{review_id}.html").read_text(encoding="utf-8")


def _long_match(n_after=60):
    """One match: five events, the corner ``k``, then ``n_after`` events; period 2 after 30."""
    specs = [ev_(i, "Pass", id=f"pre{i}") for i in range(5)]
    specs.append(corner(10, id="k", player_name="Sam O'Neil <b>"))
    for i in range(n_after):
        period = 1 if i < 30 else 2
        specs.append(ev_(i, "Pass", id=f"post{i}", period=period))
    frame = tl(*specs)
    return frame.assign(
        home_team_name=pd.Series(["Home FC"] * len(frame), dtype="string"),
        away_team_name=pd.Series(["Away FC"] * len(frame), dtype="string"),
        match_date=pd.Series(["2023-10-01"] * len(frame), dtype="string"),
        team_name=pd.Series(["Home FC"] * len(frame), dtype="string"),
    )


def _decoy_sample(match_id=1, corner_id="k", review_id=7):
    """A sample row whose rule values are all non-zero, to catch any that leak into the output."""
    return pd.DataFrame(
        {
            "review_id": [review_id],
            "match_id": pd.array([match_id], dtype="Int64"),
            "corner_event_id": [corner_id],
            "last_included_event_id": ["post3"],
            "shot_count": pd.array([3], dtype="Int64"),
            "goal": pd.array([True], dtype="boolean"),
            "lost_possession": pd.array([True], dtype="boolean"),
            "end_reason": ["possession_change"],
            "outcome_status": ["complete"],
            "stratum": ["short"],
        }
    )


def test_html_has_no_rule_output(tmp_path, review_fixture):
    s = sample_chains(*review_fixture, per_stratum=1, seed=0)
    export_review(s, review_fixture[2], tmp_path)
    html = next((tmp_path / "chains").glob("*.html")).read_text()
    for leak in ["end_reason", "time_cap", "possession_change", "lost_possession", "outcome_status"]:
        assert leak not in html


def test_export_writes_a_blind_sheet_and_one_page_per_chain(tmp_path, review_fixture):
    s = sample_chains(*review_fixture, per_stratum=2, seed=0)
    export_review(s, review_fixture[2], tmp_path)

    text = (tmp_path / "sheet.csv").read_text(encoding="utf-8")
    sheet = pd.read_csv(tmp_path / "sheet.csv", dtype={"corner_event_id": str})
    assert list(sheet.columns) == SHEET_COLUMNS
    assert sheet["review_id"].tolist() == s["review_id"].tolist()
    assert sheet["corner_event_id"].tolist() == s["corner_event_id"].tolist()
    assert sheet["match_id"].tolist() == s["match_id"].tolist()
    assert sheet[LABEL_COLUMNS].isna().all().all()  # the rule's values never reach the labels
    assert s["shot_count"].gt(0).any()  # so the check above could have caught a leak
    body = "\n".join(text.splitlines()[1:])
    for leak in LEAKS:
        assert leak not in body

    pages = sorted(p.name for p in (tmp_path / "chains").glob("*"))
    assert pages == sorted(f"{i}.html" for i in s["review_id"])
    for review_id in s["review_id"]:
        html = _page(tmp_path, review_id)
        for leak in LEAKS:
            assert leak not in html
        assert "<NA>" not in html and "&lt;NA&gt;" not in html


def test_the_page_shows_the_corner_and_the_next_40_events(tmp_path):
    events = _long_match()
    export_review(_decoy_sample(), events, tmp_path)
    page = _parse(_page(tmp_path, 7))
    assert page.headers == [
        "event_id", "event_index", "period", "timestamp", "team_name", "player_name", "type",
        "outcome", "possession_id", "out",
    ]  # fmt: skip
    ids = [cells[0] for _, cells in page.rows]
    assert ids == ["k"] + [f"post{i}" for i in range(40)]
    assert [is_corner for is_corner, _ in page.rows] == [True] + [False] * 40
    indexes = [int(cells[1]) for _, cells in page.rows]
    assert indexes == list(range(5, 46))  # the rows follow event_index; the periods change inside


def test_the_page_stops_at_the_end_of_the_match(tmp_path):
    events = _long_match(n_after=7)
    export_review(_decoy_sample(), events, tmp_path)
    ids = [cells[0] for _, cells in _parse(_page(tmp_path, 7)).rows]
    assert ids == ["k"] + [f"post{i}" for i in range(7)]


def test_the_page_follows_event_index_not_row_order_and_stays_in_its_match(tmp_path):
    events = _long_match(n_after=45)
    other = events.assign(match_id=pd.array([2] * len(events), dtype="Int64"))
    mixed = _shuffled(pd.concat([events, other.assign(event_id="other-" + other["event_id"])]), 0)
    export_review(_decoy_sample(), mixed, tmp_path)
    ids = [cells[0] for _, cells in _parse(_page(tmp_path, 7)).rows]
    assert ids == ["k"] + [f"post{i}" for i in range(40)]


def test_the_page_header_names_the_teams_date_attacker_and_corner_time(tmp_path):
    events = _long_match()
    events.loc[events["event_id"] == "k", ["period", "timestamp"]] = [2, "00:41:07.250"]
    export_review(_decoy_sample(), events, tmp_path)
    html = _page(tmp_path, 7)
    head = html.split("<table", 1)[0]
    for text in ["Home FC vs Away FC", "2023-10-01", "Home FC", "00:41:07.250", "Chain 7"]:
        assert text in head
    assert "Attacking team</dt><dd>Home FC</dd>" in head
    assert re.search(r"[Pp]eriod 2", head)


def test_the_type_and_outcome_columns_combine_detail_fields(tmp_path):
    events = tl(
        corner(0, id="k", pass_outcome="Incomplete"),
        ev_(2, "Pass", id="e1", pass_type="Throw-in"),
        ev_(3, "Shot", id="e2", shot_type="Open Play", shot_outcome="Saved"),
        ev_(4, "Duel", id="e3"),
    )
    export_review(_decoy_sample(corner_id="k"), events, tmp_path)
    rows = {cells[0]: cells for _, cells in _parse(_page(tmp_path, 7)).rows}
    type_col, outcome_col = 6, 7
    assert rows["k"][type_col] == "Pass (Corner)" and rows["k"][outcome_col] == "Incomplete"
    assert rows["e1"][type_col] == "Pass (Throw-in)" and rows["e1"][outcome_col] == ""
    assert rows["e2"][type_col] == "Shot (Open Play)" and rows["e2"][outcome_col] == "Saved"
    assert rows["e3"][type_col] == "Duel"


def test_null_cells_are_blank_and_text_is_escaped(tmp_path):
    events = _long_match(n_after=3)
    events.loc[events["event_id"] == "post1", ["player_name", "timestamp", "possession_id"]] = (
        pd.NA
    )
    export_review(_decoy_sample(), events, tmp_path)
    html = _page(tmp_path, 7)
    assert not re.search(r">\s*(<NA>|nan|NaN|None|NaT)\s*<", html)
    assert "&lt;NA&gt;" not in html and "NaN" not in html and "None" not in html
    rows = {cells[0]: cells for _, cells in _parse(html).rows}
    assert rows["post1"][3] == "" and rows["post1"][5] == "" and rows["post1"][8] == ""
    assert rows["k"][5] == "Sam O'Neil <b>"  # the parser unescapes, so it was written escaped
    assert "<b>" not in html


def test_accented_names_survive_as_utf8(tmp_path):
    events = _long_match(n_after=2)
    events.loc[events["event_id"] == "post0", "player_name"] = "Ruben Dias Álvaro Moráta"
    export_review(_decoy_sample(), events, tmp_path)
    assert "Álvaro Moráta" in _page(tmp_path, 7)


def test_export_raises_for_a_corner_that_is_not_in_events(tmp_path):
    events = _long_match()
    with pytest.raises(ValueError, match="nowhere"):
        export_review(_decoy_sample(corner_id="nowhere"), events, tmp_path)
    with pytest.raises(ValueError, match="k"):
        export_review(_decoy_sample(match_id=99), events, tmp_path)


def test_a_failed_export_leaves_the_old_files(tmp_path):
    events = _long_match()
    export_review(_decoy_sample(), events, tmp_path)
    sheet = (tmp_path / "sheet.csv").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        export_review(_decoy_sample(review_id=8, corner_id="nowhere"), events, tmp_path)
    assert (tmp_path / "sheet.csv").read_text(encoding="utf-8") == sheet
    assert (tmp_path / "chains" / "7.html").exists()
    assert not (tmp_path / "chains" / "8.html").exists()


def test_export_replaces_the_pages_of_an_earlier_export(tmp_path):
    events = _long_match()
    export_review(_decoy_sample(review_id=7), events, tmp_path)
    export_review(_decoy_sample(review_id=2), events, tmp_path)
    assert sorted(p.name for p in (tmp_path / "chains").glob("*")) == ["2.html"]
    assert pd.read_csv(tmp_path / "sheet.csv")["review_id"].tolist() == [2]


# --- agreement -------------------------------------------------------------------------------


def labels(**columns):
    n = len(next(iter(columns.values())))
    frame = {"review_id": list(range(1, n + 1)), "last_event_id": list("abcdefgh"[:n]),
             "shot_count": [0] * n, "goal": [0] * n, "lost_possession": [0] * n}  # fmt: skip
    frame.update(columns)
    return pd.DataFrame(frame)


def test_agreement_perfect_and_kappa():
    a = pd.DataFrame(
        {
            "review_id": [1, 2, 3, 4],
            "last_event_id": list("abcd"),
            "shot_count": [0, 1, 0, 2],
            "goal": [0, 0, 0, 1],
            "lost_possession": [1, 0, 1, 0],
        }
    )
    r = agreement(a, a.copy())
    assert r["boundary_exact"] == 1.0 and r["kappa_goal"] == 1.0
    assert r["kappa_shot_count"] == 1.0 and r["kappa_lost_possession"] == 1.0
    assert r["n"] == 4
    assert (r["exact_shot_count"], r["exact_goal"], r["exact_lost_possession"]) == (1.0, 1.0, 1.0)


def test_agreement_partial_values():
    a = labels(last_event_id=list("abcd"), goal=[0, 0, 1, 1], shot_count=[0, 1, 2, 0],
               lost_possession=[1, 1, 0, 0])  # fmt: skip
    b = labels(last_event_id=list("abxx"), goal=[0, 1, 1, 1], shot_count=[0, 1, 1, 0],
               lost_possession=[0, 0, 1, 1])  # fmt: skip
    r = agreement(a, b)
    assert r["boundary_exact"] == 0.5
    assert r["exact_goal"] == 0.75 and r["kappa_goal"] == pytest.approx(0.5)
    assert r["exact_shot_count"] == 0.75
    assert r["exact_lost_possession"] == 0.0 and r["kappa_lost_possession"] == pytest.approx(-1.0)
    # the shot-count kappa is the unweighted one over the integer codes
    assert r["kappa_shot_count"] == pytest.approx(0.6)


def test_agreement_matches_rows_by_review_id_not_position():
    a = labels(goal=[0, 0, 1, 1])
    b = a.iloc[::-1].reset_index(drop=True)
    assert agreement(a, b)["kappa_goal"] == 1.0


def test_agreement_accepts_bool_and_0_1_labels_and_compares_ids_as_strings():
    a = labels(last_event_id=[1, 2, 3, 4], goal=[0, 0, 1, 1], lost_possession=[0, 1, 0, 1])
    b = labels(last_event_id=["1", "2", "3", "4"], goal=[False, False, True, True],
               lost_possession=pd.array([False, True, False, True], dtype="boolean"))  # fmt: skip
    r = agreement(a, b)
    assert r["boundary_exact"] == 1.0
    assert r["kappa_goal"] == 1.0 and r["kappa_lost_possession"] == 1.0
    object_bools = labels(goal=pd.Series([False, False, True, True], dtype=object))
    assert agreement(labels(goal=[0, 0, 1, 1]), object_bools)["kappa_goal"] == 1.0


def test_ids_pasted_with_stray_whitespace_still_match():
    a = labels(last_event_id=list("abcd"))
    b = labels(last_event_id=[" a", "b ", "c", "d	"])
    assert agreement(a, b)["boundary_exact"] == 1.0


def test_an_undefined_kappa_is_nan_without_a_warning():
    a = labels(last_event_id=list("abcd"))  # everything 0 for every label: one shared category
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = agreement(a, a.copy())
    assert np.isnan(r["kappa_goal"]) and np.isnan(r["kappa_shot_count"])
    assert np.isnan(r["kappa_lost_possession"])
    assert r["exact_goal"] == 1.0 and r["boundary_exact"] == 1.0  # the shares are still defined


def test_kappa_is_defined_when_the_raters_differ_on_a_single_category():
    a = labels(goal=[0, 0, 0, 0])
    b = labels(goal=[0, 0, 0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = agreement(a, b)
    assert r["kappa_goal"] == 0.0 and r["exact_goal"] == 0.75


def test_agreement_raises_when_the_review_ids_differ():
    a = labels(goal=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="review_id"):
        agreement(a, a.iloc[:3])
    with pytest.raises(ValueError, match="review_id"):
        agreement(a, a.assign(review_id=[1, 2, 3, 5]))


def test_agreement_raises_on_a_duplicated_review_id_or_a_missing_column():
    a = labels(goal=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="review_id"):
        agreement(a, a.assign(review_id=[1, 2, 3, 3]))
    with pytest.raises(ValueError, match="goal"):
        agreement(a, a.drop(columns="goal"))


def test_agreement_raises_on_unlabelled_or_invalid_labels():
    a = labels(goal=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="goal"):
        agreement(a, a.assign(goal=[0, 0, 1, np.nan]))
    with pytest.raises(ValueError, match="last_event_id"):
        agreement(a, a.assign(last_event_id=["a", "b", "c", None]))
    with pytest.raises(ValueError, match="goal"):
        agreement(a, a.assign(goal=[0, 0, 1, 2]))
    with pytest.raises(ValueError, match="shot_count"):
        agreement(a, a.assign(shot_count=[0, 0, 1, -1]))
    with pytest.raises(ValueError, match="shot_count"):
        agreement(a, a.assign(shot_count=[0, 0, 1, 1.5]))
    with pytest.raises(ValueError, match="goal"):
        agreement(a, a.assign(goal=["no", "no", "yes", "yes"]))


def test_agreement_with_no_rows_raises():
    empty = labels(goal=[0]).iloc[:0]
    with pytest.raises(ValueError, match="no rows"):
        agreement(empty, empty)


def test_agreement_ignores_the_other_columns_of_a_sheet():
    a = labels(goal=[0, 0, 1, 1]).assign(match_id=[5, 5, 6, 6], corner_event_id=list("wxyz"))
    b = labels(goal=[0, 0, 1, 1]).assign(match_id=[5, 5, 6, 6], corner_event_id=list("wxyz"))
    assert agreement(a, b)["kappa_goal"] == 1.0


# --- rule_labels and rule_vs_consensus -------------------------------------------------------


def test_rule_labels_are_the_rules_answers_in_sheet_format(review_fixture):
    s = sample_chains(*review_fixture, per_stratum=100, seed=0)
    rule = rule_labels(s)
    assert list(rule.columns) == ["review_id", *LABEL_COLUMNS]
    assert rule["review_id"].tolist() == s["review_id"].tolist()
    assert rule["last_event_id"].tolist() == s["last_included_event_id"].tolist()
    for column in ("shot_count", "goal", "lost_possession"):
        assert rule[column].dtype == np.dtype("int64")
        assert rule[column].tolist() == s[column].astype(int).tolist()
    assert set(rule["goal"]) <= {0, 1} and set(rule["lost_possession"]) <= {0, 1}
    assert rule["shot_count"].max() == 1 and rule["lost_possession"].sum() > 0
    assert rule["goal"].sum() == 1
    # a chain with no events after the corner: the rule's last event is the corner itself
    z1 = rule[s["corner_event_id"] == "z1"]
    assert z1["last_event_id"].item() == "z1"


def test_rule_labels_raises_on_a_null_rule_value(review_fixture):
    s = sample_chains(*review_fixture, per_stratum=2, seed=0)
    with pytest.raises(ValueError, match="goal"):
        rule_labels(s.assign(goal=pd.array([pd.NA] * len(s), dtype="boolean")))


def test_rule_vs_consensus_gives_the_agreement_metrics(review_fixture):
    s = sample_chains(*review_fixture, per_stratum=100, seed=0)
    rule = rule_labels(s)
    same = rule_vs_consensus(rule, rule.copy())
    assert pd.Series(same).equals(pd.Series(agreement(rule, rule.copy())))
    assert same["kappa_goal"] == 1.0  # one chain of the sample is a goal
    consensus = rule.assign(last_event_id=rule["last_event_id"].where(rule.index != 0, "other"))
    r = rule_vs_consensus(rule, consensus)
    assert r["boundary_exact"] == pytest.approx(1 - 1 / len(rule))
    assert set(r) == set(agreement(rule, rule.copy()))


def test_a_filled_sheet_read_from_csv_compares_with_the_rules_labels(tmp_path, review_fixture):
    s = sample_chains(*review_fixture, per_stratum=100, seed=0)
    export_review(s, review_fixture[2], tmp_path)
    rule = rule_labels(s)

    # a labeller's copy: the exported sheet with the labels typed in, saved and read back
    filled = pd.read_csv(tmp_path / "sheet.csv", dtype={"corner_event_id": str})
    filled["last_event_id"] = rule["last_event_id"].to_numpy()
    filled["shot_count"] = rule["shot_count"].to_numpy()
    filled["goal"] = rule["goal"].astype(bool).map({True: "TRUE", False: "FALSE"}).to_numpy()
    filled["lost_possession"] = rule["lost_possession"].to_numpy()
    path = tmp_path / "filled.csv"
    filled.to_csv(path, index=False)
    read_back = pd.read_csv(path, dtype={"last_event_id": str})

    perfect = agreement(read_back, rule)
    assert perfect["boundary_exact"] == 1.0 and perfect["n"] == len(s)
    assert perfect["kappa_goal"] == 1.0 and perfect["kappa_shot_count"] == 1.0

    changed = read_back.copy()
    changed.loc[0, "shot_count"] += 1
    changed.loc[0, "last_event_id"] = "somewhere-else"
    r = rule_vs_consensus(rule, changed)
    assert r["boundary_exact"] == pytest.approx(1 - 1 / len(s))
    assert r["exact_shot_count"] == pytest.approx(1 - 1 / len(s))


# --- the runner ------------------------------------------------------------------------------

KEY_COLUMNS = [
    "review_id", "corner_event_id", "match_id", "competition_id", "season_id", "team_id", "stratum",
]  # fmt: skip


@pytest.fixture
def review_store(cfg_tmp, cfg):
    """``cfg_tmp`` with a store: core slices 43/106 and 9/281, and a non-core one, 2/27."""
    restarts, cap = restarts_from(cfg), cfg.p("cap_seconds")
    per_slice = {
        (43, 106): _match_events(1, 43, 106, 10, 20, MATCH_1),
        (9, 281): _match_events(2, 9, 281, 30, 40, MATCH_2),
        (2, 27): _match_events(3, 2, 27, 10, 20, MATCH_3),
    }
    slices = [
        Slice(43, 106, "World Cup", "tournament", True),
        Slice(9, 281, "Bundesliga", "league", True, 34),
        Slice(2, 27, "Premier League", "league", False, 38, core=False),
    ]
    store = dataclasses.replace(cfg_tmp, slices=slices)
    for key, events in per_slice.items():
        write_table(events, "events", store, key)
    outcomes = pd.concat(
        [build_rq2_outcomes(events, cap, restarts) for events in per_slice.values()],
        ignore_index=True,
    )
    write_table(outcomes, "derived/rq2_outcomes", store, None)
    write_table(_rq1(outcomes), "derived/rq1_corners", store, None)
    return store


def _with_per_stratum(cfg, n):
    params = {**cfg.params, "rq2_review_per_stratum": Param(n, "design_choice", "test")}
    return dataclasses.replace(cfg, params=params)


def test_runner_writes_the_review_material(review_store):
    cfg = _with_per_stratum(review_store, 2)
    counts = run_review_export_rq2(cfg)

    out = cfg.reports_dir / "review" / "rq2"
    sheet = pd.read_csv(out / "sheet.csv", dtype={"corner_event_id": str})
    assert list(sheet.columns) == SHEET_COLUMNS
    assert len(sheet) == 8 and sheet[LABEL_COLUMNS].isna().all().all()
    pages = sorted(int(p.stem) for p in (out / "chains").glob("*.html"))
    assert pages == list(range(1, 9)) == sorted(sheet["review_id"])
    assert sorted(p.name for p in out.iterdir()) == ["chains", "sheet.csv"]

    # the key sits next to the labelling folder, not in it
    key = pd.read_csv(cfg.reports_dir / "review" / "rq2_key.csv", dtype={"corner_event_id": str})
    assert list(key.columns) == KEY_COLUMNS
    assert key["review_id"].tolist() == list(range(1, 9))
    assert key["corner_event_id"].tolist() == sheet["corner_event_id"].tolist()
    assert key["stratum"].tolist() == [EXPECTED[c] for c in key["corner_event_id"]]
    assert key.groupby("stratum").size().to_dict() == dict.fromkeys(STRATA, 2)

    # complete chains of the core slices only: z1 is incomplete, x1 and x2 are in a non-core slice
    assert counts == {
        "pool": 15,
        "available": {"short": 3, "shot": 6, "clearance_first": 3, "repeat_other": 3},
        "drawn": dict.fromkeys(STRATA, 2),
        "total": 8,
    }


def test_runner_draws_everything_when_a_stratum_has_fewer_than_asked(review_store):
    cfg = _with_per_stratum(review_store, 100)
    counts = run_review_export_rq2(cfg)
    assert counts["drawn"] == counts["available"] and counts["total"] == 15
    key = pd.read_csv(cfg.reports_dir / "review" / "rq2_key.csv", dtype={"corner_event_id": str})
    assert sorted(key["corner_event_id"]) == sorted(set(EXPECTED) - {"z1"})
    assert not {"x1", "x2"} & set(key["corner_event_id"])


def test_runner_uses_the_configured_sample_size_and_seed(review_store):
    cfg = review_store
    assert cfg.p("rq2_review_per_stratum") == 20
    assert run_review_export_rq2(cfg)["total"] == 15  # 20 per stratum asked, fewer available

    small = _with_per_stratum(cfg, 2)
    key = cfg.reports_dir / "review" / "rq2_key.csv"
    run_review_export_rq2(small)
    first = key.read_text(encoding="utf-8")
    run_review_export_rq2(small)
    assert key.read_text(encoding="utf-8") == first  # repeatable
    run_review_export_rq2(dataclasses.replace(small, seed=small.seed + 1))
    assert key.read_text(encoding="utf-8") != first  # and seeded


def test_runner_pages_and_sheet_are_blind(review_store):
    cfg = _with_per_stratum(review_store, 100)
    run_review_export_rq2(cfg)
    out = cfg.reports_dir / "review" / "rq2"
    body = "\n".join((out / "sheet.csv").read_text(encoding="utf-8").splitlines()[1:])
    for leak in LEAKS:
        assert leak not in body
    for page in (out / "chains").glob("*.html"):
        html = page.read_text(encoding="utf-8")
        for leak in LEAKS:
            assert leak not in html, (page.name, leak)
    key = cfg.reports_dir / "review" / "rq2_key.csv"
    assert "stratum" in key.read_text(encoding="utf-8")  # the key is where the strata live


def test_runner_removes_the_pages_and_sheet_of_an_earlier_run(review_store):
    cfg = _with_per_stratum(review_store, 2)
    out = cfg.reports_dir / "review" / "rq2"
    (out / "chains").mkdir(parents=True)
    (out / "chains" / "99.html").write_text("stale", encoding="utf-8")
    (out / "sheet.csv").write_text("stale", encoding="utf-8")
    run_review_export_rq2(cfg)
    assert not (out / "chains" / "99.html").exists()
    assert "stale" not in (out / "sheet.csv").read_text(encoding="utf-8")
    run_review_export_rq2(_with_per_stratum(review_store, 1))  # a smaller sample drops pages
    assert sorted(int(p.stem) for p in (out / "chains").glob("*.html")) == [1, 2, 3, 4]


def test_runner_reads_only_the_core_slices_events(review_store, monkeypatch):
    from setpiece.rq2 import review

    seen = []
    real = review.read_table

    def spy(table, cfg, slices=None):
        seen.append((table, None if slices is None else tuple(map(tuple, slices))))
        return real(table, cfg, slices)

    monkeypatch.setattr(review, "read_table", spy)
    run_review_export_rq2(_with_per_stratum(review_store, 2))
    event_reads = [slices for table, slices in seen if table == "events"]
    assert event_reads  # one core slice at a time, never all of events at once, never a non-core
    assert all(s is not None and len(s) == 1 and s[0] in {(43, 106), (9, 281)} for s in event_reads)
    assert {s[0] for s in event_reads} == {(43, 106), (9, 281)}


def test_runner_without_its_inputs_says_what_to_run(cfg_tmp):
    with pytest.raises(FileNotFoundError, match=r"python -m setpiece rq2"):
        run_review_export_rq2(cfg_tmp)


def test_runner_without_the_rq1_table_says_what_to_run(review_store):
    (review_store.processed_dir / "derived" / "rq1_corners.parquet").unlink()
    with pytest.raises(FileNotFoundError, match=r"python -m setpiece rq1"):
        run_review_export_rq2(review_store)


def test_runner_with_no_core_slice_raises(review_store):
    slices = [dataclasses.replace(s, core=False) for s in review_store.slices]
    with pytest.raises(ValueError, match="core"):
        run_review_export_rq2(dataclasses.replace(review_store, slices=slices))


# --- packaging -------------------------------------------------------------------------------


def test_the_chain_template_is_packaged():
    template = resources.files("setpiece.rq2") / "templates" / "chain.html.j2"
    assert template.is_file()
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["setuptools"]["package-data"]["setpiece"] == ["rq2/templates/*.j2"]
    assert template.read_text(encoding="utf-8").isascii()
