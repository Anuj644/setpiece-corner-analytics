import dataclasses
import json

import pandas as pd
import pytest
from helpers import events_df, read_fixture

from setpiece.config import Slice
from setpiece.ingest.events import EVENTS_COLUMNS
from setpiece.io import read_table
from setpiece.qa import QAResult, check_events, run_qa
from setpiece.sources import Fetcher
from setpiece.stages.ingest import run_ingest

WARNING_KEYS = [
    "coords_out_of_range",
    "corner_start_mid_pitch",
    "technique_height_mismatch",
    "legacy_flag_without_technique",
    "timestamp_decreases",
]
KEY_COLUMNS = ["match_id", "event_id", "event_index", "period", "team_id", "possession_id"]
WORLD_CUP = Slice(43, 106, "FIFA World Cup 2022", "tournament", True)
LA_LIGA = Slice(11, 90, "La Liga 2020/21", "league", False)


# The four tests of the task brief, kept as written.
# fmt: off
def test_duplicate_key_is_error():
    df = events_df([{"match_id": 1, "event_id": "a", "event_index": 1, "period": 1, "team_id": 1, "possession_id": 1},
                    {"match_id": 1, "event_id": "a", "event_index": 2, "period": 1, "team_id": 1, "possession_id": 1}])
    assert any("duplicate" in e for e in check_events(df).errors)

def test_non_increasing_index_is_error():
    df = events_df([{"match_id": 1, "event_id": "a", "event_index": 2, "period": 1, "team_id": 1, "possession_id": 1},
                    {"match_id": 1, "event_id": "b", "event_index": 2, "period": 1, "team_id": 1, "possession_id": 1}])
    assert any("event_index" in e for e in check_events(df).errors)

def test_out_of_range_is_warning_not_error():
    df = events_df([{"match_id": 1, "event_id": "a", "event_index": 1, "period": 1, "team_id": 1,
                     "possession_id": 1, "start_x": 121.0, "start_y": 40.0}])
    r = check_events(df)
    assert r.errors == [] and r.warnings["coords_out_of_range"] == 1

def test_shot_corners_counted_separately():
    df = events_df([{"match_id": 1, "event_id": "a", "event_index": 1, "period": 1, "team_id": 1,
                     "possession_id": 1, "event_type": "Shot", "shot_type": "Corner"}])
    assert check_events(df).shot_corner_count == 1
# fmt: on


# --- helpers for the checks below ---------------------------------------------------------------


def row(event_index, **overrides):
    """A valid event of match 1, period 1, whose id follows its index."""
    return {"match_id": 1, "event_id": f"e{event_index}", "event_index": event_index, "period": 1,
            "team_id": 1, "possession_id": 1, **overrides}  # fmt: skip


def corner(event_index, **overrides):
    """A corner pass: a ground pass with no technique, taken from the corner flag."""
    return row(event_index, **{"event_type": "Pass", "pass_type": "Corner",
                               "pass_height": "Ground Pass", "start_x": 120.0, "start_y": 0.1,
                               **overrides})  # fmt: skip


def warnings_of(*rows):
    return check_events(events_df(list(rows))).warnings


def only(warning, count):
    """Every warning at 0 except ``warning``."""
    return {key: count if key == warning else 0 for key in WARNING_KEYS}


# --- structural errors ---------------------------------------------------------------------------


def test_duplicate_key_error_names_the_columns_and_counts_the_rows():
    df = events_df([row(1, event_id="a"), row(2, event_id="a"), row(3, event_id="b")])
    assert check_events(df).errors == ["duplicate (match_id, event_id): 2 rows"]


def test_the_same_event_id_in_two_matches_is_not_a_duplicate():
    df = events_df([row(1, event_id="a"), row(1, event_id="a", match_id=2)])
    assert check_events(df).errors == []


def test_non_increasing_index_is_judged_within_each_match():
    interleaved = events_df([row(1), row(1, match_id=2), row(2), row(2, match_id=2)])
    assert check_events(interleaved).errors == []
    backwards = events_df([row(1), row(1, match_id=2), row(3), row(2, match_id=2), row(2)])
    errors = check_events(backwards).errors
    assert len(errors) == 1 and "event_index" in errors[0] and "strictly increasing" in errors[0]


@pytest.mark.parametrize("column", KEY_COLUMNS)
def test_a_null_in_a_key_column_is_an_error(column):
    errors = check_events(events_df([{**row(1), column: None}, row(2)])).errors
    assert any(f"null {column}: 1 row" in e for e in errors)


def test_a_null_key_is_not_also_reported_as_a_duplicate():
    df = events_df([row(1, event_id=None), row(2, event_id=None)])
    errors = check_events(df).errors
    assert errors == ["null event_id: 2 rows"]


def test_a_null_event_index_is_not_also_an_ordering_error():
    errors = check_events(events_df([row(1), row(None, event_id="x"), row(3)])).errors
    assert errors == ["null event_index: 1 row"]


def test_nulls_in_other_columns_are_not_errors():
    df = events_df([row(1, start_x=None, start_y=None, timestamp=None, event_type=None)])
    assert check_events(df).errors == []


def test_a_frame_with_a_repeated_index_is_checked_by_position():
    both = pd.concat([events_df([row(1), row(2)]), events_df([row(1, match_id=2)])])
    assert list(both.index) == [0, 1, 0]
    assert check_events(both).errors == []


# --- warnings ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column, value",
    [("start_x", 120.1), ("start_x", -0.1), ("start_y", 80.1), ("start_y", -0.1),
     ("pass_end_x", 121.0), ("pass_end_x", -1.0), ("pass_end_y", 80.5), ("pass_end_y", -2.0)],
)  # fmt: skip
def test_each_coordinate_column_is_checked_against_the_pitch(column, value):
    assert warnings_of(row(1, **{column: value})) == only("coords_out_of_range", 1)


def test_the_pitch_edges_are_inside_and_nulls_are_not_out_of_range():
    edges = row(1, start_x=0.0, start_y=0.0, pass_end_x=120.0, pass_end_y=80.0)
    nulls = row(2, start_x=None, start_y=None, pass_end_x=None, pass_end_y=None)
    assert warnings_of(edges, nulls) == only("coords_out_of_range", 0)


def test_coordinates_are_counted_in_rows_not_cells():
    two_bad_cells = row(1, start_x=130.0, pass_end_y=90.0)
    assert warnings_of(two_bad_cells, row(2, start_x=121.0))["coords_out_of_range"] == 2


def test_corner_start_mid_pitch_counts_corner_passes_strictly_inside_the_band():
    warnings = warnings_of(
        corner(1, start_y=40.0),  # counted
        corner(2, start_y=2.5),  # counted
        corner(3, start_y=77.9),  # counted
        corner(4, start_y=2.0),  # on the band edge: not counted
        corner(5, start_y=78.0),  # on the band edge: not counted
        corner(6, start_y=None),  # no start: not counted
        row(7, event_type="Pass", pass_type="Free Kick", start_y=40.0),  # not a corner
        row(8, event_type="Shot", shot_type="Corner", start_y=40.0),  # a shot corner is not a pass
    )
    assert warnings == only("corner_start_mid_pitch", 3)


def test_technique_height_mismatch_is_counted_both_ways():
    warnings = warnings_of(
        corner(1, pass_height="High Pass"),  # not ground, no technique: counted
        corner(2, pass_height="Low Pass"),  # not ground, no technique: counted
        corner(3, pass_height="Ground Pass", pass_technique="Inswinging"),  # counted
        corner(4, pass_height="Ground Pass"),  # fine
        corner(5, pass_height="High Pass", pass_technique="Outswinging"),  # fine
        row(6, event_type="Pass", pass_type="Free Kick", pass_height="High Pass"),  # not a corner
        row(7, event_type="Pass", pass_type="Free Kick", pass_height="Ground Pass",
            pass_technique="Straight"),  # not a corner
    )  # fmt: skip
    assert warnings == only("technique_height_mismatch", 3)


def test_a_corner_of_unknown_height_without_technique_is_a_mismatch():
    assert warnings_of(corner(1, pass_height=None)) == only("technique_height_mismatch", 1)


@pytest.mark.parametrize("flag", ["inswinging", "outswinging", "straight"])
def test_a_legacy_flag_without_technique_is_counted(flag):
    details = json.dumps({"pass": {flag: True, "cross": True}})
    assert warnings_of(corner(1, event_details=details)) == only("legacy_flag_without_technique", 1)


def test_legacy_flag_only_counts_corners_without_technique_that_carry_a_flag():
    flagged = json.dumps({"pass": {"inswinging": True}})
    warnings = warnings_of(
        corner(1, event_details=flagged),  # counted
        corner(2, event_details=flagged, pass_height="High Pass", pass_technique="Inswinging"),
        corner(3, event_details=json.dumps({"pass": {"cross": True}})),  # no swing flag
        corner(4, event_details=json.dumps({"pass": {"inswinging": False}})),  # flag not set
        corner(5, event_details=json.dumps({"carry": {"inswinging": True}})),  # not the pass
        corner(6, event_details=json.dumps(["inswinging"])),  # not an object
        corner(7, event_details="not json"),
        corner(8, event_details=None),
        row(9, event_type="Pass", pass_type="Free Kick", event_details=flagged),  # not a corner
    )
    assert warnings["legacy_flag_without_technique"] == 1


def test_timestamp_decrease_is_an_event_earlier_than_the_previous_one_of_its_period():
    warnings = warnings_of(
        row(1, timestamp="00:10:00.000"),
        row(2, timestamp="00:09:59.999"),  # earlier than event 1: counted
        row(3, timestamp="00:09:59.999"),  # equal to the previous: not counted
        row(4, timestamp="00:12:00.000"),
        row(5, timestamp="00:11:00.000"),  # counted
        row(6, timestamp="00:11:30.000"),  # later than the previous one, though not than event 4
    )
    assert warnings == only("timestamp_decreases", 2)


def test_timestamps_restart_with_each_period_and_each_match():
    warnings = warnings_of(
        row(1, timestamp="00:44:30.000"),
        row(2, timestamp="00:45:00.000"),
        row(3, timestamp="00:00:00.000", period=2),  # a new period starts the clock again
        row(1, timestamp="00:30:00.000", match_id=2),
        row(2, timestamp="00:00:01.000", match_id=2),  # counted: same match and period
    )
    assert warnings == only("timestamp_decreases", 1)


def test_timestamps_are_compared_in_event_index_order_and_nulls_are_skipped():
    out_of_row_order = events_df([
        row(2, timestamp="00:00:05.000"),
        row(1, timestamp="00:00:01.000"),
    ])  # fmt: skip
    assert check_events(out_of_row_order).warnings["timestamp_decreases"] == 0
    gaps = [row(1, timestamp="00:00:09.000"), row(2, timestamp=None), row(3, timestamp="bad"),
            row(4, timestamp="00:00:08.000")]  # fmt: skip
    assert warnings_of(*gaps) == only("timestamp_decreases", 1)


def test_a_clean_frame_has_every_warning_at_zero():
    result = check_events(events_df([
        row(1, timestamp="00:00:00.000", event_type="Starting XI"),
        corner(2, timestamp="00:10:00.000", start_y=79.9),
        corner(3, timestamp="00:20:00.000", start_y=0.1, pass_height="High Pass",
               pass_technique="Inswinging",
               event_details=json.dumps({"pass": {"inswinging": True}})),
        row(4, timestamp="00:20:01.000", event_type="Shot", shot_type="Open Play"),
        row(5, timestamp="00:50:00.000", period=2, start_x=120.0, start_y=80.0),
    ]))  # fmt: skip
    assert result.errors == []
    assert result.warnings == only("coords_out_of_range", 0)
    assert list(result.warnings) == WARNING_KEYS
    assert result.shot_corner_count == 0


def test_shot_corners_are_not_corner_passes():
    df = events_df([row(1, event_type="Shot", shot_type="Corner", start_y=40.0),
                    row(2, event_type="Shot", shot_type="Open Play"),
                    row(3, event_type="Pass", shot_type="Corner"),
                    row(4, event_type="Pass", pass_type="Corner",
                        pass_height="Ground Pass")])  # fmt: skip
    result = check_events(df)
    assert result.shot_corner_count == 1
    assert result.warnings == only("corner_start_mid_pitch", 0)


# --- null rates and empty tables ------------------------------------------------------------------


def test_null_rates_cover_every_column_as_plain_floats():
    df = events_df([row(1, start_x=10.0), row(2)])
    rates = check_events(df).null_rates
    assert list(rates) == EVENTS_COLUMNS
    assert rates["match_id"] == 0.0 and rates["start_x"] == 0.5 and rates["pass_type"] == 1.0
    assert all(type(rate) is float for rate in rates.values())
    json.dumps(rates, allow_nan=False)


def test_list_columns_are_null_only_when_missing():
    df = events_df([row(1, related_event_ids=["a"]), row(2, related_event_ids=[]), row(3)])
    assert check_events(df).null_rates["related_event_ids"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "frame",
    [events_df([]), pd.DataFrame(columns=EVENTS_COLUMNS)],
    ids=["typed", "untyped"],
)
def test_an_empty_frame_is_clean_with_zero_null_rates(frame):
    result = check_events(frame)
    assert result == QAResult(
        errors=[],
        warnings=only("coords_out_of_range", 0),
        null_rates={column: 0.0 for column in EVENTS_COLUMNS},
        shot_corner_count=0,
    )
    assert all(type(rate) is float for rate in result.null_rates.values())


# --- run_qa ---------------------------------------------------------------------------------------


class NoNetwork:
    """A session that fails the test on any request, for runs that must be served from disk."""

    def get(self, url, **kwargs):
        raise AssertionError(f"unexpected network request: {url}")


def requested(session):
    """The ``rel_path`` of every request the fake session served, in order."""
    return [url.split("/data/", 1)[1] for url in session.urls]


def ingest(cfg, session):
    run_ingest(cfg, Fetcher(cfg, session))


def qa_report(cfg):
    return json.loads((cfg.processed_dir / "qa_report.json").read_text(encoding="utf-8"))


def raw_event(index, type_name, **extra):
    """A raw event of the second half, later than everything in the one-match fixture."""
    return {"id": f"x{index}", "index": index, "period": 2, "timestamp": "00:30:00.000",
            "type": {"name": type_name}, "team": {"id": 10}, "possession": 14, **extra}  # fmt: skip


def match_copy(match_id, comp, season):
    match = read_fixture("raw_matches_min.json")[0]
    match.update(match_id=match_id)
    match["competition"]["competition_id"] = comp
    match["season"]["season_id"] = season
    return match


def serve_two_slices(session):
    """Match 1 in the World Cup slice: 1 shot corner, 1 stray coordinate, all else clean.

    Match 3 in the La Liga slice: 2 shot corners, 1 stray coordinate and a repeated event.
    """
    shot_corner = {"shot": {"type": {"name": "Corner"}}}
    stray = {"location": [121.0, 40.0]}
    fixture = read_fixture("raw_events_min.json")
    session.serve("events/1.json", [*fixture, raw_event(13, "Shot", **shot_corner),
                                    raw_event(14, "Pass", **stray)])  # fmt: skip
    session.serve("matches/11/90.json", [match_copy(3, 11, 90)])
    session.serve("events/3.json", [*fixture, dict(fixture[1]),
                                    raw_event(13, "Shot", **shot_corner),
                                    raw_event(14, "Shot", **shot_corner),
                                    raw_event(15, "Pass", **stray)])  # fmt: skip


@pytest.fixture
def two_slice_run(cfg_tmp, fake_session_with_fixtures):
    cfg = dataclasses.replace(cfg_tmp, slices=[WORLD_CUP, LA_LIGA])
    serve_two_slices(fake_session_with_fixtures)
    ingest(cfg, fake_session_with_fixtures)
    return cfg, run_qa(cfg, Fetcher(cfg, NoNetwork()))


def test_run_qa_on_the_ingested_fixture_is_clean(cfg_tmp_one_slice, fake_session_with_fixtures):
    ingest(cfg_tmp_one_slice, fake_session_with_fixtures)
    result = run_qa(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, NoNetwork()))
    assert isinstance(result, QAResult)
    assert result.errors == []
    assert result.warnings == only("coords_out_of_range", 0)
    assert result.shot_corner_count == 0
    assert list(result.null_rates) == EVENTS_COLUMNS


def test_run_qa_reads_the_match_list_through_the_fetcher_and_never_the_raw_events(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    session = fake_session_with_fixtures
    ingest(cfg_tmp_one_slice, session)
    before = len(session.urls)
    cold = dataclasses.replace(cfg_tmp_one_slice, raw_dir=cfg_tmp_one_slice.raw_dir / "cold")
    run_qa(cold, Fetcher(cold, session))
    assert requested(session)[before:] == ["matches/43/106.json"]
    run_qa(cold, Fetcher(cold, session))  # now cached
    assert len(session.urls) == before + 1


def test_a_listed_match_missing_from_events_is_an_error(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    session = fake_session_with_fixtures
    ingest(cfg_tmp_one_slice, session)
    session.serve("matches/43/106.json", [match_copy(1, 43, 106), match_copy(2, 43, 106),
                                          match_copy(7, 43, 106)])  # fmt: skip
    fresh = dataclasses.replace(cfg_tmp_one_slice, raw_dir=cfg_tmp_one_slice.raw_dir / "fresh")
    result = run_qa(fresh, Fetcher(fresh, session))
    assert result.errors == ["43/106: listed matches missing from events (n=2): 2, 7"]
    report = qa_report(fresh)
    assert report["43/106"]["errors"] == ["listed matches missing from events (n=2): 2, 7"]
    assert report["overall"]["errors"] == result.errors


def test_a_long_list_of_missing_matches_is_cut_short(cfg_tmp_one_slice, fake_session_with_fixtures):
    session = fake_session_with_fixtures
    ingest(cfg_tmp_one_slice, session)
    session.serve("matches/43/106.json", [match_copy(m, 43, 106) for m in range(1, 26)])
    fresh = dataclasses.replace(cfg_tmp_one_slice, raw_dir=cfg_tmp_one_slice.raw_dir / "fresh")
    [error] = run_qa(fresh, Fetcher(fresh, session)).errors
    assert "(n=24)" in error and "2, 3, 4" in error and error.endswith("... and 14 more")
    assert "25" not in error


def test_a_configured_slice_that_was_never_ingested_misses_all_its_matches(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    session = fake_session_with_fixtures
    ingest(cfg_tmp_one_slice, session)  # the World Cup only
    session.serve("matches/11/90.json", [match_copy(3, 11, 90), match_copy(4, 11, 90)])
    cfg = dataclasses.replace(cfg_tmp_one_slice, slices=[WORLD_CUP, LA_LIGA])
    result = run_qa(cfg, Fetcher(cfg, session))
    assert result.errors == ["11/90: listed matches missing from events (n=2): 3, 4"]
    report = qa_report(cfg)
    assert report["43/106"]["errors"] == []
    assert report["11/90"]["events"] == 0
    assert set(report["11/90"]["null_rates"].values()) == {0.0}


def test_a_slice_without_matches_reports_zero_rows_and_zero_rates(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    fake_session_with_fixtures.serve("matches/43/106.json", [])
    ingest(cfg_tmp_one_slice, fake_session_with_fixtures)
    result = run_qa(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, NoNetwork()))
    assert result.errors == []
    section = qa_report(cfg_tmp_one_slice)["43/106"]
    assert section["events"] == 0 and section["shot_corner_count"] == 0
    assert section["null_rates"] == {column: 0.0 for column in EVENTS_COLUMNS}


def test_run_qa_without_slices_is_a_value_error(cfg_tmp):
    with pytest.raises(ValueError, match="no slices configured"):
        run_qa(dataclasses.replace(cfg_tmp, slices=[]), Fetcher(cfg_tmp, NoNetwork()))


def test_the_report_has_a_section_per_slice_and_an_overall_one(two_slice_run):
    cfg, _ = two_slice_run
    report = qa_report(cfg)
    assert list(report) == ["43/106", "11/90", "overall"]
    for section in report.values():
        assert list(section) == ["events", "errors", "warnings", "null_rates", "shot_corner_count"]
        assert list(section["warnings"]) == WARNING_KEYS
        assert list(section["null_rates"]) == EVENTS_COLUMNS
    assert report["43/106"]["events"] == 14 and report["11/90"]["events"] == 16
    assert report["43/106"]["errors"] == []
    assert report["43/106"]["shot_corner_count"] == 1 and report["11/90"]["shot_corner_count"] == 2
    assert report["43/106"]["warnings"] == only("coords_out_of_range", 1)
    assert report["11/90"]["warnings"] == only("coords_out_of_range", 1)


def test_the_overall_section_combines_the_slices(two_slice_run):
    cfg, _ = two_slice_run
    report = qa_report(cfg)
    overall = report["overall"]
    assert overall["events"] == 30
    assert overall["shot_corner_count"] == 3
    assert overall["warnings"] == only("coords_out_of_range", 2)
    [duplicate, ordering] = overall["errors"]
    assert duplicate.startswith("11/90: duplicate (match_id, event_id)")
    assert ordering.startswith("11/90: event_index")
    assert [e.removeprefix("11/90: ") for e in overall["errors"]] == report["11/90"]["errors"]
    # null rates are over all events, not an average of the slices' rates
    events = read_table("events", cfg)
    assert len(events) == 30
    for column, rate in overall["null_rates"].items():
        assert rate == pytest.approx(events[column].isna().mean()), column
    assert overall["null_rates"]["start_x"] not in (
        report["43/106"]["null_rates"]["start_x"],
        report["11/90"]["null_rates"]["start_x"],
    )


def test_run_qa_returns_what_the_overall_section_holds(two_slice_run):
    cfg, result = two_slice_run
    overall = qa_report(cfg)["overall"]
    assert result == QAResult(overall["errors"], overall["warnings"], overall["null_rates"],
                              overall["shot_corner_count"])  # fmt: skip


def test_the_report_is_valid_json_without_nan_and_leaves_no_temp_file(two_slice_run):
    cfg, _ = two_slice_run

    def refuse(constant):
        raise AssertionError(f"{constant} in the QA report")

    text = (cfg.processed_dir / "qa_report.json").read_text(encoding="utf-8")
    json.loads(text, parse_constant=refuse)
    assert sorted(path.name for path in cfg.processed_dir.glob("*.tmp")) == []


def test_the_qa_manifest_holds_the_error_count_and_each_warning(two_slice_run):
    cfg, _ = two_slice_run
    manifest = json.loads((cfg.processed_dir / "manifests" / "qa.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "qa"
    assert manifest["counts"] == {"errors": 2, **only("coords_out_of_range", 2)}
    assert all(type(count) is int for count in manifest["counts"].values())


def test_a_second_run_replaces_the_report(cfg_tmp_one_slice, fake_session_with_fixtures):
    ingest(cfg_tmp_one_slice, fake_session_with_fixtures)
    first = run_qa(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, NoNetwork()))
    second = run_qa(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, NoNetwork()))
    assert first == second
    assert qa_report(cfg_tmp_one_slice)["overall"]["events"] == 12
