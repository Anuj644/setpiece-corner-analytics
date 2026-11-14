import dataclasses
import json

import pandas as pd
import pytest
from helpers import read_fixture

from setpiece.config import Slice
from setpiece.ingest.events import EVENTS_COLUMNS, EVENTS_DTYPES
from setpiece.ingest.frames import CORNER_360_COLUMNS, CORNER_360_DTYPES
from setpiece.io import list_partitions, read_table
from setpiece.sources import Fetcher, FetchError, NotFound
from setpiece.stages import ingest as ingest_stage
from setpiece.stages.ingest import run_fetch, run_ingest

N_EVENTS = 12  # rows of raw_events_min.json: the fixture match has 2 corners, 1 with a 360 frame
WORLD_CUP = Slice(43, 106, "FIFA World Cup 2022", "tournament", True)


class NoNetwork:
    """A session that fails the test on any request, for runs that must be served from disk."""

    def get(self, url, **kwargs):
        raise AssertionError(f"unexpected network request: {url}")


def requested(session):
    """The ``rel_path`` of every request the fake session served, in order."""
    return [url.split("/data/", 1)[1] for url in session.urls]


def with_slices(cfg, *slices):
    return dataclasses.replace(cfg, slices=list(slices))


def serve_copy_of_the_fixture_match(session, match_id, comp=43, season=106):
    """Serve ``match_id`` in slice ``comp/season`` with the fixture's events (and no 360 file)."""
    match = read_fixture("raw_matches_min.json")[0]
    match.update(match_id=match_id)
    match["competition"]["competition_id"] = comp
    match["season"]["season_id"] = season
    session.serve(f"events/{match_id}.json", read_fixture("raw_events_min.json"))
    return match


def assert_dtypes_match(frame, dtypes):
    assert list(frame.columns) == list(dtypes)
    for col, expected in dtypes.items():
        if expected == "object":
            assert frame[col].dtype == object, col
        else:
            assert frame[col].dtype == pd.api.types.pandas_dtype(expected), col


# --- ingest ------------------------------------------------------------------------------------


def test_ingest_end_to_end(cfg_tmp_one_slice, fake_session_with_fixtures):
    counts = run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    s = counts["43/106"]
    assert s["matches"] == 1 and s["corners"] == 2 and s["corners_matched"] == 1
    assert len(read_table("events", cfg_tmp_one_slice)) == s["events"]
    assert (cfg_tmp_one_slice.processed_dir / "manifests" / "ingest.json").exists()


def test_ingest_counts_are_plain_ints_under_string_slice_keys(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    counts = run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    assert counts == {"43/106": {"matches": 1, "events": N_EVENTS, "corners": 2,
                                 "corners_matched": 1, "frames_unmatched": 1}}  # fmt: skip
    assert all(type(value) is int for value in counts["43/106"].values())


def test_ingest_manifest_holds_the_counts_and_the_pruned_slices(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    counts = run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    path = cfg_tmp_one_slice.processed_dir / "manifests" / "ingest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "ingest"
    assert manifest["counts"] == {**counts, "pruned": []}


def test_ingest_writes_the_three_tables(cfg_tmp_one_slice, fake_session_with_fixtures):
    run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    events = read_table("events", cfg_tmp_one_slice)
    assert_dtypes_match(events, EVENTS_DTYPES)
    assert set(events.match_id) == {1}
    assert set(zip(events.competition_id, events.season_id)) == {(43, 106)}

    corner_360 = read_table("corner_360", cfg_tmp_one_slice)
    assert_dtypes_match(corner_360, CORNER_360_DTYPES)
    assert len(corner_360) == 1
    assert corner_360.event_id.isin(events.event_id).all()

    stats = read_table("frame_stats", cfg_tmp_one_slice, [(43, 106)])
    assert stats.to_dict("records") == [{
        "match_id": 1, "n_frames": 3, "n_unmatched": 1, "n_corners": 2,
        "n_corners_matched": 1, "has_file": True, "competition_id": 43, "season_id": 106,
    }]  # fmt: skip


def test_matches_are_streamed_one_at_a_time_in_file_order(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    session = fake_session_with_fixtures
    session.serve(
        "matches/43/106.json",
        [
            serve_copy_of_the_fixture_match(session, 1),
            serve_copy_of_the_fixture_match(session, 3),
        ],
    )
    run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, session))
    assert requested(session) == ["matches/43/106.json", "events/1.json", "three-sixty/1.json",
                                  "events/3.json", "three-sixty/3.json"]  # fmt: skip


def test_a_slice_of_several_matches_is_concatenated_with_the_stored_dtypes(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    """Match 3 has no 360 file, so its corner_360 part is empty next to match 1's full one."""
    session = fake_session_with_fixtures
    session.serve(
        "matches/43/106.json",
        [
            serve_copy_of_the_fixture_match(session, 1),
            serve_copy_of_the_fixture_match(session, 3),
        ],
    )
    counts = run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, session))
    assert counts["43/106"] == {"matches": 2, "events": 2 * N_EVENTS, "corners": 4,
                                "corners_matched": 1, "frames_unmatched": 1}  # fmt: skip

    events = read_table("events", cfg_tmp_one_slice)
    assert_dtypes_match(events, EVENTS_DTYPES)
    assert events.match_id.tolist() == [1] * N_EVENTS + [3] * N_EVENTS
    corner_360 = read_table("corner_360", cfg_tmp_one_slice)
    assert_dtypes_match(corner_360, CORNER_360_DTYPES)
    assert corner_360.match_id.tolist() == [1]

    stats = read_table("frame_stats", cfg_tmp_one_slice)
    assert stats.match_id.tolist() == [1, 3]
    assert stats.has_file.tolist() == [True, False]
    assert stats.n_frames.tolist() == [3, 0]
    assert stats.dtypes.astype(str).to_dict() == {
        "match_id": "Int64", "n_frames": "Int64", "n_unmatched": "Int64", "n_corners": "Int64",
        "n_corners_matched": "Int64", "has_file": "boolean", "competition_id": "Int64",
        "season_id": "Int64",
    }  # fmt: skip


def test_a_slice_without_360_data_never_asks_for_it_and_still_gets_an_empty_corner_360(
    cfg_tmp, fake_session_with_fixtures
):
    cfg = with_slices(cfg_tmp, dataclasses.replace(WORLD_CUP, has_360=False))
    counts = run_ingest(cfg, Fetcher(cfg, fake_session_with_fixtures))
    assert not any(path.startswith("three-sixty") for path in requested(fake_session_with_fixtures))
    assert counts["43/106"]["corners"] == 2 and counts["43/106"]["corners_matched"] == 0
    assert counts["43/106"]["frames_unmatched"] == 0

    assert list_partitions("corner_360", cfg) == [(43, 106)]
    corner_360 = read_table("corner_360", cfg)
    assert corner_360.empty
    assert list(corner_360.columns) == CORNER_360_COLUMNS
    stats = read_table("frame_stats", cfg)
    assert stats.has_file.tolist() == [False] and stats.n_frames.tolist() == [0]


def test_a_missing_three_sixty_file_in_a_360_slice_means_no_frames(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    fake_session_with_fixtures.add_404("three-sixty/1.json")
    counts = run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    assert counts["43/106"]["corners"] == 2 and counts["43/106"]["corners_matched"] == 0
    assert counts["43/106"]["frames_unmatched"] == 0
    assert read_table("corner_360", cfg_tmp_one_slice).empty
    assert read_table("frame_stats", cfg_tmp_one_slice).has_file.tolist() == [False]
    assert len(read_table("events", cfg_tmp_one_slice)) == N_EVENTS


def test_a_missing_events_file_aborts_naming_the_match_and_the_slice(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    fake_session_with_fixtures.add_404("events/1.json")
    with pytest.raises(NotFound, match=r"match 1 .*43/106"):
        run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    processed = cfg_tmp_one_slice.processed_dir
    assert not (processed / "events").exists() and not (processed / "manifests").exists()


def test_a_bad_three_sixty_body_is_an_error_not_a_missing_file(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    fake_session_with_fixtures.files["three-sixty/1.json"] = b"<html>rate limited</html>"
    with pytest.raises(json.JSONDecodeError):
        run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))


def test_a_three_sixty_download_that_keeps_failing_is_an_error_not_a_missing_file(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    fetcher = Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures, sleep=lambda seconds: None)
    fetcher.get_json("matches/43/106.json")
    fetcher.get_json("events/1.json")  # cached, so only the 360 request fails
    fake_session_with_fixtures.fail_next(4, status=503)  # the first try and all three retries
    with pytest.raises(FetchError, match="three-sixty/1.json"):
        run_ingest(cfg_tmp_one_slice, fetcher)


def test_ingest_without_a_fetcher_builds_one_from_the_config(
    cfg_tmp_one_slice, fake_session_with_fixtures, monkeypatch
):
    monkeypatch.setattr(
        ingest_stage, "Fetcher", lambda cfg: Fetcher(cfg, fake_session_with_fixtures)
    )
    assert run_ingest(cfg_tmp_one_slice)["43/106"]["matches"] == 1


def test_a_slice_with_no_matches_gets_empty_partitions(cfg_tmp_one_slice, fake_session):
    fake_session.serve("matches/43/106.json", [])
    counts = run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session))
    assert counts["43/106"] == {"matches": 0, "events": 0, "corners": 0,
                                "corners_matched": 0, "frames_unmatched": 0}  # fmt: skip
    events = read_table("events", cfg_tmp_one_slice)
    assert events.empty and list(events.columns) == EVENTS_COLUMNS
    assert read_table("corner_360", cfg_tmp_one_slice).empty
    assert read_table("frame_stats", cfg_tmp_one_slice).empty


def test_rerunning_ingest_replaces_the_tables_instead_of_appending(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    fetcher = Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures)
    first = run_ingest(cfg_tmp_one_slice, fetcher)
    before = {t: read_table(t, cfg_tmp_one_slice) for t in ("events", "corner_360", "frame_stats")}
    second = run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    assert second == first
    for table, frame in before.items():
        pd.testing.assert_frame_equal(read_table(table, cfg_tmp_one_slice), frame)


def test_a_slice_removed_from_the_config_is_pruned_from_the_tables(
    cfg_tmp, fake_session_with_fixtures
):
    session = fake_session_with_fixtures
    liga = Slice(11, 90, "La Liga 2020/21", "league", False)
    session.serve("matches/11/90.json", [serve_copy_of_the_fixture_match(session, 2, 11, 90)])
    both = with_slices(cfg_tmp, WORLD_CUP, liga)
    run_ingest(both, Fetcher(both, session))
    assert list_partitions("events", both) == [(11, 90), (43, 106)]
    assert list_partitions("corner_360", both) == [(11, 90), (43, 106)]

    only_world_cup = with_slices(cfg_tmp, WORLD_CUP)
    counts = run_ingest(only_world_cup, Fetcher(only_world_cup, session))
    assert list(counts) == ["43/106"]
    for table in ("events", "corner_360"):
        assert list_partitions(table, only_world_cup) == [(43, 106)]
    assert set(read_table("events", only_world_cup).match_id) == {1}
    assert read_table("frame_stats", only_world_cup).match_id.tolist() == [1]
    manifest = json.loads(
        (cfg_tmp.processed_dir / "manifests" / "ingest.json").read_text(encoding="utf-8")
    )
    assert manifest["counts"]["pruned"] == ["11/90"]


def test_pruning_also_removes_an_interrupted_partition_of_a_removed_slice(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    (cfg_tmp_one_slice.processed_dir / "events" / "competition_id=7" / "season_id=235").mkdir(
        parents=True
    )
    run_ingest(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    assert list_partitions("events", cfg_tmp_one_slice) == [(43, 106)]


def test_reingest_after_fetch_needs_no_network_even_with_a_missing_three_sixty_file(
    cfg_tmp, fake_session_with_fixtures
):
    """The cache holds the files and the 404 markers hold the absent 360 file of match 3."""
    session = fake_session_with_fixtures
    session.serve(
        "matches/43/106.json",
        [
            serve_copy_of_the_fixture_match(session, 1),
            serve_copy_of_the_fixture_match(session, 3),
        ],
    )
    cfg = with_slices(cfg_tmp, WORLD_CUP)
    online = run_fetch(cfg, Fetcher(cfg, session))
    assert online["43/106"]["three_sixty_404"] == 1
    calls = session.calls

    offline = Fetcher(cfg, NoNetwork())
    counts = run_ingest(cfg, offline)
    assert counts["43/106"]["matches"] == 2 and counts["43/106"]["corners_matched"] == 1
    assert read_table("frame_stats", cfg).has_file.tolist() == [True, False]
    assert session.calls == calls


# --- fetch -------------------------------------------------------------------------------------


def test_fetch_downloads_every_file_and_saves_the_manifest(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    cfg = cfg_tmp_one_slice
    counts = run_fetch(cfg, Fetcher(cfg, fake_session_with_fixtures))
    assert counts == {"43/106": {"matches": 1, "events_files": 1, "three_sixty_files": 1,
                                 "three_sixty_404": 0}}  # fmt: skip
    assert all(type(value) is int for value in counts["43/106"].values())

    cache = cfg.raw_dir / cfg.sha
    for rel_path in ("matches/43/106.json", "events/1.json", "three-sixty/1.json"):
        assert (cache / f"{rel_path}.gz").is_file()
    saved = json.loads((cache / "download_manifest.json").read_text(encoding="utf-8"))
    assert set(saved) == {"matches/43/106.json", "events/1.json", "three-sixty/1.json"}

    manifest = json.loads(
        (cfg.processed_dir / "manifests" / "fetch.json").read_text(encoding="utf-8")
    )
    assert manifest["stage"] == "fetch" and manifest["counts"] == counts
    assert not (cfg.processed_dir / "events").exists()  # fetching builds no tables


def test_fetch_counts_a_missing_three_sixty_file_and_remembers_the_404(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    cfg = cfg_tmp_one_slice
    fake_session_with_fixtures.add_404("three-sixty/1.json")
    counts = run_fetch(cfg, Fetcher(cfg, fake_session_with_fixtures))
    assert counts["43/106"] == {"matches": 1, "events_files": 1, "three_sixty_files": 0,
                                "three_sixty_404": 1}  # fmt: skip
    cache = cfg.raw_dir / cfg.sha
    assert (cache / "three-sixty" / "1.json.404").is_file()
    saved = json.loads((cache / "download_manifest.json").read_text(encoding="utf-8"))
    assert "three-sixty/1.json" not in saved


def test_fetch_of_a_slice_without_360_data_never_asks_for_it(cfg_tmp, fake_session_with_fixtures):
    cfg = with_slices(cfg_tmp, dataclasses.replace(WORLD_CUP, has_360=False))
    counts = run_fetch(cfg, Fetcher(cfg, fake_session_with_fixtures))
    assert counts["43/106"] == {"matches": 1, "events_files": 1, "three_sixty_files": 0,
                                "three_sixty_404": 0}  # fmt: skip
    assert requested(fake_session_with_fixtures) == ["matches/43/106.json", "events/1.json"]


def test_fetch_aborts_on_a_missing_events_file(cfg_tmp_one_slice, fake_session_with_fixtures):
    fake_session_with_fixtures.add_404("events/1.json")
    with pytest.raises(NotFound, match=r"match 1 .*43/106"):
        run_fetch(cfg_tmp_one_slice, Fetcher(cfg_tmp_one_slice, fake_session_with_fixtures))
    assert not (cfg_tmp_one_slice.processed_dir / "manifests").exists()


def test_fetch_is_served_from_the_cache_the_second_time(
    cfg_tmp_one_slice, fake_session_with_fixtures
):
    cfg = cfg_tmp_one_slice
    first = run_fetch(cfg, Fetcher(cfg, fake_session_with_fixtures))
    calls = fake_session_with_fixtures.calls
    assert run_fetch(cfg, Fetcher(cfg, NoNetwork())) == first
    assert fake_session_with_fixtures.calls == calls


def test_fetch_walks_the_slices_and_matches_in_order(cfg_tmp, fake_session_with_fixtures):
    session = fake_session_with_fixtures
    liga = Slice(11, 90, "La Liga 2020/21", "league", False)
    session.serve(
        "matches/11/90.json",
        [
            serve_copy_of_the_fixture_match(session, 2, 11, 90),
            serve_copy_of_the_fixture_match(session, 4, 11, 90),
        ],
    )
    cfg = with_slices(cfg_tmp, WORLD_CUP, liga)
    counts = run_fetch(cfg, Fetcher(cfg, session))
    assert requested(session) == ["matches/43/106.json", "events/1.json", "three-sixty/1.json",
                                  "matches/11/90.json", "events/2.json", "events/4.json"]  # fmt: skip
    assert counts["11/90"]["matches"] == 2 and counts["11/90"]["events_files"] == 2
