import gzip
import hashlib
import json

import pytest
import requests

from setpiece import sources
from setpiece.config import Slice
from setpiece.sources import (
    Fetcher,
    FetchError,
    NotFound,
    SliceRaw,
    discover_slices,
    fetch_slice,
    write_catalogue,
)

COMPETITIONS = [{"competition_id": 43}]


def test_second_read_uses_cache(fake_cfg, fake_session):
    f = Fetcher(fake_cfg, fake_session)
    assert f.get_json("competitions.json") == [{"competition_id": 43}]
    assert f.get_json("competitions.json") == [{"competition_id": 43}]
    assert fake_session.calls == 1


def test_url_uses_pinned_sha(fake_cfg, fake_session):
    Fetcher(fake_cfg, fake_session).get_json("competitions.json")
    assert fake_session.urls[0].endswith(
        "/4b73468fc5b0f1950f9f66fada70ad3a4f9327cb/data/competitions.json")


def test_missing_360_file_is_recorded_not_fatal(fake_cfg, fake_session, slice_with_one_match_and_360):
    fake_session.add_404("three-sixty/1.json")
    raw = fetch_slice(Fetcher(fake_cfg, fake_session), slice_with_one_match_and_360)
    assert raw.frames[1] is None and len(raw.events[1]) > 0


def test_corrupt_cache_is_refetched(fake_cfg, fake_session):
    f = Fetcher(fake_cfg, fake_session)
    f.get_json("competitions.json")
    path = fake_cfg.raw_dir / fake_cfg.sha / "competitions.json.gz"
    path.write_bytes(path.read_bytes()[:10])        # truncate
    assert f.get_json("competitions.json") == [{"competition_id": 43}]
    assert fake_session.calls == 2


def test_manifest_has_sha256(fake_cfg, fake_session):
    f = Fetcher(fake_cfg, fake_session)
    f.get_json("competitions.json")
    assert len(f.manifest()["competitions.json"]) == 64


# --- cache ---------------------------------------------------------------------------------

def test_cache_is_gzip_json_at_pinned_path(fake_cfg, fake_session):
    Fetcher(fake_cfg, fake_session).get_json("competitions.json")
    path = fake_cfg.raw_dir / fake_cfg.sha / "competitions.json.gz"
    assert json.loads(gzip.decompress(path.read_bytes())) == COMPETITIONS


def test_nested_rel_path_creates_parent_dirs(fake_cfg, fake_session):
    fake_session.serve("events/1.json", [{"id": "e1"}])
    Fetcher(fake_cfg, fake_session).get_json("events/1.json")
    assert (fake_cfg.raw_dir / fake_cfg.sha / "events" / "1.json.gz").is_file()


def test_new_fetcher_reads_existing_cache_without_network(fake_cfg, fake_session):
    Fetcher(fake_cfg, fake_session).get_json("competitions.json")
    assert Fetcher(fake_cfg, fake_session).get_json("competitions.json") == COMPETITIONS
    assert fake_session.calls == 1


@pytest.mark.parametrize("corrupt", [
    pytest.param(lambda good: good[:10], id="truncated-EOFError"),
    pytest.param(lambda good: b"this is not gzip data", id="not-gzip-BadGzipFile"),
    pytest.param(lambda good: good[:10] + b"\xff\xff\xff\xff" + good[14:], id="bad-deflate-zlib"),
    pytest.param(lambda good: gzip.compress(b"<html>"), id="not-json-JSONDecodeError"),
    pytest.param(lambda good: b"", id="empty-file"),
])
def test_each_kind_of_corrupt_cache_is_replaced(fake_cfg, fake_session, corrupt):
    Fetcher(fake_cfg, fake_session).get_json("competitions.json")
    path = fake_cfg.raw_dir / fake_cfg.sha / "competitions.json.gz"
    path.write_bytes(corrupt(path.read_bytes()))
    assert Fetcher(fake_cfg, fake_session).get_json("competitions.json") == COMPETITIONS
    assert fake_session.calls == 2
    assert json.loads(gzip.decompress(path.read_bytes())) == COMPETITIONS   # cache repaired


def test_failed_cache_write_leaves_no_partial_or_temp_file(fake_cfg, fake_session, monkeypatch):
    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(sources.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        Fetcher(fake_cfg, fake_session).get_json("competitions.json")
    assert list((fake_cfg.raw_dir / fake_cfg.sha).iterdir()) == []


# --- HTTP errors ---------------------------------------------------------------------------

def test_404_raises_not_found_immediately_without_retry(fake_cfg, fake_session):
    sleeps = []
    f = Fetcher(fake_cfg, fake_session, sleep=sleeps.append)
    with pytest.raises(NotFound, match="events/9.json"):
        f.get_json("events/9.json")
    assert fake_session.calls == 1
    assert sleeps == []


@pytest.mark.parametrize("failure", [
    pytest.param({"status": 503}, id="http-503"),
    pytest.param({"exc": requests.ConnectionError("boom")}, id="connection-error"),
    pytest.param({"exc": requests.Timeout("slow")}, id="timeout"),
])
def test_transient_error_is_retried_with_backoff(fake_cfg, fake_session, failure):
    sleeps = []
    fake_session.fail_next(2, **failure)
    f = Fetcher(fake_cfg, fake_session, sleep=sleeps.append)
    assert f.get_json("competitions.json") == COMPETITIONS
    assert fake_session.calls == 3
    assert sleeps == [1, 2]


def test_gives_up_after_three_retries(fake_cfg, fake_session):
    sleeps = []
    fake_session.fail_next(10, status=500)
    f = Fetcher(fake_cfg, fake_session, sleep=sleeps.append)
    with pytest.raises(FetchError, match="competitions.json"):
        f.get_json("competitions.json")
    assert fake_session.calls == 4                    # first try + 3 retries
    assert sleeps == [1, 2, 4]
    assert not (fake_cfg.raw_dir / fake_cfg.sha / "competitions.json.gz").exists()


def test_requests_carry_a_timeout(fake_cfg, fake_session):
    seen = {}
    original_get = fake_session.get

    def spy(url, **kwargs):
        seen.update(kwargs)
        return original_get(url, **kwargs)

    fake_session.get = spy
    Fetcher(fake_cfg, fake_session).get_json("competitions.json")
    assert seen["timeout"] == sources.REQUEST_TIMEOUT_SECONDS


# --- manifest ------------------------------------------------------------------------------

def test_manifest_is_sha256_of_decompressed_bytes(fake_cfg, fake_session):
    f = Fetcher(fake_cfg, fake_session)
    f.get_json("competitions.json")
    assert f.manifest() == {
        "competitions.json": hashlib.sha256(fake_session.files["competitions.json"]).hexdigest()}


def test_manifest_includes_files_read_from_cache(fake_cfg, fake_session):
    Fetcher(fake_cfg, fake_session).get_json("competitions.json")
    again = Fetcher(fake_cfg, fake_session)
    again.get_json("competitions.json")
    assert fake_session.calls == 1
    assert list(again.manifest()) == ["competitions.json"]


def test_manifest_is_a_copy(fake_cfg, fake_session):
    f = Fetcher(fake_cfg, fake_session)
    f.get_json("competitions.json")
    f.manifest().clear()
    assert "competitions.json" in f.manifest()


def test_save_manifest_writes_json_next_to_the_cache(fake_cfg, fake_session):
    f = Fetcher(fake_cfg, fake_session)
    f.get_json("competitions.json")
    path = f.save_manifest()
    assert path == fake_cfg.raw_dir / fake_cfg.sha / "download_manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == f.manifest()
    assert [p.name for p in path.parent.iterdir() if p.suffix == ".tmp"] == []


def test_save_manifest_accumulates_across_runs(fake_cfg, fake_session):
    fake_session.serve("matches/43/106.json", [])
    first = Fetcher(fake_cfg, fake_session)
    first.get_json("competitions.json")
    first.save_manifest()
    second = Fetcher(fake_cfg, fake_session)
    second.get_json("matches/43/106.json")
    path = second.save_manifest()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved) == {"competitions.json", "matches/43/106.json"}
    assert saved["competitions.json"] == first.manifest()["competitions.json"]


def test_failed_manifest_write_keeps_the_old_manifest(fake_cfg, fake_session, monkeypatch):
    first = Fetcher(fake_cfg, fake_session)
    first.get_json("competitions.json")
    path = first.save_manifest()
    before = path.read_text(encoding="utf-8")
    fake_session.serve("matches/43/106.json", [])
    second = Fetcher(fake_cfg, fake_session)
    second.get_json("matches/43/106.json")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(sources.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        second.save_manifest()
    assert path.read_text(encoding="utf-8") == before
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "competitions.json.gz", "download_manifest.json", "matches"]


# --- fetch_slice ---------------------------------------------------------------------------

def test_fetch_slice_returns_matches_events_and_frames(
        fake_cfg, fake_session, slice_with_one_match_and_360):
    fake_session.serve("three-sixty/1.json", [{"event_uuid": "e1"}])
    raw = fetch_slice(Fetcher(fake_cfg, fake_session), slice_with_one_match_and_360)
    assert isinstance(raw, SliceRaw)
    assert raw.matches == [{"match_id": 1}]
    assert list(raw.events) == [1] and len(raw.events[1]) > 0
    assert raw.frames == {1: [{"event_uuid": "e1"}]}


def test_fetch_slice_skips_360_when_slice_has_none(fake_cfg, fake_session):
    fake_session.serve("matches/11/90.json", [{"match_id": 7}, {"match_id": 8}])
    fake_session.serve("events/7.json", [{"id": "a"}])
    fake_session.serve("events/8.json", [{"id": "b"}])
    league = Slice(11, 90, "liga", "league", False)
    raw = fetch_slice(Fetcher(fake_cfg, fake_session), league)
    assert set(raw.events) == {7, 8}
    assert raw.frames == {}
    assert not any("three-sixty" in url for url in fake_session.urls)


def test_fetch_slice_missing_events_file_still_raises(
        fake_cfg, fake_session, slice_with_one_match_and_360):
    fake_session.add_404("events/1.json")
    with pytest.raises(NotFound, match="events/1.json"):
        fetch_slice(Fetcher(fake_cfg, fake_session), slice_with_one_match_and_360)


# --- negative cache ------------------------------------------------------------------------

def test_a_three_sixty_404_is_cached_so_the_second_read_makes_no_network_call(
        fake_cfg, fake_session):
    f = Fetcher(fake_cfg, fake_session)
    for _ in range(2):
        with pytest.raises(NotFound, match="three-sixty/9.json"):
            f.get_json("three-sixty/9.json")
    assert fake_session.calls == 1


@pytest.mark.parametrize("rel_path", ["events/9.json", "matches/43/999.json", "competitions.json"])
def test_a_404_outside_three_sixty_is_not_cached(fake_cfg, fake_session, rel_path):
    """Only a missing 360 file is a tolerated 404; any other is likely a wrong URL, so retry it."""
    fake_session.add_404(rel_path)
    f = Fetcher(fake_cfg, fake_session)
    for _ in range(2):
        with pytest.raises(NotFound, match=rel_path):
            f.get_json(rel_path)
    assert fake_session.calls == 2
    assert not list((fake_cfg.raw_dir / fake_cfg.sha).glob("**/*.404"))


def test_a_cached_404_message_names_the_marker_file_and_says_to_delete_it(
        fake_cfg, fake_session):
    marker = fake_cfg.raw_dir / fake_cfg.sha / "three-sixty" / "9.json.404"
    with pytest.raises(NotFound):
        Fetcher(fake_cfg, fake_session).get_json("three-sixty/9.json")
    with pytest.raises(NotFound) as excinfo:
        Fetcher(fake_cfg, fake_session).get_json("three-sixty/9.json")
    message = str(excinfo.value)
    assert str(marker) in message
    assert "delete" in message and "retry" in message
    marker.unlink()                                    # doing as the message says retries the fetch
    fake_session.serve("three-sixty/9.json", [{"event_uuid": "e9"}])
    assert Fetcher(fake_cfg, fake_session).get_json("three-sixty/9.json") == [{"event_uuid": "e9"}]


def test_404_marker_sits_next_to_the_cache_path_and_survives_a_new_fetcher(
        fake_cfg, fake_session):
    with pytest.raises(NotFound):
        Fetcher(fake_cfg, fake_session).get_json("three-sixty/9.json")
    assert (fake_cfg.raw_dir / fake_cfg.sha / "three-sixty" / "9.json.404").is_file()
    with pytest.raises(NotFound, match="three-sixty/9.json"):
        Fetcher(fake_cfg, fake_session).get_json("three-sixty/9.json")
    assert fake_session.calls == 1


def test_404_marker_is_not_in_the_download_manifest(fake_cfg, fake_session):
    f = Fetcher(fake_cfg, fake_session)
    f.get_json("competitions.json")
    with pytest.raises(NotFound):
        f.get_json("three-sixty/9.json")
    assert list(f.manifest()) == ["competitions.json"]
    saved = json.loads(f.save_manifest().read_text(encoding="utf-8"))
    assert list(saved) == ["competitions.json"]


def test_404_marker_covers_only_its_own_path(fake_cfg, fake_session):
    fake_session.serve("events/2.json", [{"id": "e2"}])
    f = Fetcher(fake_cfg, fake_session)
    with pytest.raises(NotFound):
        f.get_json("events/1.json")
    assert f.get_json("events/2.json") == [{"id": "e2"}]
    assert fake_session.calls == 2


def test_failed_requests_are_not_cached_as_404(fake_cfg, fake_session):
    fake_session.fail_next(4, status=500)                 # the first try and all three retries
    f = Fetcher(fake_cfg, fake_session, sleep=lambda seconds: None)
    with pytest.raises(FetchError):
        f.get_json("competitions.json")
    assert not list((fake_cfg.raw_dir / fake_cfg.sha).glob("**/*.404"))
    assert f.get_json("competitions.json") == COMPETITIONS   # retried afresh, now served


def test_a_cached_file_wins_over_a_stray_404_marker(fake_cfg, fake_session):
    Fetcher(fake_cfg, fake_session).get_json("competitions.json")
    marker = fake_cfg.raw_dir / fake_cfg.sha / "competitions.json.404"
    marker.write_bytes(b"404\n")
    assert Fetcher(fake_cfg, fake_session).get_json("competitions.json") == COMPETITIONS
    assert fake_session.calls == 1


# --- catalogue discovery ---------------------------------------------------------------------


def test_discover_slices_from_catalogue(fake_cfg, fake_session):
    fake_session.serve("competitions.json", [
        {"competition_id": 43, "season_id": 106, "competition_name": "FIFA World Cup", "season_name": "2022"},
        {"competition_id": 12, "season_id": 27, "competition_name": "Serie A", "season_name": "2015/2016"}])
    fake_session.serve("matches/43/106.json", [{"match_id": 1, "match_status_360": "available"}])
    fake_session.serve("matches/12/27.json", [{"match_id": 2, "match_status_360": "scheduled"}])
    s = {(x.competition_id, x.season_id): x for x in discover_slices(Fetcher(fake_cfg, fake_session))}
    assert s[(43, 106)].has_360 is True and s[(12, 27)].has_360 is False
    assert s[(12, 27)].name == "Serie A 2015/2016" and s[(12, 27)].core is False


def serve_catalogue(session):
    session.serve("competitions.json", [
        {"competition_id": 55, "season_id": 43, "competition_name": "UEFA Euro", "season_name": "2020"},
        {"competition_id": 12, "season_id": 27, "competition_name": "Serie A", "season_name": "2015/2016"},
        {"competition_id": 37, "season_id": 4, "competition_name": "FA Women's Super League", "season_name": "2018/2019"}])
    session.serve("matches/55/43.json", [{"match_id": 1, "match_status_360": "scheduled"},
                                         {"match_id": 2, "match_status_360": "available"}])
    session.serve("matches/12/27.json", [{"match_id": 3, "match_status_360": "unavailable"}])
    session.serve("matches/37/4.json", [{"match_id": 4}, {"match_id": 5}])  # no status key at all


def test_discovered_slices_are_in_catalogue_order_without_overrides(fake_cfg, fake_session):
    serve_catalogue(fake_session)
    slices = discover_slices(Fetcher(fake_cfg, fake_session))
    assert slices == [
        Slice(55, 43, "UEFA Euro 2020", "league", True, None, core=False),
        Slice(12, 27, "Serie A 2015/2016", "league", False, None, core=False),
        Slice(37, 4, "FA Women's Super League 2018/2019", "league", False, None, core=False),
    ]


def test_a_slice_has_360_if_any_of_its_matches_is_available(fake_cfg, fake_session):
    serve_catalogue(fake_session)
    by_key = {(s.competition_id, s.season_id): s.has_360
              for s in discover_slices(Fetcher(fake_cfg, fake_session))}
    assert by_key == {(55, 43): True, (12, 27): False, (37, 4): False}


def test_discovery_reads_through_the_cache_and_lists_every_file_in_the_manifest(fake_cfg, fake_session):
    serve_catalogue(fake_session)
    fetcher = Fetcher(fake_cfg, fake_session)
    discover_slices(fetcher)
    assert fake_session.calls == 4  # the catalogue and one match list per entry, nothing else
    assert set(fetcher.manifest()) == {
        "competitions.json", "matches/55/43.json", "matches/12/27.json", "matches/37/4.json"}
    discover_slices(Fetcher(fake_cfg, fake_session))
    assert fake_session.calls == 4  # the second discovery is served from the cache


def test_write_catalogue_stores_the_slices_as_json_in_the_cache_of_the_pinned_revision(
    fake_cfg, fake_session
):
    serve_catalogue(fake_session)
    slices = discover_slices(Fetcher(fake_cfg, fake_session))
    path = write_catalogue(fake_cfg, slices)
    assert path == fake_cfg.raw_dir / fake_cfg.sha / "catalogue.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert [Slice(**entry) for entry in stored] == slices
    assert all(entry["core"] is False and entry["expected_matches"] is None for entry in stored)
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_a_failed_catalogue_write_keeps_the_old_file_and_leaves_no_temp_file(
    fake_cfg, fake_session, monkeypatch
):
    serve_catalogue(fake_session)
    slices = discover_slices(Fetcher(fake_cfg, fake_session))
    path = write_catalogue(fake_cfg, slices[:1])
    before = path.read_bytes()

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(sources.os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        write_catalogue(fake_cfg, slices)
    assert path.read_bytes() == before
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []
