import gzip
import hashlib
import json

import pytest
import requests

from setpiece import sources
from setpiece.config import Slice
from setpiece.sources import Fetcher, FetchError, NotFound, SliceRaw, fetch_slice

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
