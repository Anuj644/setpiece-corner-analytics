"""Download raw StatsBomb Open Data JSON at the pinned commit and cache it gzip-compressed."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from setpiece.config import Config, Slice

# Network behaviour is infrastructure, not a method parameter, so it is not in the config register.
RETRY_BACKOFF_SECONDS = (1, 2, 4)  # one entry per retry: three retries after the first attempt
REQUEST_TIMEOUT_SECONDS = 60

MANIFEST_NAME = "download_manifest.json"

# What a damaged cache file looks like when read back: cut short, not gzip, bad deflate or not JSON.
_CORRUPT_CACHE = (gzip.BadGzipFile, EOFError, zlib.error, json.JSONDecodeError)


class NotFound(Exception):
    """The pinned revision has no file at this path (HTTP 404)."""


class FetchError(Exception):
    """A request kept failing (not a 404) after every retry."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file in the same directory, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class Fetcher:
    """Read ``data/<rel_path>`` JSON files from the pinned commit, through a local gzip cache."""

    def __init__(
        self,
        cfg: Config,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = cfg
        self.session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._manifest: dict[str, str] = {}

    @property
    def _cache_root(self) -> Path:
        return self.cfg.raw_dir / self.cfg.sha

    def url(self, rel_path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}/{self.cfg.sha}/data/{rel_path}"

    def get_json(self, rel_path: str) -> Any:
        """Return the parsed JSON for ``rel_path`` (relative to ``data/``), fetching it if uncached."""
        cached = self._read_cache(rel_path)
        if cached is None:
            body = self._download(rel_path)
            obj = json.loads(body)  # parse before caching, so a bad body never reaches the cache
            _atomic_write(self._cache_root / f"{rel_path}.gz", gzip.compress(body, mtime=0))
        else:
            body, obj = cached
        self._manifest[rel_path] = _sha256(body)
        return obj

    def manifest(self) -> dict[str, str]:
        """``rel_path`` -> sha256 of the decompressed bytes, for every file this fetcher has read."""
        return dict(self._manifest)

    def save_manifest(self) -> Path:
        """Merge this fetcher's manifest into ``download_manifest.json`` next to the cache."""
        path = self._cache_root / MANIFEST_NAME
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        merged = {**saved, **self._manifest}
        text = json.dumps(merged, indent=2, sort_keys=True) + "\n"
        _atomic_write(path, text.encode("utf-8"))
        return path

    def _read_cache(self, rel_path: str) -> tuple[bytes, Any] | None:
        path = self._cache_root / f"{rel_path}.gz"
        try:
            body = gzip.decompress(path.read_bytes())
            return body, json.loads(body)
        except FileNotFoundError:
            return None
        except _CORRUPT_CACHE:
            path.unlink(missing_ok=True)
            return None

    def _download(self, rel_path: str) -> bytes:
        url = self.url(rel_path)
        last_problem = "no attempt made"
        for delay in (*RETRY_BACKOFF_SECONDS, None):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                last_problem = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response.content
                if response.status_code == 404:
                    raise NotFound(f"{rel_path} not found at pinned revision ({url})")
                last_problem = f"HTTP {response.status_code}"
            if delay is not None:
                self._sleep(delay)
        raise FetchError(
            f"{rel_path}: gave up after {len(RETRY_BACKOFF_SECONDS)} retries ({last_problem})"
        )


@dataclass
class SliceRaw:
    matches: list[dict]
    events: dict[int, list[dict]]
    frames: dict[int, list[dict] | None]  # None: the slice has 360 data but not for this match


def fetch_slice(fetcher: Fetcher, slice: Slice) -> SliceRaw:
    """Fetch a slice's match list and each match's events (and 360 frames if the slice has them)."""
    matches = fetcher.get_json(f"matches/{slice.competition_id}/{slice.season_id}.json")
    events: dict[int, list[dict]] = {}
    frames: dict[int, list[dict] | None] = {}
    for match in matches:
        match_id = match["match_id"]
        events[match_id] = fetcher.get_json(f"events/{match_id}.json")
        if slice.has_360:
            try:
                frames[match_id] = fetcher.get_json(f"three-sixty/{match_id}.json")
            except NotFound:
                frames[match_id] = None
    return SliceRaw(matches=matches, events=events, frames=frames)
