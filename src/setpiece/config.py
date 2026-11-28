"""Load ``configs/default.yaml`` into a typed ``Config`` that every pipeline stage reads."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

STATUSES = ("locked", "starting_choice", "design_choice", "kept")
KINDS = ("league", "tournament")
REQUIRED_KEYS = ("source", "paths", "seed", "slices", "params", "dossier")

# Corpus expansion: the parameter that switches it on, the cached list of discovered slices
# (inside the raw cache of the pinned revision) and the stage message while that list is missing.
EXPANSION_PARAM = "corpus_expansion"
CATALOGUE_NAME = "catalogue.json"
CATALOGUE_PENDING_MESSAGE = "catalogue not built; run fetch first"
OVERRIDE_FIELDS = ("kind", "expected_matches")
_OVERRIDE_KEY = re.compile(r"[0-9]+/[0-9]+")  # "<competition_id>/<season_id>"


class ConfigError(ValueError):
    """The config file is missing something required or holds an invalid entry."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a mapping key written twice (PyYAML keeps the last silently)."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r}", key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


@dataclass
class Slice:
    competition_id: int
    season_id: int
    name: str
    kind: Literal["league", "tournament"]
    has_360: bool
    expected_matches: int | None = None
    core: bool = True


@dataclass
class Param:
    value: Any
    status: Literal["locked", "starting_choice", "design_choice", "kept"]
    note: str


@dataclass
class Config:
    sha: str
    base_url: str
    root: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    paper_dir: Path
    seed: int
    slices: list[Slice]
    params: dict[str, Param]
    dossier_teams: list[dict]
    # True while corpus expansion is on but catalogue.json is not built yet: ``slices`` then holds
    # the core slices only, and ``fetch`` is the only stage that may run (see ``expand_slices``).
    catalogue_pending: bool = False
    # The parsed YAML document, kept only so config_hash() can hash the file content.
    _doc: dict[str, Any] = field(default_factory=dict, repr=False, compare=False, kw_only=True)

    def p(self, name: str) -> Any:
        """Return the value of parameter ``name``."""
        return self.params[name].value

    @property
    def core_slices(self) -> list[Slice]:
        """The slices defined in ``default.yaml``, in their configured order."""
        return [s for s in self.slices if s.core]

    def config_hash(self) -> str:
        """First 12 hex characters of the sha256 of the canonical dump of the parsed YAML.

        With corpus expansion on, the parsed overrides file and the parsed ``catalogue.json`` (or
        None before it is built) are hashed with it, so different overrides or catalogues never
        share a hash. With expansion off, only the parsed YAML is hashed.
        """
        document: Any = self._doc
        settings = _expansion_settings(self)
        if settings is not None:
            document = {
                "config": self._doc,
                "overrides": _read_overrides(_overrides_path(self, settings)),
                "catalogue": _read_catalogue(self),
            }
        canonical = yaml.safe_dump(document, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _require_kind(label: str, kind: Any) -> None:
    if kind not in KINDS:
        raise ConfigError(f"{label} has unknown kind {kind!r}; expected one of {KINDS}")


def load_config(path: str | Path) -> Config:
    """Read the YAML at ``path``; ``root`` is the parent of its ``configs/`` directory.

    With ``corpus_expansion`` on, the slices are expanded with the cached catalogue (see
    ``expand_slices``); with it off, neither the overrides file nor ``catalogue.json`` is read.
    """
    path = Path(path).resolve()
    if path.parent.name != "configs":
        raise ConfigError(f"{path}: a config file must sit in a directory named 'configs'")
    root = path.parent.parent
    try:
        doc = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    missing = [key for key in REQUIRED_KEYS if key not in doc]
    if missing:
        raise ConfigError(f"{path}: missing required top-level key(s): {', '.join(missing)}")

    params = {}
    for name, entry in doc["params"].items():
        if not isinstance(entry, dict) or "value" not in entry:
            raise ConfigError(f"param {name!r} has no value")
        status = entry.get("status")
        if status not in STATUSES:
            raise ConfigError(
                f"param {name!r} has unknown status {status!r}; expected one of {STATUSES}"
            )
        params[name] = Param(entry["value"], status, entry.get("note", ""))

    for entry in doc["slices"]:
        _require_kind(f"slice {entry.get('name')!r}", entry.get("kind"))

    paths = doc["paths"]
    cfg = Config(
        sha=doc["source"]["sha"],
        base_url=doc["source"]["base_url"],
        root=root,
        raw_dir=root / paths["raw"],
        processed_dir=root / paths["processed"],
        reports_dir=root / paths["reports"],
        paper_dir=root / paths["paper"],
        seed=doc["seed"],
        slices=[Slice(**s) for s in doc["slices"]],
        params=params,
        dossier_teams=list(doc["dossier"].get("teams") or []),
        _doc=doc,
    )
    return expand_slices(cfg)


# --- corpus expansion ---------------------------------------------------------------------------


def catalogue_path(cfg: Config) -> Path:
    """Where the discovered slices are cached: next to the raw cache of the pinned revision."""
    return cfg.raw_dir / cfg.sha / CATALOGUE_NAME


def expand_slices(cfg: Config) -> Config:
    """``cfg`` with the core slices followed by every cached catalogue slice that is not core.

    Does nothing (returns ``cfg``) while ``corpus_expansion.enabled`` is false. Otherwise the core
    slices of ``cfg`` come first in their order, then the slices of ``catalogue.json`` in its
    order, skipping any whose (competition_id, season_id) is already core (``default.yaml`` wins),
    each with its entry of the overrides file applied. Applying it again gives the same config.
    ``ConfigError`` if the overrides file or the catalogue is missing a required piece or invalid.

    If ``catalogue.json`` does not exist yet, the result has the core slices only and
    ``catalogue_pending = True``: ``fetch`` then discovers the catalogue and calls this again.
    """
    settings = _expansion_settings(cfg)
    if settings is None:
        return cfg
    overrides_file = _overrides_path(cfg, settings)
    overrides = _parse_overrides(overrides_file, _read_overrides(overrides_file))
    core = cfg.core_slices
    catalogue = _read_catalogue(cfg)
    if catalogue is None:
        return dataclasses.replace(cfg, slices=core, catalogue_pending=True)
    taken = {(s.competition_id, s.season_id) for s in core}
    discovered = []
    for slice in _catalogue_slices(catalogue_path(cfg), catalogue):
        key = (slice.competition_id, slice.season_id)
        if key in taken:
            continue
        taken.add(key)
        discovered.append(dataclasses.replace(slice, **overrides.get(key, {})))
    return dataclasses.replace(cfg, slices=[*core, *discovered], catalogue_pending=False)


def _expansion_settings(cfg: Config) -> dict[str, Any] | None:
    """The ``corpus_expansion`` value if expansion is enabled, else None."""
    param = cfg.params.get(EXPANSION_PARAM)
    if param is None:
        return None
    if not isinstance(param.value, dict):
        raise ConfigError(f"param {EXPANSION_PARAM!r} must be a mapping with enabled and overrides")
    return param.value if param.value.get("enabled") else None


def _overrides_path(cfg: Config, settings: dict[str, Any]) -> Path:
    """The overrides file, resolved against the repo root (an absolute path stays absolute)."""
    overrides = settings.get("overrides")
    if not isinstance(overrides, str) or not overrides:
        raise ConfigError(f"param {EXPANSION_PARAM!r} needs an 'overrides' file path")
    return cfg.root / overrides


def _read_overrides(path: Path) -> dict:
    """The parsed overrides file as written (an empty file is an empty mapping)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: cannot read the catalogue overrides file: {exc}") from exc
    try:
        doc = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if doc is None:
        return {}
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: expected a mapping of '<competition_id>/<season_id>' entries")
    return doc


def _parse_overrides(path: Path, doc: dict) -> dict[tuple[int, int], dict[str, Any]]:
    """Validate the overrides; return ``(competition_id, season_id) -> {kind, expected_matches}``."""
    parsed: dict[tuple[int, int], dict[str, Any]] = {}
    for key, entry in doc.items():
        if not isinstance(key, str) or not _OVERRIDE_KEY.fullmatch(key):
            raise ConfigError(
                f"{path}: malformed key {key!r}; expected '<competition_id>/<season_id>' integers"
            )
        label = f"override {key!r}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: {label} must be a mapping with {OVERRIDE_FIELDS}")
        unknown = sorted(set(entry) - set(OVERRIDE_FIELDS))
        if unknown:
            raise ConfigError(f"{path}: {label} has unknown field(s) {unknown}")
        _require_kind(f"{path}: {label}", entry.get("kind"))
        expected = entry.get("expected_matches")
        if expected is not None and (type(expected) is not int or expected <= 0):
            raise ConfigError(
                f"{path}: {label} has expected_matches {expected!r}; expected a positive integer"
                " or null"
            )
        competition_id, season_id = (int(part) for part in key.split("/"))
        if (competition_id, season_id) in parsed:
            raise ConfigError(f"{path}: {label} repeats another key for the same slice")
        parsed[(competition_id, season_id)] = {"kind": entry["kind"], "expected_matches": expected}
    return parsed


def _read_catalogue(cfg: Config) -> Any | None:
    """The parsed ``catalogue.json`` as written, or None if it does not exist yet."""
    path = catalogue_path(cfg)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: cannot read the catalogue: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: not valid JSON ({exc}); delete it and run fetch") from exc


def _catalogue_slices(path: Path, catalogue: Any) -> list[Slice]:
    """The slices in ``catalogue.json``, always ``core=False`` whatever the file says."""
    if not isinstance(catalogue, list):
        raise ConfigError(f"{path}: expected a list of slices")
    slices = []
    for entry in catalogue:
        try:
            slice = Slice(**{**entry, "core": False})
        except TypeError as exc:  # not a mapping, or fields missing or unknown
            raise ConfigError(f"{path}: invalid slice entry {entry!r}: {exc}") from exc
        _require_kind(f"{path}: slice {slice.name!r}", slice.kind)
        slices.append(slice)
    return slices
