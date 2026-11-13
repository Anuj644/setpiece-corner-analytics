"""Load ``configs/default.yaml`` into a typed ``Config`` that every pipeline stage reads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

STATUSES = ("locked", "starting_choice", "design_choice", "kept")
KINDS = ("league", "tournament")
REQUIRED_KEYS = ("source", "paths", "seed", "slices", "params", "dossier")


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
    # The parsed YAML document, kept only so config_hash() can hash the file content.
    _doc: dict[str, Any] = field(default_factory=dict, repr=False, compare=False, kw_only=True)

    def p(self, name: str) -> Any:
        """Return the value of parameter ``name``."""
        return self.params[name].value

    def config_hash(self) -> str:
        """First 12 hex characters of the sha256 of the canonical dump of the parsed YAML."""
        canonical = yaml.safe_dump(self._doc, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    """Read the YAML at ``path``; ``root`` is the parent of its ``configs/`` directory."""
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
        if entry.get("kind") not in KINDS:
            raise ConfigError(
                f"slice {entry.get('name')!r} has unknown kind {entry.get('kind')!r}; "
                f"expected one of {KINDS}"
            )

    paths = doc["paths"]
    return Config(
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
