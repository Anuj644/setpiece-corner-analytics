"""Load ``configs/default.yaml`` into a typed ``Config`` that every pipeline stage reads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

STATUSES = ("locked", "starting_choice", "design_choice", "kept")
REQUIRED_KEYS = ("source", "paths", "seed", "slices", "params", "dossier")


class ConfigError(ValueError):
    """The config file is missing something required or holds an invalid entry."""


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
    doc: dict[str, Any] = field(default_factory=dict, repr=False, compare=False, kw_only=True)

    def p(self, name: str) -> Any:
        """Return the value of parameter ``name``."""
        return self.params[name].value

    def config_hash(self) -> str:
        """First 12 hex characters of the sha256 of the canonical dump of the parsed YAML."""
        canonical = yaml.safe_dump(self.doc, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    """Read the YAML at ``path``; ``root`` is the parent of its ``configs/`` directory."""
    path = Path(path).resolve()
    root = path.parent.parent
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

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
        doc=doc,
    )
