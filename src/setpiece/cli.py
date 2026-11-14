"""Command-line entry point: one subcommand per pipeline stage."""

import argparse
import sys
from pathlib import Path

from setpiece.config import ConfigError, load_config
from setpiece.stages.ingest import run_fetch, run_ingest

STAGES = (
    "fetch",
    "ingest",
    "qa",
    "rq1",
    "forecast",
    "rq2",
    "rq3",
    "coverage",
    "dossier",
    "paper-assets",
    "all",
    "review-export",
)

# The stages that exist so far: each is called with the loaded Config. The rest exit with 2.
RUNNERS = {"fetch": run_fetch, "ingest": run_ingest}

# src/setpiece/cli.py -> repo root
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setpiece",
        description="Corner-kick analytics pipeline on StatsBomb Open Data.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    for stage in STAGES:
        sub = subparsers.add_parser(stage, help=f"run the '{stage}' stage")
        sub.add_argument(
            "--config",
            type=Path,
            default=DEFAULT_CONFIG,
            help="path to the YAML config file (default: %(default)s)",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = RUNNERS.get(args.stage)
    if run is None:  # nothing to run, so the config is not read either
        print(f"stage '{args.stage}' not implemented yet", file=sys.stderr)
        return 2
    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"setpiece: {exc}", file=sys.stderr)
        return 1
    run(cfg)
    return 0
