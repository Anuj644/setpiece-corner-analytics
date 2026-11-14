"""Command-line entry point: one subcommand per pipeline stage."""

import argparse
import sys
from pathlib import Path

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

    if args.stage == "rq1":
        from setpiece.config import load_config
        from setpiece.stages.rq1 import run_rq1

        cfg = load_config(args.config)
        run_rq1(cfg)
        return 0

    if args.stage == "forecast":
        from setpiece.config import load_config
        from setpiece.stages.forecast import run_forecast

        cfg = load_config(args.config)
        run_forecast(cfg)
        return 0

    print(f"stage '{args.stage}' not implemented yet", file=sys.stderr)
    return 2
