"""Command-line entry point: one subcommand per pipeline stage."""

import argparse
import sys
from pathlib import Path

from setpiece.config import ConfigError, load_config
from setpiece.qa import QAResult, run_qa
from setpiece.rq1.review import run_review_export_rq1
from setpiece.stages.forecast import run_forecast
from setpiece.stages.ingest import run_fetch, run_ingest
from setpiece.stages.rq1 import run_rq1
from setpiece.stages.rq2 import run_rq2

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
RUNNERS = {
    "fetch": run_fetch,
    "ingest": run_ingest,
    "qa": run_qa,
    "rq1": run_rq1,
    "forecast": run_forecast,
    "rq2": run_rq2,
}


# `review-export <target>`: the review material of one research question. The others are not built.
REVIEW_TARGETS = ("rq1", "rq2", "rq3")
REVIEW_RUNNERS = {"rq1": run_review_export_rq1}


def _qa_exit_code(result: QAResult) -> int:
    """Print what QA found; the run fails (1) only on structural errors, not on warnings."""
    for error in result.errors:
        print(f"qa error: {error}", file=sys.stderr)
    print("qa warnings: " + " ".join(f"{name}={n}" for name, n in result.warnings.items()))
    print(f"qa shot corners: {result.shot_corner_count}")
    return 1 if result.errors else 0


# What a stage's return value means for the exit code. A stage not listed exits 0 once it returns.
EXIT_CODES = {
    "qa": _qa_exit_code,
}

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
        if stage == "review-export":
            sub.add_argument("target", choices=REVIEW_TARGETS, help="whose review material")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "review-export":
        name, run = f"review-export {args.target}", REVIEW_RUNNERS.get(args.target)
    else:
        name, run = args.stage, RUNNERS.get(args.stage)
    if run is None:  # nothing to run, so the config is not read either
        print(f"stage '{name}' not implemented yet", file=sys.stderr)
        return 2
    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"setpiece: {exc}", file=sys.stderr)
        return 1
    result = run(cfg)
    exit_code = EXIT_CODES.get(args.stage)
    return exit_code(result) if exit_code is not None else 0
