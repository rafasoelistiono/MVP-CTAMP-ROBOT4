from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ctamp.experiments.run_pddlstream import run as run_pddlstream

from .common import exit_with_errors
from .run_simulation import ROOT_DIR

DOMAINS = ("blocksworld", "kitchen")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CTAMP PDDLStream challenge planner.")
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", default=ROOT_DIR / "runs", type=Path)
    parser.add_argument("--num-objects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-planning-time", type=float, default=300.0)
    parser.add_argument("--max-retries-per-object", type=int)
    parser.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    config = args.config
    if not config.exists():
        raise FileNotFoundError(f"scene config not found: {config}")
    output = args.output or args.log_dir / (
        f"{args.domain}_pddlstream_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    metrics = run_pddlstream(
        args.domain,
        config,
        output,
        num_objects=args.num_objects,
        seed=args.seed,
        max_retries=args.max_retries_per_object,
        project_root=ROOT_DIR,
        viewer=bool(args.viewer),
        dry_run=args.dry_run,
        max_planning_time=args.max_planning_time,
    )
    sys.stdout.write(json.dumps(metrics, indent=2) + "\n")
    return 0 if metrics["solution_found"] else 2


def cli() -> None:
    exit_with_errors(main)


if __name__ == "__main__":
    cli()
