"""Thin import and metrics boundary around vendored PDDLStream."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .problems import GeneratedProblem
from .streams import StreamContext

DOMAINS = Path(__file__).resolve().parent / "domains"
_SOLVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PDDLStreamResult:
    plan: tuple[Any, ...] | None
    cost: float
    certificate: Any
    metrics: dict[str, Any]

    @property
    def success(self) -> bool:
        return self.plan is not None


def _load_pddlstream(project_root: Path):
    configured = os.environ.get("PDDLSTREAM_PATH")
    root = Path(configured).expanduser() if configured else project_root / "third_party/pddlstream"
    if not root.is_absolute():
        root = project_root / root
    if not (root / "pddlstream").is_dir():
        raise RuntimeError(
            f"PDDLStream not found at {root}; run `git submodule update --init --recursive`"
        )
    build = root / "downward/builds/release/bin/downward"
    if not build.is_file():
        raise RuntimeError(
            f"Fast Downward not built at {build}; run `python3 {root / 'downward/build.py'}`"
        )
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from pddlstream.algorithms.meta import solve as pddlstream_solve
    from pddlstream.language.constants import PDDLProblem
    from pddlstream.language.generator import from_gen_fn, from_test

    fd_bin = os.environ.get("FD_BIN_PATH")
    if fd_bin:
        from pddlstream.algorithms import downward

        candidate = Path(fd_bin).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        executable = candidate if candidate.is_file() else candidate / "downward"
        if not executable.is_file():
            raise RuntimeError(f"Fast Downward binary not found: {executable}")
        downward.FD_BIN = str(executable.parent)
    return pddlstream_solve, PDDLProblem, from_gen_fn, from_test


def solve(
    problem: GeneratedProblem,
    streams: StreamContext,
    project_root: Path,
    algorithm: str = "adaptive",
    planner: str = "ff-eager",
    max_time: float = 300.0,
) -> PDDLStreamResult:
    pddlstream_solve, PDDLProblem, from_gen_fn, from_test = _load_pddlstream(
        project_root
    )
    domain_pddl = (DOMAINS / f"{problem.domain}_domain.pddl").read_text(encoding="utf-8")
    stream_pddl = (DOMAINS / "streams.pddl").read_text(encoding="utf-8")
    stream_map = {
        "sample-grasp": from_gen_fn(streams.sample_grasp),
        "sample-pick-pose": from_gen_fn(streams.sample_pick_pose),
        "sample-place-pose": from_gen_fn(streams.sample_place_pose),
        "sample-stack-pose": from_gen_fn(streams.sample_stack_pose),
        "inverse-kinematics": from_gen_fn(streams.inverse_kinematics),
        "plan-transit": from_gen_fn(streams.plan_transit),
        "test-motion": from_test(streams.test_motion),
        "plan-pick": from_gen_fn(streams.plan_pick),
        "plan-place": from_gen_fn(streams.plan_place),
        "plan-stack": from_gen_fn(streams.plan_stack),
    }
    constant_map = {"table": "table"} if problem.domain == "blocksworld" else {}
    pddl_problem = PDDLProblem(
        domain_pddl,
        constant_map,
        stream_pddl,
        stream_map,
        list(problem.init),
        problem.goal,
    )
    started = time.perf_counter()
    with _SOLVE_LOCK, tempfile.TemporaryDirectory(prefix="ctamp_pddlstream_") as work_dir:
        previous_directory = Path.cwd()
        try:
            os.chdir(work_dir)
            raw_plan, cost, certificate = pddlstream_solve(
                pddl_problem,
                algorithm=algorithm,
                planner=planner,
                max_time=max_time,
                unit_costs=True,
                verbose=False,
                clean=True,
            )
        finally:
            os.chdir(previous_directory)
    elapsed = time.perf_counter() - started
    plan = None if raw_plan is None or raw_plan is False else tuple(raw_plan)
    metrics = {
        "solver": "pddlstream",
        "algorithm": algorithm,
        "planner": planner,
        "planning_time_seconds": elapsed,
        "plan_length": 0 if plan is None else len(plan),
        "stream": streams.metrics(),
    }
    return PDDLStreamResult(plan, float(cost), certificate, metrics)
