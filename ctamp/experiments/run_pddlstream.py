"""PDDLStream challenge orchestrator using run_scene_v2 as executor."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import yaml

from ..pddlstream.plan_adapter import AdaptedPlan, adapt_plan, serializable_action
from ..pddlstream.problems import GeneratedProblem
from ..pddlstream.problems.blocksworld_generator import generate_problem as generate_blocksworld
from ..pddlstream.problems.kitchen_generator import generate_problem as generate_kitchen
from ..pddlstream.solver import PDDLStreamResult, solve
from ..pddlstream.streams import StreamContext
from ..simulation.panda_ik import PandaIKSolver
from .run_scene_v2 import run as run_scene_v2


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _generate(
    domain: str, config: dict[str, Any], num_objects: int, seed: int
) -> GeneratedProblem:
    if domain == "blocksworld":
        return generate_blocksworld(config, num_objects=num_objects, seed=seed)
    if domain == "kitchen":
        return generate_kitchen(config, num_objects=num_objects, seed=seed)
    raise ValueError(f"unsupported PDDLStream domain: {domain}")


def _planning_ik(
    problem: GeneratedProblem, project_root: Path, dry_run: bool
) -> tuple[PandaIKSolver | None, tuple[float, float, float]]:
    solver = PandaIKSolver.from_scene_config(problem.scene_config, project_root)
    initial_ee_xyz = tuple(float(value) for value in solver.site_position())
    return (None if dry_run else solver), initial_ee_xyz


def _placement_config(
    base: dict[str, Any],
    object_poses: dict[str, list[float]],
    object_id: str,
    target_pose: tuple[float, float, float],
    phase_index: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["scene"]["scene_id"] = f"{base['scene']['scene_id']}_placement_{phase_index:03d}"
    for obj in config["objects"]:
        if obj["id"] in object_poses:
            obj["pose"] = list(object_poses[obj["id"]])
    config["task"]["target_objects"] = [object_id]
    config["task"]["preserve_order"] = True
    config["grouped_tidy"]["axis"] = "z"
    config["grouped_tidy"]["slot_prefix"] = "pddlstream"
    config["tidy_groups"] = [
        {
            "id": f"placement_{phase_index:03d}",
            "color": "mixed",
            "objects": [object_id],
            "center": list(target_pose),
            "positions": {object_id: list(target_pose)},
        }
    ]
    return config


def _placement_sequence_config(
    problem: GeneratedProblem, adapted: AdaptedPlan
) -> dict[str, Any]:
    config = copy.deepcopy(problem.scene_config)
    config["scene"]["scene_id"] = f"{config['scene']['scene_id']}_continuous"
    config["task"]["target_objects"] = adapted.object_order
    config["task"]["preserve_order"] = True
    config["task"]["placement_sequence"] = [
        {
            "object_id": step.object_id,
            "slot_name": f"pddlstream_{index:03d}_{step.surface_id}",
            "target_pose": list(step.target_pose),
        }
        for index, step in enumerate(adapted.placements, start=1)
    ]
    config["tidy_groups"] = []
    return config


def _execute_continuous_placements(
    problem: GeneratedProblem,
    adapted: AdaptedPlan,
    output: Path,
    project_root: Path,
    max_retries: int | None,
    viewer: bool,
) -> dict[str, Any]:
    config = _placement_sequence_config(problem, adapted)
    config_path = output / "continuous_sequence.yaml"
    _write_yaml(config_path, config)
    result = run_scene_v2(
        config_path,
        output / "continuous_sequence",
        max_retries=max_retries,
        project_root=project_root,
        viewer=viewer,
    )
    per_object = result.get("per_object_result", [])
    phases: list[dict[str, Any]] = []
    for index, step in enumerate(adapted.placements, start=1):
        placement_result = per_object[index - 1] if index <= len(per_object) else {}
        target_pose = list(step.target_pose)
        placement_error = placement_result.get("placement_error")
        measured_pose = (
            [target_pose[axis] + float(placement_error[axis]) for axis in range(3)]
            if placement_error is not None and len(placement_error) == 3
            else target_pose
        )
        phases.append(
            {
                "index": index,
                "action": step.action,
                "object_id": step.object_id,
                "surface_id": step.surface_id,
                "target_pose": target_pose,
                "measured_pose": measured_pose,
                "solution_found": bool(placement_result.get("success", False)),
                "metrics": placement_result,
            }
        )
    return {
        "execution_mode": "continuous_scene",
        "solution_found": bool(result.get("solution_found", False)),
        "completed_placements": int(result.get("completed_objects", 0)),
        "target_placements": len(adapted.placements),
        "phases": phases,
        "continuous_metrics": result,
    }


def _execute_placements(
    problem: GeneratedProblem,
    adapted: AdaptedPlan,
    output: Path,
    project_root: Path,
    max_retries: int | None,
    viewer: bool,
) -> dict[str, Any]:
    object_poses = {
        obj["id"]: list(obj["pose"][:3]) for obj in problem.scene_config["objects"]
    }
    expected_poses = copy.deepcopy(object_poses)
    phases: list[dict[str, Any]] = []
    for index, step in enumerate(adapted.placements, start=1):
        target_pose = list(step.target_pose)
        if step.surface_id in object_poses:
            target_pose = [
                target_pose[axis]
                + object_poses[step.surface_id][axis]
                - expected_poses[step.surface_id][axis]
                for axis in range(3)
            ]
        phase_config = _placement_config(
            problem.scene_config,
            object_poses,
            step.object_id,
            tuple(target_pose),
            index,
        )
        phase_path = output / f"placement_{index:03d}.yaml"
        _write_yaml(phase_path, phase_config)
        result = run_scene_v2(
            phase_path,
            output / f"placement_{index:03d}",
            max_retries=max_retries,
            max_objects=1,
            project_root=project_root,
            viewer=viewer,
        )
        phases.append(
            {
                "index": index,
                "action": step.action,
                "object_id": step.object_id,
                "surface_id": step.surface_id,
                "symbolic_target_pose": list(step.target_pose),
                "target_pose": target_pose,
                "solution_found": bool(result.get("solution_found", False)),
                "metrics": result,
            }
        )
        if not result.get("solution_found", False):
            break
        per_object = result.get("per_object_result", [])
        placement_error = per_object[0].get("placement_error") if per_object else None
        measured_pose = (
            [target_pose[axis] + float(placement_error[axis]) for axis in range(3)]
            if placement_error is not None and len(placement_error) == 3
            else target_pose
        )
        phases[-1]["measured_pose"] = measured_pose
        object_poses[step.object_id] = measured_pose
        expected_poses[step.object_id] = list(step.target_pose)
    return {
        "solution_found": len(phases) == len(adapted.placements)
        and all(phase["solution_found"] for phase in phases),
        "completed_placements": sum(phase["solution_found"] for phase in phases),
        "target_placements": len(adapted.placements),
        "phases": phases,
    }


def _result_metrics(
    problem: GeneratedProblem,
    solved: PDDLStreamResult,
    adapted: AdaptedPlan | None,
    dry_run: bool,
    execution: dict[str, Any] | None,
    initial_ee_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    symbolic_success = solved.success
    execution_success = execution is None or bool(execution["solution_found"])
    return {
        "planner_arm": "pddlstream",
        "domain": problem.domain,
        "seed": problem.seed,
        "num_objects": len(problem.object_ids),
        "robot": {"initial_end_effector_xyz": list(initial_ee_xyz)},
        "dry_run": dry_run,
        "symbolic_solution_found": symbolic_success,
        "solution_found": symbolic_success and execution_success,
        "planning": solved.metrics,
        "plan_cost": solved.cost if math.isfinite(solved.cost) else None,
        "object_order": [] if adapted is None else adapted.object_order,
        "state_actions": [] if adapted is None else list(adapted.state_actions),
        "problem": problem.metadata,
        "execution": execution,
    }


def run(
    domain: str,
    config_path: Path,
    output: Path,
    num_objects: int = 3,
    seed: int = 0,
    max_retries: int | None = None,
    project_root: Path | None = None,
    viewer: bool = False,
    dry_run: bool = False,
    max_planning_time: float = 300.0,
) -> dict[str, Any]:
    project_root = project_root or config_path.resolve().parents[2]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    problem = _generate(domain, config, num_objects, seed)
    output.mkdir(parents=True, exist_ok=True)
    _write_yaml(output / "generated_scene.yaml", problem.scene_config)
    _write_json(output / "generated_problem.json", problem.metadata)

    ik_solver, initial_ee_xyz = _planning_ik(problem, project_root, dry_run)
    stream_context = StreamContext(problem, ik_solver=ik_solver, dry_run=dry_run)
    solved = solve(
        problem,
        stream_context,
        project_root=project_root,
        max_time=max_planning_time,
    )
    adapted = adapt_plan(domain, solved.plan) if solved.plan is not None else None
    raw_plan = [] if solved.plan is None else [serializable_action(action) for action in solved.plan]
    _write_json(output / "pddlstream_plan.json", raw_plan)
    if adapted is not None:
        _write_json(
            output / "adapted_plan.json",
            {
                "object_order": adapted.object_order,
                "placements": [
                    {
                        "action_index": step.action_index,
                        "action": step.action,
                        "object_id": step.object_id,
                        "surface_id": step.surface_id,
                        "target_pose": list(step.target_pose),
                    }
                    for step in adapted.placements
                ],
                "state_actions": list(adapted.state_actions),
            },
        )
    execution = None
    if solved.success and not dry_run and adapted is not None:
        execution = (
            _execute_continuous_placements(
                problem, adapted, output, project_root, max_retries, viewer
            )
            if problem.domain == "kitchen"
            else _execute_placements(
                problem, adapted, output, project_root, max_retries, viewer
            )
        )
    metrics = _result_metrics(
        problem, solved, adapted, dry_run, execution, initial_ee_xyz
    )
    _write_json(output / "metrics.json", metrics)
    return metrics
