"""Deprecated custom-search CTAMP v3 stacking runner.

V3 keeps v1/v2 runnable and adds a real Algorithm-1-shaped pass before
execution: build TMM, run A*, motion-plan edges during expansion, confirm the
candidate path, then hand the confirmed stacking order to the v2 executor.

Use :mod:`ctamp.experiments.run_pddlstream` for benchmark planning. This module
remains runnable as the custom TMM/A* reference path.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..domain.models import (
    Action,
    Edge,
    JointSpace,
    MotionPlan,
    ObjectState,
    Pose,
    RobotState,
    Shape,
    Vertex,
    WorkspaceState,
)
from ..motion_planning.mujoco import MuJoCoMotionPlanner
from ..planning.confirmation import CompletePlan, EmptyPlan, confirm_solution
from ..search.tmm_astar import SearchVisitor, TMMAStar, TMMHeuristic
from ..tmm.multigraph import TaskMotionMultigraph
from .run_scene_v2 import run as run_scene_v2
from .run_stacking_v2 import build_phase_configs


STACKING_V3_STRATEGY = "ctamp_algorithm1_confirmed_stack_then_v2_execute"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _polyline_length(points: tuple[tuple[float, float], ...]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:], strict=False))


@dataclass(frozen=True)
class EdgeMotionSpec:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]


@dataclass(frozen=True)
class DewantoCostConfig:
    max_joint_dim: int = 7
    length_scale: float = 1.0
    clearance_scale: float = 1.0
    time_scale: float = 1.0
    max_smoothing_time: float = 0.25
    max_result_prediction: float = 1_000_000.0


@dataclass
class V3LearningSample:
    action_id: str
    source_id: str
    target_id: str
    joint_space_dim: int
    success: bool
    cost: float


class DewantoEdgeCost:
    """Deprecated custom-search edge cost retained for the v3 reference path."""

    def __init__(self, config: DewantoCostConfig | None = None) -> None:
        self.config = config or DewantoCostConfig()

    def compute(self, edge: Edge, plan: MotionPlan, attempt_index: int) -> float:
        c = self.config
        success = bool(plan.success)
        result_components = {
            "smoothness": 0.0,
            "length": 0.0,
            "clearance": 0.0,
        }
        if success:
            # MotionPlan.smoothness is a quality score here, so invert it to cost.
            result_components = {
                "smoothness": max(0.0, 1.0 - float(plan.smoothness)),
                "length": self._normalize(float(plan.length), c.length_scale),
                "clearance": self._normalize(
                    1.0 / max(float(plan.clearance), 1e-6),
                    c.clearance_scale,
                ),
            }
        result_cost = sum(result_components.values())

        n_a = max(1, int(attempt_index))
        process_components = {
            "iteration": math.exp((n_a - 1) / n_a),
            "joint_space": math.exp(edge.joint_space.dimension / c.max_joint_dim),
            "planning_time": self._normalize(float(plan.planning_time), c.time_scale),
            "penalty": 0.0
            if success
            else c.max_smoothing_time + c.max_result_prediction,
        }
        process_cost = sum(process_components.values())
        total = result_cost + process_cost
        plan.metadata["dewanto_cost"] = {
            "result_cost": result_cost,
            "process_cost": process_cost,
            "total_cost": total,
            "result_components": result_components,
            "process_components": process_components,
        }
        return total

    @staticmethod
    def _normalize(value: float, scale: float) -> float:
        return max(0.0, value) / max(float(scale), 1e-6)


class OnlineMeanRemainingCost(TMMHeuristic):
    """Deprecated online heuristic retained for the v3 reference path."""

    def __init__(self, initial_mean: float = 1.0) -> None:
        self.graph: TaskMotionMultigraph | None = None
        self.mean_cost = float(initial_mean)
        self.update_count = 0

    def set_graph(self, graph: TaskMotionMultigraph) -> None:
        self.graph = graph

    def update(self, cost: float) -> None:
        if not math.isfinite(cost) or cost >= 1_000_000.0:
            return
        self.update_count += 1
        self.mean_cost += (cost - self.mean_cost) / self.update_count

    def evaluate(self, vertex: Vertex, goal_ids: set[str]) -> float:
        if vertex.vertex_id in goal_ids:
            return 0.0
        steps = self._unit_steps_to_goal(vertex.vertex_id, goal_ids)
        if steps is None:
            return float("inf")
        return max(steps * self.mean_cost, 1e-6)

    def _unit_steps_to_goal(self, start_id: str, goal_ids: set[str]) -> int | None:
        if self.graph is None:
            return None
        frontier = [(start_id, 0)]
        seen = {start_id}
        for vertex_id, depth in frontier:
            if vertex_id in goal_ids:
                return depth
            for edge in self.graph.get_outgoing_edges(vertex_id):
                if edge.target in seen:
                    continue
                seen.add(edge.target)
                frontier.append((edge.target, depth + 1))
        return None


class StackingV3MotionVisitor(SearchVisitor):
    """Deprecated motion visitor retained for the v3 reference path."""

    def __init__(
        self,
        planner: MuJoCoMotionPlanner,
        edge_specs: dict[str, EdgeMotionSpec],
        heuristic: OnlineMeanRemainingCost,
        cost_calculator: DewantoEdgeCost | None = None,
    ) -> None:
        self.planner = planner
        self.edge_specs = edge_specs
        self.heuristic = heuristic
        self.cost_calculator = cost_calculator or DewantoEdgeCost()
        self.samples: list[V3LearningSample] = []

    def on_expand(self, graph: TaskMotionMultigraph, vertex: Vertex) -> None:
        by_target: dict[str, list[Edge]] = {}
        for edge in graph.get_outgoing_edges(vertex.vertex_id):
            by_target.setdefault(edge.target, []).append(edge)

        for edges in by_target.values():
            edges.sort(key=lambda edge: edge.joint_space.dimension)
            for attempt_index, edge in enumerate(edges, start=1):
                if edge.flag_motion_planned:
                    break
                plan = self._plan(edge)
                edge.flag_motion_planned = True
                edge.motion_plan = plan
                edge.cost = self.cost_calculator.compute(edge, plan, attempt_index)
                self.samples.append(
                    V3LearningSample(
                        action_id=edge.action.action_id,
                        source_id=edge.source,
                        target_id=edge.target,
                        joint_space_dim=edge.joint_space.dimension,
                        success=plan.success,
                        cost=edge.cost,
                    )
                )
                if plan.success:
                    self.heuristic.update(edge.cost)
                    break

    def on_edge(self, edge: Edge, motion_plan: MotionPlan | None) -> None:
        pass

    def _plan(self, edge: Edge) -> MotionPlan:
        spec = self.edge_specs[edge.action.action_id]
        plan = self.planner.plan_xy(spec.start_xy, spec.goal_xy)
        if plan.success:
            return plan
        return self._plan_x_corridor(spec, plan)

    def _plan_x_corridor(
        self, spec: EdgeMotionSpec, fallback: MotionPlan
    ) -> MotionPlan:
        probe = self.planner.probe
        started = time.perf_counter()
        margin = 0.12
        candidates: list[tuple[tuple[float, float], ...]] = []
        for xmin, xmax, _, _ in probe.rectangles:
            for x in (xmin - margin, xmax + margin):
                candidates.append(
                    (
                        spec.start_xy,
                        (x, spec.start_xy[1]),
                        (x, spec.goal_xy[1]),
                        spec.goal_xy,
                    )
                )
        valid = [path for path in candidates if probe.path_clear(path)]
        if not valid:
            return fallback
        path = min(valid, key=_polyline_length)
        return MotionPlan(
            success=True,
            waypoints=[list(point) for point in path],
            length=_polyline_length(path),
            smoothness=0.8,
            clearance=probe.clearance,
            planning_time=fallback.planning_time + time.perf_counter() - started,
            iterations=max(1, len(path) - 1),
            metadata={
                "route_type": "v3_x_corridor",
                "reason": "v3 reach-safe side corridor",
                "fallback_reason": fallback.metadata.get("reason"),
                "validation_level": "geometric_2d_probe_v3",
            },
        )


def _workspace_from_config(
    config: dict[str, Any], target_order: list[str]
) -> WorkspaceState:
    target_ids = set(target_order)
    default_size = tuple(float(v) for v in config["geometry"]["cube_size_xyz"])
    objects: dict[str, ObjectState] = {}
    for raw in config["objects"]:
        size = tuple(float(v) for v in raw.get("size_xyz", default_size))
        pose = raw["pose"]
        objects[raw["id"]] = ObjectState(
            object_id=raw["id"],
            pose=Pose(x=float(pose[0]), y=float(pose[1])),
            shape=Shape(
                type=str(raw.get("class", "box")),
                width=size[0],
                height=size[1],
                radius=max(size[0], size[1]) / 2.0,
            ),
            movable=raw["id"] in target_ids,
        )
    return WorkspaceState(objects=objects)


def _robot_from_config(config: dict[str, Any]) -> RobotState:
    qpos = config.get("robot", {}).get("physical_start_qpos", [])
    return RobotState(
        joint_values={f"panda_joint{i + 1}": float(q) for i, q in enumerate(qpos)},
        active_arm="left",
    )


def _joint_spaces() -> tuple[JointSpace, JointSpace]:
    return (
        JointSpace(name="panda_xy_probe", joints=["x", "y"]),
        JointSpace(
            name="panda_left_arm", joints=[f"panda_joint{i}" for i in range(1, 8)]
        ),
    )


def _home_xy(config: dict[str, Any]) -> tuple[float, float]:
    robot = config["robot"]
    return (
        float(robot["base_xy"][0]) + float(robot["reach_min_xy"]) + 0.02,
        float(robot["base_xy"][1]),
    )


def _build_v3_tmm(
    stack_config: dict[str, Any],
) -> tuple[TaskMotionMultigraph, dict[str, EdgeMotionSpec]]:
    target_order = list(stack_config["task"]["target_objects"])
    workspace = _workspace_from_config(stack_config, target_order)
    robot = _robot_from_config(stack_config)
    graph = TaskMotionMultigraph()
    edge_specs: dict[str, EdgeMotionSpec] = {}
    graph.add_vertex(
        Vertex(vertex_id="root", robot_state=robot, workspace_state=workspace, is_root=True)
    )
    previous_id = "root"
    current_xy = _home_xy(stack_config)
    objects = {obj["id"]: obj for obj in stack_config["objects"]}
    target_positions = stack_config["tidy_groups"][0]["positions"]

    for object_id in target_order:
        pick_id = f"pick_{object_id}"
        place_id = f"place_{object_id}"
        graph.add_vertex(
            Vertex(vertex_id=pick_id, robot_state=robot, workspace_state=workspace)
        )
        graph.add_vertex(
            Vertex(vertex_id=place_id, robot_state=robot, workspace_state=workspace)
        )

        start_xy = tuple(float(v) for v in objects[object_id]["pose"][:2])
        goal_xy = tuple(float(v) for v in target_positions[object_id][:2])
        _add_action_edges(
            graph,
            edge_specs,
            previous_id,
            pick_id,
            f"transit_{object_id}",
            "transit",
            object_id,
            current_xy,
            start_xy,
        )
        _add_action_edges(
            graph,
            edge_specs,
            pick_id,
            place_id,
            f"transfer_{object_id}",
            "transfer",
            object_id,
            start_xy,
            goal_xy,
        )
        previous_id = place_id
        current_xy = goal_xy

    graph.add_vertex(
        Vertex(vertex_id="goal", robot_state=robot, workspace_state=workspace, is_goal=True)
    )
    _add_action_edges(
        graph,
        edge_specs,
        previous_id,
        "goal",
        "done",
        "transit",
        "",
        current_xy,
        current_xy,
    )
    return graph, edge_specs


def _add_action_edges(
    graph: TaskMotionMultigraph,
    edge_specs: dict[str, EdgeMotionSpec],
    source: str,
    target: str,
    action_id: str,
    action_type: str,
    object_id: str,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> None:
    edge_specs[action_id] = EdgeMotionSpec(start_xy=start_xy, goal_xy=goal_xy)
    for joint_space in _joint_spaces():
        graph.add_edge(
            Edge(
                source=source,
                target=target,
                action=Action(
                    action_id=action_id,
                    action_type="transfer" if action_type == "transfer" else "transit",
                    object_id=object_id,
                    arm="left",
                ),
                joint_space=joint_space,
                cost=1_000_000.0,
                flag_motion_planned=False,
            )
        )


def _run_algorithm1(
    stack_config: dict[str, Any],
) -> tuple[dict[str, Any], CompletePlan | None]:
    graph, edge_specs = _build_v3_tmm(stack_config)
    heuristic = OnlineMeanRemainingCost()
    heuristic.set_graph(graph)
    visitor = StackingV3MotionVisitor(
        MuJoCoMotionPlanner(stack_config), edge_specs, heuristic
    )
    search = TMMAStar(heuristic=heuristic, visitor=visitor, max_time=30.0)
    search_result = search.search(graph)
    complete_plan: CompletePlan | None = None
    confirmation: CompletePlan | EmptyPlan | None = None
    if search_result.success:
        confirmation = confirm_solution(search_result.path_edges)
        if isinstance(confirmation, CompletePlan):
            complete_plan = confirmation

    metrics = {
        "search_success": search_result.success,
        "confirmation_success": isinstance(confirmation, CompletePlan),
        "search_error": search_result.error,
        "confirmation_error": None
        if confirmation is None or isinstance(confirmation, CompletePlan)
        else confirmation.reason,
        "path_vertex_ids": search_result.path_vertex_ids,
        "path_edges": [_edge_dict(edge) for edge in search_result.path_edges],
        "total_cost": complete_plan.total_cost if complete_plan is not None else None,
        "tmm_vertices": graph.vertex_count,
        "tmm_edges": graph.edge_count,
        "tmm_is_dag": graph.is_dag,
        "nodes_expanded": search_result.nodes_expanded,
        "nodes_generated": search_result.nodes_generated,
        "time_elapsed": search_result.time_elapsed,
        "heuristic": {
            "name": "online_mean_remaining_path_cost",
            "mean_edge_cost": heuristic.mean_cost,
            "online_updates": heuristic.update_count,
        },
        "learning_samples": [sample.__dict__ for sample in visitor.samples],
        "algorithm1": {
            "search_then_confirmation": True,
            "empty_on_failed_confirmation": complete_plan is None,
        },
    }
    return metrics, complete_plan


def _edge_dict(edge: Edge) -> dict[str, Any]:
    cost_meta = {}
    if edge.motion_plan is not None:
        cost_meta = edge.motion_plan.metadata.get("dewanto_cost", {})
    return {
        "source": edge.source,
        "target": edge.target,
        "action_id": edge.action.action_id,
        "action_type": edge.action.action_type,
        "object_id": edge.action.object_id,
        "joint_space": edge.joint_space.name,
        "joint_space_dim": edge.joint_space.dimension,
        "motion_planned": edge.flag_motion_planned,
        "motion_success": bool(edge.motion_plan and edge.motion_plan.success),
        "cost": edge.cost,
        "dewanto_cost": cost_meta,
    }


def _confirmed_object_order(plan: CompletePlan | None) -> list[str]:
    if plan is None:
        return []
    return [
        action.object_id
        for action in plan.actions
        if action.action_type == "transfer" and action.object_id
    ]


def _apply_confirmed_order(stack_config: dict[str, Any], order: list[str]) -> None:
    if not order:
        return
    stack_config["task"]["target_objects"] = order
    for group in stack_config.get("tidy_groups", []):
        group["objects"] = [
            object_id for object_id in order if object_id in group["objects"]
        ]


def _metrics(
    summary: dict[str, Any],
    ctamp_metrics: dict[str, Any],
    plan: CompletePlan | None,
    dry_run: bool,
    continuous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    confirmed = ctamp_metrics["confirmation_success"]
    return {
        "ctamp_version": "v3",
        "task": "stack",
        "strategy": STACKING_V3_STRATEGY,
        "dry_run": dry_run,
        "solution_found": confirmed
        if continuous is None
        else confirmed and bool(continuous.get("solution_found")),
        "confirmed_order": _confirmed_object_order(plan),
        "ctamp_v3": ctamp_metrics,
        "continuous_stack": continuous,
        **summary,
    }


def run(
    config_path: Path,
    output: Path,
    max_retries: int | None = None,
    max_objects: int | None = None,
    project_root: Path | None = None,
    viewer: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    safe_zone_config, stack_config, summary = build_phase_configs(
        config, max_objects=max_objects
    )
    stack_config = copy.deepcopy(stack_config)

    ctamp_metrics, complete_plan = _run_algorithm1(stack_config)
    confirmed_order = _confirmed_object_order(complete_plan)
    _apply_confirmed_order(stack_config, confirmed_order)

    _write_yaml(output / "safe_zone_preview.yaml", safe_zone_config)
    stack_path = output / "continuous_stack_v3.yaml"
    _write_yaml(stack_path, stack_config)
    _write_json(output / "ctamp_v3_plan.json", ctamp_metrics)
    _write_json(output / "stacking_plan.json", summary)

    if dry_run or complete_plan is None:
        metrics = _metrics(summary, ctamp_metrics, complete_plan, dry_run=dry_run)
        _write_json(output / "metrics.json", metrics)
        return metrics

    continuous = run_scene_v2(
        stack_path,
        output / "continuous_stack",
        max_retries=max_retries,
        max_objects=max_objects,
        project_root=project_root,
        viewer=viewer,
    )
    metrics = _metrics(
        summary,
        ctamp_metrics,
        complete_plan,
        dry_run=False,
        continuous=continuous,
    )
    _write_json(output / "metrics.json", metrics)
    return metrics
