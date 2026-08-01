from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.models import Action, Edge, JointSpace, RobotState, Vertex, WorkspaceState
from ..motion_planning.mujoco import MuJoCoMotionPlanner
from ..tmm.multigraph import TaskMotionMultigraph


@dataclass
class ObjectExecution:
    ik_success: bool = True
    ik_reason: str | None = None
    transit_joint_waypoints: list[list[float]] = field(default_factory=list)
    transfer_joint_waypoints: list[list[float]] = field(default_factory=list)
    grasp_style: str | None = None
    physical_grip_success: bool | None = None
    physical_tidy_success: bool | None = None
    physical_stage: str | None = None
    physical_lift_height: float | None = None
    placement_error: list[float] | None = None


GRASP_APPROACHES = {
    "top": (0.0, 0.0, -1.0),
    "side_pos_x": (-1.0, 0.0, 0.0),
    "side_neg_x": (1.0, 0.0, 0.0),
    "side_pos_y": (0.0, -1.0, 0.0),
    "side_neg_y": (0.0, 1.0, 0.0),
}

DEFAULT_PLACEMENT_SUCCESS_TOLERANCE_XY = 0.07


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def placement_within_tolerance(error_xyz: Iterable[float], config: dict) -> bool:
    error = tuple(float(value) for value in error_xyz)
    if len(error) < 2:
        raise ValueError("placement error must contain at least x and y")
    tolerance = float(
        config.get("physical_execution", {}).get(
            "placement_success_tolerance_xy",
            DEFAULT_PLACEMENT_SUCCESS_TOLERANCE_XY,
        )
    )
    if tolerance < 0:
        raise ValueError("placement_success_tolerance_xy must be non-negative")
    return math.hypot(error[0], error[1]) <= tolerance


def route_type(motion: object) -> str:
    return motion.metadata["route_type"]


def object_reach_ok(config: dict, obj: dict, start: list[float]) -> bool:
    reach = math.dist(config["robot"]["base_xy"], start)
    return bool(obj.get("reachable", True)) and (
        float(config["robot"]["reach_min_xy"])
        <= reach
        <= float(config["robot"]["reach_max_xy"])
    )


def probe_transfer(
    planner: MuJoCoMotionPlanner,
    start: list[float],
    goal: tuple[float, ...],
    retry_limit: int,
):
    attempts = 0
    collision_failures = 0
    motion = None
    while attempts <= retry_limit:
        attempts += 1
        motion = planner.plan_xy(start, goal)
        if motion.success:
            break
        collision_failures += 1
    return motion, collision_failures, max(0, attempts - 1)


def completion_status(
    per_object: list[dict], config: dict
) -> tuple[bool, int, float, str, bool]:
    all_objects_solved = all(item["success"] for item in per_object)
    completed_objects = sum(item["success"] for item in per_object)
    completion_ratio = completed_objects / len(per_object) if per_object else 1.0
    physical_config = config.get("physical_execution", {})
    completion_policy = physical_config.get("completion_policy", "strict")
    minimum_ratio = float(physical_config.get("minimum_completion_ratio", 1.0))
    accepted_completion = (
        all_objects_solved
        if completion_policy == "strict"
        else completion_ratio >= minimum_ratio
    )
    return (
        all_objects_solved,
        completed_objects,
        completion_ratio,
        completion_policy,
        accepted_completion,
    )


def plan_action(
    object_id: str,
    slot: object,
    route: str,
    transit: object,
    motion: object,
    execution: ObjectExecution,
    grip_width: float,
) -> dict:
    return {
        "object_id": object_id,
        "slot": slot.name,
        "route_type": route,
        "transit_route_type": route_type(transit),
        "transit_waypoints": transit.waypoints,
        "transfer_waypoints": motion.waypoints,
        "transit_joint_waypoints": execution.transit_joint_waypoints,
        "transfer_joint_waypoints": execution.transfer_joint_waypoints,
        "waypoints": motion.waypoints,
        "z": slot.position[2],
        "grasp_width": grip_width,
    }


def per_object_result(
    object_id: str,
    slot: object,
    object_success: bool,
    route: str,
    retries: int,
    reason: str | None,
    transit: object,
    reach: float,
    motion: object,
    reach_ok: bool,
    execution: ObjectExecution,
) -> dict:
    return {
        "object_id": object_id,
        "slot": slot.name,
        "success": object_success,
        "route_type": route,
        "retries_used": retries,
        "reason": reason,
        "transit_route_type": route_type(transit),
        "transit_length": transit.length,
        "ik_success": execution.ik_success,
        "ik_reason": execution.ik_reason,
        "grasp_style": execution.grasp_style,
        "start_reach": reach,
        "motion_length": motion.length,
        "reach_ok": reach_ok,
        "physical_stage": execution.physical_stage,
        "physical_grip_success": execution.physical_grip_success,
        "physical_tidy_success": execution.physical_tidy_success,
        "physical_lift_height": execution.physical_lift_height,
        "placement_error": execution.placement_error,
    }


def build_ordered_tmm(
    object_ids: list[str], motions: dict[str, object]
) -> TaskMotionMultigraph:
    graph = TaskMotionMultigraph()
    workspace = WorkspaceState()
    robot = RobotState(active_arm="left")
    root = Vertex(
        vertex_id="root", robot_state=robot, workspace_state=workspace, is_root=True
    )
    graph.add_vertex(root)
    previous = root.vertex_id
    joint_spaces = (
        JointSpace(name="left_arm", joints=[f"panda_joint{i}" for i in range(1, 8)]),
        JointSpace(
            name="left_arm_redundant", joints=[f"panda_joint{i}" for i in range(1, 8)]
        ),
    )
    for object_id in object_ids:
        pick_id, place_id = f"pick_{object_id}", f"place_{object_id}"
        graph.add_vertex(
            Vertex(vertex_id=pick_id, robot_state=robot, workspace_state=workspace)
        )
        graph.add_vertex(
            Vertex(vertex_id=place_id, robot_state=robot, workspace_state=workspace)
        )
        motion = motions[object_id]
        for joint_space in joint_spaces:
            graph.add_edge(
                Edge(
                    source=previous,
                    target=pick_id,
                    action=Action(
                        action_id=f"transit_{object_id}",
                        action_type="transit",
                        object_id=object_id,
                        arm="left",
                    ),
                    joint_space=joint_space,
                    motion_plan=motion,
                    cost=0.0,
                    flag_motion_planned=True,
                )
            )
            graph.add_edge(
                Edge(
                    source=pick_id,
                    target=place_id,
                    action=Action(
                        action_id=f"transfer_{object_id}",
                        action_type="transfer",
                        object_id=object_id,
                        arm="left",
                    ),
                    joint_space=joint_space,
                    motion_plan=motion,
                    cost=motion.length if motion.success else 1e6,
                    flag_motion_planned=True,
                )
            )
        previous = place_id
    goal = Vertex(
        vertex_id="goal", robot_state=robot, workspace_state=workspace, is_goal=True
    )
    graph.add_vertex(goal)
    for joint_space in joint_spaces:
        graph.add_edge(
            Edge(
                source=previous,
                target="goal",
                action=Action(
                    action_id="done", action_type="transit", object_id="", arm="left"
                ),
                joint_space=joint_space,
                cost=0.0,
                flag_motion_planned=True,
            )
        )
    return graph


def interpolate_polyline(points: Iterable[list[float]], frames_per_segment: int = 8):
    pts = list(points)
    for start, goal in zip(pts, pts[1:], strict=False):
        for frame in range(frames_per_segment):
            t = frame / frames_per_segment
            yield [
                start[0] + (goal[0] - start[0]) * t,
                start[1] + (goal[1] - start[1]) * t,
            ]
    if pts:
        yield pts[-1]


def dense_xyz(points: list[list[float]], z: float, steps: int = 4):
    return [(xy[0], xy[1], z) for xy in interpolate_polyline(points, steps)]


def interpolate_joints(points: list[list[float]], frames_per_segment: int = 1):
    return interpolate_polyline_nd(points, frames_per_segment)


def interpolate_polyline_nd(points, frames_per_segment: int):
    pts = list(points)
    for start, goal in zip(pts, pts[1:], strict=False):
        for frame in range(frames_per_segment):
            alpha = frame / frames_per_segment
            yield [a + (b - a) * alpha for a, b in zip(start, goal, strict=True)]
    if pts:
        yield list(pts[-1])
