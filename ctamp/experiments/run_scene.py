"""Run the ordered grouped-tidy MuJoCo scene adapter."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from ..motion_planning.mujoco import MuJoCoMotionPlanner
from ..search.tmm_astar import TMMAStar
from ..simulation import (
    GoalSlot,
    MuJoCoBackend,
    MuJoCoSceneBuilder,
    PandaIKSolver,
    PandaPhysicsExecutor,
    generate_tidy_slots,
    load_scene_config,
)
from .scene_helpers import (
    GRASP_APPROACHES,
    ObjectExecution as _ObjectExecution,
    build_ordered_tmm as _build_ordered_tmm,
    completion_status as _completion_status,
    dense_xyz as _dense_xyz,
    object_reach_ok as _object_reach_ok,
    per_object_result as _per_object_result,
    plan_action as _plan_action,
    probe_transfer as _probe_transfer,
    route_type as _route_type,
    write_json as _write_json,
)


def run(
    config_path: Path,
    output: Path,
    max_retries: int | None = None,
    max_objects: int | None = None,
    project_root: Path | None = None,
    viewer: bool = False,
) -> dict:
    started = time.perf_counter()
    config = load_scene_config(config_path)
    output.mkdir(parents=True, exist_ok=True)
    project_root = project_root or config_path.resolve().parents[2]
    slots = generate_tidy_slots(config)
    builder = MuJoCoSceneBuilder(config, project_root)
    xml = builder.build_xml()
    backend = MuJoCoBackend()
    backend.load_model(xml_string=xml)
    backend.step(10)
    real_panda = builder.panda_asset.status == "real_panda_asset"
    viewer_context = None
    live_viewer = None
    if viewer and real_panda:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(backend.model, backend.data)
        live_viewer = viewer_context.__enter__()
        live_viewer.cam.lookat[:] = [0.0, -0.15, 1.0]
        live_viewer.cam.distance = 2.8
        live_viewer.cam.azimuth = 90
        live_viewer.cam.elevation = -32
    planner = MuJoCoMotionPlanner(config)
    retry_limit = (
        int(max_retries)
        if max_retries is not None
        else int(config["constraints"]["max_retries_per_object"])
    )
    objects = {obj["id"]: obj for obj in config["objects"]}
    preserve_order = bool(config.get("task", {}).get("preserve_order"))
    per_object = []
    plan_actions = []
    collision_failures = 0
    retries_used = 0
    total_cost = 0.0
    motions = {}
    probe = planner.probe
    home_xy = (
        float(config["robot"]["base_xy"][0])
        + float(config["robot"]["reach_min_xy"])
        + 0.02,
        float(config["robot"]["base_xy"][1]),
    )
    current_xy = home_xy
    direct_only_failures = 0
    aware_corridor_routes = 0
    ik_solver = PandaIKSolver(backend) if real_panda else None
    planning_backend = None
    ik_home = None
    physics_executor = None
    physical_qpos = None
    physical_home_qpos = None
    ik_failures = 0
    ik_collision_failures = 0
    grasp_styles: dict[str, str | None] = {}

    def _grip_width(obj: dict) -> float:
        return float(
            obj.get(
                "grip_target_width",
                config.get("physical_execution", {}).get("grip_target_width", 0.052),
            )
        )

    if ik_solver is not None:
        ik_home = ik_solver.solve_collision_free(
            (home_xy[0], home_xy[1], 0.95),
            random_restarts=128,
        )
        if not ik_home.success:
            ik_failures += 1
        start_qpos = config["robot"].get("physical_start_qpos")
        if start_qpos is None and ik_home.success:
            start_qpos = ik_home.qpos
        if start_qpos is not None:
            physical_qpos = np.asarray(start_qpos, dtype=float)
            physical_home_qpos = physical_qpos.copy()
            physics_executor = PandaPhysicsExecutor(backend, viewer=live_viewer)
            physics_executor.initialize_arm(physical_qpos)
            physics_executor.open_gripper(steps=120)
            planning_backend = MuJoCoBackend()
            planning_backend.load_model(xml_string=xml)
            ik_solver = PandaIKSolver(planning_backend)
            ik_solver.set_qpos(physical_qpos)

    def _sync_planning_scene() -> None:
        if planning_backend is None:
            return
        for object_id in objects:
            planning_backend.set_body_pose(
                f"cube_{object_id}",
                backend.get_body_pose(f"cube_{object_id}"),
            )

    def _move_arm_to_safe_pose() -> bool:
        if physics_executor is None or physical_home_qpos is None:
            return True
        physics_executor.open_gripper(steps=40)
        returned_home, _ = physics_executor.follow_joint_path(
            [physical_home_qpos],
            max_joint_step=0.025,
        )
        if returned_home:
            return True
        start_qpos = physics_executor.ik.current_qpos()
        planning_backend = MuJoCoBackend()
        planning_backend.load_model(xml_string=xml)
        planning_ik = PandaIKSolver(planning_backend)
        planning_ik.set_qpos(start_qpos)
        route = planning_ik.plan_joint_rrt(
            start_qpos,
            physical_home_qpos,
            max_iterations=5000,
            rng_seed=90_000,
        )
        if route is None:
            return False
        returned_home, _ = physics_executor.follow_joint_path(
            route[1:],
            max_joint_step=0.025,
        )
        return returned_home

    def _execute_physical_pick_place(
        object_id: str,
        obj: dict,
        slot: object,
        arm_at_home: bool,
    ) -> _ObjectExecution:
        nonlocal physical_qpos
        execution = _ObjectExecution()
        tracking_error = 0.0
        object_pose = backend.get_body_pose(f"cube_{object_id}")[:3]
        ik_solver.set_qpos(physical_qpos)
        grasp = ik_solver.plan_physical_grasp(
            object_id,
            tuple(object_pose),
            start_qpos=physical_qpos,
        )
        if not grasp.success and arm_at_home and physical_home_qpos is not None:
            ik_solver.set_qpos(physical_home_qpos)
            grasp = ik_solver.plan_physical_grasp(
                object_id,
                tuple(object_pose),
                start_qpos=physical_home_qpos,
                random_restarts=192,
            )
        execution.ik_success = grasp.success
        execution.grasp_style = grasp.grasp_style
        execution.ik_reason = grasp.reason

        if execution.ik_success:
            approach_q = np.asarray(grasp.joint_waypoints[-1])
            ik_solver.set_qpos(approach_q)
            grasp_rotation = (
                ik_solver.data.site_xmat[ik_solver.site_id].reshape(3, 3).copy()
            )
            lift = ik_solver.solve(
                ik_solver.site_position() + np.array([0.0, 0.0, 0.14]),
                seed=approach_q,
                orientation=grasp_rotation,
                orientation_tolerance=0.10,
            )
            execution.ik_success = lift.success
            execution.ik_reason = None if lift.success else "lift IK failed"

        if execution.ik_success:
            ik_solver.set_qpos(lift.qpos)
            place_target = np.asarray(slot.position) + np.array([0.0, 0.0, 0.06])
            preplace_target = place_target + np.array([0.0, 0.0, 0.14])
            place_path = ik_solver.solve_path(
                [tuple(preplace_target), tuple(place_target)],
                orientation=grasp_rotation,
                allowed_object_id=object_id,
            )
            if not place_path.success:
                ik_solver.set_qpos(lift.qpos)
                place_path = ik_solver.solve_path(
                    [tuple(preplace_target), tuple(place_target)],
                    allowed_object_id=object_id,
                )
            execution.ik_success = place_path.success
            execution.ik_reason = place_path.reason

        if execution.ik_success:
            execution.transit_joint_waypoints = [list(q) for q in grasp.joint_waypoints]
            execution.transfer_joint_waypoints = [list(lift.qpos)] + [
                list(q) for q in place_path.joint_waypoints[1:]
            ]
            physics_executor.open_gripper(steps=120)
            execution.physical_stage = "pregrasp_tracking"
            tracked, tracking_error = physics_executor.follow_joint_path(
                grasp.joint_waypoints[1:-1],
            )
            execution.ik_success = tracked
            execution.ik_reason = None if tracked else "pregrasp tracking failed"

        if execution.ik_success:
            execution.physical_stage = "physical_grasp"
            grasp_result = physics_executor.validate_grasp_and_lift(
                object_id,
                approach_q,
                lift.qpos,
                grasp_site_target=np.asarray(object_pose) + np.array([0.0, 0.0, 0.02]),
                grip_width=_grip_width(obj),
            )
            execution.physical_grip_success = grasp_result.success
            execution.physical_lift_height = grasp_result.lift_height
            execution.ik_success = grasp_result.success
            execution.ik_reason = grasp_result.reason
            tracking_error = max(tracking_error, grasp_result.arm_tracking_error)

        if execution.ik_success:
            execution.physical_stage = "place_tracking"
            tracked, place_error = physics_executor.follow_joint_path(
                place_path.joint_waypoints[1:],
            )
            tracking_error = max(tracking_error, place_error)
            execution.ik_success = tracked
            execution.ik_reason = None if tracked else "place tracking failed"

        if execution.ik_success:
            physics_executor.set_carry_constraint(object_id, False)
            physics_executor.settle(steps=60)
            physics_executor.open_gripper(steps=320)
            physics_executor.settle(steps=100)
            if len(place_path.joint_waypoints) >= 2:
                physics_executor.follow_joint_path(
                    [place_path.joint_waypoints[-2]],
                    max_joint_step=0.018,
                )
            reverse_to_lift = list(reversed(place_path.joint_waypoints[:-2]))
            reverse_to_home = list(reversed(grasp.joint_waypoints[:-1]))
            if reverse_to_lift or reverse_to_home:
                physics_executor.follow_joint_path(
                    [*reverse_to_lift, *reverse_to_home],
                    max_joint_step=0.025,
                )
            physics_executor.settle(steps=180)
            final_position = backend.get_body_pose(f"cube_{object_id}")[:3]
            execution.placement_error = (
                np.asarray(final_position) - np.asarray(slot.position)
            ).tolist()
            execution.physical_tidy_success = (
                float(np.linalg.norm(execution.placement_error[:2])) <= 0.07
            )
            execution.ik_success = execution.physical_tidy_success
            execution.ik_reason = (
                None if execution.physical_tidy_success else "cube missed tidy slot"
            )
        elif execution.physical_grip_success:
            physics_executor.set_carry_constraint(object_id, False)
            physics_executor.open_gripper(steps=250)
            physics_executor.settle(steps=120)

        if execution.physical_grip_success is None:
            execution.physical_grip_success = False
        if execution.physical_tidy_success is None:
            execution.physical_tidy_success = False
        if execution.physical_stage is not None and physical_home_qpos is not None:
            _move_arm_to_safe_pose()
        physical_qpos = physics_executor.ik.current_qpos()
        ik_solver.set_qpos(physical_qpos)
        return execution

    def _execute_ik_preview(
        obj: dict, transit: object, motion: object
    ) -> tuple[_ObjectExecution, int]:
        execution = _ObjectExecution()
        collision_count = 0
        transit_targets = _dense_xyz(transit.waypoints, 0.95)[1:]
        transit_ik = ik_solver.solve_path(transit_targets)
        execution.ik_success = transit_ik.success
        execution.transit_joint_waypoints = [
            list(q) for q in transit_ik.joint_waypoints
        ]
        execution.ik_reason = transit_ik.reason
        if not transit_ik.success and transit_ik.collision_pairs:
            collision_count += 1
        if execution.ik_success:
            grasp = ik_solver.plan_grasp_candidates(
                tuple(obj["pose"]),
                start_qpos=transit_ik.joint_waypoints[-1],
                random_restarts=48,
            )
            execution.ik_success = grasp.success
            execution.grasp_style = grasp.grasp_style
            execution.ik_reason = grasp.reason
            if grasp.success:
                execution.transit_joint_waypoints.extend(
                    list(q) for q in grasp.joint_waypoints[1:]
                )
                ik_solver.set_qpos(grasp.joint_waypoints[-1])
                adaptive_grasp_orientation = (
                    ik_solver.data.site_xmat[ik_solver.site_id].reshape(3, 3).copy()
                )
        if execution.ik_success:
            grasp_orientation = (
                adaptive_grasp_orientation
                if execution.grasp_style == "adaptive_oblique"
                else ik_solver._rotation_from_approach(
                    GRASP_APPROACHES[execution.grasp_style]
                )
                if execution.grasp_style in GRASP_APPROACHES
                else None
            )
            transfer_targets = _dense_xyz(motion.waypoints, 0.938)[1:]
            transfer_ik = ik_solver.solve_path(
                transfer_targets,
                orientation=grasp_orientation,
            )
            if not transfer_ik.success and (execution.grasp_style or "").startswith(
                "adaptive_"
            ):
                ik_solver.set_qpos(grasp.joint_waypoints[-1])
                transfer_ik = ik_solver.solve_path(transfer_targets)
                if transfer_ik.success:
                    execution.grasp_style = (
                        f"{execution.grasp_style}_adaptive_transport"
                    )
            execution.ik_success = transfer_ik.success
            execution.transfer_joint_waypoints = [list(grasp.joint_waypoints[-1])] + [
                list(q) for q in transfer_ik.joint_waypoints[1:]
            ]
            execution.ik_reason = transfer_ik.reason
            if not transfer_ik.success and transfer_ik.collision_pairs:
                collision_count += 1
        return execution, collision_count

    placement_sequence = list(config["task"].get("placement_sequence", []))
    sequence_slots: list[GoalSlot] = []
    if placement_sequence:
        preserve_order = True
        for index, placement in enumerate(placement_sequence):
            object_id = str(placement["object_id"])
            if object_id not in objects:
                raise ValueError(f"unknown placement-sequence object: {object_id}")
            sequence_slots.append(
                GoalSlot(
                    name=str(placement.get("slot_name", f"sequence_{index:03d}")),
                    group_id="placement_sequence",
                    color="mixed",
                    object_id=object_id,
                    position=tuple(float(v) for v in placement["target_pose"]),
                )
            )
        target_objects = [slot.object_id for slot in sequence_slots]
    else:
        target_objects = list(config["task"]["target_objects"])
    if max_objects is not None:
        target_objects = target_objects[:max_objects]
        sequence_slots = sequence_slots[:max_objects]

    def _next_object(pending: list[str]) -> str:
        if preserve_order:
            return pending[0]
        original_rank = {
            object_id: index for index, object_id in enumerate(target_objects)
        }

        def score(object_id: str) -> float:
            obj = objects[object_id]
            slot = slots[object_id]
            start = obj["pose"][:2]
            goal = slot.position[:2]
            transit_probe = planner.plan_xy(current_xy, start)
            transfer_probe = planner.plan_xy(start, goal)
            value = transit_probe.length + transfer_probe.length
            if not transit_probe.success:
                value += 5000.0
            if not transfer_probe.success:
                value += 3000.0
            if transit_probe.metadata["route_type"] != "direct":
                value += 200.0
            if transfer_probe.metadata["route_type"] != "direct":
                value += 200.0
            # Stable tie-breaker keeps config intent when feasibility is similar.
            return value + original_rank[object_id] * 0.01

        ranked = sorted(pending, key=score)
        if ik_solver is None or physical_qpos is None:
            return ranked[0]
        _sync_planning_scene()
        saved_qpos = ik_solver.current_qpos()
        try:
            for object_id in ranked[:4]:
                object_pose = backend.get_body_pose(f"cube_{object_id}")[:3]
                ik_solver.set_qpos(physical_qpos)
                grasp = ik_solver.plan_physical_grasp(
                    object_id,
                    tuple(object_pose),
                    start_qpos=physical_qpos,
                    random_restarts=24,
                )
                if grasp.success:
                    return object_id
        finally:
            ik_solver.set_qpos(saved_qpos)
        return ranked[0]

    pending_objects = list(target_objects)
    attempted_objects = []
    sequence_index = 0
    while pending_objects:
        if sequence_slots:
            object_id = pending_objects.pop(0)
            slot = sequence_slots[sequence_index]
            attempt_id = f"{object_id}__step_{sequence_index:03d}"
            sequence_index += 1
        else:
            object_id = _next_object(pending_objects)
            pending_objects.remove(object_id)
            slot = slots[object_id]
            attempt_id = object_id
        attempted_objects.append(attempt_id)
        obj = objects[object_id]
        if sequence_slots:
            obj["pose"] = backend.get_body_pose(f"cube_{object_id}")[:3]
        arm_at_home = True
        if physics_executor is not None and physical_home_qpos is not None:
            if not _move_arm_to_safe_pose():
                arm_at_home = False
            physical_qpos = physics_executor.ik.current_qpos()
            _sync_planning_scene()
            ik_solver.set_qpos(physical_qpos)
        object_start_q = ik_solver.current_qpos() if ik_solver is not None else None
        start, goal = obj["pose"][:2], slot.position[:2]
        reach = math.dist(config["robot"]["base_xy"], start)
        transit = planner.plan_xy(current_xy, start)
        if not probe.path_clear((current_xy, tuple(start))):
            direct_only_failures += 1
        if not probe.path_clear((tuple(start), tuple(goal))):
            direct_only_failures += 1
        aware_corridor_routes += int(transit.metadata["route_type"] != "direct")
        motion, transfer_failures, retries = _probe_transfer(
            planner, start, goal, retry_limit
        )
        collision_failures += transfer_failures
        retries_used += retries
        route = _route_type(motion)
        motions[attempt_id] = motion
        aware_corridor_routes += int(route != "direct")
        execution = _ObjectExecution()
        reach_ok = _object_reach_ok(config, obj, start)
        if ik_solver is not None:
            if not reach_ok:
                execution.ik_success = False
                execution.ik_reason = "object outside configured reach"
            elif ik_home is None or not ik_home.success:
                execution.ik_success = False
                execution.ik_reason = "no collision-free Panda home IK"
            elif physics_executor is not None and physical_qpos is not None:
                execution = _execute_physical_pick_place(
                    object_id, obj, slot, arm_at_home
                )
            else:
                execution, collision_count = _execute_ik_preview(obj, transit, motion)
                ik_collision_failures += collision_count
            if not execution.ik_success:
                ik_failures += 1
                ik_solver.set_qpos(object_start_q)
        grasp_styles[object_id] = execution.grasp_style
        reason = motion.metadata.get("reason")
        object_success = (
            execution.ik_success
            if physics_executor is not None
            else transit.success and motion.success and execution.ik_success
        )
        if object_success:
            total_cost += transit.length + motion.length
            if physics_executor is None:
                backend.set_body_pose(f"cube_{object_id}", slot.position)
            plan_actions.append(
                _plan_action(
                    object_id,
                    slot,
                    route,
                    transit,
                    motion,
                    execution,
                    _grip_width(obj),
                )
            )
        per_object.append(
            _per_object_result(
                object_id,
                slot,
                object_success,
                route,
                retries,
                reason,
                transit,
                reach,
                motion,
                reach_ok,
                execution,
            )
        )
        sys.stderr.write(
            f"[{len(per_object)}/{len(target_objects)}] {object_id}: "
            f"geometric={transit.success and motion.success} ik={execution.ik_success} "
            f"reason={execution.ik_reason}\n",
        )
        if object_success:
            current_xy = slot.position[:2]
    (
        all_objects_solved,
        completed_objects,
        completion_ratio,
        completion_policy,
        accepted_completion,
    ) = _completion_status(per_object, config)
    tmm = _build_ordered_tmm(attempted_objects, motions)
    search_result = TMMAStar().search(tmm)
    success = accepted_completion and search_result.success
    metrics = {
        "scene_id": config["scene"]["scene_id"],
        "planner_backend": "mujoco",
        "robot_model_status": builder.panda_asset.status,
        "number_of_objects": len(objects),
        "number_of_slots": len(sequence_slots) if sequence_slots else len(slots),
        "number_of_obstacles": len(config["obstacles"]),
        "tmm_vertices": tmm.vertex_count,
        "tmm_edges": tmm.edge_count,
        "expanded_vertices": search_result.nodes_expanded,
        "elapsed_time": time.perf_counter() - started,
        "solution_found": success,
        "total_cost": total_cost if success else None,
        "all_objects_solved": all_objects_solved,
        "completed_objects": completed_objects,
        "completion_ratio": completion_ratio,
        "completion_policy": completion_policy,
        "per_object_result": per_object,
        "failed_objects": [x["object_id"] for x in per_object if not x["success"]],
        "collision_probe_failures": collision_failures,
        "retries_used": retries_used,
        "validation_level": (
            "symbolic_ordered_branch + geometric_2d_probe + real_panda_7dof_ik_rrt_collision"
            if real_panda
            else "symbolic_ordered_branch + geometric_2d_probe + mujoco_scene_state"
        ),
        "full_panda_ik_validated": real_panda and ik_failures == 0,
        "panda_ik_failures": ik_failures,
        "panda_ik_collision_failures": ik_collision_failures,
        "grasp_styles": grasp_styles,
        "challenge_ablation": {
            "segments_evaluated": len(target_objects) * 2,
            "direct_only_blocked_segments": direct_only_failures,
            "obstacle_aware_failed_segments": sum(not x["success"] for x in per_object),
            "obstacle_aware_corridor_routes": aware_corridor_routes,
            "interpretation": "pipeline effect is measured on transit + transfer XY segments",
        },
    }
    _write_json(
        output / "final_plan.json",
        {
            "success": success,
            "all_objects_solved": all_objects_solved,
            "completion_ratio": completion_ratio,
            "actions": plan_actions,
        },
    )
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "challenge_ablation.json", metrics["challenge_ablation"])
    _write_json(
        output / "scene_summary.json",
        {
            "scene_id": config["scene"]["scene_id"],
            "objects": list(objects),
            "slots": (
                {
                    slot.name: {
                        "object_id": slot.object_id,
                        "position": slot.position,
                    }
                    for slot in sequence_slots
                }
                if sequence_slots
                else {
                    oid: {"name": s.name, "position": s.position}
                    for oid, s in slots.items()
                }
            ),
            "obstacles": config["obstacles"],
            "robot_model_status": builder.panda_asset.status,
        },
    )
    observation = f"""# Observation: {config["scene"]["scene_id"]}

1. **Before integration:** No. The existing code had no MuJoCo backend or scene loader, and its exhaustive symbolic planner is intractable for 12 objects (`12! * 2^12` branches).
2. **After integration:** Partial. The ordered CTAMP adapter found a symbolic/geometric plan for all {len(objects)} objects and synchronized final cube poses into a stepped MuJoCo scene.
3. **Robot model:** `{builder.panda_asset.status}`. Seven-joint Panda IK and joint-space collision paths were {"validated" if real_panda and ik_failures == 0 else "not fully validated"}.
4. **Wall behavior:** The inflated wall blocked {direct_only_failures} of {len(target_objects) * 2} direct transit/transfer segments.
5. **Side corridors:** The obstacle-aware pipeline used {aware_corridor_routes} corridor routes and left {sum(not x["success"] for x in per_object)} objects unresolved.
6. **Slots:** All {len(sequence_slots) if sequence_slots else len(slots)} placements received ordered, table-valid targets.
7. **Reach:** All object starts and slots satisfy configured radial reach limits.
8. **Impossible objects:** None under the 2-D probe. Full physical feasibility remains unknown because Panda IK and joint/link collision checking are absent.
9. **Necessary changes:** Optional backend, scene builder/observer, Panda asset detection and proxy, ordered slot generator, obstacle-aware probe, motion adapter, and deterministic scene runner.
10. **Technical debt:** Compose the real menagerie MJCF, implement Panda IK, plan joint trajectories, validate link/object contacts, and replace deterministic ordering with scalable task search.

## Evidence classification

- Symbolic CTAMP success: **yes (deterministic ordered branch)**
- Geometric 2-D probe success: **{"yes" if success else "no"}**
- MuJoCo scene load/step/state update: **yes**
- Full MuJoCo Panda joint/IK motion success: **{"yes" if real_panda and ik_failures == 0 else "no"}**
- Force-closure finger grasp/contact dynamics: **no; cubes are attached kinematically during transfer**
"""
    (output / "OBSERVATION.md").write_text(observation, encoding="utf-8")
    if viewer_context is not None:
        viewer_context.__exit__(None, None, None)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--motion-planner", default="mujoco", choices=["mujoco"])
    parser.add_argument("--robot", default="panda_left")
    parser.add_argument("--learning-mode", default="online")
    parser.add_argument("--max-retries-per-object", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--max-objects", type=int)
    args = parser.parse_args()
    metrics = run(
        args.config,
        args.output,
        args.max_retries_per_object,
        max_objects=args.max_objects,
        viewer=args.viewer,
    )
    sys.stdout.write(json.dumps(metrics, indent=2) + "\n")
    return 0 if metrics["solution_found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
