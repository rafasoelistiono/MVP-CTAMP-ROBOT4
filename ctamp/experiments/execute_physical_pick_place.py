"""Execute one contact-validated Panda pick-and-place in MuJoCo physics."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ..simulation import (
    MuJoCoBackend,
    MuJoCoSceneBuilder,
    PandaIKSolver,
    PandaPhysicsExecutor,
    generate_tidy_slots,
    load_scene_config,
)
from .scene_helpers import placement_within_tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--object", default="e")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.5,
        help="viewer real-time factor; 0.5 runs at half speed",
    )
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")

    config = load_scene_config(args.config)
    objects = {obj["id"]: obj for obj in config["objects"]}
    slots = generate_tidy_slots(config)
    if args.object not in objects:
        parser.error(f"unknown object: {args.object}")
    builder = MuJoCoSceneBuilder(config, args.config.resolve().parents[2])
    xml = builder.build_xml()
    planning_backend = MuJoCoBackend()
    planning_backend.load_model(xml_string=xml)
    ik = PandaIKSolver(planning_backend)
    try:
        safe_q = np.asarray(config["robot"]["physical_start_qpos"], dtype=float)
    except KeyError:
        parser.error("robot.physical_start_qpos is required for physical execution")
    if safe_q.shape != (7,):
        parser.error("robot.physical_start_qpos must contain seven joint values")
    ik.set_qpos(safe_q)
    obj = objects[args.object]
    grasp = ik.plan_physical_grasp(
        args.object,
        tuple(obj["pose"]),
        start_qpos=safe_q,
    )
    if not grasp.success:
        payload = {
            "object_id": args.object,
            "success": False,
            "grip_success": False,
            "tidy_success": False,
            "stage": "grasp_planning",
            "reason": grasp.reason,
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 2
    approach_q = grasp.joint_waypoints[-1]
    ik.set_qpos(approach_q)
    grasp_rotation = ik.data.site_xmat[ik.site_id].reshape(3, 3).copy()
    lift_target = ik.site_position() + np.array([0.0, 0.0, 0.14])
    lift = ik.solve(
        lift_target,
        seed=approach_q,
        orientation=grasp_rotation,
        orientation_tolerance=0.10,
    )
    if not lift.success:
        payload = {
            "object_id": args.object,
            "success": False,
            "grip_success": False,
            "tidy_success": False,
            "stage": "lift_planning",
            "reason": "lift IK failed",
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 2
    ik.set_qpos(lift.qpos)
    slot = slots[args.object]
    place_target = np.asarray(slot.position) + np.array([0.0, 0.0, 0.06])
    preplace_target = place_target + np.array([0.0, 0.0, 0.14])
    place_path = ik.solve_path(
        [tuple(preplace_target), tuple(place_target)],
        orientation=grasp_rotation,
        allowed_object_id=args.object,
    )
    if not place_path.success:
        place_path = ik.solve_path(
            [tuple(preplace_target), tuple(place_target)],
            allowed_object_id=args.object,
        )
    place_failure_reason = None if place_path.success else place_path.reason

    physics_backend = MuJoCoBackend()
    physics_backend.load_model(xml_string=xml)
    viewer_context = nullcontext(None)
    if args.viewer:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(
            physics_backend.model,
            physics_backend.data,
        )
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = [0.0, -0.15, 1.0]
            viewer.cam.distance = 2.8
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -32
        executor = PandaPhysicsExecutor(
            physics_backend,
            viewer=viewer,
            realtime_factor=args.speed,
        )
        executor.initialize_arm(safe_q)
        executor.open_gripper(steps=120)
        tracked, error = executor.follow_joint_path(grasp.joint_waypoints[1:-1])
        if not tracked:
            payload = {
                "object_id": args.object,
                "success": False,
                "grip_success": False,
                "tidy_success": False,
                "stage": "pregrasp_tracking",
                "arm_tracking_error": error,
                "reason": "pregrasp tracking failed",
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return 2
        grasp_result = executor.validate_grasp_and_lift(
            args.object,
            approach_q,
            lift.qpos,
            grasp_site_target=np.asarray(obj["pose"]) + np.array([0.0, 0.0, 0.02]),
        )
        if not grasp_result.success:
            payload = {
                "object_id": args.object,
                "success": False,
                "grip_success": False,
                "tidy_success": False,
                "stage": "physical_grasp",
                **grasp_result.__dict__,
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return 2
        if place_failure_reason is not None:
            executor.set_carry_constraint(args.object, False)
            executor.open_gripper(steps=250)
            executor.settle(steps=180)
            payload = {
                "object_id": args.object,
                "success": False,
                "grip_success": True,
                "tidy_success": False,
                "stage": "place_planning",
                "bilateral_contact": grasp_result.bilateral_contact,
                "lift_height": grasp_result.lift_height,
                "reason": place_failure_reason,
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return 2
        tracked, error = executor.follow_joint_path(place_path.joint_waypoints[1:])
        if not tracked:
            executor.set_carry_constraint(args.object, False)
            executor.open_gripper(steps=250)
            payload = {
                "object_id": args.object,
                "success": False,
                "grip_success": True,
                "tidy_success": False,
                "stage": "place_tracking",
                "bilateral_contact": grasp_result.bilateral_contact,
                "lift_height": grasp_result.lift_height,
                "arm_tracking_error": error,
                "reason": "place tracking failed",
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
            return 2
        executor.set_carry_constraint(args.object, False)
        executor.open_gripper(steps=320)
        if len(place_path.joint_waypoints) >= 2:
            executor.follow_joint_path(
                [place_path.joint_waypoints[-2]], max_joint_step=0.025
            )
        executor.settle(steps=180)
        final_position = physics_backend.get_body_pose(f"cube_{args.object}")[:3]
        error_xyz = np.asarray(final_position) - np.asarray(slot.position)
        success = placement_within_tolerance(error_xyz, config)
        payload = {
            "object_id": args.object,
            "success": success,
            "grip_success": True,
            "tidy_success": success,
            "grasp_style": grasp.grasp_style,
            "bilateral_contact": grasp_result.bilateral_contact,
            "lift_height": grasp_result.lift_height,
            "slot": slot.name,
            "target_position": list(slot.position),
            "final_position": final_position,
            "placement_error": error_xyz.tolist(),
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        if viewer is not None:
            while viewer.is_running():
                viewer.sync()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
