"""Failure-safe PDDLStream wrappers around existing Panda and motion APIs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import AbstractSet, Any

import numpy as np

from ..simulation.panda_ik import PandaIKSolver
from ..simulation.scene import MotionProbe
from .problems import GeneratedProblem, Pose

Configuration = tuple[float, ...]
Grasp = tuple[float, float, float]
Trajectory = tuple[Configuration, ...]


class StreamContext:
    def __init__(
        self,
        problem: GeneratedProblem,
        ik_solver: PandaIKSolver | None = None,
        dry_run: bool = False,
    ) -> None:
        self.problem = problem
        self.config = problem.scene_config
        self.ik_solver = ik_solver
        self.dry_run = dry_run
        self.probe = MotionProbe(self.config)
        self.home = tuple(float(v) for v in self.config["robot"]["physical_start_qpos"])
        self.objects = {obj["id"]: obj for obj in self.config["objects"]}
        self.evaluations: Counter[str] = Counter()
        self.samples: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self._configuration_poses: dict[Configuration, Pose] = {
            self.home: (
                float(self.config["robot"]["base_xy"][0])
                + float(self.config["robot"]["reach_min_xy"])
                + 0.02,
                float(self.config["robot"]["base_xy"][1]),
                0.95,
            )
        }
        self._known_ik: dict[Pose, Configuration] = {}

    def _record(self, name: str, success: bool) -> None:
        self.evaluations[name] += 1
        if success:
            self.samples[name] += 1
        else:
            self.failures[name] += 1

    def sample_grasp(self, object_id: str) -> Iterator[tuple[Grasp]]:
        name = "sample-grasp"
        if object_id not in self.objects:
            self._record(name, False)
            return
        self._record(name, True)
        yield ((0.0, 0.0, 0.02),)

    def sample_pick_pose(
        self, object_id: str, object_pose: Pose, grasp: Grasp
    ) -> Iterator[tuple[Pose]]:
        name = "sample-pick-pose"
        if object_id not in self.objects:
            self._record(name, False)
            return
        position = tuple(float(v) for v in object_pose)
        relative = tuple(float(v) for v in grasp)
        pick_pose = tuple(
            position[index] + relative[index] for index in range(3)
        )
        if self.dry_run:
            self._record(name, True)
            yield (pick_pose,)
            return
        if self.ik_solver is None:
            self._record(name, False)
            return
        previous = self.ik_solver.current_qpos()
        try:
            result = self.ik_solver.plan_physical_grasp(
                object_id, position, self.home, random_restarts=32
            )
            if not result.success:
                self._record(name, False)
                return
            pick_pose = (
                position[0],
                position[1],
                position[2] + (0.02 if result.grasp_style == "top" else 0.04),
            )
            self._known_ik[pick_pose] = tuple(
                float(v) for v in result.joint_waypoints[-1]
            )
            self._record(name, True)
            yield (pick_pose,)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            self._record(name, False)
        finally:
            self.ik_solver.set_qpos(previous)

    def sample_place_pose(
        self, object_id: str, surface_id: str
    ) -> Iterator[tuple[Pose]]:
        name = "sample-place-pose"
        pose = self.problem.placement_poses.get((object_id, surface_id))
        self._record(name, pose is not None)
        if pose is not None:
            yield (pose,)

    def sample_stack_pose(
        self, object_id: str, support_id: str, support_pose: Pose
    ) -> Iterator[tuple[Pose]]:
        name = "sample-stack-pose"
        obj = self.objects.get(object_id)
        support = self.objects.get(support_id)
        if obj is None or support is None:
            self._record(name, False)
            return
        default_size = self.config["geometry"]["cube_size_xyz"]
        object_height = float(obj.get("size_xyz", default_size)[2])
        support_height = float(support.get("size_xyz", default_size)[2])
        x, y, z = (float(value) for value in support_pose)
        pose = (x, y, z + support_height / 2.0 + object_height / 2.0)
        self._record(name, True)
        yield (pose,)

    @staticmethod
    def _first_output(generator: Iterator[tuple[Any, ...]]) -> tuple[Any, ...] | None:
        return next(generator, None)

    def plan_pick(
        self,
        object_id: str,
        object_pose: Pose,
        home: Configuration,
        fluents: AbstractSet[tuple[Any, ...]] = frozenset(),
    ) -> Iterator[tuple[Grasp, Pose, Configuration, Trajectory]]:
        name = "plan-pick"
        grasp_output = self._first_output(self.sample_grasp(object_id))
        if grasp_output is None:
            self._record(name, False)
            return
        grasp = grasp_output[0]
        pose_output = self._first_output(
            self.sample_pick_pose(object_id, object_pose, grasp)
        )
        if pose_output is None:
            self._record(name, False)
            return
        pick_pose = pose_output[0]
        ik_output = self._first_output(self.inverse_kinematics(pick_pose))
        if ik_output is None:
            self._record(name, False)
            return
        pick_q = ik_output[0]
        transit_output = self._first_output(
            self.plan_transit(home, pick_q, fluents=fluents)
        )
        if transit_output is None:
            self._record(name, False)
            return
        self._record(name, True)
        yield grasp, pick_pose, pick_q, transit_output[0]

    def plan_place(
        self,
        object_id: str,
        surface_id: str,
        pick_q: Configuration,
        home: Configuration,
        fluents: AbstractSet[tuple[Any, ...]] = frozenset(),
    ) -> Iterator[tuple[Pose, Configuration, Trajectory, Trajectory]]:
        name = "plan-place"
        pose_output = self._first_output(
            self.sample_place_pose(object_id, surface_id)
        )
        if pose_output is None:
            self._record(name, False)
            return
        yield from self._plan_placement(
            name, pose_output[0], pick_q, home, fluents
        )

    def plan_stack(
        self,
        object_id: str,
        support_id: str,
        support_pose: Pose,
        pick_q: Configuration,
        home: Configuration,
        fluents: AbstractSet[tuple[Any, ...]] = frozenset(),
    ) -> Iterator[tuple[Pose, Configuration, Trajectory, Trajectory]]:
        name = "plan-stack"
        pose_output = self._first_output(
            self.sample_stack_pose(object_id, support_id, support_pose)
        )
        if pose_output is None:
            self._record(name, False)
            return
        yield from self._plan_placement(
            name, pose_output[0], pick_q, home, fluents
        )

    def _plan_placement(
        self,
        name: str,
        pose: Pose,
        pick_q: Configuration,
        home: Configuration,
        fluents: AbstractSet[tuple[Any, ...]],
    ) -> Iterator[tuple[Pose, Configuration, Trajectory, Trajectory]]:
        ik_output = self._first_output(self.inverse_kinematics(pose))
        if ik_output is None:
            self._record(name, False)
            return
        place_q = ik_output[0]
        transfer_output = self._first_output(
            self.plan_transit(pick_q, place_q, fluents=fluents)
        )
        return_output = self._first_output(
            self.plan_transit(place_q, home, fluents=fluents)
        )
        if transfer_output is None or return_output is None:
            self._record(name, False)
            return
        self._record(name, True)
        yield pose, place_q, transfer_output[0], return_output[0]

    def inverse_kinematics(self, pose: Pose) -> Iterator[tuple[Configuration]]:
        name = "inverse-kinematics"
        xyz = tuple(float(v) for v in pose)
        known = self._known_ik.get(xyz)
        if known is not None:
            self._configuration_poses[known] = xyz
            self._record(name, True)
            yield (known,)
            return
        if self.dry_run:
            pseudo = tuple(round(v, 6) for v in (*xyz, 0.0, 0.0, 0.0, 0.0))
            self._configuration_poses[pseudo] = xyz
            self._record(name, True)
            yield (pseudo,)
            return
        if self.ik_solver is None:
            self._record(name, False)
            return
        previous = self.ik_solver.current_qpos()
        try:
            ik_target = (xyz[0], xyz[1], xyz[2] + 0.06)
            result = self.ik_solver.solve_collision_free(
                ik_target, preferred_seed=self.home, random_restarts=32
            )
            if not result.success:
                self._record(name, False)
                return
            configuration = tuple(float(v) for v in result.qpos)
            self._configuration_poses[configuration] = xyz
            self._record(name, True)
            yield (configuration,)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            self._record(name, False)
        finally:
            self.ik_solver.set_qpos(previous)

    def _probe_clear(self, start: Configuration, goal: Configuration) -> bool:
        start_pose = self._configuration_poses.get(tuple(start))
        goal_pose = self._configuration_poses.get(tuple(goal))
        if start_pose is None or goal_pose is None:
            return False
        return self.probe.probe(start_pose[:2], goal_pose[:2]).success

    def _apply_fluent_poses(
        self, fluents: AbstractSet[tuple[Any, ...]]
    ) -> dict[str, list[float]]:
        if self.ik_solver is None:
            return {}
        saved: dict[str, list[float]] = {}
        for fact in fluents:
            predicate = str(fact[0]).lower()
            if predicate not in {"at-pose", "atpose"} or len(fact) != 3:
                continue
            object_id, pose = str(fact[1]), fact[2]
            if object_id not in self.objects or not isinstance(pose, tuple):
                continue
            body_name = f"cube_{object_id}"
            if object_id not in saved:
                saved[object_id] = self.ik_solver.backend.get_body_pose(body_name)
            self.ik_solver.backend.set_body_pose(body_name, pose)
        return saved

    def _restore_fluent_poses(self, saved: dict[str, list[float]]) -> None:
        if self.ik_solver is None:
            return
        for object_id, pose in saved.items():
            self.ik_solver.backend.set_body_pose(f"cube_{object_id}", pose)

    def test_motion(
        self,
        start: Configuration,
        goal: Configuration,
        fluents: AbstractSet[tuple[Any, ...]] = frozenset(),
    ) -> bool:
        name = "test-motion"
        if self.ik_solver is None and not self.dry_run:
            self._record(name, False)
            return False
        success = self._probe_clear(tuple(start), tuple(goal))
        saved = self._apply_fluent_poses(fluents)
        if success and self.ik_solver is not None and not self.dry_run:
            previous = self.ik_solver.current_qpos()
            try:
                success = not bool(
                    self.ik_solver.validate_joint_segment(
                        np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
                    )
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                success = False
            finally:
                self.ik_solver.set_qpos(previous)
                self._restore_fluent_poses(saved)
        self._record(name, success)
        return success

    def plan_transit(
        self,
        start: Configuration,
        goal: Configuration,
        fluents: AbstractSet[tuple[Any, ...]] = frozenset(),
    ) -> Iterator[tuple[Trajectory]]:
        name = "plan-transit"
        start_q, goal_q = tuple(start), tuple(goal)
        if self.ik_solver is None and not self.dry_run:
            self._record(name, False)
            return
        if not self._probe_clear(start_q, goal_q):
            self._record(name, False)
            return
        route: Trajectory = (start_q, goal_q)
        saved = self._apply_fluent_poses(fluents)
        if self.ik_solver is not None and not self.dry_run:
            previous = self.ik_solver.current_qpos()
            try:
                collisions = self.ik_solver.validate_joint_segment(
                    np.asarray(start_q, dtype=float), np.asarray(goal_q, dtype=float)
                )
                if collisions:
                    planned = self.ik_solver.plan_joint_rrt(
                        np.asarray(start_q, dtype=float),
                        np.asarray(goal_q, dtype=float),
                        max_iterations=4000,
                        rng_seed=71,
                    )
                    if planned is None:
                        self._record(name, False)
                        return
                    route = tuple(tuple(float(v) for v in q) for q in planned)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                self._record(name, False)
                return
            finally:
                self.ik_solver.set_qpos(previous)
                self._restore_fluent_poses(saved)
        self._record(name, True)
        yield (route,)

    def metrics(self) -> dict[str, Any]:
        return {
            "evaluations": sum(self.evaluations.values()),
            "samples": sum(self.samples.values()),
            "failures": sum(self.failures.values()),
            "by_stream": {
                name: {
                    "evaluations": self.evaluations[name],
                    "samples": self.samples[name],
                    "failures": self.failures[name],
                }
                for name in sorted(self.evaluations)
            },
        }
