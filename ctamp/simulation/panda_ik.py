"""Damped-least-squares position IK for the real seven-joint Panda model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .mujoco_backend import MuJoCoBackend


@dataclass(frozen=True)
class IKResult:
    success: bool
    qpos: tuple[float, ...]
    residual: float
    iterations: int
    reason: str | None = None
    grasp_style: str | None = None


@dataclass(frozen=True)
class IKPathResult:
    success: bool
    joint_waypoints: tuple[tuple[float, ...], ...]
    max_residual: float
    collision_pairs: tuple[tuple[str, str], ...]
    reason: str | None = None


@dataclass(frozen=True)
class GraspPlanResult:
    success: bool
    grasp_style: str | None
    joint_waypoints: tuple[tuple[float, ...], ...]
    residual: float
    reason: str | None = None


class PandaIKSolver:
    """Solve gripper-site position IK while respecting Panda joint limits."""

    JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))

    @classmethod
    def from_scene_config(
        cls, config: dict[str, Any], project_root: str | Path
    ) -> PandaIKSolver:
        from .mujoco_scene_builder import MuJoCoSceneBuilder

        builder = MuJoCoSceneBuilder(config, project_root)
        if builder.panda_asset.status != "real_panda_asset":
            raise RuntimeError("Panda scene requires real Panda MJCF assets")
        backend = MuJoCoBackend()
        backend.load_model(xml_string=builder.build_xml())
        solver = cls(backend)
        home = np.asarray(config["robot"]["physical_start_qpos"], dtype=float)
        if home.shape != (7,):
            raise ValueError("robot.physical_start_qpos must contain seven values")
        solver.set_qpos(home)
        return solver

    def __init__(
        self,
        backend: MuJoCoBackend,
        site_name: str = "gripper",
        tolerance: float = 2e-3,
        damping: float = 2e-3,
        step_size: float = 0.6,
        max_iterations: int = 250,
    ) -> None:
        self.backend = backend
        self.model = backend.model
        self.data = backend.data
        self.mujoco = backend._mujoco()
        self.tolerance = tolerance
        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.site_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if self.site_id < 0:
            raise ValueError(
                f"site {site_name!r} not found; real Panda MJCF is required"
            )
        self.joint_ids = [self.model.joint(name).id for name in self.JOINT_NAMES]
        self.qpos_indices = np.array(
            [self.model.jnt_qposadr[j] for j in self.joint_ids]
        )
        self.dof_indices = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])
        self.lower = np.array([self.model.jnt_range[j, 0] for j in self.joint_ids])
        self.upper = np.array([self.model.jnt_range[j, 1] for j in self.joint_ids])
        if self.model.nkey:
            self.data.qpos[self.qpos_indices] = self.model.key_qpos[
                0, self.qpos_indices
            ]
        self.set_gripper_width(0.08)

    def current_qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].copy()

    def set_qpos(self, qpos: np.ndarray | tuple[float, ...]) -> None:
        self.data.qpos[self.qpos_indices] = np.asarray(qpos, dtype=float)
        self.mujoco.mj_forward(self.model, self.data)

    def site_position(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    def set_gripper_width(self, width: float) -> None:
        """Set total finger opening in meters for kinematic replay."""
        half_width = float(np.clip(width / 2.0, 0.0, 0.04))
        for name in ("finger_joint1", "finger_joint2"):
            joint_id = self.model.joint(name).id
            qpos_index = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_index] = half_width
        self.mujoco.mj_forward(self.model, self.data)

    def solve(
        self,
        target: np.ndarray | tuple[float, float, float],
        seed=None,
        orientation: np.ndarray | None = None,
        orientation_tolerance: float = 0.035,
    ) -> IKResult:
        target_array = np.asarray(target, dtype=float)
        if target_array.shape != (3,):
            raise ValueError("IK target must be xyz")
        if seed is not None:
            self.set_qpos(np.asarray(seed, dtype=float))
        if orientation is not None and np.asarray(orientation).shape != (3, 3):
            raise ValueError("orientation must be a 3x3 rotation matrix")
        target_rotation = (
            None if orientation is None else np.asarray(orientation, dtype=float)
        )
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for iteration in range(1, self.max_iterations + 1):
            position_error = target_array - self.site_position()
            rotation_error = np.zeros(3)
            if target_rotation is not None:
                current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
                rotation_error = 0.5 * sum(
                    np.cross(current_rotation[:, axis], target_rotation[:, axis])
                    for axis in range(3)
                )
            residual = float(np.linalg.norm(position_error))
            if (
                residual <= self.tolerance
                and np.linalg.norm(rotation_error) <= orientation_tolerance
            ):
                return IKResult(True, tuple(self.current_qpos()), residual, iteration)
            jacp.fill(0.0)
            jacr.fill(0.0)
            self.mujoco.mj_jacSite(
                self.model,
                self.data,
                jacp,
                jacr if target_rotation is not None else None,
                self.site_id,
            )
            if target_rotation is None:
                error = position_error
                jacobian = jacp[:, self.dof_indices]
            else:
                orientation_weight = 0.35
                error = np.concatenate(
                    (position_error, orientation_weight * rotation_error)
                )
                jacobian = np.vstack(
                    (
                        jacp[:, self.dof_indices],
                        orientation_weight * jacr[:, self.dof_indices],
                    )
                )
            system = jacobian @ jacobian.T + self.damping * np.eye(jacobian.shape[0])
            delta = jacobian.T @ np.linalg.solve(system, error)
            qpos = np.clip(
                self.current_qpos() + self.step_size * delta,
                self.lower + 1e-5,
                self.upper - 1e-5,
            )
            self.set_qpos(qpos)
        residual = float(np.linalg.norm(target_array - self.site_position()))
        return IKResult(
            False,
            tuple(self.current_qpos()),
            residual,
            self.max_iterations,
            "IK did not converge within tolerance",
        )

    def solve_collision_free(
        self,
        target: np.ndarray | tuple[float, float, float],
        preferred_seed=None,
        random_restarts: int = 64,
        rng_seed: int = 42,
        grasp_style: str | None = None,
        orientation: np.ndarray | None = None,
        orientation_tolerance: float = 0.035,
    ) -> IKResult:
        """Search multiple Panda postures and reject link/environment collisions."""
        rng = np.random.default_rng(rng_seed)
        seeds = []
        if preferred_seed is not None:
            seeds.append(np.asarray(preferred_seed, dtype=float))
        seeds.append(self.current_qpos())
        if self.model.nkey:
            seeds.append(self.model.key_qpos[0, self.qpos_indices].copy())
        seeds.extend(
            rng.uniform(self.lower, self.upper) for _ in range(random_restarts)
        )
        best: IKResult | None = None
        for seed in seeds:
            result = self.solve(
                target,
                seed=seed,
                orientation=orientation,
                orientation_tolerance=orientation_tolerance,
            )
            if best is None or result.residual < best.residual:
                best = result
            if result.success and not self.robot_collision_pairs():
                return IKResult(
                    True,
                    result.qpos,
                    result.residual,
                    result.iterations,
                    grasp_style=grasp_style,
                )
        assert best is not None
        return IKResult(
            False,
            best.qpos,
            best.residual,
            best.iterations,
            "no collision-free IK solution across multi-start seeds",
            grasp_style=grasp_style,
        )

    def collision_free_candidates(
        self,
        target,
        preferred_seed=None,
        random_restarts: int = 64,
        rng_seed: int = 42,
        orientation: np.ndarray | None = None,
        max_results: int = 8,
        orientation_tolerance: float = 0.035,
        allowed_object_id: str | None = None,
    ) -> list[IKResult]:
        rng = np.random.default_rng(rng_seed)
        seeds = []
        if preferred_seed is not None:
            seeds.append(np.asarray(preferred_seed, dtype=float))
        seeds.extend(
            [
                np.array([0.0, 0.529, 0.0, -1.98, 0.0, 2.495, -0.75]),
                np.array([0.0, 0.20, 0.0, -2.40, 0.0, 2.60, -0.75]),
                np.array([2.827, -1.553, -1.694, -1.797, -1.581, 1.695, -1.079]),
            ]
        )
        seeds.extend(
            rng.uniform(self.lower, self.upper) for _ in range(random_restarts)
        )
        results: list[IKResult] = []
        for seed in seeds:
            result = self.solve(
                target,
                seed=seed,
                orientation=orientation,
                orientation_tolerance=orientation_tolerance,
            )
            if not result.success or self.robot_collision_pairs(
                allowed_object_id=allowed_object_id,
            ):
                continue
            qpos = np.asarray(result.qpos)
            if any(
                np.linalg.norm(qpos - np.asarray(old.qpos)) < 0.15 for old in results
            ):
                continue
            results.append(result)
            if len(results) >= max_results:
                break
        return results

    def solve_grasp_candidates(
        self,
        object_position: tuple[float, float, float],
        preferred_seed=None,
        random_restarts: int = 64,
    ) -> IKResult:
        """Try top and four side grasp-site targets around a cube.

        The current solver constrains end-effector position; style-specific offsets
        represent distinct approach sides. Orientation constraints are applied in
        the subsequent trajectory layer, so a returned style is not a force-closure
        grasp claim.
        """
        x, y, z = object_position
        candidates = (
            ("top", (x, y, z + 0.105), (0.0, 0.0, -1.0)),
            ("side_pos_x", (x + 0.075, y, z + 0.035), (-1.0, 0.0, 0.0)),
            ("side_neg_x", (x - 0.075, y, z + 0.035), (1.0, 0.0, 0.0)),
            ("side_pos_y", (x, y + 0.075, z + 0.035), (0.0, -1.0, 0.0)),
            ("side_neg_y", (x, y - 0.075, z + 0.035), (0.0, 1.0, 0.0)),
        )
        best: IKResult | None = None
        for index, (style, target, approach) in enumerate(candidates):
            result = self.solve_collision_free(
                target,
                preferred_seed=preferred_seed,
                random_restarts=random_restarts,
                rng_seed=42 + index,
                grasp_style=style,
                orientation=self._rotation_from_approach(approach),
            )
            if result.success:
                return result
            if best is None or result.residual < best.residual:
                best = result
        assert best is not None
        return best

    def plan_grasp_candidates(
        self,
        object_position: tuple[float, float, float],
        start_qpos,
        random_restarts: int = 64,
    ) -> GraspPlanResult:
        """Plan collision-free reorientation at pre-grasp, then a short approach."""
        x, y, z = object_position
        candidates = (
            ("top", (x, y, z + 0.105), (0.0, 0.0, -1.0)),
            ("side_pos_x", (x + 0.075, y, z + 0.035), (-1.0, 0.0, 0.0)),
            ("side_neg_x", (x - 0.075, y, z + 0.035), (1.0, 0.0, 0.0)),
            ("side_pos_y", (x, y + 0.075, z + 0.035), (0.0, -1.0, 0.0)),
            ("side_neg_y", (x, y - 0.075, z + 0.035), (0.0, 1.0, 0.0)),
        )
        start = np.asarray(start_qpos, dtype=float)
        best_residual = float("inf")
        self.set_qpos(start)
        adaptive_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        tool_axis = adaptive_rotation[:, 2]
        cube = np.asarray(object_position, dtype=float)
        adaptive_pregrasp = cube - tool_axis * 0.15
        adaptive_grasp = cube - tool_axis * 0.075
        minimum_site_z = float(object_position[2]) + 0.025
        if (
            adaptive_pregrasp[2] >= minimum_site_z
            and adaptive_grasp[2] >= minimum_site_z
        ):
            adaptive_route = self.solve_path(
                [tuple(adaptive_pregrasp), tuple(adaptive_grasp)],
                orientation=adaptive_rotation,
            )
            if adaptive_route.success:
                return GraspPlanResult(
                    True,
                    "adaptive_oblique",
                    adaptive_route.joint_waypoints,
                    adaptive_route.max_residual,
                )
        self.set_qpos(start)
        adaptive_free = self.solve_path([(x, y, z + 0.075)])
        if adaptive_free.success:
            return GraspPlanResult(
                True,
                "adaptive_free_orientation",
                adaptive_free.joint_waypoints,
                adaptive_free.max_residual,
            )
        self.set_qpos(start)
        for style_index, (style, target, approach) in enumerate(candidates):
            orientation = self._rotation_from_approach(approach)
            approach_vector = np.asarray(approach)
            pregrasp = np.asarray(target) - approach_vector * 0.10
            incremental = self._incremental_grasp_route(
                start,
                pregrasp,
                np.asarray(target),
                orientation,
            )
            if incremental is not None:
                route, residual = incremental
                return GraspPlanResult(
                    True, style, tuple(tuple(q) for q in route), residual
                )
            pregrasp_solutions = self.collision_free_candidates(
                pregrasp,
                preferred_seed=start,
                random_restarts=random_restarts,
                rng_seed=30_000 + style_index,
                orientation=orientation,
                max_results=8,
            )
            pregrasp_solutions.sort(
                key=lambda result: np.linalg.norm(np.asarray(result.qpos) - start),
            )
            for candidate_index, pregrasp_result in enumerate(pregrasp_solutions):
                pregrasp_q = np.asarray(pregrasp_result.qpos)
                route = None
                if not self.validate_joint_segment(start, pregrasp_q):
                    route = [start, pregrasp_q]
                else:
                    route = self.plan_joint_rrt(
                        start,
                        pregrasp_q,
                        max_iterations=2500,
                        rng_seed=40_000 + style_index * 20 + candidate_index,
                    )
                if route is None:
                    continue
                grasp_result = self.solve(
                    target, seed=pregrasp_q, orientation=orientation
                )
                best_residual = min(best_residual, grasp_result.residual)
                if not grasp_result.success or self.robot_collision_pairs():
                    continue
                grasp_q = np.asarray(grasp_result.qpos)
                if self.validate_joint_segment(pregrasp_q, grasp_q, steps=16):
                    continue
                full_route = [tuple(q) for q in route]
                full_route.append(tuple(grasp_q))
                return GraspPlanResult(
                    True, style, tuple(full_route), grasp_result.residual
                )
        return GraspPlanResult(
            False,
            None,
            (),
            best_residual,
            "no collision-free top/side pre-grasp and approach",
        )

    def plan_physical_grasp(
        self,
        object_id: str,
        object_position: tuple[float, float, float],
        start_qpos,
        random_restarts: int = 96,
    ) -> GraspPlanResult:
        """Plan a finger-centered top/side grasp allowing target-cube contacts."""
        cube = np.asarray(object_position, dtype=float)
        styles = (
            ("top", (0.0, 0.0, -1.0)),
            ("side_pos_x", (-1.0, 0.0, 0.0)),
            ("side_neg_x", (1.0, 0.0, 0.0)),
            ("side_pos_y", (0.0, -1.0, 0.0)),
            ("side_neg_y", (0.0, 1.0, 0.0)),
        )
        start = np.asarray(start_qpos, dtype=float)
        for style_index, (style, approach) in enumerate(styles):
            approach_vector = np.asarray(approach, dtype=float)
            orientation = self._rotation_from_approach(approach_vector)
            grasp_target = cube.copy()
            grasp_target[2] += 0.02 if style == "top" else 0.04
            pregrasp = grasp_target - approach_vector * 0.14
            solutions = self.collision_free_candidates(
                pregrasp,
                preferred_seed=start,
                random_restarts=random_restarts,
                rng_seed=50_000 + style_index,
                orientation=orientation,
                max_results=10,
                orientation_tolerance=0.35,
            )
            solutions.sort(
                key=lambda item: np.linalg.norm(np.asarray(item.qpos) - start)
            )
            for candidate_index, pregrasp_result in enumerate(solutions):
                pregrasp_q = np.asarray(pregrasp_result.qpos)
                route = None
                if not self.validate_joint_segment(start, pregrasp_q):
                    route = [start, pregrasp_q]
                else:
                    route = self.plan_joint_rrt(
                        start,
                        pregrasp_q,
                        max_iterations=4000,
                        rng_seed=60_000 + style_index * 20 + candidate_index,
                    )
                if route is None:
                    continue
                grasp = self.solve(
                    grasp_target,
                    seed=pregrasp_q,
                    orientation=orientation,
                    orientation_tolerance=0.08,
                )
                if not grasp.success or self.robot_collision_pairs(
                    allowed_object_id=object_id,
                ):
                    continue
                grasp_q = np.asarray(grasp.qpos)
                if self.validate_joint_segment(
                    pregrasp_q,
                    grasp_q,
                    steps=24,
                    allowed_object_id=object_id,
                ):
                    continue
                full_route = [tuple(q) for q in route]
                full_route.append(tuple(grasp_q))
                return GraspPlanResult(True, style, tuple(full_route), grasp.residual)
        self.set_qpos(start)
        return GraspPlanResult(
            False, None, (), float("inf"), "no contact-valid physical grasp path"
        )

    def _incremental_grasp_route(
        self,
        start: np.ndarray,
        pregrasp: np.ndarray,
        target: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[list[np.ndarray], float] | None:
        """Move to pre-grasp, rotate gradually, then descend along tool approach."""
        position_result = self.solve(pregrasp, seed=start)
        if not position_result.success or self.robot_collision_pairs():
            return None
        position_q = np.asarray(position_result.qpos)
        if self.validate_joint_segment(start, position_q, steps=16):
            return None
        route = [start, position_q]
        current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        current_q = position_q
        max_residual = position_result.residual
        for alpha in np.linspace(0.12, 1.0, 9):
            mixed = (1.0 - alpha) * current_rotation + alpha * target_rotation
            u, _, vt = np.linalg.svd(mixed)
            intermediate_rotation = u @ vt
            if np.linalg.det(intermediate_rotation) < 0:
                u[:, -1] *= -1
                intermediate_rotation = u @ vt
            result = self.solve(
                pregrasp, seed=current_q, orientation=intermediate_rotation
            )
            if not result.success or self.robot_collision_pairs():
                return None
            next_q = np.asarray(result.qpos)
            if self.validate_joint_segment(current_q, next_q, steps=8):
                return None
            route.append(next_q)
            current_q = next_q
            max_residual = max(max_residual, result.residual)
        for alpha in np.linspace(0.2, 1.0, 5):
            point = pregrasp + alpha * (target - pregrasp)
            result = self.solve(point, seed=current_q, orientation=target_rotation)
            if not result.success or self.robot_collision_pairs():
                return None
            next_q = np.asarray(result.qpos)
            if self.validate_joint_segment(current_q, next_q, steps=8):
                return None
            route.append(next_q)
            current_q = next_q
            max_residual = max(max_residual, result.residual)
        return route, max_residual

    @staticmethod
    def _rotation_from_approach(approach) -> np.ndarray:
        z_axis = np.asarray(approach, dtype=float)
        z_axis /= np.linalg.norm(z_axis)
        reference = (
            np.array([1.0, 0.0, 0.0])
            if abs(z_axis[2]) > 0.8
            else np.array([0.0, 0.0, 1.0])
        )
        x_axis = np.cross(reference, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def colliding_geom_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            name1 = self.mujoco.mj_id2name(
                self.model,
                self.mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom1,
            )
            name2 = self.mujoco.mj_id2name(
                self.model,
                self.mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom2,
            )
            pairs.append((name1 or str(contact.geom1), name2 or str(contact.geom2)))
        return pairs

    def robot_collision_pairs(
        self,
        allowed_object_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return contacts involving Panda links, excluding its mounted base/table contact."""
        pairs = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if allowed_object_id is not None:
                allowed_body = self.model.body(f"cube_{allowed_object_id}").id
                if allowed_body in (body1, body2):
                    continue
            robot1, robot2 = self._is_robot_body(body1), self._is_robot_body(body2)
            if not (robot1 or robot2):
                continue
            name1 = self._contact_label(contact.geom1, body1)
            name2 = self._contact_label(contact.geom2, body2)
            if {self._body_name(body1), self._body_name(body2)} == {"link0", "world"}:
                continue
            if "table" in (name1, name2) and "link0" in (name1, name2):
                continue
            pairs.append((name1, name2))
        return pairs

    def solve_path(
        self,
        targets: list[tuple[float, float, float]],
        orientation: np.ndarray | None = None,
        allowed_object_id: str | None = None,
    ) -> IKPathResult:
        joint_waypoints = [tuple(self.current_qpos())]
        max_residual = 0.0
        all_collisions: list[tuple[str, str]] = []
        for target_index, target in enumerate(targets):
            start = np.asarray(joint_waypoints[-1])
            local_result = self.solve(target, seed=start, orientation=orientation)
            if local_result.success and not self.robot_collision_pairs(
                allowed_object_id=allowed_object_id,
            ):
                local_goal = np.asarray(local_result.qpos)
                if not self.validate_joint_segment(
                    start,
                    local_goal,
                    allowed_object_id=allowed_object_id,
                ):
                    max_residual = max(max_residual, local_result.residual)
                    joint_waypoints.append(local_result.qpos)
                    continue
            candidates = self.collision_free_candidates(
                target,
                preferred_seed=start,
                random_restarts=48,
                rng_seed=1000 + target_index,
                max_results=8,
                orientation=orientation,
                allowed_object_id=allowed_object_id,
            )
            if not candidates:
                return IKPathResult(
                    False,
                    tuple(joint_waypoints),
                    max_residual,
                    tuple(all_collisions),
                    "no collision-free IK candidate",
                )
            route = None
            selected = None
            for result in candidates:
                candidate = np.asarray(result.qpos)
                if not self.validate_joint_segment(
                    start,
                    candidate,
                    allowed_object_id=allowed_object_id,
                ):
                    route, selected = [start, candidate], result
                    break
            if route is None:
                for candidate_index, result in enumerate(candidates):
                    route = self.plan_joint_rrt(
                        start,
                        np.asarray(result.qpos),
                        rng_seed=10_000 + target_index * 10 + candidate_index,
                        allowed_object_id=allowed_object_id,
                    )
                    if route is not None:
                        selected = result
                        break
            if route is None or selected is None:
                return IKPathResult(
                    False,
                    tuple(joint_waypoints),
                    max_residual,
                    tuple(all_collisions),
                    "joint-space RRT failed",
                )
            max_residual = max(max_residual, selected.residual)
            joint_waypoints.extend(tuple(q) for q in route[1:])
            self.set_qpos(np.asarray(joint_waypoints[-1]))
        return IKPathResult(
            True, tuple(joint_waypoints), max_residual, tuple(all_collisions)
        )

    def plan_joint_rrt(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_iterations: int = 2500,
        step_size: float = 0.28,
        rng_seed: int = 0,
        allowed_object_id: str | None = None,
    ) -> list[np.ndarray] | None:
        """Bidirectional RRT-Connect between two valid seven-joint states."""
        self.set_qpos(start)
        if self.robot_collision_pairs(allowed_object_id=allowed_object_id):
            return None
        if not self.validate_joint_segment(
            start,
            goal,
            allowed_object_id=allowed_object_id,
        ):
            return [start, goal]
        rng = np.random.default_rng(rng_seed)
        nodes_a, parents_a = [start.copy()], [-1]
        nodes_b, parents_b = [goal.copy()], [-1]
        a_from_start = True
        scale = np.maximum(self.upper - self.lower, 1e-6)

        def extend(nodes, parents, target):
            distances = [np.linalg.norm((node - target) / scale) for node in nodes]
            nearest_index = int(np.argmin(distances))
            nearest = nodes[nearest_index]
            direction = target - nearest
            distance = float(np.linalg.norm(direction))
            if distance == 0.0:
                return nearest_index
            new = nearest + direction / distance * min(step_size, distance)
            new = np.clip(new, self.lower + 1e-5, self.upper - 1e-5)
            if self.validate_joint_segment(
                nearest,
                new,
                steps=5,
                allowed_object_id=allowed_object_id,
            ):
                return None
            nodes.append(new)
            parents.append(nearest_index)
            return len(nodes) - 1

        def root_path(nodes, parents, index):
            path = []
            while index >= 0:
                path.append(nodes[index])
                index = parents[index]
            path.reverse()
            return path

        for _ in range(max_iterations):
            sample = rng.uniform(self.lower, self.upper)
            index_a = extend(nodes_a, parents_a, sample)
            if index_a is not None:
                target = nodes_a[index_a]
                while True:
                    index_b = extend(nodes_b, parents_b, target)
                    if index_b is None:
                        break
                    if np.linalg.norm(nodes_b[index_b] - target) < 1e-6:
                        path_a = root_path(nodes_a, parents_a, index_a)
                        path_b = root_path(nodes_b, parents_b, index_b)
                        if a_from_start:
                            path = path_a + list(reversed(path_b))[1:]
                        else:
                            path = path_b + list(reversed(path_a))[1:]
                        self.set_qpos(goal)
                        return path
                    if len(nodes_b) > max_iterations * 4:
                        break
            nodes_a, nodes_b = nodes_b, nodes_a
            parents_a, parents_b = parents_b, parents_a
            a_from_start = not a_from_start
        self.set_qpos(start)
        return None

    def validate_joint_segment(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        steps: int = 12,
        allowed_object_id: str | None = None,
    ) -> list[tuple[str, str]]:
        for alpha in np.linspace(0.0, 1.0, steps):
            self.set_qpos(start + alpha * (goal - start))
            collisions = self.robot_collision_pairs(allowed_object_id=allowed_object_id)
            if collisions:
                self.set_qpos(goal)
                return collisions
        self.set_qpos(goal)
        return []

    def _is_robot_body(self, body_id: int) -> bool:
        while body_id > 0:
            name = self._body_name(body_id)
            if name == "link0":
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def _body_name(self, body_id: int) -> str:
        if body_id == 0:
            return "world"
        return self.mujoco.mj_id2name(
            self.model,
            self.mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        ) or str(body_id)

    def _contact_label(self, geom_id: int, body_id: int) -> str:
        return self.mujoco.mj_id2name(
            self.model,
            self.mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        ) or self._body_name(body_id)
