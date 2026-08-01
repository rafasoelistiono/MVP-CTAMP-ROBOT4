"""Configuration, tidy-slot generation, and conservative 2-D motion probes."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

Point = tuple[float, float]


@dataclass(frozen=True)
class GoalSlot:
    name: str
    group_id: str
    color: str
    object_id: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class ProbeResult:
    success: bool
    route_type: str
    waypoints: tuple[Point, ...]
    length: float
    clearance: float
    reason: str | None = None


def load_scene_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("scene config must contain a mapping")
    return config


def generate_tidy_slots(config: dict[str, Any]) -> dict[str, GoalSlot]:
    tidy = config["grouped_tidy"]
    axis = tidy.get("axis", "y")
    if axis not in {"x", "y", "z"}:
        raise ValueError("grouped tidy axis must be x, y, or z")
    prefix = tidy.get("slot_prefix", "tidy_slot")
    spacing = float(tidy["spacing"])
    slots: dict[str, GoalSlot] = {}
    for group in config["tidy_groups"]:
        objects = list(group["objects"])
        explicit_positions = group.get("positions", {})
        if explicit_positions:
            for index, object_id in enumerate(objects):
                position = tuple(float(v) for v in explicit_positions[object_id])
                slots[object_id] = GoalSlot(
                    name=f"{prefix}_{group['id']}_{index}",
                    group_id=group["id"],
                    color=group["color"],
                    object_id=object_id,
                    position=position,
                )
            continue
        center = tuple(float(v) for v in group["center"])
        midpoint = (len(objects) - 1) / 2.0
        for index, object_id in enumerate(objects):
            offset = (index - midpoint) * spacing
            if axis == "x":
                position = (center[0] + offset, center[1], center[2])
            elif axis == "y":
                position = (center[0], center[1] + offset, center[2])
            else:
                position = (center[0], center[1], center[2] + offset)
            slots[object_id] = GoalSlot(
                name=f"{prefix}_{group['id']}_{index}",
                group_id=group["id"],
                color=group["color"],
                object_id=object_id,
                position=position,
            )
    return slots


def _segment_intersects_rect(
    a: Point, b: Point, rect: tuple[float, float, float, float]
) -> bool:
    """Liang-Barsky closed-segment/axis-aligned-rectangle intersection."""
    xmin, xmax, ymin, ymax = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    p = (-dx, dx, -dy, dy)
    q = (a[0] - xmin, xmax - a[0], a[1] - ymin, ymax - a[1])
    low, high = 0.0, 1.0
    for pi, qi in zip(p, q, strict=True):
        if pi == 0.0:
            if qi < 0.0:
                return False
            continue
        t = qi / pi
        if pi < 0.0:
            low = max(low, t)
        else:
            high = min(high, t)
        if low > high:
            return False
    return True


def _polyline_length(points: Iterable[Point]) -> float:
    pts = list(points)
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))


class MotionProbe:
    """Probe direct and left/right detours around inflated rectangular obstacles."""

    def __init__(self, config: dict[str, Any], clearance: float = 0.055) -> None:
        self.config = config
        self.clearance = float(clearance)
        self.table_x = tuple(config["table"]["x_range"])
        self.table_y = tuple(config["table"]["y_range"])
        robot = config["robot"]
        self.base = tuple(robot["base_xy"])
        self.reach_min = float(robot["reach_min_xy"])
        self.reach_max = float(robot["reach_max_xy"])
        self.rectangles = [
            self._inflated_rect(obstacle)
            for obstacle in config.get("obstacles", [])
            if obstacle.get("blocks_motion_probe", True)
        ]

    def _inflated_rect(
        self, obstacle: dict[str, Any]
    ) -> tuple[float, float, float, float]:
        x, y = (float(v) for v in obstacle["pose"][:2])
        sx, sy = (float(v) for v in obstacle["size"][:2])
        return (
            x - sx / 2 - self.clearance,
            x + sx / 2 + self.clearance,
            y - sy / 2 - self.clearance,
            y + sy / 2 + self.clearance,
        )

    def point_valid(self, point: Point) -> bool:
        if not (
            self.table_x[0] <= point[0] <= self.table_x[1]
            and self.table_y[0] <= point[1] <= self.table_y[1]
        ):
            return False
        reach = math.dist(self.base, point)
        return self.reach_min <= reach <= self.reach_max

    def path_clear(self, points: Iterable[Point]) -> bool:
        pts = tuple(points)
        if not pts or any(not self.point_valid(p) for p in pts):
            return False
        return not any(
            _segment_intersects_rect(a, b, rect)
            for a, b in zip(pts, pts[1:], strict=False)
            for rect in self.rectangles
        )

    def probe(self, start: Point, goal: Point) -> ProbeResult:
        direct = (start, goal)
        if self.path_clear(direct):
            return ProbeResult(
                True, "direct", direct, _polyline_length(direct), self.clearance
            )
        margin = 0.01
        candidates: list[tuple[str, tuple[Point, ...]]] = []
        for rect in self.rectangles:
            xmin, xmax, ymin, ymax = rect
            # Detour through the two table corridors on either side of the wall.
            # Naming is from the robot's viewpoint; geometrically these are the
            # lower/upper y sides of the rectangular footprint.
            candidates.extend(
                [
                    (
                        "left_corridor",
                        (
                            start,
                            (start[0], ymax + margin),
                            (goal[0], ymax + margin),
                            goal,
                        ),
                    ),
                    (
                        "right_corridor",
                        (
                            start,
                            (start[0], ymin - margin),
                            (goal[0], ymin - margin),
                            goal,
                        ),
                    ),
                ]
            )
        valid = [(name, path) for name, path in candidates if self.path_clear(path)]
        if valid:
            name, path = min(valid, key=lambda item: _polyline_length(item[1]))
            return ProbeResult(True, name, path, _polyline_length(path), self.clearance)
        return ProbeResult(
            False,
            "failed",
            direct,
            _polyline_length(direct),
            self.clearance,
            "no collision-free route within table and reach limits",
        )
