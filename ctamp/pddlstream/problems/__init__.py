"""Procedural challenge problem representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Pose = tuple[float, float, float]


@dataclass(frozen=True)
class GeneratedProblem:
    domain: str
    seed: int
    object_ids: tuple[str, ...]
    scene_config: dict[str, Any]
    init: tuple[tuple[Any, ...], ...]
    goal: tuple[Any, ...]
    placement_poses: dict[tuple[str, str], Pose]
    metadata: dict[str, Any]


__all__ = ["GeneratedProblem", "Pose"]
