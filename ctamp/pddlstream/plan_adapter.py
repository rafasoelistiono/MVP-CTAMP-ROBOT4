"""Translate raw PDDLStream actions into executor placement phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .problems import Pose


@dataclass(frozen=True)
class PlacementStep:
    action_index: int
    action: str
    object_id: str
    surface_id: str
    target_pose: Pose


@dataclass(frozen=True)
class AdaptedPlan:
    placements: tuple[PlacementStep, ...]
    state_actions: tuple[dict[str, Any], ...]

    @property
    def object_order(self) -> list[str]:
        return [step.object_id for step in self.placements]


def adapt_plan(domain: str, raw_plan: tuple[Any, ...]) -> AdaptedPlan:
    placements: list[PlacementStep] = []
    state_actions: list[dict[str, Any]] = []
    for index, raw_action in enumerate(raw_plan):
        name = str(raw_action.name).lower()
        args = raw_action.args
        if domain == "blocksworld" and name == "putdown":
            placements.append(PlacementStep(index, name, str(args[0]), "table", tuple(args[1])))
        elif domain == "blocksworld" and name == "stack":
            placements.append(
                PlacementStep(index, name, str(args[0]), str(args[1]), tuple(args[3]))
            )
        elif domain == "kitchen" and name == "place":
            placements.append(
                PlacementStep(index, name, str(args[1]), str(args[2]), tuple(args[4]))
            )
        elif domain == "kitchen" and name in {"clean", "cook"}:
            state_actions.append(
                {"action_index": index, "action": name, "object_id": str(args[0])}
            )
    return AdaptedPlan(tuple(placements), tuple(state_actions))


def serializable_action(raw_action: Any) -> dict[str, Any]:
    return {"name": str(raw_action.name), "args": _json_value(raw_action.args)}


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value
