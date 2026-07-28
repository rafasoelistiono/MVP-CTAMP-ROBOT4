"""Seeded IPC-style Blocksworld instances grounded in MuJoCo cube poses."""

from __future__ import annotations

import copy
import random
from typing import Any

from . import GeneratedProblem, Pose


def _random_stacks(
    object_ids: list[str],
    rng: random.Random,
    minimum: int = 2,
    maximum: int | None = None,
) -> list[list[str]]:
    upper = min(3, len(object_ids), maximum or len(object_ids))
    count = rng.randint(min(minimum, upper), upper)
    shuffled = rng.sample(object_ids, len(object_ids))
    stacks = [[object_id] for object_id in shuffled[:count]]
    for object_id in shuffled[count:]:
        rng.choice(stacks).append(object_id)
    rng.shuffle(stacks)
    return stacks


def _relations(stacks: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    facts: list[tuple[str, ...]] = []
    for stack in stacks:
        facts.append(("on-table", stack[0]))
        facts.extend(("on", above, below) for below, above in zip(stack, stack[1:]))
        facts.append(("clear", stack[-1]))
    return tuple(facts)


def _stack_poses(
    stacks: list[list[str]],
    origins: list[list[float]],
    sizes: dict[str, tuple[float, float, float]],
    table_z: float,
) -> dict[str, Pose]:
    poses: dict[str, Pose] = {}
    for index, stack in enumerate(stacks):
        x, y = origins[index]
        surface_z = table_z
        for object_id in stack:
            height = sizes[object_id][2]
            poses[object_id] = (float(x), float(y), surface_z + height / 2.0)
            surface_z += height
    return poses


def generate_problem(
    base_config: dict[str, Any], num_objects: int = 3, seed: int = 0
) -> GeneratedProblem:
    if not 2 <= num_objects <= 6:
        raise ValueError("Blocksworld num_objects must be between 2 and 6")
    config = copy.deepcopy(base_config)
    candidates = [obj for obj in config["objects"] if obj["id"].startswith("c")]
    if num_objects > len(candidates):
        raise ValueError(f"scene only defines {len(candidates)} block objects")
    objects = candidates[:num_objects]
    object_ids = [obj["id"] for obj in objects]
    config["objects"] = objects

    rng = random.Random(seed)
    initial_stacks = _random_stacks(object_ids, rng)
    goal_stacks = _random_stacks(
        object_ids, rng, minimum=1, maximum=max(1, len(object_ids) - 1)
    )
    while _relations(goal_stacks) == _relations(initial_stacks):
        goal_stacks = _random_stacks(
            object_ids, rng, minimum=1, maximum=max(1, len(object_ids) - 1)
        )

    challenge = config["blocksworld"]
    table_z = float(config["table"]["z_top"])
    default_size = tuple(float(v) for v in config["geometry"]["cube_size_xyz"])
    sizes = {
        obj["id"]: tuple(float(v) for v in obj.get("size_xyz", default_size))
        for obj in objects
    }
    initial_poses = _stack_poses(
        initial_stacks, challenge["initial_stack_origins"], sizes, table_z
    )
    initial_bottoms = {stack[0] for stack in initial_stacks}
    goal_origins: list[list[float]] = []
    for index, stack in enumerate(goal_stacks):
        bottom = stack[0]
        if bottom in initial_bottoms:
            goal_origins.append(list(initial_poses[bottom][:2]))
        else:
            goal_origins.append(list(challenge["goal_stack_origins"][index]))
    goal_poses = _stack_poses(
        goal_stacks, goal_origins, sizes, table_z
    )
    for obj in objects:
        obj["pose"] = list(initial_poses[obj["id"]])

    safe_origin = challenge["temporary_table_origin"]
    safe_spacing = float(challenge.get("temporary_table_spacing", 0.11))
    placements: dict[tuple[str, str], Pose] = {}
    goal_bottoms = {stack[0] for stack in goal_stacks}
    for index, object_id in enumerate(object_ids):
        placements[(object_id, "table")] = (
            goal_poses[object_id]
            if object_id in goal_bottoms
            else (
                float(safe_origin[0]) + index * safe_spacing,
                float(safe_origin[1]),
                table_z + sizes[object_id][2] / 2.0,
            )
        )
        for support_id in object_ids:
            if object_id == support_id:
                continue
            support_pose = goal_poses.get(support_id, initial_poses[support_id])
            placements[(object_id, support_id)] = (
                support_pose[0],
                support_pose[1],
                support_pose[2]
                + sizes[support_id][2] / 2.0
                + sizes[object_id][2] / 2.0,
            )
    for stack in goal_stacks:
        for object_id in stack:
            if object_id != stack[0]:
                support_id = stack[stack.index(object_id) - 1]
                placements[(object_id, support_id)] = goal_poses[object_id]

    home = tuple(float(v) for v in config["robot"]["physical_start_qpos"])
    init: list[tuple[Any, ...]] = [("arm-empty",), ("is-surface", "table"), ("home-config", home), ("is-config", home)]
    init.extend(("is-object", object_id) for object_id in object_ids)
    init.extend(("is-surface", object_id) for object_id in object_ids)
    init.extend(
        ("at-pose", object_id, initial_poses[object_id])
        for object_id in object_ids
    )
    init.extend(("is-pose", initial_poses[object_id]) for object_id in object_ids)
    init.extend(_relations(initial_stacks))
    goal_facts = [fact for fact in _relations(goal_stacks) if fact[0] in {"on", "on-table"}]

    config["task"]["target_objects"] = object_ids
    config["task"]["preserve_order"] = False
    metadata = {
        "initial_stacks_bottom_to_top": initial_stacks,
        "goal_stacks_bottom_to_top": goal_stacks,
        "initial_poses": {key: list(value) for key, value in initial_poses.items()},
        "goal_poses": {key: list(value) for key, value in goal_poses.items()},
    }
    return GeneratedProblem(
        "blocksworld",
        seed,
        tuple(object_ids),
        config,
        tuple(init),
        ("and", *goal_facts),
        placements,
        metadata,
    )
