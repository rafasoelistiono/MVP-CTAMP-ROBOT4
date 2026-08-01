"""Seeded Kitchen cleaning/cooking instances grounded in box primitives."""

from __future__ import annotations

import copy
import random
from typing import Any

from . import GeneratedProblem, Pose


def generate_problem(
    base_config: dict[str, Any], num_objects: int = 3, seed: int = 0
) -> GeneratedProblem:
    if not 1 <= num_objects <= 6:
        raise ValueError("Kitchen num_objects must be between 1 and 6")
    config = copy.deepcopy(base_config)
    rng = random.Random(seed)
    foods = [obj for obj in config["objects"] if obj.get("class") == "food"]
    if num_objects > len(foods):
        raise ValueError(f"scene only defines {len(foods)} food objects")
    selected = rng.sample(foods, num_objects)
    selected_ids = [obj["id"] for obj in selected]
    config["objects"] = selected

    kitchen = config["kitchen"]
    table_z = float(config["table"]["z_top"])
    positions = [tuple(float(v) for v in xy) for xy in kitchen["initial_food_xy"]]
    rng.shuffle(positions)
    placements: dict[tuple[str, str], Pose] = {}
    initial_surfaces: dict[str, str] = {}
    surface_offsets = [
        tuple(float(value) for value in offset)
        for offset in kitchen["surface_offsets"]
    ]
    for index, obj in enumerate(selected):
        object_id = obj["id"]
        height = float(obj.get("size_xyz", config["geometry"]["cube_size_xyz"])[2])
        x, y = positions[index]
        jitter = (rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015))
        obj["pose"] = [x + jitter[0], y + jitter[1], table_z + height / 2.0]
        initial_surface = f"initial-{object_id}"
        initial_surfaces[object_id] = initial_surface
        placements[(object_id, initial_surface)] = tuple(obj["pose"])
        for surface_id in ("sink", "stove"):
            surface = next(item for item in config["obstacles"] if item["id"] == surface_id)
            offset_x, offset_y = surface_offsets[index]
            placements[(object_id, surface_id)] = (
                float(surface["pose"][0]) + offset_x,
                float(surface["pose"][1]) + offset_y,
                float(surface["pose"][2]) + float(surface["size"][2]) / 2.0 + height / 2.0,
            )

    home = tuple(float(v) for v in config["robot"]["physical_start_qpos"])
    init: list[tuple[Any, ...]] = [
        ("IsGripper", "gripper"),
        ("Empty", "gripper"),
        ("IsSink", "sink"),
        ("IsStove", "stove"),
        ("is-surface", "sink"),
        ("is-surface", "stove"),
        ("home-config", home),
        ("is-config", home),
    ]
    for object_id in selected_ids:
        initial_surface = initial_surfaces[object_id]
        object_pose = tuple(
            float(value)
            for value in next(obj for obj in selected if obj["id"] == object_id)["pose"]
        )
        init.extend(
            [
                ("IsFood", object_id),
                ("is-object", object_id),
                ("is-surface", initial_surface),
                ("IsPose", object_id, object_pose),
                ("is-pose", object_pose),
                ("AtPose", object_id, object_pose),
                ("AtSurface", object_id, initial_surface),
            ]
        )
    goal = ("and", *(("Cooked", object_id) for object_id in selected_ids))
    config["task"]["target_objects"] = selected_ids
    config["task"]["preserve_order"] = False
    metadata = {
        "food_objects": selected_ids,
        "initial_surfaces": initial_surfaces,
        "goal_cooked": selected_ids,
    }
    return GeneratedProblem(
        "kitchen",
        seed,
        tuple(selected_ids),
        config,
        tuple(init),
        goal,
        placements,
        metadata,
    )
