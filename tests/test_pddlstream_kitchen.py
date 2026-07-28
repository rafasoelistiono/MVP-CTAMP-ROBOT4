from pathlib import Path

import pytest
import yaml

from ctamp.experiments.run_pddlstream import (
    _placement_sequence_config,
    run as run_pddlstream,
)
from ctamp.pddlstream.plan_adapter import AdaptedPlan, PlacementStep
from ctamp.pddlstream.problems.kitchen_generator import generate_problem


CONFIG = Path("configs/scenes/kitchen_challenge.yaml")


def test_kitchen_generator_is_seeded_for_literature_sizes():
    config = yaml.safe_load(CONFIG.read_text())
    for num_objects in range(3, 7):
        first = generate_problem(config, num_objects=num_objects, seed=11)
        repeated = generate_problem(config, num_objects=num_objects, seed=11)
        assert first.scene_config["objects"] == repeated.scene_config["objects"]
        assert len(first.object_ids) == num_objects


def test_kitchen_builds_one_ordered_continuous_sequence():
    config = yaml.safe_load(CONFIG.read_text())
    problem = generate_problem(config, num_objects=1, seed=1)
    object_id = problem.object_ids[0]
    adapted = AdaptedPlan(
        placements=(
            PlacementStep(
                1,
                "place",
                object_id,
                "sink",
                problem.placement_poses[(object_id, "sink")],
            ),
            PlacementStep(
                4,
                "place",
                object_id,
                "stove",
                problem.placement_poses[(object_id, "stove")],
            ),
        ),
        state_actions=(),
    )

    sequence = _placement_sequence_config(problem, adapted)

    assert sequence["task"]["target_objects"] == [object_id, object_id]
    assert [
        step["object_id"] for step in sequence["task"]["placement_sequence"]
    ] == [object_id, object_id]
    assert sequence["tidy_groups"] == []


def test_kitchen_dry_run_cleans_before_cooking_and_samples_streams(tmp_path):
    metrics = run_pddlstream(
        "kitchen", CONFIG, tmp_path / "kitchen", num_objects=1, seed=1, dry_run=True
    )

    assert metrics["solution_found"] is True
    assert metrics["planning"]["plan_length"] == 6
    assert metrics["planning"]["planning_time_seconds"] >= 0
    assert metrics["planning"]["stream"]["evaluations"] > 0
    assert "failures" in metrics["planning"]["stream"]
    assert metrics["planning"]["stream"]["samples"] > 0
    assert metrics["object_order"].count("egg") == 2
    actions_by_object: dict[str, list[str]] = {}
    for action in metrics["state_actions"]:
        actions_by_object.setdefault(action["object_id"], []).append(action["action"])
    assert set(actions_by_object) == set(metrics["problem"]["goal_cooked"])
    assert all(actions == ["clean", "cook"] for actions in actions_by_object.values())
    assert (tmp_path / "kitchen" / "adapted_plan.json").exists()


@pytest.mark.simulation
def test_kitchen_executes_sink_and_stove_phases_in_mujoco(tmp_path):
    metrics = run_pddlstream(
        "kitchen", CONFIG, tmp_path / "executed", num_objects=3, seed=0
    )

    assert metrics["solution_found"] is True
    assert metrics["execution"]["execution_mode"] == "continuous_scene"
    assert metrics["execution"]["completed_placements"] == 6
    assert all(phase["solution_found"] for phase in metrics["execution"]["phases"])
    assert [action["action"] for action in metrics["state_actions"]] == [
        "clean",
        "cook",
        "clean",
        "cook",
        "clean",
        "cook",
    ]
