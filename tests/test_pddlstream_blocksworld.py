import json
from pathlib import Path

import pytest
import yaml

from ctamp.experiments.run_pddlstream import run as run_pddlstream
from ctamp.pddlstream.problems.blocksworld_generator import generate_problem


CONFIG = Path("configs/scenes/blocksworld_challenge.yaml")


def test_blocksworld_generator_is_seeded_for_literature_sizes():
    config = yaml.safe_load(CONFIG.read_text())
    for num_objects in range(3, 7):
        first = generate_problem(config, num_objects=num_objects, seed=11)
        repeated = generate_problem(config, num_objects=num_objects, seed=11)
        assert first.metadata == repeated.metadata
        assert len(first.object_ids) == num_objects


def test_blocksworld_dry_run_searches_goal_order_and_samples_streams(tmp_path):
    first_output = tmp_path / "seed_1"
    second_output = tmp_path / "seed_2"
    first = run_pddlstream(
        "blocksworld", CONFIG, first_output, num_objects=3, seed=1, dry_run=True
    )
    second = run_pddlstream(
        "blocksworld", CONFIG, second_output, num_objects=3, seed=2, dry_run=True
    )

    assert first["solution_found"] is True
    assert second["solution_found"] is True
    assert first["planning"]["plan_length"] > 0
    assert first["planning"]["planning_time_seconds"] >= 0
    assert first["planning"]["stream"]["evaluations"] > 0
    assert "failures" in first["planning"]["stream"]
    assert first["planning"]["stream"]["samples"] > 0
    assert first["problem"]["goal_stacks_bottom_to_top"] != second["problem"][
        "goal_stacks_bottom_to_top"
    ]
    first_plan = json.loads((first_output / "pddlstream_plan.json").read_text())
    second_plan = json.loads((second_output / "pddlstream_plan.json").read_text())
    assert first_plan != second_plan
    generated = yaml.safe_load((first_output / "generated_scene.yaml").read_text())
    assert generated["task"]["preserve_order"] is False
    assert (first_output / "adapted_plan.json").exists()


@pytest.mark.simulation
def test_blocksworld_executes_pddlstream_placements_in_mujoco(tmp_path):
    metrics = run_pddlstream(
        "blocksworld", CONFIG, tmp_path / "executed", num_objects=3, seed=0
    )

    assert metrics["solution_found"] is True
    assert metrics["execution"]["completed_placements"] > 0
    for phase in metrics["execution"]["phases"]:
        assert phase["solution_found"] is True
        assert len(phase["measured_pose"]) == 3
