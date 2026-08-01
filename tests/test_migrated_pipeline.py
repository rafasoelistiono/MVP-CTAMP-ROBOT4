from pathlib import Path
from types import SimpleNamespace

import yaml

from cli.run_simulation import _context_config
from ctamp.cost import EdgeCostCalculator
from ctamp.experiments.run_scene import (
    _ObjectExecution,
    _completion_status,
    _per_object_result,
    _plan_action,
    _probe_transfer,
)
from ctamp.experiments.scene_helpers import placement_within_tolerance
from ctamp.experiments.run_stacking_v2 import build_phase_configs
from ctamp.learning.heuristic_models import OnlineSGDModel
from ctamp.planning.symbolic import PlanningProblem, SymbolicTaskPlanner
from ctamp.search.tmm_astar import TMMAStar
from ctamp.simulation.scene import generate_tidy_slots
from ctamp.tmm.builder import TMMGraphBuilder


def test_context_adapter_feeds_migrated_scene_pipeline():
    config = _context_config(Path("contexts/examples/align_grouped_tidy_wall_world.md"))

    assert config["scene"]["scene_id"] == "align_grouped_tidy_wall_world"
    assert config["grouped_tidy"]["axis"] == "y"
    assert len(config["objects"]) == 12
    assert len(generate_tidy_slots(config)) == 12


def test_source_learning_planning_cost_search_tmm_are_imported():
    assert OnlineSGDModel is not None
    assert EdgeCostCalculator is not None
    assert (
        SymbolicTaskPlanner(PlanningProblem(objects={}, target_poses={})).solve()
        is not None
    )
    assert TMMGraphBuilder is not None
    assert TMMAStar is not None


def test_stacking_v2_builds_placeholder_then_large_to_small_stack():
    config = yaml.safe_load(
        Path("configs/scenes/stacking_wall_world_v2.yaml").read_text()
    )

    phase1, phase2, summary = build_phase_configs(config)
    sizes = {obj["id"]: obj["size_xyz"][0] for obj in config["objects"]}
    grip_widths = {obj["id"]: obj["grip_target_width"] for obj in config["objects"]}

    assert phase1["task"]["target_objects"] == ["c6", "c5", "c4", "c3", "c2", "c1"]
    assert phase2["task"]["target_objects"] == ["c6", "c5", "c4", "c3", "c2", "c1"]
    assert phase2["task"]["preserve_order"] is True
    assert summary["largest_to_smallest_order"] == ["c6", "c5", "c4", "c3", "c2", "c1"]
    assert sizes["c6"] - sizes["c1"] == 0.04
    assert grip_widths["c1"] < grip_widths["c6"]
    assert (
        summary["safe_zone_positions"]["c6"][0]
        < summary["safe_zone_positions"]["c1"][0]
    )
    assert summary["safe_zone_positions"]["c1"][2] < 0.84
    assert summary["safe_zone_positions"]["c6"][2] < 0.85
    assert summary["final_stack_positions"]["c6"][:2] == [-0.30, -0.75]
    assert (
        summary["final_stack_positions"]["c6"][2]
        < summary["final_stack_positions"]["c1"][2]
    )
    assert len(generate_tidy_slots(phase1)) == 6
    assert len(generate_tidy_slots(phase2)) == 6


def test_scene_runner_helpers_preserve_output_payload_shape():
    slot = SimpleNamespace(name="tidy_slot_red_lane_0", position=(0.1, 0.2, 0.83))
    transit = SimpleNamespace(
        metadata={"route_type": "direct"},
        waypoints=[[0.0, 0.0], [0.1, 0.0]],
        length=0.1,
    )
    motion = SimpleNamespace(
        metadata={"route_type": "left_corridor", "reason": "detour"},
        waypoints=[[0.1, 0.0], [0.1, 0.2]],
        length=0.2,
    )
    execution = _ObjectExecution(
        ik_success=True,
        transit_joint_waypoints=[[1.0]],
        transfer_joint_waypoints=[[2.0]],
        grasp_style="top",
    )

    action = _plan_action("j", slot, "left_corridor", transit, motion, execution, 0.052)
    result = _per_object_result(
        "j",
        slot,
        True,
        "left_corridor",
        1,
        "detour",
        transit,
        0.5,
        motion,
        True,
        execution,
    )

    assert action == {
        "object_id": "j",
        "slot": "tidy_slot_red_lane_0",
        "route_type": "left_corridor",
        "transit_route_type": "direct",
        "transit_waypoints": [[0.0, 0.0], [0.1, 0.0]],
        "transfer_waypoints": [[0.1, 0.0], [0.1, 0.2]],
        "transit_joint_waypoints": [[1.0]],
        "transfer_joint_waypoints": [[2.0]],
        "waypoints": [[0.1, 0.0], [0.1, 0.2]],
        "z": 0.83,
        "grasp_width": 0.052,
    }
    assert result["object_id"] == "j"
    assert result["ik_success"] is True
    assert result["grasp_style"] == "top"
    assert result["motion_length"] == 0.2


def test_completion_status_supports_strict_and_best_effort_policy():
    per_object = [{"success": True}, {"success": False}]

    strict = _completion_status(
        per_object, {"physical_execution": {"completion_policy": "strict"}}
    )
    best_effort = _completion_status(
        per_object,
        {
            "physical_execution": {
                "completion_policy": "best_effort",
                "minimum_completion_ratio": 0.5,
            },
        },
    )

    assert strict == (False, 1, 0.5, "strict", False)
    assert best_effort == (False, 1, 0.5, "best_effort", True)


def test_placement_success_uses_scene_tolerance():
    config = {
        "physical_execution": {"placement_success_tolerance_xy": 0.015}
    }

    assert placement_within_tolerance([0.009, 0.012, 0.2], config) is True
    assert placement_within_tolerance([0.014999, 0.0, 0.0], config) is True
    assert placement_within_tolerance([0.015001, 0.0, 0.0], config) is False
    assert placement_within_tolerance([0.04, 0.0, 0.0], config) is False


def test_probe_transfer_reports_retries_and_failures():
    class Planner:
        def __init__(self):
            self.calls = 0

        def plan_xy(self, start, goal):
            self.calls += 1
            return SimpleNamespace(
                success=self.calls == 2,
                metadata={"route_type": "direct"},
                waypoints=[start, goal],
                length=1.0,
            )

    motion, failures, retries = _probe_transfer(Planner(), [0.0, 0.0], (1.0, 0.0), 2)

    assert motion.success is True
    assert failures == 1
    assert retries == 1
