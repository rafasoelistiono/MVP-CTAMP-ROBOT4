from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
import yaml

from ctamp.simulation.mujoco_scene_builder import MuJoCoSceneBuilder
from ctamp.simulation.scene import MotionProbe


ROOT = Path(__file__).resolve().parents[1]
BLOCK_CONFIG = ROOT / "configs/scenes/blocksworld_challenge.yaml"
KITCHEN_CONFIG = ROOT / "configs/scenes/kitchen_challenge.yaml"


def test_motion_probe_uses_explicit_obstacle_flag():
    config = yaml.safe_load(KITCHEN_CONFIG.read_text())
    assert all(
        obstacle["support_surface"] and not obstacle["blocks_motion_probe"]
        for obstacle in config["obstacles"]
    )

    probe = MotionProbe(config)

    assert probe.rectangles == []

    config["obstacles"][0].pop("blocks_motion_probe")

    assert len(MotionProbe(config).rectangles) == 1


def test_panda_pedestal_spans_table_to_robot_base():
    config = yaml.safe_load(BLOCK_CONFIG.read_text())
    xml = MuJoCoSceneBuilder(config, ROOT).build_xml()
    root = ET.fromstring(xml)
    pedestal = root.find("worldbody/geom[@name='panda_mount']")

    assert pedestal is not None
    assert pedestal.get("type") == "cylinder"
    assert [float(value) for value in pedestal.get("pos", "").split()] == pytest.approx(
        [-0.42, -0.08, 0.84]
    )
    assert [float(value) for value in pedestal.get("size", "").split()] == pytest.approx(
        [0.10, 0.04]
    )
