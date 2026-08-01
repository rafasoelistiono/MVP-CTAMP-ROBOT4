"""Build a self-contained MuJoCo tabletop scene from YAML configuration."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .panda_loader import PandaAsset, find_panda_asset
from .scene import GoalSlot, generate_tidy_slots

COLORS = {
    "blue": "0.1 0.25 0.9 1",
    "red": "0.9 0.1 0.1 1",
    "green": "0.1 0.75 0.2 1",
    "yellow": "0.95 0.8 0.1 1",
    "orange": "0.95 0.45 0.1 1",
    "purple": "0.55 0.25 0.85 1",
}


class MuJoCoSceneBuilder:
    def __init__(self, config: dict, project_root: str | Path) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.panda_asset: PandaAsset = find_panda_asset(self.project_root)
        self.slots: dict[str, GoalSlot] = generate_tidy_slots(config)

    def build_xml(self) -> str:
        if self.panda_asset.path is not None:
            panda_xml = self.panda_asset.path / "panda.xml"
            if not panda_xml.is_file():
                panda_xml = self.panda_asset.path / "models/panda.xml"
            root = ET.parse(panda_xml).getroot()
            root.set("model", self.config["scene"]["scene_id"])
            compiler = root.find("compiler")
            if compiler is not None:
                asset_dir = self.panda_asset.path / "assets"
                compiler.set("meshdir", str(asset_dir.resolve()))
            world = root.find("worldbody")
            if world is None:
                raise ValueError("Panda MJCF has no worldbody")
            link0 = world.find("body[@name='link0']")
            if link0 is None:
                raise ValueError("Panda MJCF has no link0 body")
            for child in list(world):
                if child is not link0:
                    world.remove(child)
            robot = self.config["robot"]
            link0.set(
                "pos", f"{robot['base_xy'][0]} {robot['base_xy'][1]} {robot['base_z']}"
            )
            hand = world.find(".//body[@name='hand']")
            if hand is None:
                raise ValueError("Panda MJCF has no hand body")
            ET.SubElement(
                hand,
                "site",
                name="gripper",
                pos="0 0 0.10",
                size="0.015",
                rgba="0.1 1 0.1 0.8",
            )
        else:
            root = ET.Element("mujoco", model=self.config["scene"]["scene_id"])
            ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
            world = ET.SubElement(root, "worldbody")
            self._add_panda_proxy(world)
        ET.SubElement(
            world,
            "light",
            pos="0 -0.4 2.8",
            directional="true",
            diffuse="0.45 0.45 0.45",
            ambient="0.12 0.12 0.12",
            specular="0.04 0.04 0.04",
        )
        ET.SubElement(
            world,
            "camera",
            name="overview",
            pos="0 -2.8 2.8",
            xyaxes="1 0 0 0 0.72 0.70",
        )
        table = self.config["table"]
        physics = self.config.get("physical_execution", {})
        table_friction = " ".join(
            str(v) for v in physics.get("table_friction", [1.0, 0.01, 0.001])
        )
        cube_friction = " ".join(
            str(v) for v in physics.get("cube_friction", [1.5, 0.02, 0.002])
        )
        cube_mass = str(physics.get("cube_mass", 0.12))
        xsize = (table["x_range"][1] - table["x_range"][0]) / 2
        ysize = (table["y_range"][1] - table["y_range"][0]) / 2
        ET.SubElement(
            world,
            "geom",
            name="table",
            type="box",
            pos=f"0 {-0.10} {table['z_top'] - 0.04}",
            size=f"{xsize} {ysize} 0.04",
            rgba="0.55 0.45 0.35 1",
            friction=table_friction,
        )
        self._add_robot_mount(world)
        if self.panda_asset.path is None:
            probe = ET.SubElement(
                world, "body", name="ee_probe", pos="-0.15 -0.08 1.02"
            )
            ET.SubElement(probe, "freejoint", name="ee_probe_free")
            ET.SubElement(
                probe,
                "geom",
                name="ee_probe_geom",
                type="sphere",
                size="0.025",
                rgba="0.1 0.95 0.2 0.9",
                contype="0",
                conaffinity="0",
            )
        for obstacle in self.config.get("obstacles", []):
            pos = " ".join(str(v) for v in obstacle["pose"])
            half = " ".join(str(float(v) / 2) for v in obstacle["size"])
            ET.SubElement(
                world,
                "geom",
                name=obstacle["id"],
                type="box",
                pos=pos,
                size=half,
                rgba="0.25 0.25 0.25 1",
            )
        self._add_tidy_tray(world)
        for obj in self.config["objects"]:
            size_xyz = obj.get("size_xyz", self.config["geometry"]["cube_size_xyz"])
            cube_half = [float(v) / 2 for v in size_xyz]
            body = ET.SubElement(
                world,
                "body",
                name=f"cube_{obj['id']}",
                pos=" ".join(str(v) for v in obj["pose"]),
            )
            ET.SubElement(body, "freejoint", name=f"cube_{obj['id']}_free")
            ET.SubElement(
                body,
                "geom",
                name=f"cube_{obj['id']}_geom",
                type="box",
                size=" ".join(str(v) for v in cube_half),
                rgba=COLORS.get(obj["color"], "0.5 0.5 0.5 1"),
                mass=cube_mass,
                friction=cube_friction,
                solref="0.005 1",
                solimp="0.95 0.99 0.001",
            )
        for slot in self.slots.values():
            ET.SubElement(
                world,
                "site",
                name=slot.name,
                pos=" ".join(str(v) for v in slot.position),
                type="cylinder",
                size="0.031 0.001",
                rgba="0.2 0.9 0.2 0.35",
            )
        if self.panda_asset.path is not None:
            equality = root.find("equality")
            if equality is None:
                equality = ET.SubElement(root, "equality")
            for obj in self.config["objects"]:
                ET.SubElement(
                    equality,
                    "weld",
                    name=f"carry_{obj['id']}",
                    body1="hand",
                    body2=f"cube_{obj['id']}",
                    active="false",
                    solref="0.005 1",
                    solimp="0.95 0.99 0.001",
                )
        return ET.tostring(root, encoding="unicode")

    def _add_robot_mount(self, world: ET.Element) -> None:
        robot = self.config["robot"]
        mount = robot.get("mount")
        if not mount:
            return
        if mount.get("type") != "pedestal":
            raise ValueError("robot.mount.type must be 'pedestal'")
        bottom = float(self.config["table"]["z_top"])
        top = float(robot["base_z"])
        if top <= bottom:
            raise ValueError("pedestal requires robot.base_z above table.z_top")
        radius = float(mount["radius"])
        if radius <= 0:
            raise ValueError("robot.mount.radius must be positive")
        x, y = (float(value) for value in robot["base_xy"])
        height = top - bottom
        ET.SubElement(
            world,
            "geom",
            name="panda_mount",
            type="cylinder",
            pos=f"{x} {y} {bottom + height / 2.0}",
            size=f"{radius} {height / 2.0}",
            rgba="0.25 0.25 0.28 1",
        )

    def _add_panda_proxy(self, world: ET.Element) -> None:
        robot = self.config["robot"]
        x, y = robot["base_xy"]
        z = float(robot["base_z"])
        # Real assets are detected and reported, but composing arbitrary menagerie
        # root MJCFs requires their compiler/assets context. This conservative proxy
        # keeps scene tests portable and does not claim Panda dynamics or IK.
        body = ET.SubElement(world, "body", name=robot["id"], pos=f"{x} {y} {z}")
        ET.SubElement(
            body,
            "geom",
            name="panda_base_proxy",
            type="cylinder",
            size="0.09 0.12",
            rgba="0.85 0.85 0.85 1",
        )
        ET.SubElement(
            body,
            "site",
            name="panda_ee_probe",
            pos="0.35 0 0.25",
            size="0.02",
            rgba="0.1 0.8 0.1 1",
        )

    def _add_tidy_tray(self, world: ET.Element) -> None:
        if not self.slots:
            return
        xs = sorted({round(slot.position[0], 6) for slot in self.slots.values()})
        ys = sorted({round(slot.position[1], 6) for slot in self.slots.values()})
        if len(xs) < 2 or len(ys) < 1:
            return
        step_x = min(b - a for a, b in zip(xs, xs[1:], strict=False))
        step_y = (
            min(b - a for a, b in zip(ys, ys[1:], strict=False))
            if len(ys) > 1
            else float(self.config["grouped_tidy"].get("row_spacing", step_x))
        )
        wall_height = 0.018
        wall_thickness = 0.006
        z = float(self.config["table"]["z_top"]) + wall_height / 2
        xmin, xmax = xs[0] - step_x / 2, xs[-1] + step_x / 2
        ymin, ymax = ys[0] - step_y / 2, ys[-1] + step_y / 2
        x_walls = [xmin, xmax] + [(a + b) / 2 for a, b in zip(xs, xs[1:], strict=False)]
        y_walls = [ymin, ymax] + [(a + b) / 2 for a, b in zip(ys, ys[1:], strict=False)]
        rgba = "0.08 0.12 0.10 0.65"
        for index, x in enumerate(x_walls):
            ET.SubElement(
                world,
                "geom",
                name=f"tidy_tray_x_{index}",
                type="box",
                pos=f"{x} {(ymin + ymax) / 2} {z}",
                size=f"{wall_thickness / 2} {(ymax - ymin) / 2 + wall_thickness / 2} {wall_height / 2}",
                rgba=rgba,
                friction="1.2 0.3 0.1",
            )
        for index, y in enumerate(y_walls):
            ET.SubElement(
                world,
                "geom",
                name=f"tidy_tray_y_{index}",
                type="box",
                pos=f"{(xmin + xmax) / 2} {y} {z}",
                size=f"{(xmax - xmin) / 2 + wall_thickness / 2} {wall_thickness / 2} {wall_height / 2}",
                rgba=rgba,
                friction="1.2 0.3 0.1",
            )
