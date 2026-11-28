"""A walker that its avoidance model holds back for a robot is waiting, not stuck.

A walker steering with ``orca`` through a doorway sees the robot driving along the far side of
the wall only once it is in the doorway, and ``orca`` holds it there, against the door frame, until
the robot has passed or moved on. That wait must not run its stuck clock: four recoveries would
have it give up on its goal and stand for good.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

#: A TurtleBot's shape: a low base that collides, a slim tower that does not.
ROBOT = """<mujoco model="robot">
  <worldbody><body name="base">
    <geom name="base" type="cylinder" size=".164 .03" pos="0 0 .049"/>
    <geom name="tower" type="cylinder" size=".03 .15" pos="0 0 .25" contype="0" conaffinity="0"/>
  </body></worldbody>
</mujoco>"""


def _wall(name, x0, y0, x1, y1):
    return {
        "box": {
            "prefix": f"{name}_",
            "pose": {"position": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2}},
            "size": [abs(x1 - x0) or 0.1, abs(y1 - y0) or 0.1, 2.2],
            "motion": "static",
        },
        "name": name,
    }


def _doorway(tmp_path, pause):
    """A walker at 0.5 m/s walking east through a 0.9 m doorway in a wall along x = 0, and a robot
    driving south along the wall's far side, 0.6 m from it, that reaches the doorway as the walker
    does and stands before it ``pause`` seconds before it drives on."""
    (tmp_path / "robot.xml").write_text(ROBOT)
    walker = {
        "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": -3.0, "y": 0.0}}},
        "name": "w",
        "components": [
            {
                "navigator": {
                    "output": "walker",
                    "speed": 0.5,
                    "goals": [[4.0, 0.0]],
                    "avoidance": {"steer": "orca", "stop": True},
                }
            }
        ],
    }
    robot = {
        "spawn_model": {
            "model": "robot.xml",
            "prefix": "r_",
            "motion": "driven",
            "pose": {"position": {"x": 0.6, "y": 1.4}, "orientation": {"yaw": -np.pi / 2}},
        },
        "name": "robot",
        "components": [
            {
                "navigator": {
                    "output": "mocap",
                    "speed": 0.46,
                    "route_mode": "exact",
                    "arrival_radius": 0.01,
                    "avoidance": {"stop": False},
                    "recovery": {"enabled": False},
                    "goals": [[0.6, 1.4], [0.6, 0.0], [0.6, -1.4]],
                    "dwell": [0.0, 4.16, pause, 0.0],
                }
            }
        ],
    }
    comps = [
        walker,
        robot,
        _wall("north", 0.0, 0.45, 0.0, 5.0),
        _wall("south", 0.0, -5.0, 0.0, -0.45),
    ]
    engine = Engine(
        load_config_from_dict({"sim": {"pacing": "asap"}, "components": comps}, base_dir=tmp_path)
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


@pytest.mark.parametrize("pause", [0.0, 4.0])
def test_a_walker_orca_holds_in_a_doorway_for_a_robot_waits_and_arrives(tmp_path, pause):
    engine = _doorway(tmp_path, pause)
    try:
        model = engine.ctx.model
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("w").body)
        nav = engine.ctx.blackboard.get("nav:w")
        data, still, last, last_t = engine.ctx.data, 0.0, None, 0.0
        for _ in range(int(30.0 / engine.ctx.dt)):
            engine.step()
            t = float(engine.ctx.sim_time)
            if t - last_t >= 0.25:
                pos = data.xpos[body][:2].copy()
                # held back at the doorway: under 30 % of its speed, within a stride of it
                slow = np.linalg.norm(pos - last) / (t - last_t) < 0.15 if last is not None else 0
                if slow and abs(pos[0]) < 0.8:
                    still += t - last_t
                last, last_t = pos, t
        end = engine.ctx.data.xpos[body][:2]
        assert still >= 1.0, "it was not held back at the doorway: the case tests no wait"
        assert not nav._core.failure, f"it gave up: {nav._core.failure}"
        assert np.linalg.norm(end - (4.0, 0.0)) < 0.3, f"it did not arrive: it is at {end}"
    finally:
        engine.shutdown()


def test_a_walker_orca_holds_before_a_robot_standing_on_its_route_still_gets_round(tmp_path):
    # held by a robot that does not move, past `yield_time` it is stalled, and recovery is its way
    # round: a hold of the model is traffic only that long
    (tmp_path / "robot.xml").write_text(ROBOT)
    walker = {
        "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": -4.0, "y": 0.0}}},
        "name": "w",
        "components": [
            {
                "navigator": {
                    "output": "walker",
                    "speed": 0.5,
                    "goals": [[4.0, 0.0]],
                    "avoidance": {"steer": "orca", "stop": True},
                }
            }
        ],
    }
    robot = {
        "spawn_model": {
            "model": "robot.xml",
            "prefix": "r_",
            "motion": "driven",
            "pose": {"position": {"x": 0.0, "y": 0.0}},
        },
        "name": "robot",
    }
    engine = Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap"}, "components": [walker, robot]}, base_dir=tmp_path
        )
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        body = mujoco.mj_name2id(
            engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("w").body
        )
        for _ in range(int(30.0 / engine.ctx.dt)):
            engine.step()
        end = engine.ctx.data.xpos[body][:2]
        assert np.linalg.norm(end - (4.0, 0.0)) < 0.3, f"it did not get round: it is at {end}"
    finally:
        engine.shutdown()
