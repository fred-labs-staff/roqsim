"""A walker on its route judges a crosser from its motion before it is in its way, as a person does.

With ``margin`` (the walker declares 0.8 m) it goes first only where the two would pass that far
apart, centre to centre, and otherwise slows at ``decel`` to let the crosser go first, from as far
ahead as its ``horizon``, rather than slowing once the crosser is in its corridor.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.presence import entity_geoms

#: A TurtleBot's shape: a low base that collides, a slim tower that does not.
ROBOT = """<mujoco model="robot">
  <worldbody><body name="base">
    <geom name="base" type="cylinder" size=".164 .03" pos="0 0 .049"/>
    <geom name="tower" type="cylinder" size=".03 .15" pos="0 0 .25" contype="0" conaffinity="0"/>
  </body></worldbody>
</mujoco>"""


def _world(tmp_path, dt, robot_speed=0.46, walker_speed=0.5, avoidance=None):
    """A walker walking east from (-4, 0) and a robot crossing its way at x = 0 from 1.2 m to its
    left, timed so its centre reaches the walker's line ``dt`` s after the walker's centre would
    reach the robot's."""
    (tmp_path / "robot.xml").write_text(ROBOT)
    reach_line = 4.0 / walker_speed + dt
    dwell = reach_line - 1.2 / robot_speed
    assert dwell >= 0.0
    walker = {
        "walker": {"walker": "MaleVisitorWalk", "pose": {"position": {"x": -4.0, "y": 0.0}}},
        "name": "w",
        "components": [
            {
                "navigator": {
                    "output": "walker",
                    "speed": walker_speed,
                    "goals": [[4.5, 0.0]],
                    "avoidance": {"steer": "none", "stop": True, **(avoidance or {})},
                }
            }
        ],
    }
    robot = {
        "spawn_model": {
            "model": "robot.xml",
            "prefix": "r_",
            "motion": "driven",
            "pose": {"position": {"x": 0.0, "y": 1.2}, "orientation": {"yaw": -np.pi / 2}},
        },
        "name": "robot",
        "components": [
            {
                "navigator": {
                    "output": "mocap",
                    "speed": robot_speed,
                    "route_mode": "exact",
                    "arrival_radius": 0.01,
                    "avoidance": {"stop": False},
                    "recovery": {"enabled": False},
                    "goals": [[0.0, 1.2], [0.0, -1.2]],
                    "dwell": [0.0, dwell, 0.0],
                }
            }
        ],
    }
    engine = Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap"}, "components": [walker, robot]}, base_dir=tmp_path
        )
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine, reach_line


def _run(engine, seconds):
    """Per 0.25 s, the walker's speed; the least gap between the walker's body and the robot's."""
    model, data = engine.ctx.model, engine.ctx.data
    solid = lambda g: model.geom_contype[g] or model.geom_conaffinity[g]  # noqa: E731
    mine = [g for g in entity_geoms(model, engine.ctx.entities.get("w"), strict=False) if solid(g)]
    its = [
        g for g in entity_geoms(model, engine.ctx.entities.get("robot"), strict=False) if solid(g)
    ]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, engine.ctx.entities.get("w").body)
    last, last_t, speeds, gap = data.xpos[body][:2].copy(), 0.0, [], np.inf
    for k in range(int(seconds / engine.ctx.dt)):
        engine.step()
        t = float(engine.ctx.sim_time)
        if k % 10 == 0:
            gap = min(
                gap,
                min(
                    mujoco.mj_geomDistance(model, data, a, b, 2.0, None) for a in mine for b in its
                ),
            )
        if t - last_t >= 0.25:
            pos = data.xpos[body][:2].copy()
            speeds.append((t, float(np.linalg.norm(pos - last)) / (t - last_t)))
            last, last_t = pos, t
    return np.array(speeds), gap


@pytest.mark.parametrize("dt", [0.0, 0.5])
def test_it_slows_early_and_gently_to_let_a_robot_crossing_ahead_go_first(tmp_path, dt):
    engine, reach_line = _world(tmp_path, dt)
    try:
        speeds, gap = _run(engine, reach_line + 6.0)
        walking = speeds[speeds[:, 0] > 1.0]
        slow = walking[walking[:, 1] < 0.85 * 0.5]
        assert len(slow), "it never slowed for the robot"
        assert slow[0, 0] <= reach_line - 1.5, (
            f"it slowed at {slow[0, 0]:.2f} s for a robot on its line at {reach_line:.2f} s"
        )
        drops = -np.diff(walking[:, 1]) / np.diff(walking[:, 0])
        assert drops.max() <= 1.1, f"it slowed at {drops.max():.2f} m/s^2"
        assert gap > 0.2, f"it came within {gap:.2f} m of the robot"
    finally:
        engine.shutdown()


def test_it_does_not_slow_for_a_robot_it_passes_well_ahead_of(tmp_path):
    # the robot reaches the walker's line 2.5 s after the walker: they pass more than 0.8 m apart
    engine, reach_line = _world(tmp_path, 2.5)
    try:
        speeds, gap = _run(engine, reach_line + 3.0)
        walking = speeds[speeds[:, 0] > 1.0]
        assert walking[:, 1].min() >= 0.85 * 0.5, "it slowed for a robot it passes well ahead of"
    finally:
        engine.shutdown()


def test_with_no_margin_it_judges_a_crosser_only_in_its_corridor(tmp_path):
    engine, reach_line = _world(tmp_path, 0.0, avoidance={"margin": 0.0})
    try:
        speeds, _ = _run(engine, reach_line + 1.0)
        walking = speeds[speeds[:, 0] > 1.0]
        slow = walking[walking[:, 1] < 0.85 * 0.5]
        assert len(slow) and slow[0, 0] > reach_line - 1.5, "it slowed before the robot was near"
    finally:
        engine.shutdown()


def test_a_negative_margin_is_refused(tmp_path):
    with pytest.raises(Exception) as err:
        _world(tmp_path, 0.0, avoidance={"margin": -0.1})
    assert "avoidance.margin" in str(err.value)


@pytest.mark.parametrize("avoidance", [{"margin": 0.8, "decel": 0.0}, {"decel": 0.0}])
def test_a_margin_without_decel_is_refused_at_load(tmp_path, avoidance):
    # the walker declares margin 0.8 m, so a world that turns its decel off has a margin that would
    # do nothing
    with pytest.raises(Exception) as err:
        _world(tmp_path, 0.0, avoidance=avoidance)
    assert "'avoidance.margin' needs 'avoidance.decel' > 0" in str(err.value)
