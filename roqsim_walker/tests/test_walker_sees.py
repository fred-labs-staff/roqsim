"""A walker meets only what it can see, after a person's reaction, and keeps what it saw in mind.

What a walker perceives is what its output declares (``WalkerOutput.perception``): from an eye
1.64 m up, within 75 degrees of its heading, what has been in sight 0.4 s, and what went out of
sight for 4 s after. Walls, shelves and other movers block its sight; its own body does not. A mover
that declares nothing, or a world's ``perception: false``, sees everything as before.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError
from roqsim.presence import entity_geoms

#: A TurtleBot's shape: a low base that collides, a slim tower that does not.
ROBOT = """<mujoco model="robot">
  <worldbody><body name="base">
    <geom name="base" type="cylinder" size=".164 .03" pos="0 0 .049"/>
    <geom name="tower" type="cylinder" size=".03 .15" pos="0 0 .25" contype="0" conaffinity="0"/>
  </body></worldbody>
</mujoco>"""


def _walker(name, start, goals, speed=0.5, avoidance=None, perception=None):
    nav = {
        "output": "walker",
        "speed": speed,
        "goals": [list(g) for g in goals],
        "avoidance": {"steer": "none", "stop": True, **(avoidance or {})},
    }
    if perception is not None:
        nav["perception"] = perception
    first = goals[0]
    yaw = float(np.arctan2(first[1] - start[1], first[0] - start[0]))
    return {
        "walker": {
            "walker": "MaleVisitorWalk",
            "pose": {"position": {"x": start[0], "y": start[1]}, "orientation": {"yaw": yaw}},
        },
        "name": name,
        "components": [{"navigator": nav}],
    }


def _robot(name, xy, goals=None, speed=0.46, dwell=0.0):
    ent = {
        "spawn_model": {
            "model": "robot.xml",
            "prefix": f"{name}_",
            "motion": "driven",
            "pose": {"position": {"x": xy[0], "y": xy[1]}},
        },
        "name": name,
    }
    if goals:
        ent["components"] = [
            {
                "navigator": {
                    "output": "mocap",
                    "speed": speed,
                    "route_mode": "exact",
                    "arrival_radius": 0.01,
                    "avoidance": {"stop": False},
                    "recovery": {"enabled": False},
                    "goals": [list(xy), *[list(g) for g in goals]],
                    "dwell": [0.0, dwell] + [0.0] * len(goals),
                }
            }
        ]
    return ent


def _wall(name, x0, y0, x1, y1, h=2.2):
    """A static wall from (x0, y0) to (x1, y1), axis-aligned, 0.1 m thick."""
    return {
        "box": {
            "prefix": f"{name}_",
            "pose": {"position": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2}},
            "size": [abs(x1 - x0) or 0.1, abs(y1 - y0) or 0.1, h],
            "motion": "static",
        },
        "name": name,
    }


def _engine(tmp_path, components):
    (tmp_path / "robot.xml").write_text(ROBOT)
    engine = Engine(
        load_config_from_dict(
            {"sim": {"pacing": "asap"}, "components": components}, base_dir=tmp_path
        )
    )
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    return engine


def _nav(engine, name):
    return engine.ctx.blackboard.get(f"nav:{name}")


def _xy(engine, name):
    model = engine.ctx.model
    body = engine.ctx.entities.get(name).body
    return engine.ctx.data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)][:2].copy()


def _in_sight(engine, walker, other, fov=75.0):
    """Whether ``other`` is in ``walker``'s sight, decided here and not by the navigator: a line from
    the centre of any of its geoms to an eye 1.64 m above the walker's centre, within ``fov``
    degrees of its heading, meets nothing before the walker."""
    model, data = engine.ctx.model, engine.ctx.data
    w = engine.ctx.entities.get(walker)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, w.body)
    mine = set(entity_geoms(model, w, strict=False))
    eye = np.array([*data.xpos[root][:2], 1.64])
    m = data.xmat[root].reshape(3, 3)
    yaw = np.arctan2(m[1, 0], m[0, 0])
    gid = np.zeros(1, dtype=np.int32)
    for g in entity_geoms(model, engine.ctx.entities.get(other), strict=False):
        p = data.geom_xpos[g].copy()
        ang = (np.arctan2(p[1] - eye[1], p[0] - eye[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(ang) > np.radians(fov):
            continue
        x = mujoco.mj_ray(model, data, p, eye - p, None, 1, int(model.geom_bodyid[g]), gid)
        if x < 0 or x >= 1 or int(gid[0]) in mine:
            return True
    return False


def _names(nav, things):
    return {nav._caution.owner_names.get(t) for t in things}


def test_it_sees_what_is_ahead_not_through_a_wall_nor_behind_its_back(tmp_path):
    engine = _engine(
        tmp_path,
        [
            _walker("w", (0.0, 0.0), [(6.0, 0.0)]),
            _robot("ahead", (3.0, 0.8)),
            _robot("hidden", (3.0, -2.0)),
            _wall("wall", 1.0, -1.2, 5.0, -1.2),
            _robot("behind", (-2.0, 0.0)),
        ],
    )
    try:
        for _ in range(int(0.2 / engine.ctx.dt)):
            engine.step()
        nav = _nav(engine, "w")
        assert _names(nav, nav._sight.in_sight) == {"ahead"}
        assert _names(nav, nav._sight.perceived) == {"ahead"}
    finally:
        engine.shutdown()


def test_another_mover_blocks_its_sight(tmp_path):
    # a walker standing between it and a robot hides the robot; the walker itself is seen
    engine = _engine(
        tmp_path,
        [
            _walker("w", (0.0, 0.0), [(6.0, 0.0)]),
            _walker("stander", (2.0, 0.0), [(2.0, 0.0)], speed=0.0),
            _robot("robot", (3.0, 0.0)),
        ],
    )
    try:
        for _ in range(int(0.2 / engine.ctx.dt)):
            engine.step()
        nav = _nav(engine, "w")
        assert "stander" in _names(nav, nav._sight.in_sight)
        assert "robot" not in _names(nav, nav._sight.in_sight)
    finally:
        engine.shutdown()


def test_what_comes_into_sight_is_perceived_after_the_reaction_time(tmp_path):
    # the robot drives out from behind a wall across the walker's view, 4 m ahead of it
    engine = _engine(
        tmp_path,
        [
            _walker("w", (0.0, 0.0), [(0.0, 0.0)], speed=0.5),
            _robot("robot", (4.0, 2.5), goals=[(4.0, -2.0)], speed=0.46),
            _wall("wall", 0.5, 1.0, 6.0, 1.0),
        ],
    )
    try:
        nav = _nav(engine, "w")
        seen = perceived = None
        for _ in range(int(8.0 / engine.ctx.dt)):
            engine.step()
            t = float(engine.ctx.sim_time)
            names_seen = _names(nav, nav._sight.in_sight)
            if seen is None and "robot" in names_seen:
                seen = t
            if perceived is None and "robot" in _names(nav, nav._sight.perceived):
                perceived = t
        assert seen is not None and seen > 0.5, "it saw the robot through the wall"
        assert perceived - seen == pytest.approx(0.4, abs=0.05)
    finally:
        engine.shutdown()


def test_what_went_out_of_sight_is_kept_in_mind_for_its_memory_then_forgotten(tmp_path):
    # the robot drives behind a wall and stays there
    engine = _engine(
        tmp_path,
        [
            _walker("w", (0.0, 0.0), [(0.0, 0.0)]),
            _robot("robot", (4.0, -0.5), goals=[(4.0, 2.5)], speed=0.46),
            _wall("wall", 0.5, 1.0, 6.0, 1.0),
        ],
    )
    try:
        nav = _nav(engine, "w")
        seen = gone = forgotten = None
        for _ in range(int(9.0 / engine.ctx.dt)):
            engine.step()
            t = float(engine.ctx.sim_time)
            in_sight = "robot" in _names(nav, nav._sight.in_sight)
            if seen is None and in_sight:
                seen = t
            if seen is not None and gone is None and not in_sight:
                gone = t
            perceived = "robot" in _names(nav, nav._sight.perceived)
            if gone is not None and forgotten is None and not perceived:
                forgotten = t
        assert gone is not None and forgotten is not None
        assert forgotten - gone == pytest.approx(4.0, abs=0.05)
    finally:
        engine.shutdown()


def _reacts(engine, speed, seconds):
    """``(seen, reacted)``: when the robot first came into the walker's sight (:func:`_in_sight`),
    and when the walker first changed its speed or heading for 0.25 s: slower than 85 % or faster
    than 115 % of its set speed over 0.25 s (its stride sways the speed over less), or 5 cm off its
    line, from 1 s on (it sets off from standing)."""
    seen = reacted = None
    start = _xy(engine, "w")
    last, last_t = start, 0.0
    for _ in range(int(seconds / engine.ctx.dt)):
        engine.step()
        t = float(engine.ctx.sim_time)
        if seen is None and _in_sight(engine, "w", "robot"):
            seen = t
        pos = _xy(engine, "w")
        if reacted is None and t > 1.0 and abs(pos[1] - start[1]) > 0.05:
            reacted = t
        if t - last_t >= 0.25:
            v = float(np.linalg.norm(pos - last)) / (t - last_t)
            last, last_t = pos, t
            if reacted is None and t > 1.0 and not 0.85 * speed <= v <= 1.15 * speed:
                reacted = t
    return seen, reacted


def _case_wall(steer):
    """Case 1: walking along a wall 0.6 m to its left, a robot drives to the 1.0 m opening ahead
    and through it across its way, hidden by the wall until it is near the opening."""
    return [
        _walker("w", (-4.0, 0.0), [(4.5, 0.0)], avoidance={"steer": steer}),
        _robot("robot", (0.0, 2.4), goals=[(0.0, -1.6)], speed=0.46, dwell=2.78),
        _wall("west", -5.0, 0.6, -0.5, 0.6),
        _wall("east", 0.5, 0.6, 5.0, 0.6),
    ]


def _case_doorway(steer):
    """Case 2: walking out of a room through a 0.9 m doorway, into the way of a robot driving along
    the far side of the wall, 0.6 m from it, which reaches its line as it does."""
    return [
        _walker("w", (-3.0, 0.0), [(4.0, 0.0)], avoidance={"steer": steer}),
        _robot("robot", (0.6, 1.4), goals=[(0.6, -1.4)], speed=0.46, dwell=4.16),
        _wall("north", 0.0, 0.45, 0.0, 5.0),
        _wall("south", 0.0, -5.0, 0.0, -0.45),
    ]


def _case_behind(steer):
    """Case 4: a robot at 0.9 m/s comes up behind it and overtakes it 0.55 m beside its line."""
    return [
        _walker("w", (-3.0, 0.0), [(4.5, 0.0)], avoidance={"steer": steer}),
        _robot("robot", (-4.8, 0.55), goals=[(4.8, 0.55)], speed=0.9),
    ]


def _case_people_at_a_corner(steer):
    """Case 7: two walkers at 1.4 m/s reach the corner of two corridors 1.6 m wide at once, one
    turning north, the other coming south and turning west; the corner's walls hide each from the
    other until they are near it."""
    y0, cx, t = -1.5, 1.0, 0.85
    return [
        _walker("w", (-4.0, y0), [(cx, y0), (cx, 3.3)], speed=1.4, avoidance={"steer": steer}),
        _walker(
            "robot",  # the other walker, in the robot's place
            (cx + 0.3, y0 + 4.7),
            [(cx + 0.3, y0 - 0.3), (-4.0, y0 - 0.3)],
            speed=1.4,
            avoidance={"steer": steer},
        ),
        _wall("a_out", -5.0, y0 - t, cx + t, y0 - t),
        _wall("a_in", -5.0, y0 + t, cx - t, y0 + t),
        _wall("b_in", cx - t, y0 + t, cx - t, 4.9),
        _wall("b_out", cx + t, y0 - t, cx + t, 4.9),
    ]


@pytest.mark.parametrize("steer", ["give_way", "orca"])
@pytest.mark.parametrize(
    ("case", "speed", "seconds"),
    [
        (_case_wall, 0.5, 14.0),
        (_case_doorway, 0.5, 12.0),
        (_case_behind, 0.5, 10.0),
        (_case_people_at_a_corner, 1.4, 4.0),
    ],
)
def test_steering_it_does_not_react_to_what_it_has_not_seen(tmp_path, steer, case, speed, seconds):
    engine = _engine(tmp_path, case(steer))
    try:
        seen, reacted = _reacts(engine, speed, seconds)
        assert seen is not None, "it never saw what it meets"
        assert reacted is None or reacted >= seen + 0.3, (
            f"it reacted at {reacted:.2f} s to what it could see only from {seen:.2f} s"
        )
    finally:
        engine.shutdown()


def test_with_perception_false_it_sees_everything(tmp_path):
    engine = _engine(
        tmp_path,
        [_walker("w", (0.0, 0.0), [(4.0, 0.0)], perception=False), _robot("r", (2.0, 0.5))],
    )
    try:
        nav = _nav(engine, "w")
        engine.step()
        assert nav._sight is None
        assert nav._caution.perceived is None
    finally:
        engine.shutdown()


def test_a_mover_whose_output_declares_none_perceives_everything(tmp_path):
    engine = _engine(
        tmp_path,
        [
            _robot("mover", (0.0, 0.0), goals=[(4.0, 0.0)]),
            _robot("hidden", (2.0, -2.0)),
            _wall("wall", 1.0, -1.2, 3.0, -1.2),
        ],
    )
    try:
        nav = _nav(engine, "mover")
        assert nav._sight is None
        engine.step()
        assert nav._caution.perceived is None
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("perception", "says"),
    [
        ({"fov": 200.0}, "perception.fov"),
        ({"memory": -1.0}, "perception.memory"),
        (True, "perception: true"),
        ({"eye": 1.6}, "eye"),
    ],
)
def test_a_bad_perception_is_refused(tmp_path, perception, says):
    with pytest.raises((PluginError, ValueError)) as err:
        _engine(tmp_path, [_walker("w", (0.0, 0.0), [(4.0, 0.0)], perception=perception)])
    assert says in str(err.value)
