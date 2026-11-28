"""What a mover perceives: what it can see from its eye, and what it saw a moment ago.

A navigator's caution probe and its avoidance model look at everything that moves in the world,
through walls and behind the mover's back. That is right for a robot under a navigator, which is
apparatus, and wrong for something that stands for a person: a walker that reacts to a robot it
could not see makes the dangerous encounters look safe. So what a mover perceives is a capability
its embodiment declares (:attr:`~roqsim_nav.outputs.NavOutput.perception`), and a mover that
declares none perceives as before: everything, at once.

**Seen** is a clear sight line from the eye to a part of the thing, within the field of view. The
eye stands ``eye_height`` above the bottom of the mover's body, over its centre; the field of view
is ``fov`` degrees to either side of its heading, so nothing behind its back is seen. A sight line
runs to the centres of up to :data:`POINTS` of the thing's geoms, visual ones included (a robot's
tower over a shelf, its base under one), and is blocked by whatever it meets first: a wall, a
shelf, a door frame, another mover. The mover's own geoms and what it holds do not block it: a
sight line is cast from the thing to the eye, and reaching the mover is reaching the eye.

**Perceived** is what the mover acts on. What comes into sight is perceived once it has stayed in
sight ``reaction`` seconds; what goes out of sight stays perceived ``memory`` seconds, where it
really is, and is then forgotten at once. At the first look of a run, what is in sight is perceived
at once: the mover has been standing there.

Cost matters, because this runs on the simulation thread: a mover looks only at the things within
the reach its navigator gives (what its probe and its model could act on), casts at most
:data:`POINTS` rays to each, and none to a thing whose every point is outside its field of view.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim import raycast

#: How many points of a thing a sight line is cast to at most: its highest and lowest geoms'
#: centres and two between, so a tower above a shelf and a base below one are both looked for.
POINTS = 4

#: How many other parts of the thing itself a sight line passes before it is taken as blocked.
_PASSES = 4

#: Defaults of ``perception:``, for the schema and for a mover that declares only some keys.
DEFAULTS = {"eye_height": 1.6, "fov": 75.0, "memory": 4.0, "reaction": 0.4}


class Sight:
    """What one mover perceives, looked at once per navigator tick."""

    def __init__(self, config: dict):
        cfg = {**DEFAULTS, **(config or {})}
        self.eye_height = float(cfg["eye_height"])
        self.fov = math.radians(float(cfg["fov"]))
        self.memory = float(cfg["memory"])
        self.reaction = float(cfg["reaction"])
        self._own: set[int] = set()
        self._bottom: float | None = None
        self.reset()

    @staticmethod
    def validate(cfg: dict | None) -> list[str]:
        """The ranges of ``perception:``; a value that is not a number is the schema's to report."""
        errors = []
        for key, value in (cfg or {}).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if key == "fov" and not 0.0 < value <= 180.0:
                errors.append("'perception.fov' must be in (0, 180] degrees to each side")
            elif key != "fov" and value < 0.0:
                errors.append(f"'perception.{key}' must be >= 0")
        return errors

    def attach(self, own_geoms) -> None:
        """The mover's own geoms, which never block its sight."""
        self._own = {int(g) for g in own_geoms}
        self._bottom = None

    def reset(self) -> None:
        self._since: dict[int, float] = {}  # thing -> since when it has been in sight
        self._last: dict[int, float] = {}  # thing -> when it was last in sight
        self.perceived: set[int] = set()
        self.in_sight: set[int] = set()
        self._looked = False

    def eye(self, model, data, xy) -> np.ndarray:
        """The eye, ``eye_height`` above the bottom of the mover's body (measured once a run)."""
        if self._bottom is None:
            own = np.array(sorted(self._own), dtype=int)
            if own.size:
                low = data.geom_xpos[own, 2] - model.geom_rbound[own]
                self._bottom = max(0.0, float(low.min()))
            else:
                self._bottom = 0.0
        return np.array([float(xy[0]), float(xy[1]), self._bottom + self.eye_height])

    def look(self, ctx, now: float, xy, yaw: float, things: dict, reach: float, held=()) -> set:
        """Look at ``things`` (thing -> its geoms) within ``reach`` of ``xy``; return the perceived.

        ``held`` are geoms the mover holds, which do not block its sight either.
        """
        model, data = ctx.model, ctx.data
        eye = self.eye(model, data, xy)
        skip = self._own | {int(g) for g in held}
        seen = set()
        for thing, geoms in things.items():
            if geoms.size == 0:
                continue
            centres = data.geom_xpos[geoms]
            flat = np.linalg.norm(centres[:, :2] - eye[:2], axis=1)
            if float(flat.min()) > reach:
                continue
            if self._sees(model, data, eye, yaw, geoms, centres, skip):
                seen.add(thing)
        first = not self._looked
        self._looked = True
        for thing in seen:
            since = self._since.setdefault(thing, now)
            self._last[thing] = now
            if first or thing in self.perceived or now - since >= self.reaction - 1e-9:
                self.perceived.add(thing)
        for thing in list(self._since):
            if thing not in seen:
                del self._since[thing]
        for thing in list(self.perceived):
            if thing not in seen and now - self._last.get(thing, now) > self.memory:
                self.perceived.discard(thing)
        self.in_sight = seen
        return self.perceived

    def _sees(self, model, data, eye, yaw, geoms, centres, skip) -> bool:
        """Whether a sight line to a point of the thing is clear and within the field of view."""
        order = np.argsort(centres[:, 2])
        if order.size > POINTS:
            pick = np.unique(np.linspace(0, order.size - 1, POINTS).round().astype(int))
            order = order[pick]
        mine = {int(g) for g in geoms}
        gid = np.zeros(1, dtype=np.int32)
        for k in order[::-1]:  # from the top: what shows over an obstacle first
            target = centres[k]
            d = target - eye
            angle = math.atan2(float(d[1]), float(d[0])) - yaw
            angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
            if abs(angle) > self.fov:
                continue
            # Cast from the thing to the eye, not the other way: a line from the eye down to a
            # robot's base runs through the mover's own head, chest and hips, where one from the
            # base meets the first thing in between, or the mover itself.
            start = target.copy()
            body = int(model.geom_bodyid[geoms[k]])
            for _ in range(_PASSES):
                vec = eye - start
                x = mujoco.mj_ray(model, data, start, vec, raycast.VISIBLE_GROUPS, 1, body, gid)
                g = int(gid[0])
                if x < 0.0 or x >= 1.0 or g in skip:
                    return True
                if g not in mine:
                    break
                # another part of the thing itself: on from just past it
                start = start + vec * (x + 1e-4)
        return False
