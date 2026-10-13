"""Body kinematics: top-view pose of the hexapod, moved by the legs in stance."""
import numpy as np
from brian2 import ms

from config import BODY_LENGTH, BODY_WIDTH, TRACK_WIDTH, STRIDE
from legs import RAISE, LOWER, FORWARD, BACKWARD

SHOWN_RANKS = 3   # legs drawn per side, whatever N_LEGS (lighter display)


class Body:
    """
    Rigid body seen from above, pushed by its stance legs (differential drive).

    Each F (B) neuron fired by a leg in stance pushes the body forward (backward) by
    stride / N. Each side moves by the mean push of its stance legs (d_left, d_right):
        advance = (d_left + d_right) / 2
        turn    = (d_right - d_left) / track      (> 0 = counter-clockwise = left)
    The swing (R and return group) does not move the body.

    Pose = [x, y, heading] in metres / radians, heading measured from the x axis
    (default start: facing +y, i.e. "up" on the plots).
    One step() per CPG half-cycle: usable offline (trajectory) or live.

    Drawn legs (display only): ranks 0..SHOWN_RANKS-1 of each side. All legs of the
    same side and CPG phase get the same commands, so these show the whole gait.
    Foot of a drawn leg, relative to its hip, for each half-cycle:
      - fore-aft sweep centred on the hip: F moves the foot backward, B forward,
        stride / N per neuron (with an unchanged command the swing mirrors the stance)
      - raised by R, lowered by D (at their spike times); with no F / B spike the
        foot stays where it is.
    Body frame: x forward, y left.
    """

    def __init__(self, legs, length=BODY_LENGTH, width=BODY_WIDTH, track=TRACK_WIDTH,
                 stride=STRIDE):
        self.legs = legs
        self.length, self.width = length, width
        self.track, self.stride = track, stride

        per_side = legs.n_legs // 2
        ranks = [r for s in (0, 1) for r in range(SHOWN_RANKS)]
        sides = [s for s in (0, 1) for r in range(SHOWN_RANKS)]
        self.shown = [legs.legs[s * per_side + r] for s, r in zip(sides, ranks)]
        # hips spread along the body sides (front rank first), feet at the track width
        spacing = length / SHOWN_RANKS
        self.hips = np.array([[((SHOWN_RANKS - 1) / 2 - r) * spacing, (0.5 - s) * width]
                              for s, r in zip(sides, ranks)])
        self.foot_lateral = np.array([(0.5 - s) * track for s in sides])
        self.reset()

    def reset(self, x=0.0, y=0.0, heading=np.pi / 2):
        self.pose = np.array([x, y, heading])
        n = len(self.shown)
        self.foot_from = np.zeros(n)              # fore-aft offset at the half-cycle start
        self.foot_to = np.zeros(n)                # ... and at its end
        self.up_before = np.zeros(n, dtype=bool)  # raised before the R / D spike
        self.up_after = np.zeros(n, dtype=bool)   # raised after it
        self.switch_ms = np.zeros(n)              # R / D spike time after the CPG spike

    def side_push(self, counts_c, phase):
        """Mean push (m) of the stance legs of each side [left, right] in one half-cycle."""
        push = [[], []]
        for leg in self.legs.legs:
            if leg.phase == phase:
                c = counts_c[leg.index]
                push[leg.side].append((c[FORWARD] - c[BACKWARD]) * self.stride / self.legs.n_motor)
        return np.array([np.mean(p) if p else 0.0 for p in push])

    def step(self, counts_c, phase):
        """
        Move the body (and the drawn feet) by one half-cycle.
        counts_c: [leg, kind] neurons fired in this half-cycle; phase: CPG phase (stance).
        """
        d_left, d_right = self.side_push(counts_c, phase)
        advance = (d_left + d_right) / 2
        turn = (d_right - d_left) / self.track
        x, y, heading = self.pose
        mid = heading + turn / 2           # midpoint heading: good approximation of the arc
        self.pose = np.array([x + advance * np.cos(mid), y + advance * np.sin(mid), heading + turn])
        self._step_feet(counts_c)
        return self.pose

    def _step_feet(self, counts_c):
        exc_ms = float(self.legs.exc_delay / ms)
        lift_ms = float(self.legs.lift_delay / ms)
        self.up_before = self.up_after.copy()
        for k, leg in enumerate(self.shown):
            c = counts_c[leg.index]
            if c[FORWARD] or c[BACKWARD]:
                sweep = (c[BACKWARD] - c[FORWARD]) * self.stride / self.legs.n_motor
                self.foot_from[k], self.foot_to[k] = -sweep / 2, sweep / 2
            else:                          # no stroke: foot stays where it is
                self.foot_from[k] = self.foot_to[k]
            if c[LOWER]:
                self.up_after[k], self.switch_ms[k] = False, exc_ms
            if c[RAISE]:
                self.up_after[k], self.switch_ms[k] = True, exc_ms + lift_ms

    def feet_state(self):
        """Copy of the drawn feet state of the latest step() (for replays)."""
        return tuple(a.copy() for a in (self.foot_from, self.foot_to, self.up_before,
                                        self.up_after, self.switch_ms))

    def feet(self, pose, t_in_ms, state=None):
        """
        Drawn legs at a pose, t_in_ms after the CPG spike of the latest step()
        (or of the half-cycle whose feet_state() is given).
        Returns (hips [6, 2], feet [6, 2], raised [6]) in world coordinates.
        """
        foot_from, foot_to, up_before, up_after, switch_ms = state or self.feet_state()
        a = np.clip(t_in_ms / float(self.legs.cpg.period / 2 / ms), 0, 1)
        fore = foot_from + a * (foot_to - foot_from)
        local_feet = np.column_stack([self.hips[:, 0] + fore, self.foot_lateral])
        raised = np.where(t_in_ms >= switch_ms, up_after, up_before)
        return self.to_world(self.hips, pose), self.to_world(local_feet, pose), raised

    def trajectory(self, on_t, on_p, counts):
        """
        Replay a recorded run (output of LegSet.recruitment) from the current pose.
        Returns (times in ms, poses [k, 3], feet_states):
          poses at the start of every half-cycle and at the end of the last one,
          feet_states[c]: feet_state() of half-cycle c (for feet()).
        """
        half_ms = float(self.legs.cpg.period / 2 / ms)
        times = np.append(on_t, on_t[-1] + half_ms)
        poses, feet_states = [self.pose.copy()], []
        for c, p in enumerate(on_p):
            poses.append(self.step(counts[c], p).copy())
            feet_states.append(self.feet_state())
        return times, np.array(poses), feet_states

    def outline(self, pose):
        """Corners (x, y) of the body drawn as an arrow-shaped polygon at a pose."""
        l, w = self.length / 2, self.width / 2
        return self.to_world(np.array([[-l, -w], [l / 2, -w], [l, 0.0], [l / 2, w], [-l, w]]), pose)

    @staticmethod
    def to_world(local, pose):
        """Body-frame points (x forward, y left) -> world coordinates."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return local @ np.array([[c, s], [-s, c]]) + pose[:2]
