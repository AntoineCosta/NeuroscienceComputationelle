"""Body kinematics: top-view pose of the hexapod, moved by the legs in stance."""
import numpy as np
from brian2 import ms

from config import BODY_LENGTH, BODY_WIDTH, TRACK_WIDTH, STRIDE
from legs import FORWARD, BACKWARD


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
    One step() per CPG half-cycle: usable offline (trajectory) or live later.
    """

    def __init__(self, legs, length=BODY_LENGTH, width=BODY_WIDTH, track=TRACK_WIDTH,
                 stride=STRIDE):
        self.legs = legs
        self.length, self.width = length, width
        self.track, self.stride = track, stride
        self.reset()

    def reset(self, x=0.0, y=0.0, heading=np.pi / 2):
        self.pose = np.array([x, y, heading])

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
        Move the body by one half-cycle.
        counts_c: [leg, kind] neurons fired in this half-cycle; phase: CPG phase (stance).
        """
        d_left, d_right = self.side_push(counts_c, phase)
        advance = (d_left + d_right) / 2
        turn = (d_right - d_left) / self.track
        x, y, heading = self.pose
        mid = heading + turn / 2           # midpoint heading: good approximation of the arc
        self.pose = np.array([x + advance * np.cos(mid), y + advance * np.sin(mid), heading + turn])
        return self.pose

    def trajectory(self, on_t, on_p, counts):
        """
        Replay a recorded run (output of LegSet.recruitment) from the current pose.
        Returns (times in ms, poses [k, 3]): poses at the start of every half-cycle
        and at the end of the last one.
        """
        half_ms = float(self.legs.cpg.period / 2 / ms)
        times = np.append(on_t, on_t[-1] + half_ms)
        poses = [self.pose.copy()]
        for c, p in enumerate(on_p):
            poses.append(self.step(counts[c], p).copy())
        return times, np.array(poses)

    def outline(self, pose):
        """Corners (x, y) of the body drawn as an arrow-shaped polygon at a pose."""
        l, w = self.length / 2, self.width / 2
        local = np.array([[-l, -w], [l / 2, -w], [l, 0.0], [l / 2, w], [-l, w]])
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return local @ np.array([[c, s], [-s, c]]) + pose[:2]
