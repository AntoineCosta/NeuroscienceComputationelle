"""Manual control: drive the hexapod live with the arrow keys (top view)."""
import time
import numpy as np
from brian2 import Network, SpikeMonitor, network_operation, ms, second

import config
from CPG import CPG
from legs import LegSet
from intermediates import IntermediateLayer
from brain import Brain
from body import Body
import visualisation as vis

KEY_SPEED = 1.0   # up / down: walking speed
KEY_STEER = 0.5   # + left / right while walking: that side's stroke halved (curve)
KEY_SPIN = 1.0    # left / right alone: turn on the spot


class Keyboard:
    """Arrow-key state -> brain command (speed, steer). Opposite keys cancel out."""

    ARROWS = ('up', 'down', 'left', 'right')

    def __init__(self, speed=KEY_SPEED, steer=KEY_STEER, spin=KEY_SPIN):
        self.speed, self.steer, self.spin = speed, steer, spin
        self.pressed = set()
        self.quit = False

    def press(self, key):
        if key in self.ARROWS:
            self.pressed.add(key)
        elif key == 'escape':
            self.quit = True

    def release(self, key):
        self.pressed.discard(key)

    def command(self, t=None):
        """(speed, steer) for the keys currently held (t unused: same signature as a timeline)."""
        forward = ('up' in self.pressed) - ('down' in self.pressed)
        side = ('right' in self.pressed) - ('left' in self.pressed)
        if forward:
            return forward * self.speed, side * self.steer
        return 0.0, side * self.spin           # speed 0 + steer = turn on the spot


def run_manual(n_legs=config.N_LEGS, n_motor=config.N_MOTOR, fps=20, view_size=1.0,
               max_duration=3600 * second, keys=None):
    """
    Live manual control in (approximately) real time, in a matplotlib window:
      up / down                  walk forward / backward
      up / down + left / right   walk while turning towards that side
      left / right alone         turn on the spot
      Esc or close the window    quit
    The window needs the keyboard focus (click on it once).

    Commands are latched at each CPG half-cycle (as in the scripted tests). The body
    moves by one stance push per half-cycle, known once its spikes are in, and shown
    during the following half-cycle: about 1/4 to 1/2 s between key and motion.
    Returns (times in ms, poses) at the end of every half-cycle.
    keys: command source (default: a new Keyboard); anything with press(key),
          release(key), command(t) and a `quit` flag.
    """
    cpg = CPG(period=config.CYCLE_PERIOD)
    legs = LegSet(cpg, n_legs=n_legs, n_motor=n_motor)
    layer = IntermediateLayer(cpg, legs)
    brain = Brain(layer)
    body = Body(legs)
    keys = Keyboard() if keys is None else keys
    view = vis.LiveTopView(body, keys.press, keys.release, half_size=view_size)

    mon_cpg = SpikeMonitor(cpg.neurons)
    mon_motor = SpikeMonitor(legs.motors)
    half_ms = float(cpg.period / 2 / ms)
    times, poses = [0.0], [body.pose.copy()]
    n_seen = 0             # motor spikes already used by the odometry
    wall_start = None      # wall-clock time of simulation time 0

    @network_operation(dt=cpg.period / 2, when='start')
    def odometry(t):
        """At each half-cycle start: move the body by the half-cycle that just ended."""
        nonlocal n_seen
        if mon_cpg.num_spikes == 0:
            return
        phase = int(np.asarray(mon_cpg.i[:])[-1])
        body.step(legs.spike_counts(mon_motor.i[n_seen:]), phase)
        n_seen = mon_motor.num_spikes
        times.append(float(t / ms))
        poses.append(body.pose.copy())

    @network_operation(dt=1000 / fps * ms, when='end')
    def display(t):
        """Draw the body (one half-cycle behind, interpolated) and keep real time."""
        nonlocal wall_start
        if keys.quit or not view.is_open:
            net.stop()
            return
        t_ms = float(t / ms)
        a = np.clip((t_ms - times[-1]) / half_ms, 0, 1)
        start = poses[-2] if len(poses) > 1 else poses[-1]
        view.draw(start + a * (poses[-1] - start), t_ms, *keys.command(t))

        now = time.perf_counter()
        if wall_start is None or wall_start + t_ms / 1000 < now - 0.2:
            wall_start = now - t_ms / 1000     # (re)synchronise if late: no catch-up rush
        view.wait(wall_start + t_ms / 1000 - now)

    net = Network(*cpg.objects, *legs.objects, *layer.objects, brain.driver(keys.command),
                  odometry, display, mon_cpg, mon_motor)
    print("Manual control: arrows to drive, Esc or close the window to quit "
          "(click on the window first).")
    net.run(max_duration)
    if view.is_open:
        view.close()
    x, y, heading = body.pose
    print(f"Stopped at t={times[-1] / 1000:.2f} s: x={x:+.3f} m, y={y:+.3f} m, "
          f"heading={np.degrees(heading):.1f} deg")
    return np.array(times), np.array(poses)


if __name__ == '__main__':
    run_manual()
