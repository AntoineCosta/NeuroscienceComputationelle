"""Robot: the whole closed loop (network, body, world, sensors), for live and scripted runs."""
import numpy as np
from brian2 import Network, SpikeMonitor, network_operation, ms

import config
from CPG import CPG
from legs import LegSet
from intermediates import IntermediateLayer
from brain import Brain
from body import Body
from world import World
from sensors import Sensors


class Robot:
    """
    CPG + legs + intermediates + brain, the body moved by the motor spikes, and the
    distance sensors reading the world at the current body pose.

    command_fn(t) -> (speed, steer): scripted timeline or live controller.
    world: World (default layout) or None for no obstacles and no sensors.

    Closed loop, every SENSE_DT:
      - odometry: once all motor spikes of the latest half-cycle are in
        (exc_delay + lift_delay after its CPG spike), the body moves by that
        half-cycle's push, spread over the half-cycle (pose_at)
      - sensors: distances measured at the current pose -> sensor neurons
    """

    def __init__(self, command_fn, world=None, n_legs=config.N_LEGS, n_motor=config.N_MOTOR,
                 start=(0.0, 0.0, np.pi / 2)):
        self.cpg = CPG(period=config.CYCLE_PERIOD)
        self.legs = LegSet(self.cpg, n_legs=n_legs, n_motor=n_motor)
        self.layer = IntermediateLayer(self.cpg, self.legs)
        self.brain = Brain(self.layer)
        self.body = Body(self.legs)
        self.body.reset(*start)
        self.world = world
        self.sensors = Sensors(self.layer, self.body, world) if world is not None else None
        self.command_fn = command_fn

        self.mon_cpg = SpikeMonitor(self.cpg.neurons)
        self.mon_motor = SpikeMonitor(self.legs.motors)
        self.half_ms = float(self.cpg.period / 2 / ms)
        # all motor spikes caused by a CPG spike are in after this time (R is the last one)
        self.settle_ms = float((self.legs.exc_delay + self.legs.lift_delay) / ms) + 1
        # log: CPG spike time of each push, pose after it (first pose: start), feet states
        self.push_times, self.poses, self.feet_states = [], [self.body.pose.copy()], []
        self._n_motor_seen = self._n_cpg_seen = 0

        @network_operation(dt=config.SENSE_DT, when='end')
        def sense(t):
            t_ms = float(t / ms)
            self.odometry(t_ms)
            if self.sensors is not None:
                self.sensors.update(self.pose_at(t_ms))

        objects = [*self.cpg.objects, *self.legs.objects, *self.layer.objects,
                   self.brain.driver(command_fn), sense, self.mon_cpg, self.mon_motor]
        if self.sensors is not None:
            objects += self.sensors.objects
        self.net = Network(*objects)

    def odometry(self, t_ms):
        """Move the body by the latest half-cycle, once all its motor spikes are in."""
        if self.mon_cpg.num_spikes == self._n_cpg_seen:
            return
        t_cpg = float(np.asarray(self.mon_cpg.t / ms)[-1])
        if t_ms < t_cpg + self.settle_ms:
            return
        phase = int(np.asarray(self.mon_cpg.i[:])[-1])
        self.body.step(self.legs.spike_counts(self.mon_motor.i[self._n_motor_seen:]), phase)
        self._n_motor_seen, self._n_cpg_seen = self.mon_motor.num_spikes, self.mon_cpg.num_spikes
        self.push_times.append(t_cpg)
        self.poses.append(self.body.pose.copy())
        self.feet_states.append(self.body.feet_state())

    def pose_at(self, t_ms):
        """Body pose at t_ms: the latest push spread linearly over its half-cycle."""
        if not self.push_times:
            return self.poses[0]
        a = np.clip((t_ms - self.push_times[-1]) / self.half_ms, 0, 1)
        return self.poses[-2] + a * (self.poses[-1] - self.poses[-2])

    def feet_at(self, t_ms, pose):
        """Drawn legs (hips, feet, raised) at t_ms and pose."""
        t_in = t_ms - self.push_times[-1] if self.push_times else self.half_ms
        return self.body.feet(pose, t_in)

    def run(self, duration):
        self.net.run(duration)

    def trajectory(self):
        """(times, poses, feet_states) in the format of Body.trajectory()."""
        times = np.append(self.push_times, self.push_times[-1] + self.half_ms)
        return times, np.array(self.poses), self.feet_states
