"""Robot : la boucle fermée complète (réseau, corps, monde, capteurs), en direct ou scénarisée."""
import numpy as np
from brian2 import Network, SpikeMonitor, network_operation, ms

from hexapode import config
from hexapode.reseau.CPG import CPG
from hexapode.reseau.legs import LegSet
from hexapode.reseau.intermediates import IntermediateLayer
from hexapode.simulation.brain import Brain
from hexapode.simulation.body import Body
from hexapode.simulation.world import World
from hexapode.reseau.sensors import Sensors


class Robot:
    """
    CPG + pattes + intermédiaires + cerveau, le corps déplacé par les potentiels d'action
    des motoneurones, et les capteurs de distance qui lisent le monde à la pose courante.

    command_fn(t) -> (vitesse, virage) : scénario ou pilotage en direct.
    world : World (disposition par défaut) ou None pour aucun obstacle et aucun capteur.

    Boucle fermée, tous les SENSE_DT :
      - odométrie : une fois tous les potentiels d'action moteurs du dernier demi-cycle
        arrivés (exc_delay + lift_delay après le potentiel d'action du CPG), le corps
        avance de la poussée de ce demi-cycle, étalée sur le demi-cycle (pose_at)
      - capteurs : distances mesurées à la pose courante -> neurones capteurs
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
        # tous les potentiels d'action moteurs dus à un potentiel d'action du CPG sont
        # arrivés après ce temps (R est le dernier)
        self.settle_ms = float((self.legs.exc_delay + self.legs.lift_delay) / ms) + 1
        # historique : temps du potentiel d'action du CPG de chaque poussée, pose après
        # celle-ci (première pose : départ), état des pieds
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
        """Déplace le corps selon le dernier demi-cycle, une fois tous ses potentiels d'action moteurs arrivés."""
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
        """Pose du corps à t_ms : la dernière poussée étalée linéairement sur son demi-cycle."""
        if not self.push_times:
            return self.poses[0]
        a = np.clip((t_ms - self.push_times[-1]) / self.half_ms, 0, 1)
        return self.poses[-2] + a * (self.poses[-1] - self.poses[-2])

    def feet_at(self, t_ms, pose):
        """Pattes dessinées (hanches, pieds, levés) à t_ms et à la pose donnée."""
        t_in = t_ms - self.push_times[-1] if self.push_times else self.half_ms
        return self.body.feet(pose, t_in)

    def run(self, duration):
        self.net.run(duration)

    def trajectory(self):
        """(times, poses, feet_states) au même format que Body.trajectory()."""
        times = np.append(self.push_times, self.push_times[-1] + self.half_ms)
        return times, np.array(self.poses), self.feet_states
