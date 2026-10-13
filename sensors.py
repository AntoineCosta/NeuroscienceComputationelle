"""Distance sensors: 6 graded sensor neurons (1D lines of sight) feeding the intermediates."""
from brian2 import NeuronGroup, Synapses
import numpy as np

from config import (SENSOR_ANGLE, FRONT_RANGE, REAR_RANGE, SIDE_RANGE, CONTACT_MARGIN,
                    BACKOFF_DISTANCE, BACKOFF_RAMP, STEER_GAIN, SIDE_STEER_GAIN)
from intermediates import I_FWD_L, I_FWD_R, I_REV_L, I_REV_R, I_STEER_L, I_STEER_R

FL, FR, RL, RR, SL, SR = range(6)
SENSOR_NAMES = ['FL', 'FR', 'RL', 'RR', 'SL', 'SR']


class Sensors:
    """
    Six distance sensors on the body: FL / FR at the nose (+-SENSOR_ANGLE),
    RL / RR at the rear corners (180 -+ SENSOR_ANGLE), SL / SR at the middle of the
    sides (+-90 deg). Each is a graded neuron:
        near  = clip((range - distance) / (range - margin), 0, 1)   0 out of range,
                                                          1 from `margin` to contact
        close = clip((d_back - distance) / ramp, 0, 1)    back-off zone
    Their summed output is the `sensor` input of the intermediates:

      | sensor   | I-Fwd-L/R         | I-Rev-L/R         | I-Steer-R   | I-Steer-L   |
      | FL       | near / 2 + close  | - close           | + g * near  | - g * near  |
      | FR       | near / 2 + close  | - close           | - g * near  | + g * near  |
      | RL, RR   | - close           | near / 2 + close  |             |             |
      | SL       |                   |                   | + gs * near | - gs * near |
      | SR       |                   |                   | - gs * near | + gs * near |

      - obstacle ahead (behind): brakes forward (reverse) walking, graded with the
        distance; a wall straight ahead is seen by both sensors, each gives half
      - closer than d_back: that direction fully braked and the opposite one released
        (disinhibition: the sensor inhibits the opposite I-Fwd / I-Rev) -> the robot
        walks away until it leaves the zone, then the brain's command takes over
        again (yoyo), forward and backward alike
      - obstacle seen more on one side by the front sensors: the other side is braked
        -> turns away (the difference of FL and FR, so a wall straight ahead only brakes)
      - wall alongside: the same with SL and SR (steering only: the robot cannot step
        sideways). Equal on both sides in a corridor -> centres itself. Also turns
        the right way when reversing (the tail moves away from the wall).
    The Brain gates the sensors off unless walking: stopped or turning on the spot,
    the sensors never move the robot.
    The distances are written by update() (regular code: the world is not neural).
    """

    def __init__(self, layer, body, world, angle=SENSOR_ANGLE, front_range=FRONT_RANGE,
                 rear_range=REAR_RANGE, margin=CONTACT_MARGIN, d_back=BACKOFF_DISTANCE,
                 ramp=BACKOFF_RAMP, steer_gain=STEER_GAIN, side_range=SIDE_RANGE,
                 side_gain=SIDE_STEER_GAIN):
        self.body, self.world = body, world
        l, w = body.length / 2, body.width / 2
        a = np.radians(angle)
        # mount points (body frame: x forward, y left) and directions
        self.mounts = np.array([[l, 0.0], [l, 0.0], [-l, w], [-l, -w], [0.0, w], [0.0, -w]])
        self.angles = np.array([a, -a, np.pi - a, -np.pi + a, np.pi / 2, -np.pi / 2])
        self.ranges = np.array([front_range, front_range, rear_range, rear_range,
                                side_range, side_range])
        self.margin = margin

        self.eqs = '''
        distance : 1                                        # metres (written by update)
        sensor_range : 1 (constant)
        margin : 1 (constant)                               # brake full below this
        d_back : 1 (constant)                               # back-off distance
        ramp : 1 (constant)                                 # back-off ramp
        near = clip((sensor_range - distance) / (sensor_range - margin), 0, 1) : 1
        close = clip((d_back - distance) / ramp, 0, 1) : 1
        '''
        self.neurons = NeuronGroup(len(self.ranges), model=self.eqs)
        self.neurons.sensor_range = self.ranges
        self.neurons.margin = margin
        self.neurons.d_back = d_back
        self.neurons.ramp = ramp
        self.neurons.distance = self.ranges

        # (sensor, intermediate, weight of near, weight of close)
        g = steer_gain
        table = []
        for s, own_steer, other_steer in ((FL, I_STEER_R, I_STEER_L), (FR, I_STEER_L, I_STEER_R)):
            table += [(s, I_FWD_L, 0.5, 1), (s, I_FWD_R, 0.5, 1),
                      (s, I_REV_L, 0, -1), (s, I_REV_R, 0, -1),
                      (s, own_steer, g, 0), (s, other_steer, -g, 0)]
        for s, own_steer, other_steer in ((SL, I_STEER_R, I_STEER_L), (SR, I_STEER_L, I_STEER_R)):
            table += [(s, own_steer, side_gain, 0), (s, other_steer, -side_gain, 0)]
        for s in (RL, RR):
            table += [(s, I_REV_L, 0.5, 1), (s, I_REV_R, 0.5, 1),
                      (s, I_FWD_L, 0, -1), (s, I_FWD_R, 0, -1)]
        i, j, w_near, w_close = np.array(table, dtype=float).T
        self.synapses = Synapses(self.neurons, layer.neurons, model='''
            w_near : 1 (constant)
            w_close : 1 (constant)
            sensor_post = w_near * near_pre + w_close * close_pre : 1 (summed)
            ''')
        self.synapses.connect(i=i.astype(int), j=j.astype(int))
        self.synapses.w_near = w_near
        self.synapses.w_close = w_close

    @property
    def objects(self):
        """Brian2 objects to add to a Network."""
        return [self.neurons, self.synapses]

    def rays(self, pose):
        """(origins [4, 2], ends [4, 2], distances [4]) of the lines of sight at a pose."""
        origins = self.body.to_world(self.mounts, pose)
        dist = np.array([self.world.ray_distance(o, pose[2] + a, r)
                         for o, a, r in zip(origins, self.angles, self.ranges)])
        angles = pose[2] + self.angles
        ends = origins + dist[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
        return origins, ends, dist

    def near(self, dist):
        """Graded 'near' activity for distances (same formula as the neurons, for display)."""
        return np.clip((self.ranges - dist) / (self.ranges - self.margin), 0, 1)

    def update(self, pose):
        """Write the measured distances into the sensor neurons."""
        self.neurons.distance = self.rays(pose)[2]
