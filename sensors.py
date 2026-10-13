"""Capteurs de distance : 6 neurones capteurs gradués (lignes de visée 1D) reliés aux intermédiaires."""
from brian2 import NeuronGroup, Synapses
import numpy as np

from config import (SENSOR_ANGLE, FRONT_RANGE, REAR_RANGE, SIDE_RANGE, CONTACT_MARGIN,
                    BACKOFF_DISTANCE, BACKOFF_RAMP, STEER_GAIN, SIDE_STEER_GAIN)
from intermediates import I_FWD_L, I_FWD_R, I_REV_L, I_REV_R, I_STEER_L, I_STEER_R

FL, FR, RL, RR, SL, SR = range(6)
SENSOR_NAMES = ['FL', 'FR', 'RL', 'RR', 'SL', 'SR']


class Sensors:
    """
    Six capteurs de distance sur le corps : FL / FR au nez (+-SENSOR_ANGLE),
    RL / RR aux coins arrière (180 -+ SENSOR_ANGLE), SL / SR au milieu des flancs
    (+-90 deg). Chacun est un neurone gradué :
        near  = clip((range - distance) / (range - margin), 0, 1)   0 hors de portée,
                                                          1 de `margin` jusqu'au contact
        close = clip((d_back - distance) / ramp, 0, 1)    zone de recul
    Leur somme forme l'entrée `sensor` des intermédiaires :

      | capteur  | I-Fwd-L/R         | I-Rev-L/R         | I-Steer-R   | I-Steer-L   |
      | FL       | near / 2 + close  | - close           | + g * near  | - g * near  |
      | FR       | near / 2 + close  | - close           | - g * near  | + g * near  |
      | RL, RR   | - close           | near / 2 + close  |             |             |
      | SL       |                   |                   | + gs * near | - gs * near |
      | SR       |                   |                   | - gs * near | + gs * near |

      - obstacle devant (derrière) : freine la marche avant (arrière), de façon graduée
        selon la distance ; un mur droit devant est vu par les deux capteurs, chacun
        apporte la moitié
      - plus près que d_back : cette direction est freinée à fond et l'opposée est
        libérée (désinhibition : le capteur inhibe le I-Fwd / I-Rev opposé) -> le robot
        s'éloigne jusqu'à sortir de la zone, puis la commande du cerveau reprend la main
        (va-et-vient), en avant comme en arrière
      - obstacle vu davantage d'un côté par les capteurs avant : l'autre côté est freiné
        -> le robot s'en détourne (différence FL - FR, donc un mur droit devant ne fait
        que freiner)
      - mur le long du flanc : même chose avec SL et SR (virage seulement : le robot ne
        peut pas faire de pas de côté). Égal des deux côtés dans un couloir -> le robot
        se centre. Tourne aussi dans le bon sens en marche arrière (l'arrière s'éloigne du mur).
    Le Brain coupe les capteurs hors de la marche : à l'arrêt ou en rotation sur place,
    ils ne font jamais bouger le robot.
    Les distances sont écrites par update() (code classique : le monde n'est pas neuronal).
    """

    def __init__(self, layer, body, world, angle=SENSOR_ANGLE, front_range=FRONT_RANGE,
                 rear_range=REAR_RANGE, margin=CONTACT_MARGIN, d_back=BACKOFF_DISTANCE,
                 ramp=BACKOFF_RAMP, steer_gain=STEER_GAIN, side_range=SIDE_RANGE,
                 side_gain=SIDE_STEER_GAIN):
        self.body, self.world = body, world
        l, w = body.length / 2, body.width / 2
        a = np.radians(angle)
        # points de fixation (repère du corps : x vers l'avant, y vers la gauche) et directions
        self.mounts = np.array([[l, 0.0], [l, 0.0], [-l, w], [-l, -w], [0.0, w], [0.0, -w]])
        self.angles = np.array([a, -a, np.pi - a, -np.pi + a, np.pi / 2, -np.pi / 2])
        self.ranges = np.array([front_range, front_range, rear_range, rear_range,
                                side_range, side_range])
        self.margin = margin

        self.eqs = '''
        distance : 1                                        # mètres (écrit par update)
        sensor_range : 1 (constant)
        margin : 1 (constant)                               # frein maximal en dessous
        d_back : 1 (constant)                               # distance de recul
        ramp : 1 (constant)                                 # rampe du recul
        near = clip((sensor_range - distance) / (sensor_range - margin), 0, 1) : 1
        close = clip((d_back - distance) / ramp, 0, 1) : 1
        '''
        self.neurons = NeuronGroup(len(self.ranges), model=self.eqs)
        self.neurons.sensor_range = self.ranges
        self.neurons.margin = margin
        self.neurons.d_back = d_back
        self.neurons.ramp = ramp
        self.neurons.distance = self.ranges

        # (capteur, intermédiaire, poids de near, poids de close)
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
        """Objets Brian2 à ajouter à un Network."""
        return [self.neurons, self.synapses]

    def rays(self, pose):
        """(origines [6, 2], extrémités [6, 2], distances [6]) des lignes de visée à une pose."""
        origins = self.body.to_world(self.mounts, pose)
        dist = np.array([self.world.ray_distance(o, pose[2] + a, r)
                         for o, a, r in zip(origins, self.angles, self.ranges)])
        angles = pose[2] + self.angles
        ends = origins + dist[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
        return origins, ends, dist

    def near(self, dist):
        """Activité graduée « near » pour des distances (même formule que les neurones, pour l'affichage)."""
        return np.clip((self.ranges - dist) / (self.ranges - self.margin), 0, 1)

    def update(self, pose):
        """Écrit les distances mesurées dans les neurones capteurs."""
        self.neurons.distance = self.rays(pose)[2]
