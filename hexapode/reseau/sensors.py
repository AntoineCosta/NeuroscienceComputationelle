"""Capteurs de distance : 4 neurones capteurs gradués (lignes de visée 1D) reliés aux intermédiaires."""
from brian2 import NeuronGroup, Synapses
import numpy as np

from hexapode.config import (FRONT_RANGE, REAR_RANGE, SIDE_RANGE, CONTACT_MARGIN,
                    BACKOFF_DISTANCE, BACKOFF_RAMP, SIDE_STEER_GAIN)
from hexapode.reseau.intermediates import I_FWD_L, I_FWD_R, I_REV_L, I_REV_R, I_STEER_L, I_STEER_R

FRONT, REAR, SL, SR = range(4)
SENSOR_NAMES = ['Avant', 'Arrière', 'SL', 'SR']


class Sensors:
    """
    Quatre capteurs de distance sur le corps : avant au nez (0 deg), arrière à la queue
    (180 deg), SL / SR au milieu des flancs (+-90 deg). Chacun est un neurone gradué :
        near  = clip((range - distance) / (range - margin), 0, 1)   0 hors de portée,
                                                          1 de `margin` jusqu'au contact
        close = clip((d_back - distance) / ramp, 0, 1)    zone de recul
    Leur somme forme l'entrée `sensor` des intermédiaires :

      | capteur  | I-Fwd-L/R     | I-Rev-L/R     | I-Steer-R   | I-Steer-L   |
      | Avant    | near + close  | - close       |             |             |
      | Arrière  | - close       | near + close  |             |             |
      | SL       |               |               | + gs * near | - gs * near |
      | SR       |               |               | - gs * near | + gs * near |

      - obstacle devant (derrière) : freine la marche avant (arrière), de façon graduée
        selon la distance
      - plus près que d_back : cette direction est freinée à fond et l'opposée est
        libérée (désinhibition : le capteur inhibe le I-Fwd / I-Rev opposé) -> le robot
        s'éloigne jusqu'à sortir de la zone, puis la commande du cerveau reprend la main
        (va-et-vient), en avant comme en arrière
      - mur le long du flanc : le côté opposé est freiné -> le robot s'en détourne
        (virage seulement : il ne peut pas faire de pas de côté). Dans un couloir, le
        robot avance sans toucher les murs mais ondule autour du centre : à 90 deg, les
        capteurs mesurent la position latérale mais pas le cap, rien n'amortit le virage.
        Tourne aussi dans le bon sens en marche arrière (l'arrière s'éloigne du mur).
    Le Brain coupe les capteurs hors de la marche : à l'arrêt ou en rotation sur place,
    ils ne font jamais bouger le robot.
    Les distances sont écrites par update() (code classique : le monde n'est pas neuronal).
    """

    def __init__(self, layer, body, world, front_range=FRONT_RANGE, rear_range=REAR_RANGE,
                 margin=CONTACT_MARGIN, d_back=BACKOFF_DISTANCE, ramp=BACKOFF_RAMP,
                 side_range=SIDE_RANGE, side_gain=SIDE_STEER_GAIN):
        self.body, self.world = body, world
        l, w = body.length / 2, body.width / 2
        # points de fixation (repère du corps : x vers l'avant, y vers la gauche) et directions
        self.mounts = np.array([[l, 0.0], [-l, 0.0], [0.0, w], [0.0, -w]])
        self.angles = np.array([0.0, np.pi, np.pi / 2, -np.pi / 2])
        self.ranges = np.array([front_range, rear_range, side_range, side_range])
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
        table = []
        for s, own, other in ((FRONT, (I_FWD_L, I_FWD_R), (I_REV_L, I_REV_R)),
                              (REAR, (I_REV_L, I_REV_R), (I_FWD_L, I_FWD_R))):
            table += [(s, i, 1, 1) for i in own]       # frein de sa direction, puis recul
            table += [(s, i, 0, -1) for i in other]    # désinhibition de la direction opposée
        for s, own_steer, other_steer in ((SL, I_STEER_R, I_STEER_L), (SR, I_STEER_L, I_STEER_R)):
            table += [(s, own_steer, side_gain, 0), (s, other_steer, -side_gain, 0)]
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
        """(origines [4, 2], extrémités [4, 2], distances [4]) des lignes de visée à une pose."""
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
