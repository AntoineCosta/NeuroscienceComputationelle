"""Le « cerveau » : code classique qui traduit les commandes en entrées de la couche intermédiaire."""
import numpy as np
from brian2 import network_operation

from hexapode.reseau.intermediates import I_FWD_L, I_FWD_R, I_REV_L, I_REV_R, I_STEER_L, I_STEER_R


class Brain:
    """
    Traduit une commande (vitesse, virage) en entrée `brain` des 6 intermédiaires.
    Toutes les entrées signifient « frein » (0 = pas de frein, 1 = frein total).
    Par défaut (tonique) : arrêt.

      speed dans [-1, 1] : > 0 marche avant, < 0 marche arrière
      steer dans [-1, 1] : < 0 gauche, > 0 droite
        - en marche : ralentit les pattes de ce côté
        - à vitesse 0 : rotation sur place (ce côté recule, l'autre avance)

    | intermédiaire | arrêt | avant s | arrière s | virage gauche x | rotation gauche x (vitesse 0) |
    | I-Fwd-L       |   1   |  1 - s  |     1     |   inchangé      |              1                |
    | I-Fwd-R       |   1   |  1 - s  |     1     |   inchangé      |            1 - x              |
    | I-Rev-L       |   1   |    1    |   1 - s   |   inchangé      |            1 - x              |
    | I-Rev-R       |   1   |    1    |   1 - s   |   inchangé      |              1                |
    | I-Steer-L     |   0   |    0    |     0     |    |s| * x      |              0                |
    | I-Steer-R     |   0   |    0    |     0     |      0          |              0                |
    -> en marche : amplitude de chaque côté = |s| * (1 - virage de ce côté), poussée et retour.
    -> en rotation : amplitude = |x| des deux côtés, dans des sens opposés.
    """

    STOP = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    def __init__(self, layer):
        self.layer = layer
        self.layer.neurons.brain = self.STOP
        self.layer.neurons.gate = 0.0            # à l'arrêt : capteurs coupés

    @staticmethod
    def is_spin(speed, steer):
        """Rotation sur place : vitesse nulle avec un virage."""
        return speed == 0 and steer != 0

    @staticmethod
    def outputs(speed, steer=0.0):
        """Entrée `brain` des 6 intermédiaires pour une commande."""
        speed = float(np.clip(speed, -1, 1))
        steer = float(np.clip(steer, -1, 1))
        s = abs(speed)
        out = Brain.STOP.copy()
        if Brain.is_spin(speed, steer):         # rotation sur place
            a = abs(steer)
            if steer < 0:
                out[[I_REV_L, I_FWD_R]] = 1 - a  # pattes gauches en arrière, droites en avant
            else:
                out[[I_FWD_L, I_REV_R]] = 1 - a
            return out
        if speed > 0:
            out[[I_FWD_L, I_FWD_R]] = 1 - s
        elif speed < 0:
            out[[I_REV_L, I_REV_R]] = 1 - s
        out[I_STEER_L] = s * max(0.0, -steer)
        out[I_STEER_R] = s * max(0.0, steer)
        return out

    def set_command(self, speed, steer=0.0):
        """
        Applique une commande (effective au prochain potentiel d'action du CPG).
        Les capteurs ne sont activés qu'en marche (vitesse != 0) : à l'arrêt ou en
        rotation sur place, ils ne font jamais bouger le robot.
        """
        self.layer.neurons.brain = self.outputs(speed, steer)
        self.layer.neurons.gate = 1.0 if speed != 0 else 0.0

    def driver(self, command_fn):
        """
        NetworkOperation qui lit command_fn(t) -> (vitesse, virage) au début de chaque
        demi-cycle (juste avant le potentiel d'action du CPG). À ajouter au Network.
        """
        @network_operation(dt=self.layer.cpg.period / 2, when='start')
        def apply_command(t):
            self.set_command(*command_fn(t))
        return apply_command
