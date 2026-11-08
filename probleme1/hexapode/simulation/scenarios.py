"""Scénarios de commande : listes de (temps de début, vitesse, virage) -> command_fn(t)."""
from brian2 import ms


DEFAULT_TIMELINE = [  # (temps de début, vitesse, virage)
    (0 * ms, 1.0, 0.0),       # avant 100 %
    (1000 * ms, 0.5, 0.0),    # avant 50 %
    (2000 * ms, 1.0, -0.5),   # avant 100 %, virage gauche 50 % -> pattes gauches à 50 %
    (3000 * ms, 0.5, 0.4),    # avant 50 %, virage droite 40 % -> pattes droites à 30 %
    (4000 * ms, 0.0, 0.0),    # arrêt
    (5000 * ms, -0.7, 0.0),   # arrière 70 %
    (6000 * ms, 0.0, -1.0),   # rotation sur place vers la gauche, 100 %
    (7000 * ms, -1.0, 0.5),   # arrière 100 %, virage droite 50 %
]


def timeline_command(timeline):
    """command_fn(t) -> (vitesse, virage) à partir d'une liste de (temps de début, vitesse, virage)."""
    def command_fn(t):
        speed, steer = 0.0, 0.0
        for t_start, sp, st in timeline:
            if t >= t_start:
                speed, steer = sp, st
        return speed, steer
    return command_fn
