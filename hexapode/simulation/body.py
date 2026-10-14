"""Cinématique du corps : pose de l'hexapode vue de dessus, déplacée par les pattes en appui."""
import numpy as np
from brian2 import ms

from hexapode.config import BODY_LENGTH, BODY_WIDTH, TRACK_WIDTH, STRIDE
from hexapode.reseau.legs import RAISE, LOWER, FORWARD, BACKWARD

SHOWN_RANKS = 3   # pattes dessinées par côté, quel que soit N_LEGS (affichage plus léger)


class Body:
    """
    Corps rigide vu de dessus, poussé par ses pattes en appui (comme un char).

    Chaque neurone F (B) qui décharge sur une patte en appui pousse le corps vers
    l'avant (l'arrière) de stride / N. Chaque côté avance de la poussée moyenne de ses
    pattes en appui (d_left, d_right) :
        avance = (d_left + d_right) / 2
        virage = (d_right - d_left) / track      (> 0 = sens trigonométrique = gauche)
    Le balancement (R et groupe de retour) ne déplace pas le corps.

    Pose = [x, y, cap] en mètres / radians, cap mesuré depuis l'axe x
    (départ par défaut : orienté vers +y, c.-à-d. « vers le haut » sur les figures).
    Un step() par demi-cycle du CPG : utilisable hors ligne (trajectoire) ou en direct.

    Pattes dessinées (affichage seulement) : rangs 0..SHOWN_RANKS-1 de chaque côté. Toutes
    les pattes d'un même côté et d'une même phase du CPG reçoivent les mêmes commandes,
    donc elles suffisent à montrer toute la démarche.
    Pied d'une patte dessinée, par rapport à sa hanche, à chaque demi-cycle :
      - balayage avant-arrière centré sur la hanche : F recule le pied, B l'avance,
        stride / N par neurone (à commande inchangée, le balancement est symétrique de l'appui)
      - levé par R, posé par D (au moment de leurs potentiels d'action) ; sans potentiel
        d'action F / B, le pied reste où il est.
    Repère du corps : x vers l'avant, y vers la gauche.
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
        # hanches réparties le long des flancs (rang avant en premier), pieds à l'écartement track
        spacing = length / SHOWN_RANKS
        self.hips = np.array([[((SHOWN_RANKS - 1) / 2 - r) * spacing, (0.5 - s) * width]
                              for s, r in zip(sides, ranks)])
        self.foot_lateral = np.array([(0.5 - s) * track for s in sides])
        self.reset()

    def reset(self, x=0.0, y=0.0, heading=np.pi / 2):
        self.pose = np.array([x, y, heading])
        n = len(self.shown)
        self.foot_from = np.zeros(n)              # décalage avant-arrière au début du demi-cycle
        self.foot_to = np.zeros(n)                # ... et à sa fin
        self.up_before = np.zeros(n, dtype=bool)  # levé avant le potentiel d'action R / D
        self.up_after = np.zeros(n, dtype=bool)   # levé après
        self.switch_ms = np.zeros(n)              # temps du potentiel d'action R / D après celui du CPG

    def side_push(self, counts_c, phase):
        """Poussée moyenne (m) des pattes en appui de chaque côté [gauche, droite] sur un demi-cycle."""
        push = [[], []]
        for leg in self.legs.legs:
            if leg.phase == phase:
                c = counts_c[leg.index]
                push[leg.side].append((c[FORWARD] - c[BACKWARD]) * self.stride / self.legs.n_motor)
        return np.array([np.mean(p) if p else 0.0 for p in push])

    def step(self, counts_c, phase):
        """
        Déplace le corps (et les pieds dessinés) d'un demi-cycle.
        counts_c : [patte, type] neurones qui ont déchargé ce demi-cycle ; phase : phase du CPG (appui).
        """
        d_left, d_right = self.side_push(counts_c, phase)
        advance = (d_left + d_right) / 2
        turn = (d_right - d_left) / self.track
        x, y, heading = self.pose
        mid = heading + turn / 2           # cap à mi-parcours : bonne approximation de l'arc
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
            else:                          # pas de mouvement : le pied reste où il est
                self.foot_from[k] = self.foot_to[k]
            if c[LOWER]:
                self.up_after[k], self.switch_ms[k] = False, exc_ms
            if c[RAISE]:
                self.up_after[k], self.switch_ms[k] = True, exc_ms + lift_ms

    def feet_state(self):
        """Copie de l'état des pieds dessinés au dernier step() (pour les rejeux)."""
        return tuple(a.copy() for a in (self.foot_from, self.foot_to, self.up_before,
                                        self.up_after, self.switch_ms))

    def feet(self, pose, t_in_ms, state=None):
        """
        Pattes dessinées à une pose, t_in_ms après le potentiel d'action du CPG du dernier
        step() (ou du demi-cycle dont feet_state() est donné).
        Renvoie (hanches [6, 2], pieds [6, 2], levés [6]) en coordonnées du monde.
        """
        foot_from, foot_to, up_before, up_after, switch_ms = state or self.feet_state()
        a = np.clip(t_in_ms / float(self.legs.cpg.period / 2 / ms), 0, 1)
        fore = foot_from + a * (foot_to - foot_from)
        local_feet = np.column_stack([self.hips[:, 0] + fore, self.foot_lateral])
        raised = np.where(t_in_ms >= switch_ms, up_after, up_before)
        return self.to_world(self.hips, pose), self.to_world(local_feet, pose), raised

    def trajectory(self, on_t, on_p, counts):
        """
        Rejoue une simulation enregistrée (sortie de LegSet.recruitment) depuis la pose courante.
        Renvoie (temps en ms, poses [k, 3], feet_states) :
          poses au début de chaque demi-cycle et à la fin du dernier,
          feet_states[c] : feet_state() du demi-cycle c (pour feet()).
        """
        half_ms = float(self.legs.cpg.period / 2 / ms)
        times = np.append(on_t, on_t[-1] + half_ms)
        poses, feet_states = [self.pose.copy()], []
        for c, p in enumerate(on_p):
            poses.append(self.step(counts[c], p).copy())
            feet_states.append(self.feet_state())
        return times, np.array(poses), feet_states

    def outline(self, pose):
        """Coins (x, y) du corps, dessiné comme un polygone en forme de flèche, à une pose."""
        l, w = self.length / 2, self.width / 2
        return self.to_world(np.array([[-l, -w], [l / 2, -w], [l, 0.0], [l / 2, w], [-l, w]]), pose)

    @staticmethod
    def to_world(local, pose):
        """Points dans le repère du corps (x vers l'avant, y vers la gauche) -> coordonnées du monde."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return local @ np.array([[c, s], [-s, c]]) + pose[:2]
