"""Couche intermédiaire : 6 interneurones inhibiteurs gradués entre le cerveau et les pattes."""
from brian2 import *
import numpy as np

from hexapode.reseau.legs import RAISE, FORWARD

# Indices des neurones intermédiaires
I_FWD_L, I_FWD_R, I_REV_L, I_REV_R, I_STEER_L, I_STEER_R = range(6)
INTER_NAMES = ['I-Fwd-L', 'I-Fwd-R', 'I-Rev-L', 'I-Rev-R', 'I-Steer-L', 'I-Steer-R']
# Types d'intermédiaires
T_DRIVE, T_STEER = 0, 1


class IntermediateLayer:
    """
    Six interneurones inhibiteurs gradués (sans potentiel d'action), un par commande du cerveau.

    Chaque intermédiaire est silencieux au repos. Son activité est la somme de ses
    entrées, saturée entre 0 et 1 :
        level = clip(brain + gate * sensor, 0, 1)
    brain  : fixé par le Brain (code classique)
    sensor : somme des entrées des neurones capteurs (sensors.py). Surtout excitatrice
             (frein) ; le réflexe de recul inhibe la direction opposée (désinhibition).
    gate   : fixé par le Brain, à 1 seulement en marche (arrêt ou rotation : capteurs ignorés).

    À chaque potentiel d'action du CPG (signal de phase), chaque intermédiaire libère une
    inhibition de level * N paliers de seuil sur ses cibles, qui arrive avant l'excitation du CPG :

      I-Fwd-X (côté X) : marche avant du côté X
          F des pattes en appui, B des pattes en balancement, R
      I-Rev-X (côté X) : marche arrière du côté X
          B des pattes en appui, F des pattes en balancement, R
      I-Steer-X (côté X) : F et B du côté X, dans les deux phases
    D n'est jamais freiné : chaque début d'appui pose le pied.

    Exemple, marche avant à vitesse s (I-Fwd = 1 - s, I-Rev = 1, I-Steer = 0) :
      F en appui        : N - (1 - s)N = sN déchargent    (poussée)
      B en balancement  : N - (1 - s)N = sN déchargent    (retour, symétrique de la poussée)
      B en appui, F en balancement : N - N = 0
      R (excitation 2N) : 2N - (1 - s)N - N = sN > 0 -> la patte se lève
    Avec tous les intermédiaires à 0, tout déchargerait : le Brain doit donc envoyer
    le motif « arrêt » par défaut.

    Les intermédiaires savent quelles pattes sont en appui grâce à `stance_phase`, écrit
    par le potentiel d'action du CPG (remplace des neurones de signal de phase non modélisés).
    """

    def __init__(self, cpg, legs):
        dt = defaultclock.dt
        if legs.exc_delay < 2 * dt:
            raise ValueError("exc_delay doit valoir au moins 2 pas de temps (latence de libération des intermédiaires)")
        self.cpg, self.legs = cpg, legs

        self.eqs = '''
        brain : 1                            # entrée du cerveau (code classique)
        sensor : 1                           # somme des entrées des neurones capteurs
        gate : 1                             # porte des capteurs, fixée par le cerveau
        level = clip(brain + gate * sensor, 0, 1) : 1   # activité graduée dans [0, 1]
        release_now : boolean                # mis à True par le signal de phase à chaque potentiel d'action du CPG
        stance_phase : integer               # phase du CPG des pattes en appui
        itype : integer (constant)           # marche / virage
        direction : integer (constant)       # marche : 0 = avant, 1 = arrière
        side : integer (constant)            # 0 = gauche, 1 = droite
        '''
        self.neurons = NeuronGroup(6, model=self.eqs, events={'release': 'release_now'})
        self.neurons.run_on_event('release', 'release_now = False')
        self.neurons.itype = [T_DRIVE] * 4 + [T_STEER] * 2
        self.neurons.direction = [0, 0, 1, 1, 0, 0]
        self.neurons.side = [0, 1, 0, 1, 0, 1]

        # --- Signal de phase : chaque potentiel d'action du CPG déclenche une libération (un pas plus tard)
        self.trigger = Synapses(cpg.neurons, self.neurons,
                                on_pre='release_now_post = True; stance_phase_post = i')
        self.trigger.connect()

        # Libération un pas après le potentiel d'action du CPG -> avance = exc_delay - dt
        ns = {'w_full': legs.inhibition_weight(legs.n_motor, lead=legs.exc_delay - dt)}

        # --- Intermédiaires de marche : poussée (appui) et retour (balancement) de leur direction
        # (Brian analyse une instruction par ligne : les variables temporaires gardent le code lisible)
        self.drive = Synapses(self.neurons, legs.motors, on_event='release', namespace=ns, on_pre=f'''
            in_stance = int(kind_post == {FORWARD} + direction_pre) * int(phase_post == stance_phase_pre)
            in_swing = int(kind_post == {FORWARD + 1} - direction_pre) * int(phase_post != stance_phase_pre)
            v_post -= w_full * level_pre * (in_stance + in_swing + int(kind_post == {RAISE}))''')
        self.drive.connect(condition=f'itype_pre == {T_DRIVE} and side_post == side_pre')
        self.drive.delay = legs.input_delay(self.drive, 0*ms)

        # --- Intermédiaires de virage : F et B d'un côté, dans les deux phases ------
        self.steer = Synapses(self.neurons, legs.motors, on_event='release', namespace=ns,
                              on_pre='v_post -= w_full * level_pre')
        self.steer.connect(condition=f'itype_pre == {T_STEER} and side_post == side_pre '
                                     f'and kind_post >= {FORWARD}')

    @property
    def objects(self):
        """Objets Brian2 à ajouter à un Network (CPG et pattes non inclus)."""
        return [self.neurons, self.trigger, self.drive, self.steer]
