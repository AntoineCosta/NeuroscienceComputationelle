"""Motoneurones des pattes : groupes lever (R), poser (D), avant (F) et arrière (B)."""
from brian2 import *
import numpy as np

from hexapode.config import N_LEGS, N_MOTOR, TAU_M, EXC_DELAY, LIFT_DELAY

# Types de motoneurones
RAISE, LOWER, FORWARD, BACKWARD = 0, 1, 2, 3
KIND_NAMES = ['R', 'D', 'F', 'B']


class Leg:
    """
    Vue sur les motoneurones d'une patte (ne possède aucun objet Brian2).

    Disposition du bloc dans le NeuronGroup moteur partagé :
        [R, D, F_0 .. F_{N-1}, B_0 .. B_{N-1}]
    """

    def __init__(self, motors, index, n_motor, n_legs):
        per_side = n_legs // 2
        self.index = index
        self.side = 0 if index < per_side else 1        # 0 = gauche, 1 = droite
        self.rank = index % per_side                     # 0 = patte avant
        self.phase = (self.rank + self.side) % 2         # neurone du CPG qui pilote son appui
        self.name = f"{'LR'[self.side]}{self.rank + 1}"

        self.start = index * (2 + 2 * n_motor)
        s = self.start
        self.raise_n = motors[s:s + 1]
        self.lower_n = motors[s + 1:s + 2]
        self.forward = motors[s + 2:s + 2 + n_motor]
        self.backward = motors[s + 2 + n_motor:s + 2 + 2 * n_motor]

    def __repr__(self):
        return f"Leg({self.name}, phase d'appui {'AB'[self.phase]})"


class LegSet:
    """
    Tous les motoneurones des pattes, pilotés par le CPG.

    Motoneurone : LIF (v relatif au repos), seuils régulièrement espacés :
        theta_k = (k + 1/2) * u     (R et D : theta_0)
    Chaque potentiel d'action du CPG donne une seule excitation de N*u à ses cibles F/B
    (un PPSE, potentiel postsynaptique excitateur ; une inhibition est un PPSI) : sans
    inhibition, les N neurones d'un groupe déchargent (une fois, tous ensemble). Une
    inhibition arrivant juste avant abaisse v de m*u au moment de l'excitation : seuls les
    N - m seuils les plus bas sont atteints. Le poids inhibiteur est donné par
    inhibition_weight(m).

    Câblage du CPG, pour une patte dont l'appui correspond à la phase p du CPG :
      - D (poser) <- CPG p (début de l'appui)
      - R (lever) <- CPG 1-p (début du balancement), retardé de lift_delay
      - F et B    <- les deux neurones du CPG (direction choisie par la couche intermédiaire)
    R reçoit une excitation de 2N*u : il est freiné par les deux intermédiaires de direction
    (avant et arrière), donc la patte se lève dès qu'une direction a une vitesse non nulle.
    D reçoit N*u et n'est jamais freiné : chaque début d'appui pose le pied, donc à
    l'arrêt tous les pieds finissent au sol (poser un pied déjà posé ne change rien).
    """

    def __init__(self, cpg, n_legs=N_LEGS, n_motor=N_MOTOR, tau_m=TAU_M,
                 exc_delay=EXC_DELAY, lift_delay=LIFT_DELAY, u=1*mV, t_ref=2*ms):
        if n_legs % 2 or not 6 <= n_legs <= 400:
            raise ValueError("n_legs doit être pair et compris entre 6 et 400")
        if not defaultclock.dt <= exc_delay < cpg.period / 2:
            raise ValueError("exc_delay doit être compris entre dt et une demi-période")
        if lift_delay + exc_delay >= cpg.period / 2:
            raise ValueError("lift_delay + exc_delay doit être plus court qu'une demi-période")
        if tau_m > cpg.period / 20:
            print(f"ATTENTION : tau_m={tau_m} est long par rapport au demi-cycle : les motoneurones "
                  f"ne reviennent pas au repos entre deux potentiels d'action du CPG, le recrutement sera faux.")
        self.cpg = cpg
        self.n_legs = n_legs
        self.n_motor = n = n_motor
        self.tau_m = tau_m
        self.exc_delay = exc_delay
        self.lift_delay = lift_delay
        self.u = u
        block = 2 + 2 * n

        # --- Constantes par neurone ------------------------------------------------
        theta = (np.arange(n) + 0.5) * float(u / volt)
        kind = np.array([RAISE, LOWER] + [FORWARD] * n + [BACKWARD] * n)
        rank = np.array([0, 0] + list(range(n)) * 2)
        v_th = np.concatenate([[theta[0], theta[0]], theta, theta])
        legs_phase = np.array([(i % (n_legs // 2) + i // (n_legs // 2)) % 2 for i in range(n_legs)])
        legs_side = np.array([i // (n_legs // 2) for i in range(n_legs)])

        self.eqs = '''
        dv/dt = -v / tau_m : volt (unless refractory)
        v_th : volt (constant)
        kind : integer (constant)
        rank : integer (constant)
        leg : integer (constant)
        side : integer (constant)   # 0 = gauche, 1 = droite
        phase : integer (constant)  # neurone du CPG qui pilote l'appui de cette patte
        '''
        self.motors = NeuronGroup(n_legs * block, model=self.eqs, threshold='v > v_th',
                                  reset='v = 0*volt', refractory=t_ref, method='exact',
                                  namespace={'tau_m': tau_m})
        self.motors.kind = np.tile(kind, n_legs)
        self.motors.rank = np.tile(rank, n_legs)
        self.motors.v_th = np.tile(v_th, n_legs) * volt
        self.motors.leg = np.repeat(np.arange(n_legs), block)
        self.motors.side = np.repeat(legs_side, block)
        self.motors.phase = np.repeat(legs_phase, block)

        self.legs = [Leg(self.motors, i, n, n_legs) for i in range(n_legs)]

        # --- Excitation par le CPG : excitation de N*u (F/B/D) ou 2N*u (R) par potentiel d'action
        # v fuit pendant un pas entre le saut synaptique et le test du seuil :
        # le poids compense cette fuite pour avoir v = N*u (2N*u) exactement au test.
        self.excitation = Synapses(cpg.neurons, self.motors, model='w : volt (constant)',
                                   on_pre='v_post += w')
        self.excitation.connect(condition=f'kind_post >= {FORWARD} '
                                          f'or (kind_post == {LOWER} and i == phase_post) '
                                          f'or (kind_post == {RAISE} and i != phase_post)')
        is_r = np.asarray(self.motors.kind[:])[np.asarray(self.excitation.j[:])] == RAISE
        self.excitation.w = (np.where(is_r, 2 * n, n) * float(u / volt)
                             * np.exp(float(defaultclock.dt / tau_m))) * volt
        self.excitation.delay = self.input_delay(self.excitation, exc_delay)

    @property
    def objects(self):
        """Objets Brian2 à ajouter à un Network (CPG non inclus)."""
        return [self.motors, self.excitation]

    def input_delay(self, syn, base_delay):
        """Tableau des délais par synapse : base_delay, + lift_delay pour les cibles R."""
        is_raise = np.asarray(self.motors.kind[:])[np.asarray(syn.j[:])] == RAISE
        return (float(base_delay / second)
                + np.where(is_raise, float(self.lift_delay / second), 0.0)) * second

    def inhibition_weight(self, m, lead=None):
        """
        Poids d'UN potentiel d'action qui décale le potentiel d'un groupe de m paliers de
        seuil au moment de l'excitation du CPG : en inhibition, il fait taire les m seuils les
        plus hauts (m = N fait tout taire, R/D compris) ; en excitation, il annule m paliers
        d'inhibition. m peut être fractionnaire (le recrutement arrondit au palier le plus proche).
        lead : temps entre l'arrivée du potentiel d'action et l'excitation du CPG
        (par défaut : exc_delay, c.-à-d. une arrivée au moment du potentiel d'action du CPG).
        """
        lead = self.exc_delay if lead is None else lead
        # fuite pendant `lead`, plus le pas qui précède le test du seuil
        return m * self.u * np.exp(float((lead + defaultclock.dt) / self.tau_m))

    def spike_counts(self, indices):
        """[patte, type] nombre de potentiels d'action des motoneurones donnés (usage en direct)."""
        indices = np.asarray(indices, dtype=int)
        counts = np.zeros((self.n_legs, 4), dtype=int)
        np.add.at(counts, (np.asarray(self.motors.leg[:])[indices],
                           np.asarray(self.motors.kind[:])[indices]), 1)
        return counts

    def recruitment(self, mon_cpg, mon_motor):
        """
        Compte les motoneurones qui ont déchargé à chaque demi-cycle du CPG.

        Renvoie (on_t, on_p, counts, max_rank) :
          on_t     : temps des potentiels d'action du CPG (ms), un par demi-cycle
          on_p     : phase du CPG (0 = A, 1 = B) de chaque demi-cycle
          counts   : [demi-cycle, patte, type] nombre de neurones qui ont déchargé
          max_rank : [demi-cycle, patte, type] rang le plus haut qui a déchargé (-1 si aucun)
        """
        # copies numpy simples (les vues Brian2 ne gèrent pas l'indexation avancée avec répétitions)
        kind_arr = np.asarray(self.motors.kind[:])
        leg_arr = np.asarray(self.motors.leg[:])
        rank_arr = np.asarray(self.motors.rank[:])

        cpg_t = np.asarray(mon_cpg.t / ms)
        order = np.argsort(cpg_t)
        on_t, on_p = cpg_t[order], np.asarray(mon_cpg.i[:])[order]
        half_ms = float(self.cpg.period / 2 / ms)

        # Ramène chaque potentiel d'action moteur au potentiel d'action du CPG qui l'a causé
        idx = np.asarray(mon_motor.i[:])
        t_eff = (np.asarray(mon_motor.t / ms) - float(self.exc_delay / ms)
                 - np.where(kind_arr[idx] == RAISE, float(self.lift_delay / ms), 0.0))
        cycle = np.searchsorted(on_t, t_eff + 1e-6, side='right') - 1
        keep = cycle >= 0
        keep[keep] &= t_eff[keep] < on_t[cycle[keep]] + half_ms
        idx, cycle = idx[keep], cycle[keep]

        counts = np.zeros((len(on_t), self.n_legs, 4), dtype=int)
        np.add.at(counts, (cycle, leg_arr[idx], kind_arr[idx]), 1)
        max_rank = -np.ones((len(on_t), self.n_legs, 4), dtype=int)
        np.maximum.at(max_rank, (cycle, leg_arr[idx], kind_arr[idx]), rank_arr[idx])
        return on_t, on_p, counts, max_rank
