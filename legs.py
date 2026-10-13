"""Leg motor neurons: raise (R), lower (D), forward (F) and backward (B) groups."""
from brian2 import *
import numpy as np

from config import N_LEGS, N_MOTOR, TAU_M, EXC_DELAY, LIFT_DELAY

# Motor neuron kinds
RAISE, LOWER, FORWARD, BACKWARD = 0, 1, 2, 3
KIND_NAMES = ['R', 'D', 'F', 'B']


class Leg:
    """
    View on the motor neurons of one leg (owns no Brian2 object).

    Block layout inside the shared motor NeuronGroup:
        [R, D, F_0 .. F_{N-1}, B_0 .. B_{N-1}]
    """

    def __init__(self, motors, index, n_motor, n_legs):
        per_side = n_legs // 2
        self.index = index
        self.side = 0 if index < per_side else 1        # 0 = left, 1 = right
        self.rank = index % per_side                     # 0 = front leg
        self.phase = (self.rank + self.side) % 2         # CPG neuron driving its stance
        self.name = f"{'LR'[self.side]}{self.rank + 1}"

        self.start = index * (2 + 2 * n_motor)
        s = self.start
        self.raise_n = motors[s:s + 1]
        self.lower_n = motors[s + 1:s + 2]
        self.forward = motors[s + 2:s + 2 + n_motor]
        self.backward = motors[s + 2 + n_motor:s + 2 + 2 * n_motor]

    def __repr__(self):
        return f"Leg({self.name}, stance phase {'AB'[self.phase]})"


class LegSet:
    """
    All leg motor neurons, driven by the CPG.

    Motor neuron: LIF (v relative to rest), thresholds evenly spaced:
        theta_k = (k + 1/2) * u     (R and D: theta_0)
    Each CPG spike gives a single EPSP of N*u to its F/B targets, so without inhibition
    all N neurons of a group fire (once, all together). An inhibitory spike arriving
    just before lowers v by m*u at the EPSP time: only the N - m lowest thresholds
    are reached. The inhibitory weight is given by inhibition_weight(m).

    CPG wiring, for a leg whose stance is CPG phase p:
      - D (lower) <- CPG p (stance onset)
      - R (raise) <- CPG 1-p (swing onset), delayed by lift_delay
      - F and B   <- both CPG neurons (direction selected by the intermediate layer)
    R gets an EPSP of 2N*u: it is braked by both direction intermediates (forward and
    reverse), so the leg lifts as soon as either direction has some speed.
    D gets N*u and is never braked: every stance onset puts the foot down, so at stop
    all feet end up on the ground (lowering a foot already down changes nothing).
    """

    def __init__(self, cpg, n_legs=N_LEGS, n_motor=N_MOTOR, tau_m=TAU_M,
                 exc_delay=EXC_DELAY, lift_delay=LIFT_DELAY, u=1*mV, t_ref=2*ms):
        if n_legs % 2 or not 6 <= n_legs <= 200:
            raise ValueError("n_legs must be even and between 6 and 200")
        if not defaultclock.dt <= exc_delay < cpg.period / 2:
            raise ValueError("exc_delay must be between dt and half a period")
        if lift_delay + exc_delay >= cpg.period / 2:
            raise ValueError("lift_delay + exc_delay must be shorter than half a period")
        if tau_m > cpg.period / 20:
            print(f"WARNING: tau_m={tau_m} is long compared with the half-cycle: motor neurons "
                  f"do not return to rest between CPG spikes, recruitment will be wrong.")
        self.cpg = cpg
        self.n_legs = n_legs
        self.n_motor = n = n_motor
        self.tau_m = tau_m
        self.exc_delay = exc_delay
        self.lift_delay = lift_delay
        self.u = u
        block = 2 + 2 * n

        # --- Per-neuron constants -------------------------------------------------
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
        side : integer (constant)   # 0 = left, 1 = right
        phase : integer (constant)  # CPG neuron driving this leg's stance
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

        # --- CPG excitation: one EPSP of N*u (F/B/D) or 2N*u (R) per CPG spike -------
        # v leaks for one step between the synaptic jump and the threshold test:
        # the weight compensates it so v = N*u (2N*u) exactly when tested.
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
        """Brian2 objects to add to a Network (CPG not included)."""
        return [self.motors, self.excitation]

    def input_delay(self, syn, base_delay):
        """Per-synapse delay array: base_delay, + lift_delay for R targets."""
        is_raise = np.asarray(self.motors.kind[:])[np.asarray(syn.j[:])] == RAISE
        return (float(base_delay / second)
                + np.where(is_raise, float(self.lift_delay / second), 0.0)) * second

    def inhibition_weight(self, m, lead=None):
        """
        Weight of ONE spike shifting a group's potential by m threshold steps at the
        CPG EPSP time: as an inhibition it silences the m highest thresholds (m = N
        silences everything, R/D included); as an excitation it cancels m steps of
        inhibition. m may be fractional (recruitment rounds to the nearest step).
        lead: time between the spike's arrival and the CPG EPSP
        (default: exc_delay, i.e. arriving at the CPG spike time).
        """
        lead = self.exc_delay if lead is None else lead
        # decays during `lead`, plus the one step before the threshold test
        return m * self.u * np.exp(float((lead + defaultclock.dt) / self.tau_m))

    def spike_counts(self, indices):
        """[leg, kind] number of spikes of the given motor neuron indices (live use)."""
        indices = np.asarray(indices, dtype=int)
        counts = np.zeros((self.n_legs, 4), dtype=int)
        np.add.at(counts, (np.asarray(self.motors.leg[:])[indices],
                           np.asarray(self.motors.kind[:])[indices]), 1)
        return counts

    def recruitment(self, mon_cpg, mon_motor):
        """
        Count the motor neurons fired in each CPG half-cycle.

        Returns (on_t, on_p, counts, max_rank):
          on_t     : CPG spike times (ms), one per half-cycle
          on_p     : CPG phase (0 = A, 1 = B) of each half-cycle
          counts   : [half-cycle, leg, kind] number of neurons fired
          max_rank : [half-cycle, leg, kind] highest rank fired (-1 if none)
        """
        # plain numpy copies (Brian2 views do not support fancy indexing with repeats)
        kind_arr = np.asarray(self.motors.kind[:])
        leg_arr = np.asarray(self.motors.leg[:])
        rank_arr = np.asarray(self.motors.rank[:])

        cpg_t = np.asarray(mon_cpg.t / ms)
        order = np.argsort(cpg_t)
        on_t, on_p = cpg_t[order], np.asarray(mon_cpg.i[:])[order]
        half_ms = float(self.cpg.period / 2 / ms)

        # Shift each motor spike back to the CPG spike that caused it
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
