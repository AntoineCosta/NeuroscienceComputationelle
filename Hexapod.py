from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
N_LEGS = 6                 # Number of legs: even, 6 <= N_LEGS <= 200
N_MOTOR = 10               # Neurons per forward/backward motor group = spikes per CPG burst (<= 50)
CYCLE_PERIOD = 500 * ms    # Full CPG cycle (phase A + phase B), fixed rhythm
BURST_DUTY = 0.8           # Fraction of each half-cycle covered by the burst (rest is silence)
TAU_LEAK = 1 * second      # Leak time constant of the leg motor neurons
LIFT_DELAY = 10 * ms       # Raise (R) delay after swing onset (other tripod touching down)

prefs.codegen.target = 'numpy'
defaultclock.dt = 0.5 * ms

# =============================================================================
# TODO (project roadmap)
# - Motor sequence timing inside a phase (D before F/B? F/B after R?).
# - Descending inhibition onto R must be shifted by LIFT_DELAY (like R's reset).
# - Descending neurons (Fwd-L/R, Rev-L/R, L, R) acting through tonically active
#   inhibitory interneurons (no command = everything inhibited = stand still).
#   Inhibition delivered in the silent gap, after the motor reset and before the
#   next burst (k inhibitory spikes silence the k highest thresholds).
# - Leg position (x, z) computed from motor spikes.
# - Manual input: scripted timeline, then keyboard / controller.
# - Sensor neurons hijacking descending commands for obstacle avoidance.
# =============================================================================


class CPG:
    """
    Half-center oscillator producing alternating bursts of exactly `n_spikes` spikes.

    Each of the 2 neurons (phase A = 0, phase B = 1) is a LIF neuron with a plateau
    state `plateau`:
      - while the plateau is on, a drive current makes it fire tonically at a fixed ISI;
      - each spike increments a calcium-like counter `ca`;
      - when `ca` reaches `n_spikes`, a 'burst_end' event fires (Ca-activated K+
        current in a biological neuron): the plateau shuts off and the counter resets;
      - the 'burst_end' event excites the partner after a silent gap, starting its
        plateau (handoff), so the rhythm is fixed: gap + (n_spikes-1)*ISI = period/2.
    Mutual inhibition shuts off the partner's plateau on every spike (guarantees
    that only one half-center bursts at a time).

    The 'burst_end' event is also meant to actively reset the leg motor neurons.
    """

    def __init__(self, n_spikes=N_MOTOR, period=CYCLE_PERIOD, duty=BURST_DUTY,
                 tau_m=20*ms, v_rest=-65*mV, v_th=-50*mV, v_reset=-65*mV, t_ref=2*ms):
        if n_spikes < 1:
            raise ValueError("n_spikes must be >= 1")
        dt = defaultclock.dt
        half = period / 2
        self.n_spikes = n_spikes
        self.period = period

        # --- Timing: ISI snapped to the simulation grid -----------------------
        if n_spikes > 1:
            isi_steps = int(round(float(duty * half / (n_spikes - 1) / dt)))
        else:
            isi_steps = int(round(float(half / dt)))  # irrelevant: burst ends after 1 spike
        isi_steps = max(isi_steps, int(np.ceil(float(t_ref / dt))) + 2)
        self.isi = isi_steps * dt
        self.burst_duration = (n_spikes - 1) * self.isi  # first to last spike
        self.gap = half - self.burst_duration            # last spike -> partner's first spike
        if self.gap < 2 * dt:
            raise ValueError(f"Burst too long for the period: lower duty or n_spikes "
                             f"(ISI={self.isi}, gap={self.gap})")

        # --- Drive current giving exactly this ISI (exact LIF solution) -------
        # After reset + refractory, v reaches v_th at t' with:
        #   v_inf = (a*v_th - v_reset)/(a - 1),  a = exp(t'/tau_m)
        # Brian records a spike at the start of the step in which v crosses v_th, so target
        # the crossing half a step AFTER the grid point (robust to rounding).
        a = np.exp(float((self.isi + dt/2 - t_ref) / tau_m))
        I_drive = (a * v_th - v_reset) / (a - 1) - v_rest

        self.eqs = '''
        dv/dt = (v_rest - v + plateau * I_drive) / tau_m : volt (unless refractory)
        ca : 1        # spike counter of the current burst (calcium-like)
        plateau : 1   # plateau state: 1 = bursting, 0 = silent
        '''
        params = {'tau_m': tau_m, 'v_rest': v_rest, 'v_th': v_th, 'v_reset': v_reset,
                  'I_drive': I_drive, 'n_spikes': n_spikes}

        self.neurons = NeuronGroup(2, model=self.eqs, threshold='v > v_th',
                                   reset='v = v_reset; ca += 1', refractory=t_ref,
                                   events={'burst_end': 'ca >= n_spikes'},
                                   method='exact', namespace=params)
        # Burst termination: plateau off, counter reset
        self.neurons.run_on_event('burst_end', 'ca = 0; plateau = 0; v = v_rest')

        # Initial state: phase A starts bursting immediately, phase B is silent
        self.neurons.v = [float(v_th + 1*mV), float(v_rest)] * volt
        self.neurons.plateau = [1, 0]
        self.neurons.ca = 0

        # Handoff: end of A's burst starts B's burst after the gap (and vice versa).
        # Latency = 1 step (event detection) + delay + 1 step (partner threshold).
        self.handoff = Synapses(self.neurons, self.neurons,
                                on_pre='plateau_post = 1; v_post = v_th + 1*mV',
                                on_event='burst_end', delay=self.gap - 2*dt,
                                namespace=params)
        self.handoff.connect(i=[0, 1], j=[1, 0])

        # Mutual inhibition: every spike shuts off the partner's plateau
        self.inhibition = Synapses(self.neurons, self.neurons,
                                   on_pre='plateau_post = 0; v_post = v_rest',
                                   namespace=params)
        self.inhibition.connect(i=[0, 1], j=[1, 0])

    @property
    def objects(self):
        """Brian2 objects to add to a Network."""
        return [self.neurons, self.handoff, self.inhibition]

    def burst_stats(self, spike_mon):
        """Split each neuron's spikes into bursts; return spike counts and burst onsets."""
        stats = {}
        trains = spike_mon.spike_trains()
        for idx in (0, 1):
            t = np.asarray(trains[idx] / ms)
            if len(t) == 0:
                stats[idx] = {'counts': [], 'onsets': []}
                continue
            # A new burst starts when the interval is clearly longer than the ISI
            breaks = np.where(np.diff(t) > 1.5 * float(self.isi / ms))[0] + 1
            bursts = np.split(t, breaks)
            stats[idx] = {'counts': [len(b) for b in bursts],
                          'onsets': [b[0] for b in bursts]}
        return stats

    def test_and_plot(self, duration=4*CYCLE_PERIOD, filename="cpg_bursts_class.png", show=True):
        """Isolated simulation with temporary monitors, checks and plots."""
        mon_state = StateMonitor(self.neurons, ['v', 'ca', 'plateau'], record=True)
        mon_spikes = SpikeMonitor(self.neurons)
        mon_end = EventMonitor(self.neurons, 'burst_end')

        net = Network(*self.objects, mon_state, mon_spikes, mon_end)
        print(f"Running isolated CPG: n_spikes={self.n_spikes}, period={self.period}, "
              f"ISI={self.isi}, gap={self.gap}")
        net.run(duration)

        # ----------------------------------------------------------------------
        # CHECKS
        # ----------------------------------------------------------------------
        stats = self.burst_stats(mon_spikes)
        for idx, name in ((0, 'A'), (1, 'B')):
            counts, onsets = stats[idx]['counts'], stats[idx]['onsets']
            # Last burst may be cut by the end of the simulation
            complete = counts[:-1] if len(counts) > 1 else counts
            periods = np.diff(onsets)
            print(f"Phase {name}: spikes per burst = {counts}, "
                  f"measured periods (ms) = {np.round(periods, 2).tolist()}")
            if [c for c in complete if c != self.n_spikes]:  # (brian2 shadows any())
                print(f"  WARNING: phase {name} bursts do not all contain {self.n_spikes} spikes")
        if stats[0]['onsets'] and stats[1]['onsets']:
            print(f"Phase lag B - A = {stats[1]['onsets'][0] - stats[0]['onsets'][0]:.2f} ms "
                  f"(expected {float(self.period/2/ms):.2f} ms)")

        # ----------------------------------------------------------------------
        # PLOTS
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 8))

        # 1: Membrane potential
        plt.subplot(3, 1, 1)
        plt.plot(mon_state.t/ms, mon_state.v[0]/mV, color='blue', label='N0 (phase A)')
        plt.plot(mon_state.t/ms, mon_state.v[1]/mV, color='red', linestyle='--', label='N1 (phase B)')
        plt.title(f'CPG membrane potential ({self.n_spikes} spikes / burst, '
                  f'period {float(self.period/ms):.0f} ms)')
        plt.ylabel('v (mV)')
        plt.legend(loc='upper right')

        # 2: Spike counter and plateau state
        plt.subplot(3, 1, 2)
        plt.plot(mon_state.t/ms, mon_state.ca[0], color='blue', label='Counter N0')
        plt.plot(mon_state.t/ms, mon_state.ca[1], color='red', linestyle='--', label='Counter N1')
        plt.step(mon_state.t/ms, mon_state.plateau[0] * self.n_spikes, color='blue', alpha=0.3,
                 label='Plateau N0 (scaled)')
        plt.step(mon_state.t/ms, mon_state.plateau[1] * self.n_spikes, color='red', alpha=0.3,
                 label='Plateau N1 (scaled)')
        plt.axhline(self.n_spikes, color='grey', linestyle=':', label='Burst end threshold')
        plt.title('Spike counter (ca) terminating each burst')
        plt.ylabel('Spikes in burst')
        plt.legend(loc='upper right', fontsize=8)

        # 3: Raster + burst_end events
        plt.subplot(3, 1, 3)
        plt.plot(mon_spikes.t/ms, mon_spikes.i, 'k|', markersize=12, label='Spikes')
        plt.plot(mon_end.t/ms, mon_end.i, 'rx', markersize=8, label="'burst_end' event")
        plt.title('Raster plot')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron')
        plt.yticks([0, 1], ['0 (phase A)', '1 (phase B)'])
        plt.ylim(-0.5, 1.5)
        plt.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to '{filename}'.")
        if show:
            plt.show()
        return stats


# =============================================================================
# LEGS
# =============================================================================
# Motor neuron kinds
RAISE, LOWER, FORWARD, BACKWARD = 0, 1, 2, 3
KIND_NAMES = ['R', 'D', 'F', 'B']
KIND_COLORS = ['green', 'saddlebrown', 'blue', 'red']


class Leg:
    """
    View on the motor neurons of one leg (owns no Brian2 object).

    Block layout inside the shared motor NeuronGroup:
        [R, D, F_0 .. F_{N-1}, B_0 .. B_{N-1}]
    F_k / B_k fire on the (k+1)-th CPG spike of a burst (increasing thresholds).
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

    Motor neuron: leaky integrator, fires at most once per burst.
      - every CPG spike adds w_exc to v;
      - neuron fires when v > v_th (thresholds evenly spaced w.r.t. the potential reached
        at the end of the burst, so F_k / B_k fire on about the (k+1)-th spike; R and D
        on the 1st; m inhibitory spikes recruit exactly N - m neurons);
      - after firing it is blocked (done = True) until the next reset;
      - active reset (v = 0, done = False) one step after every CPG 'burst_end' event.

    CPG wiring, for a leg whose stance is CPG phase p:
      - D (lower) <- CPG p (stance onset)
      - R (raise) <- CPG 1-p (swing onset), delayed by lift_delay (reset delayed too)
      - F and B   <- both CPG neurons (direction selected later by descending inhibition)
    """

    def __init__(self, cpg, n_legs=N_LEGS, tau_leak=TAU_LEAK, lift_delay=LIFT_DELAY,
                 w_exc=1*mV):
        if n_legs % 2 or not 6 <= n_legs <= 200:
            raise ValueError("n_legs must be even and between 6 and 200")
        if lift_delay >= cpg.gap:
            raise ValueError(f"lift_delay ({lift_delay}) must be shorter than the CPG gap ({cpg.gap})")
        self.cpg = cpg
        self.n_legs = n_legs
        self.n_motor = n = cpg.n_spikes
        self.tau_leak = tau_leak
        self.lift_delay = lift_delay
        self.w_exc = w_exc
        block = 2 + 2 * n

        # --- Thresholds calibrated for the leak --------------------------------
        # v just after the j-th spike of a burst: S_j = w * sum_{i<=j} r^i, r = exp(-ISI/tau).
        # v is maximal at the last spike (S_end). Thresholds are evenly spaced below it:
        #   theta_k = (k + 1/2) * u,  u = S_end / N
        # so an inhibition lowering the final potential by m*u recruits exactly N - m neurons,
        # whatever the leak (with a strong leak, F_k may fire one CPG spike early or late).
        r = np.exp(-float(cpg.isi / tau_leak))
        S_end = float(w_exc / volt) * np.sum(r ** np.arange(n))           # in volt
        self.u = S_end / n                                                # in volt
        theta = (np.arange(n) + 0.5) * self.u

        # --- Per-neuron constants -----------------------------------------------
        kind = np.array([RAISE, LOWER] + [FORWARD] * n + [BACKWARD] * n)
        rank = np.array([0, 0] + list(range(n)) * 2)
        v_th = np.concatenate([[theta[0], theta[0]], theta, theta])
        legs_phase = np.array([(i % (n_legs // 2) + i // (n_legs // 2)) % 2 for i in range(n_legs)])

        self.eqs = '''
        dv/dt = -v / tau_leak : volt
        v_th : volt (constant)
        done : boolean           # already fired during this burst
        kind : integer (constant)
        rank : integer (constant)
        leg : integer (constant)
        phase : integer (constant)  # CPG neuron driving this leg's stance
        '''
        self.motors = NeuronGroup(n_legs * block, model=self.eqs,
                                  threshold='v > v_th and not done', reset='done = True',
                                  method='exact', namespace={'tau_leak': tau_leak})
        self.motors.kind = np.tile(kind, n_legs)
        self.motors.rank = np.tile(rank, n_legs)
        self.motors.v_th = np.tile(v_th, n_legs) * volt
        self.motors.leg = np.repeat(np.arange(n_legs), block)
        self.motors.phase = np.repeat(legs_phase, block)

        self.legs = [Leg(self.motors, i, n, n_legs) for i in range(n_legs)]

        # Delay per synapse: R pathway shifted by lift_delay
        def r_delay(syn):
            return np.where(self.motors.kind[syn.j[:]] == RAISE, float(lift_delay / second), 0.0) * second

        # --- CPG excitation -------------------------------------------------------
        self.excitation = Synapses(cpg.neurons, self.motors, on_pre='v_post += w_exc',
                                   namespace={'w_exc': w_exc})
        self.excitation.connect(condition=f'kind_post >= {FORWARD} '
                                          f'or (kind_post == {LOWER} and i == phase_post) '
                                          f'or (kind_post == {RAISE} and i != phase_post)')
        self.excitation.delay = r_delay(self.excitation)

        # --- Active reset on every burst end ----------------------------------
        self.reset = Synapses(cpg.neurons, self.motors, on_pre='v_post = 0*volt; done_post = False',
                              on_event='burst_end')
        self.reset.connect()
        # +1 step: the last recruited neuron can fire in the same step as 'burst_end';
        # its own reset (done = True) would otherwise run after this one and block it.
        self.reset.delay = r_delay(self.reset) + defaultclock.dt

    @property
    def objects(self):
        """Brian2 objects to add to a Network (CPG not included)."""
        return [self.motors, self.excitation, self.reset]

    def inhibition_weight(self, inh_delay):
        """
        Weight of one inhibitory spike delivered inh_delay after a burst_end, so that each
        spike lowers the potential at the end of the next burst by exactly one threshold
        step u (it decays by exp(-(t_pre + t_last)/tau) until then).
        """
        t_pre = self.cpg.gap - defaultclock.dt - inh_delay   # inhibition -> first spike
        t_last = (self.n_motor - 1) * self.cpg.isi           # first -> last spike
        return self.u * np.exp(float((t_pre + t_last) / self.tau_leak)) * volt

    # --------------------------------------------------------------------------
    # TEST
    # --------------------------------------------------------------------------
    def test_and_plot(self, inhibition=None, n_cycles=4, show_legs=None, inh_delay=1*ms,
                      filename="legs_test.png", show=True):
        """
        Isolated simulation CPG + legs with a fake, constant test inhibition.

        inhibition: {leg_index: (m_F, m_B)}, number of inhibitory spikes delivered to the
                    F / B groups before every burst. Default: m_F spread from 0 to N across
                    legs, m_B = N - m_F.
        Expected per burst: N - m_F forward and N - m_B backward neurons recruited (lowest
        thresholds first), D fires once in stance, R once in swing (after lift_delay).
        """
        n, cpg = self.n_motor, self.cpg
        if inhibition is None:
            m_f = [int(np.round(i * n / (self.n_legs - 1))) for i in range(self.n_legs)]
            inhibition = {i: (m_f[i], n - m_f[i]) for i in range(self.n_legs)}
        if show_legs is None:
            show_legs = [0, self.n_legs // 2]
        if not defaultclock.dt < inh_delay < cpg.gap - defaultclock.dt:
            raise ValueError("inh_delay must fall inside the CPG gap, after the motor reset")

        # Per-neuron number of test inhibitory spikes
        m_test = np.zeros(len(self.motors))
        for li, (m_f, m_b) in inhibition.items():
            leg = self.legs[li]
            m_test[leg.start + 2:leg.start + 2 + n] = m_f
            m_test[leg.start + 2 + n:leg.start + 2 + 2 * n] = m_b

        # Temporary test inhibition, delivered in the gap after the reset
        w_inh = self.inhibition_weight(inh_delay)
        test_inh = Synapses(cpg.neurons, self.motors, model='m : 1',
                            on_pre='v_post -= m * w_inh', on_event='burst_end',
                            delay=inh_delay, namespace={'w_inh': w_inh})
        test_inh.connect(condition=f'kind_post >= {FORWARD}')
        test_inh.m = m_test[test_inh.j[:]]
        # First burst has no preceding burst_end: preload the same (decayed) inhibition
        t_pre = cpg.gap - defaultclock.dt - inh_delay
        self.motors.v = -m_test * float(w_inh / volt) * np.exp(-float(t_pre / self.tau_leak)) * volt
        self.motors.done = False

        mon_cpg = SpikeMonitor(cpg.neurons)
        mon_motor = SpikeMonitor(self.motors)
        v_leg = self.legs[show_legs[-1]]
        mon_v = StateMonitor(self.motors, 'v', record=[v_leg.start + 2])  # all F neurons share v

        # small margin so the last burst (and its delayed R spike) is complete
        duration = n_cycles * cpg.period + self.lift_delay + 2 * defaultclock.dt
        net = Network(*cpg.objects, *self.objects, test_inh, mon_cpg, mon_motor, mon_v)
        print(f"Running CPG + {self.n_legs} legs: N={n}, tau_leak={self.tau_leak}, "
              f"lift_delay={self.lift_delay}, w_inh={w_inh}")
        net.run(duration)

        # ----------------------------------------------------------------------
        # CHECKS: count recruited neurons per burst / leg / kind
        # ----------------------------------------------------------------------
        stats = cpg.burst_stats(mon_cpg)
        onsets = sorted((t, p) for p in (0, 1) for t in stats[p]['onsets'])
        # keep only bursts fully inside the simulation (+ lift delay)
        half_ms = float(cpg.period / 2 / ms)
        onsets = [(t, p) for t, p in onsets
                  if t + half_ms <= float(duration / ms) - float(self.lift_delay / ms)]
        on_t = np.array([t for t, _ in onsets])
        on_p = np.array([p for _, p in onsets])

        # plain numpy copies (Brian2 views do not support fancy indexing with repeats)
        kind_arr = np.asarray(self.motors.kind[:])
        leg_arr = np.asarray(self.motors.leg[:])
        rank_arr = np.asarray(self.motors.rank[:])

        idx = np.asarray(mon_motor.i[:])
        t_eff = np.asarray(mon_motor.t / ms) - np.where(kind_arr[idx] == RAISE,
                                                        float(self.lift_delay / ms), 0.0)
        burst = np.searchsorted(on_t, t_eff + 1e-6, side='right') - 1
        keep = burst >= 0
        keep[keep] &= t_eff[keep] < on_t[burst[keep]] + half_ms   # inside its own half-cycle
        idx, burst = idx[keep], burst[keep]
        kinds, legs_i, ranks = kind_arr[idx], leg_arr[idx], rank_arr[idx]

        counts = np.zeros((len(on_t), self.n_legs, 4), dtype=int)
        np.add.at(counts, (burst, legs_i, kinds), 1)
        max_rank = -np.ones((len(on_t), self.n_legs, 4), dtype=int)
        np.maximum.at(max_rank, (burst, legs_i, kinds), ranks)

        n_errors = 0
        for leg in self.legs:
            m_f, m_b = inhibition.get(leg.index, (0, 0))
            lines = []
            for kind, expected_fb in ((FORWARD, n - m_f), (BACKWARD, n - m_b)):
                got = counts[:, leg.index, kind]
                ordered = np.all((max_rank[:, leg.index, kind] + 1) == got)
                bad = int(np.sum(got != expected_fb)) + (0 if ordered else 1)
                n_errors += bad
                lines.append(f"{KIND_NAMES[kind]}: {got.tolist()} (exp {expected_fb})"
                             + ("" if ordered else " NOT lowest-first"))
            exp_d = (on_p == leg.phase).astype(int)
            exp_r = 1 - exp_d
            rd_ok = (np.array_equal(counts[:, leg.index, LOWER], exp_d)
                     and np.array_equal(counts[:, leg.index, RAISE], exp_r))
            n_errors += 0 if rd_ok else 1
            print(f"{leg.name} (stance {'AB'[leg.phase]}, m_F={m_f}, m_B={m_b}) | "
                  + " | ".join(lines) + f" | R/D {'OK' if rd_ok else 'WRONG'}")
        print("All checks passed." if n_errors == 0 else f"{n_errors} check(s) FAILED.")

        # ----------------------------------------------------------------------
        # PLOTS
        # ----------------------------------------------------------------------
        n_rows = 2 + len(show_legs)
        fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.2 + 2.6 * n_rows), sharex=True)

        def shade(ax):
            for t, p in zip(on_t, on_p):
                ax.axvspan(t, t + float(cpg.burst_duration / ms),
                           color='blue' if p == 0 else 'red', alpha=0.06)

        # 1: CPG
        ax = axes[0]
        shade(ax)
        ax.plot(mon_cpg.t / ms, mon_cpg.i, 'k|', markersize=10)
        ax.set_yticks([0, 1], ['A', 'B'])
        ax.set_ylim(-0.5, 1.5)
        ax.set_title('CPG bursts (shaded: blue = phase A, red = phase B)')

        # 2..: motor raster of selected legs
        all_t, all_i = np.asarray(mon_motor.t / ms), np.asarray(mon_motor.i[:])
        for ax, li in zip(axes[1:-1], show_legs):
            leg = self.legs[li]
            shade(ax)
            sel = (all_i >= leg.start) & (all_i < leg.start + 2 + 2 * n)
            local = all_i[sel] - leg.start
            local_kind = kind_arr[all_i[sel]]
            for kind in (RAISE, LOWER, FORWARD, BACKWARD):
                k = local_kind == kind
                ax.plot(all_t[sel][k], local[k], '|', color=KIND_COLORS[kind], markersize=5,
                        label=KIND_NAMES[kind])
            m_f, m_b = inhibition.get(li, (0, 0))
            ax.set_yticks([0, 1, 2 + n // 2, 2 + n + n // 2], ['R', 'D', 'F', 'B'])
            ax.set_ylim(-1, 2 + 2 * n)
            ax.set_title(f"Leg {leg.name} (stance {'AB'[leg.phase]}), test inhibition "
                         f"m_F={m_f}, m_B={m_b}")
            ax.legend(loc='upper right', fontsize=7, ncol=4)

        # last: shared membrane potential of the F group of the last shown leg
        ax = axes[-1]
        shade(ax)
        ax.plot(mon_v.t / ms, mon_v.v[0] / mV, color='blue')
        for th in self.motors.v_th[v_leg.start + 2:v_leg.start + 2 + n] / mV:
            ax.axhline(th, color='grey', linestyle=':', linewidth=0.7)
        ax.set_title(f"F group membrane potential, leg {v_leg.name} (dotted: F_k thresholds)")
        ax.set_ylabel('v (mV)')
        ax.set_xlabel('Time (ms)')

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        print(f"Plot saved to '{filename}'.")
        if show:
            plt.show()
        return counts


# --- MODULE TEST ---
if __name__ == '__main__':
    cpg = CPG(n_spikes=N_MOTOR)
    # cpg.test_and_plot(duration=4*CYCLE_PERIOD)
    legs = LegSet(cpg, n_legs=N_LEGS)
    legs.test_and_plot()
