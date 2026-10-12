"""
Hexapod controlled by a spiking neural network (Brian2) - entry point and test scenarios.

Modules:
  config.py         global parameters (legs, neurons per group, rhythm, delays, dt)
  CPG.py            rhythm generator (one spike per half-cycle)
  legs.py           leg motor neurons (R, D, F, B groups)
  visualisation.py  all plots

TODO (project roadmap)
- Descending neurons (Fwd-L/R, Rev-L/R, L, R): one inhibitory spike per half-cycle,
  triggered by the CPG spike, weight = LegSet.inhibition_weight(m) to silence m neurons.
  Tonically active interneurons: no command = everything inhibited = stand still.
  Inhibition onto R must be delayed by LIFT_DELAY (like R's excitation).
- Leg position (x, z) computed from motor spikes.
- Manual input: scripted timeline, then keyboard / controller.
- Sensor neurons hijacking descending commands for obstacle avoidance.
"""
import numpy as np
from brian2 import Network, SpikeMonitor, StateMonitor, Synapses, defaultclock, ms

import config
from CPG import CPG
from legs import LegSet, RAISE, LOWER, FORWARD, BACKWARD, KIND_NAMES
import visualisation as vis


def test_cpg(n_cycles=4, filename="cpg_test.png", show=True):
    """Isolated CPG: checks period and anti-phase, then plots."""
    cpg = CPG(period=config.CYCLE_PERIOD)
    mon_state = StateMonitor(cpg.neurons, 'v', record=True)
    mon_spikes = SpikeMonitor(cpg.neurons)
    net = Network(*cpg.objects, mon_state, mon_spikes)
    net.run(n_cycles * cpg.period)

    trains = mon_spikes.spike_trains()
    t_a, t_b = np.asarray(trains[0] / ms), np.asarray(trains[1] / ms)
    print(f"Phase A spikes (ms): {t_a.tolist()}")
    print(f"Phase B spikes (ms): {t_b.tolist()}")
    print(f"Periods A: {np.diff(t_a).tolist()}, B: {np.diff(t_b).tolist()} "
          f"(expected {float(cpg.period / ms)})")
    print(f"Lag B - A: {(t_b - t_a[:len(t_b)]).tolist()} (expected {float(cpg.period / 2 / ms)})")

    vis.plot_cpg(cpg, mon_state, mon_spikes, filename=filename, show=show)
    return t_a, t_b


def test_legs(n_legs=config.N_LEGS, n_motor=config.N_MOTOR, tau_m=config.TAU_M,
              inhibition=None, n_cycles=4, show_legs=None, filename="legs_test.png", show=True):
    """
    CPG + legs with a fake test inhibition standing in for the descending neurons:
    one inhibitory spike per CPG spike, delivered at the CPG spike time (exc_delay
    before the EPSP), with weight legs.inhibition_weight(m).

    inhibition: {leg_index: (m_F, m_B)}, applied in both phases. Default: m_F spread
                from 0 to N across legs, m_B = N - m_F.
    Expected per half-cycle: N - m_F forward and N - m_B backward neurons fire
    (lowest thresholds), D fires in stance, R in swing (after lift_delay).
    """
    cpg = CPG(period=config.CYCLE_PERIOD)
    legs = LegSet(cpg, n_legs=n_legs, n_motor=n_motor, tau_m=tau_m)
    n = legs.n_motor
    if inhibition is None:
        m_f = [int(np.round(i * n / (n_legs - 1))) for i in range(n_legs)]
        inhibition = {i: (m_f[i], n - m_f[i]) for i in range(n_legs)}
    if show_legs is None:
        show_legs = [0, n_legs // 2]

    # Temporary test inhibition (F and B groups only)
    m_test = np.zeros(len(legs.motors))
    for li, (m_f, m_b) in inhibition.items():
        s = legs.legs[li].start
        m_test[s + 2:s + 2 + n] = m_f
        m_test[s + 2 + n:s + 2 + 2 * n] = m_b
    test_inh = Synapses(cpg.neurons, legs.motors, model='w : volt', on_pre='v_post -= w')
    test_inh.connect(condition=f'kind_post >= {FORWARD}')
    test_inh.w = legs.inhibition_weight(m_test[np.asarray(test_inh.j[:])])

    mon_cpg = SpikeMonitor(cpg.neurons)
    mon_motor = SpikeMonitor(legs.motors)
    v_start = legs.legs[show_legs[-1]].start + 2
    mon_v = StateMonitor(legs.motors, 'v', record=np.arange(v_start, v_start + n))

    # margin so the last half-cycle (and its delayed R spike) is complete
    duration = n_cycles * cpg.period + legs.exc_delay + legs.lift_delay + 2 * defaultclock.dt
    net = Network(*cpg.objects, *legs.objects, test_inh, mon_cpg, mon_motor, mon_v)
    print(f"Running CPG + {n_legs} legs: N={n}, tau_m={legs.tau_m}, "
          f"exc_delay={legs.exc_delay}, lift_delay={legs.lift_delay}")
    net.run(duration)

    # --- Checks ---------------------------------------------------------------
    on_t, on_p, counts, max_rank = legs.recruitment(mon_cpg, mon_motor)
    n_errors = 0
    for leg in legs.legs:
        m_f, m_b = inhibition.get(leg.index, (0, 0))
        parts = []
        for kind, expected in ((FORWARD, n - m_f), (BACKWARD, n - m_b)):
            got = counts[:, leg.index, kind]
            ordered = np.array_equal(max_rank[:, leg.index, kind] + 1, got)
            n_errors += int(np.sum(got != expected)) + (0 if ordered else 1)
            parts.append(f"{KIND_NAMES[kind]}: {got.tolist()} (exp {expected})"
                         + ("" if ordered else " NOT lowest-first"))
        exp_d = (on_p == leg.phase).astype(int)
        rd_ok = (np.array_equal(counts[:, leg.index, LOWER], exp_d)
                 and np.array_equal(counts[:, leg.index, RAISE], 1 - exp_d))
        n_errors += 0 if rd_ok else 1
        print(f"{leg.name} (stance {'AB'[leg.phase]}, m_F={m_f}, m_B={m_b}) | "
              + " | ".join(parts) + f" | R/D {'OK' if rd_ok else 'WRONG'}")
    print("All checks passed." if n_errors == 0 else f"{n_errors} check(s) FAILED.")

    vis.plot_legs(legs, mon_cpg, mon_motor, mon_v, show_legs, inhibition, duration,
                  filename=filename, show=show)
    return counts


if __name__ == '__main__':
    # test_cpg()
    test_legs()
