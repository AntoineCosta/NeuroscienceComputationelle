"""
Hexapod controlled by a spiking neural network (Brian2) - entry point and test scenarios.

Modules:
  config.py         global parameters (legs, neurons per group, rhythm, delays, dt)
  CPG.py            rhythm generator (one spike per half-cycle)
  legs.py           leg motor neurons (R, D, F, B groups)
  intermediates.py  6 graded inhibitory interneurons between the brain and the legs
  brain.py          the 'brain': regular code turning commands into intermediate inputs
  body.py           top-view body pose, moved by the stance legs (differential drive)
  manual.py         live manual control: game controller or arrow keys (python manual.py)
  visualisation.py  all plots

TODO (project roadmap)
- Leg position (x, z) computed from motor spikes (the swing mirrors the stance).
- Faster CPG to cut the command latency (a command waits up to half a period for the
  next CPG spike). Constraints: TAU_M <= period / 20, EXC_DELAY + LIFT_DELAY < period / 2,
  STRIDE per half-cycle (walking speed doubles if the period halves).
- Sensor neurons exciting the intermediates (`sensor` input) to brake or turn away
  from obstacles, whatever the brain commands.
"""
import numpy as np
from brian2 import Network, SpikeMonitor, StateMonitor, Synapses, defaultclock, ms

import config
from CPG import CPG
from legs import LegSet, RAISE, LOWER, FORWARD, BACKWARD, KIND_NAMES
from intermediates import IntermediateLayer
from brain import Brain
from body import Body
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
    CPG + legs with a fake test inhibition standing in for the intermediate layer:
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


DEFAULT_TIMELINE = [  # (start time, speed, steer)
    (0 * ms, 1.0, 0.0),       # forward 100 %
    (1000 * ms, 0.5, 0.0),    # forward 50 %
    (2000 * ms, 1.0, -0.5),   # forward 100 %, steer left 50 % -> left legs 50 %
    (3000 * ms, 0.5, 0.4),    # forward 50 %, steer right 40 % -> right legs 30 %
    (4000 * ms, 0.0, 0.0),    # stop
    (5000 * ms, -0.7, 0.0),   # reverse 70 %
    (6000 * ms, 0.0, -1.0),   # turn left on the spot, 100 %
    (7000 * ms, -1.0, 0.5),   # reverse 100 %, steer right 50 %
]


def timeline_command(timeline):
    """command_fn(t) -> (speed, steer) from a list of (start time, speed, steer)."""
    def command_fn(t):
        speed, steer = 0.0, 0.0
        for t_start, sp, st in timeline:
            if t >= t_start:
                speed, steer = sp, st
        return speed, steer
    return command_fn


def expected_recruitment(n, speed, steer, leg, in_stance):
    """
    Expected number of fired neurons [R, D, F, B] of a leg in one half-cycle.
    Walking: stroke = |speed| * (1 - steer of the leg's side) * N for the push (stance)
    and the return (swing); R/D step as soon as |speed| * N > 1/2.
    Speed 0: turn on the spot, stroke |steer| * N, the side towards the turn in reverse.
    """
    speed = float(np.clip(speed, -1, 1))
    steer = float(np.clip(steer, -1, 1))
    if speed != 0:
        side_steer = max(0.0, -steer) if leg.side == 0 else max(0.0, steer)
        side_speed, drive = speed * (1 - side_steer), abs(speed)
    else:
        side_speed, drive = (steer if leg.side == 0 else -steer), abs(steer)
    stroke = int(np.sum(np.arange(n) + 0.5 < abs(side_speed) * n - 1e-9))
    steps = int(drive * n > 0.5 + 1e-9)
    direction = 0 if side_speed >= 0 else 1
    out = np.zeros(4, dtype=int)
    if in_stance:
        out[LOWER] = steps
        out[FORWARD + direction] = stroke          # push: F forward, B reverse
    else:
        out[RAISE] = steps
        out[FORWARD + 1 - direction] = stroke      # return: B forward, F reverse
    return out


def test_control(timeline=DEFAULT_TIMELINE, n_legs=config.N_LEGS, n_motor=config.N_MOTOR,
                 duration=8000 * ms, show_legs=None, filename="control_test.png", show=True):
    """
    CPG + legs + intermediate layer + brain, driven by a scripted command timeline.
    Checks the number of R, D, F, B neurons fired by every leg in every half-cycle.
    Returns (legs, on_t, on_p, counts) as in LegSet.recruitment().
    """
    cpg = CPG(period=config.CYCLE_PERIOD)
    legs = LegSet(cpg, n_legs=n_legs, n_motor=n_motor)
    layer = IntermediateLayer(cpg, legs)
    brain = Brain(layer)
    command_fn = timeline_command(timeline)

    mon_cpg = SpikeMonitor(cpg.neurons)
    mon_level = StateMonitor(layer.neurons, 'level', record=True)
    mon_motor = SpikeMonitor(legs.motors)
    # margin so the last half-cycle (and its delayed R spike) is complete
    run_time = duration + legs.exc_delay + legs.lift_delay + 2 * defaultclock.dt
    net = Network(*cpg.objects, *legs.objects, *layer.objects, brain.driver(command_fn),
                  mon_cpg, mon_level, mon_motor)
    print(f"Running CPG + {n_legs} legs + intermediate layer for {duration}")
    net.run(run_time)

    # --- Checks ---------------------------------------------------------------
    on_t, on_p, counts, _ = legs.recruitment(mon_cpg, mon_motor)
    n_errors = 0
    for c, (t, p) in enumerate(zip(on_t, on_p)):
        speed, steer = command_fn(t * ms)
        for leg in legs.legs:
            expected = expected_recruitment(n_motor, speed, steer, leg, in_stance=(p == leg.phase))
            got = counts[c, leg.index]
            if not np.array_equal(got, expected):
                n_errors += 1
                print(f"  t={t:.0f} ms {leg.name}: got R,D,F,B={got.tolist()}, "
                      f"expected {expected.tolist()} (speed={speed}, steer={steer})")
    print(f"{len(on_t)} half-cycles x {n_legs} legs checked: "
          + ("all checks passed." if n_errors == 0 else f"{n_errors} FAILED."))

    vis.plot_control(layer, mon_cpg, mon_level, mon_motor, command_fn, counts, run_time,
                     show_legs=show_legs, filename=filename, show=show)
    return legs, on_t, on_p, counts


def test_body(timeline=DEFAULT_TIMELINE, duration=8000 * ms, animate=True, show=True):
    """Full chain (test_control), then the top view of the body moved by its stance legs."""
    legs, on_t, on_p, counts = test_control(timeline, duration=duration, show=False)
    body = Body(legs)
    times, poses = body.trajectory(on_t, on_p, counts)
    command_fn = timeline_command(timeline)

    print("Pose at each command change (x, y in m, heading in deg, 90 = start):")
    for t_start, speed, steer in timeline:
        k = np.searchsorted(times, float(t_start / ms))
        x, y, h = poses[k]
        print(f"  {float(t_start / ms):6.0f} ms  speed {speed:+.1f} steer {steer:+.1f} -> "
              f"x={x:+.3f} y={y:+.3f} heading={np.degrees(h):6.1f}")
    x, y, h = poses[-1]
    print(f"  end {times[-1]:6.0f} ms -> x={x:+.3f} y={y:+.3f} heading={np.degrees(h):6.1f}")

    vis.plot_trajectory(body, times, poses, command_fn, show=show and not animate)
    if animate:
        vis.animate_body(body, times, poses, command_fn, show=show)
    return times, poses


if __name__ == '__main__':
    # test_cpg()
    # test_legs()
    # test_control()
    test_body()
