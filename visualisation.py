"""All plotting functions (simulation classes stay free of matplotlib code)."""
import numpy as np
import matplotlib.pyplot as plt
from brian2 import ms, mV

from legs import RAISE, LOWER, FORWARD, BACKWARD, KIND_NAMES

KIND_COLORS = ['green', 'saddlebrown', 'blue', 'red']


def _finish(filename, show):
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
        print(f"Plot saved to '{filename}'.")
    if show:
        plt.show()


def _shade_half_cycles(ax, on_t, on_p, half_ms):
    """Background shading of each CPG half-cycle (blue = phase A, red = phase B)."""
    for t, p in zip(on_t, on_p):
        ax.axvspan(t, t + half_ms, color='blue' if p == 0 else 'red', alpha=0.06)


def plot_cpg(cpg, mon_state, mon_spikes, filename="cpg_test.png", show=True):
    """CPG membrane potentials and raster."""
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(mon_state.t / ms, mon_state.v[0] / mV, color='blue', label='A (pacemaker)')
    plt.plot(mon_state.t / ms, mon_state.v[1] / mV, color='red', linestyle='--', label='B (relay)')
    plt.title(f'CPG membrane potential (period {float(cpg.period / ms):.0f} ms)')
    plt.ylabel('v (mV)')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(mon_spikes.t / ms, mon_spikes.i, 'k|', markersize=15)
    plt.yticks([0, 1], ['A', 'B'])
    plt.ylim(-0.5, 1.5)
    plt.title('CPG raster: one spike per half-cycle')
    plt.xlabel('Time (ms)')
    _finish(filename, show)


def plot_legs(legs, mon_cpg, mon_motor, mon_v, show_legs, inhibition, duration,
              filename="legs_test.png", show=True):
    """
    CPG raster, motor raster of the legs in show_legs, and a zoom on the F group
    potentials (mon_v records the F group of the last leg in show_legs).
    inhibition: {leg_index: (m_F, m_B)}, only used in the titles.
    """
    n, cpg = legs.n_motor, legs.cpg
    half_ms = float(cpg.period / 2 / ms)
    end_ms = float(duration / ms)
    kind_arr = np.asarray(legs.motors.kind[:])
    cpg_t, cpg_i = np.asarray(mon_cpg.t / ms), np.asarray(mon_cpg.i[:])
    order = np.argsort(cpg_t)
    on_t, on_p = cpg_t[order], cpg_i[order]

    n_rows = 2 + len(show_legs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.2 + 2.4 * n_rows))

    # 1: CPG
    ax = axes[0]
    _shade_half_cycles(ax, on_t, on_p, half_ms)
    ax.plot(cpg_t, cpg_i, 'k|', markersize=12)
    ax.set_yticks([0, 1], ['A', 'B'])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0, end_ms)
    ax.set_title('CPG spikes (shaded half-cycles: blue = phase A, red = phase B)')

    # 2..: motor raster of selected legs
    all_t, all_i = np.asarray(mon_motor.t / ms), np.asarray(mon_motor.i[:])
    for ax, li in zip(axes[1:-1], show_legs):
        leg = legs.legs[li]
        _shade_half_cycles(ax, on_t, on_p, half_ms)
        sel = (all_i >= leg.start) & (all_i < leg.start + 2 + 2 * n)
        local = all_i[sel] - leg.start
        local_kind = kind_arr[all_i[sel]]
        for kind in (RAISE, LOWER, FORWARD, BACKWARD):
            k = local_kind == kind
            ax.plot(all_t[sel][k], local[k], '|', color=KIND_COLORS[kind], markersize=4,
                    label=KIND_NAMES[kind])
        m_f, m_b = inhibition.get(li, (0, 0))
        ax.set_yticks([0, 1, 2 + n // 2, 2 + n + n // 2], ['R', 'D', 'F', 'B'])
        ax.set_ylim(-1, 2 + 2 * n)
        ax.set_xlim(0, end_ms)
        ax.set_title(f"Leg {leg.name} (stance {'AB'[leg.phase]}), test inhibition "
                     f"m_F={m_f}, m_B={m_b}")
        ax.legend(loc='upper right', fontsize=7, ncol=4)

    # last: zoom on the F group potentials around the second CPG spike
    ax = axes[-1]
    v_leg = legs.legs[show_legs[-1]]
    t0 = on_t[1] if len(on_t) > 1 else on_t[0]
    tv = np.asarray(mon_v.t / ms)
    win = (tv >= t0 - 2) & (tv <= t0 + 5 * float(legs.tau_m / ms))
    for k in range(n):
        ax.plot(tv[win], mon_v.v[k][win] / mV, color=plt.cm.viridis(k / max(n - 1, 1)),
                linewidth=1)
    for th in np.asarray(legs.motors.v_th[v_leg.start + 2:v_leg.start + 2 + n] / mV):
        ax.axhline(th, color='grey', linestyle=':', linewidth=0.7)
    ax.axvline(t0, color='k', linestyle='--', linewidth=0.8, label='CPG spike (+ inhibition)')
    ax.axvline(t0 + float(legs.exc_delay / ms), color='orange', linestyle='--', linewidth=0.8,
               label='CPG EPSP')
    ax.set_title(f"F group potentials, leg {v_leg.name} (dotted: thresholds, colour: rank)")
    ax.set_ylabel('v (mV)')
    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right', fontsize=7)
    _finish(filename, show)
