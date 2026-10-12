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


def _plot_leg_raster(ax, legs, leg_index, mon_motor, on_t, on_p, end_ms, title):
    """Raster of one leg's motor neurons (R, D, F, B), shaded by CPG half-cycle."""
    n = legs.n_motor
    leg = legs.legs[leg_index]
    kind_arr = np.asarray(legs.motors.kind[:])
    all_t, all_i = np.asarray(mon_motor.t / ms), np.asarray(mon_motor.i[:])
    _shade_half_cycles(ax, on_t, on_p, float(legs.cpg.period / 2 / ms))
    sel = (all_i >= leg.start) & (all_i < leg.start + 2 + 2 * n)
    local = all_i[sel] - leg.start
    local_kind = kind_arr[all_i[sel]]
    for kind in (RAISE, LOWER, FORWARD, BACKWARD):
        k = local_kind == kind
        ax.plot(all_t[sel][k], local[k], '|', color=KIND_COLORS[kind], markersize=4,
                label=KIND_NAMES[kind])
    ax.set_yticks([0, 1, 2 + n // 2, 2 + n + n // 2], ['R', 'D', 'F', 'B'])
    ax.set_ylim(-1, 2 + 2 * n)
    ax.set_xlim(0, end_ms)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=7, ncol=4)


def _cpg_onsets(mon_cpg):
    """CPG spike times (ms) and phases, sorted: one entry per half-cycle."""
    cpg_t, cpg_i = np.asarray(mon_cpg.t / ms), np.asarray(mon_cpg.i[:])
    order = np.argsort(cpg_t)
    return cpg_t[order], cpg_i[order]


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
    on_t, on_p = _cpg_onsets(mon_cpg)

    n_rows = 2 + len(show_legs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.2 + 2.4 * n_rows))

    # 1: CPG
    ax = axes[0]
    _shade_half_cycles(ax, on_t, on_p, half_ms)
    ax.plot(on_t, on_p, 'k|', markersize=12)
    ax.set_yticks([0, 1], ['A', 'B'])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0, end_ms)
    ax.set_title('CPG spikes (shaded half-cycles: blue = phase A, red = phase B)')

    # 2..: motor raster of selected legs
    for ax, li in zip(axes[1:-1], show_legs):
        leg = legs.legs[li]
        m_f, m_b = inhibition.get(li, (0, 0))
        _plot_leg_raster(ax, legs, li, mon_motor, on_t, on_p, end_ms,
                         f"Leg {leg.name} (stance {'AB'[leg.phase]}), test inhibition "
                         f"m_F={m_f}, m_B={m_b}")

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


def plot_control(layer, mon_cpg, mon_level, mon_motor, command_fn, counts, duration,
                 show_legs=None, filename="control_test.png", show=True):
    """
    Commands over time, intermediate levels, stance stroke of every leg per half-cycle
    (F - B neurons fired while in stance: > 0 forward, < 0 backward) and the motor
    raster of the legs in show_legs.
    counts: [half-cycle, leg, kind] from LegSet.recruitment().
    """
    from intermediates import INTER_NAMES
    legs = layer.legs
    n = legs.n_motor
    end_ms = float(duration / ms)
    on_t, on_p = _cpg_onsets(mon_cpg)
    if show_legs is None:
        show_legs = [0, legs.n_legs // 2]

    n_rows = 3 + len(show_legs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.0 + 2.3 * n_rows), sharex=True)

    # 1: commands (latched at each half-cycle)
    ax = axes[0]
    cmds = np.array([command_fn(t * ms) for t in on_t])
    ax.step(on_t, cmds[:, 0], where='post', color='black', label='speed')
    ax.step(on_t, cmds[:, 1], where='post', color='purple', linestyle='--', label='steer (<0 left)')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Brain commands (latched at each CPG half-cycle)')
    ax.legend(loc='upper right', fontsize=7)

    # 2: intermediate levels (0 = no brake, 1 = full brake)
    ax = axes[1]
    styles = ['-', '--', '-', '--', '-', '--']
    colors = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']
    for k, name in enumerate(INTER_NAMES):
        ax.plot(mon_level.t / ms, mon_level.level[k], linestyle=styles[k],
                color=colors[k], label=name)
    ax.set_ylim(-0.1, 1.2)
    ax.set_title('Intermediate levels (0 = no brake, 1 = full brake; dashed = right side)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)

    # 3: stance stroke per leg (only half-cycles where the leg is in stance)
    ax = axes[2]
    plot_legs_idx = range(legs.n_legs) if legs.n_legs <= 12 else show_legs
    for li in plot_legs_idx:
        leg = legs.legs[li]
        stance = on_p == leg.phase
        stroke = counts[:, li, FORWARD] - counts[:, li, BACKWARD]
        ax.plot(on_t[stance], stroke[stance] / n, marker='o', markersize=3,
                linestyle='-' if leg.side == 0 else '--', label=leg.name)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Stance stroke per leg (fraction of N, > 0 forward)')
    ax.legend(loc='upper right', fontsize=7, ncol=6)

    # 4..: motor rasters
    for ax, li in zip(axes[3:], show_legs):
        leg = legs.legs[li]
        _plot_leg_raster(ax, legs, li, mon_motor, on_t, on_p, end_ms,
                         f"Leg {leg.name} (stance {'AB'[leg.phase]})")
    axes[-1].set_xlabel('Time (ms)')
    axes[-1].set_xlim(0, end_ms)
    _finish(filename, show)


def _command_changes(times, command_fn):
    """Indices of the times where the command changes (first one included)."""
    cmds = [command_fn(t * ms) for t in times]
    return [k for k in range(len(times)) if k == 0 or cmds[k] != cmds[k - 1]], cmds


def _top_view_axes(ax, body, poses):
    """Fixed, square, equal-ratio limits containing the whole trajectory, with a grid."""
    centre = (poses[:, :2].min(axis=0) + poses[:, :2].max(axis=0)) / 2
    half = np.max(np.ptp(poses[:, :2], axis=0)) / 2 + body.length
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')


def plot_trajectory(body, times, poses, command_fn, filename="body_trajectory.png", show=True):
    """
    Top view of the whole run: heading arrow at every half-cycle (colour = time),
    numbered markers at the command changes (listed in the box), body outline at the end.
    times, poses: from Body.trajectory().
    """
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(8, 8))
    _top_view_axes(ax, body, poses)
    ax.plot(poses[:, 0], poses[:, 1], color='grey', linewidth=0.8, zorder=2)
    arrows = ax.quiver(poses[:, 0], poses[:, 1], np.cos(poses[:, 2]), np.sin(poses[:, 2]),
                       times / 1000, cmap='viridis', scale=25, width=0.004, zorder=3)
    fig.colorbar(arrows, ax=ax, label='Time (s)', shrink=0.8)
    ax.add_patch(Polygon(body.outline(poses[-1]), closed=True, facecolor='none',
                         edgecolor='black', linestyle='--', zorder=4))

    # Numbered command changes; changes at the same place share one label
    changes, cmds = _command_changes(times[:-1], command_fn)
    lines, labels = [], {}
    for n, k in enumerate(changes, start=1):
        speed, steer = cmds[k]
        lines.append(f"{n}: {times[k] / 1000:4.1f} s  speed {speed:+.1f}  steer {steer:+.1f}")
        labels.setdefault(tuple(np.round(poses[k, :2], 4)), []).append(str(n))
    for (x, y), numbers in labels.items():
        ax.plot(x, y, 'ko', markersize=4, zorder=5)
        ax.annotate(",".join(numbers), (x, y), textcoords='offset points', xytext=(6, -10),
                    fontsize=9, fontweight='bold', zorder=5)
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va='top', fontsize=8,
            family='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    ax.set_title('Hexapod top view (arrows: heading every half-cycle; dashed: final pose)')
    _finish(filename, show)


def animate_body(body, times, poses, command_fn, fps=20, filename="body_animation.gif",
                 show=True):
    """
    Real-time top-view animation: body arrow, trail, time and current command.
    Poses are interpolated linearly inside each half-cycle (the stance push is
    spread over the half-cycle). Saved as a GIF when filename is given.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon
    frame_t = np.arange(times[0], times[-1] + 1e-9, 1000 / fps)
    frames = np.column_stack([np.interp(frame_t, times, poses[:, k]) for k in range(3)])

    fig, ax = plt.subplots(figsize=(7, 7))
    _top_view_axes(ax, body, poses)
    ax.plot(poses[:, 0], poses[:, 1], color='lightgrey', linewidth=0.8, linestyle=':')
    trail, = ax.plot([], [], color='tab:blue', linewidth=1.5, zorder=3)
    shape = ax.add_patch(Polygon(body.outline(frames[0]), closed=True, facecolor='tab:orange',
                                 edgecolor='black', alpha=0.6, zorder=2))
    label = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', family='monospace')
    ax.set_title('Hexapod top view (steer < 0 = left)')

    def update(f):
        trail.set_data(frames[:f + 1, 0], frames[:f + 1, 1])
        shape.set_xy(body.outline(frames[f]))
        speed, steer = command_fn(frame_t[f] * ms)
        label.set_text(f"t = {frame_t[f] / 1000:5.2f} s\nspeed {speed:+.2f}\nsteer {steer:+.2f}")
        return trail, shape, label

    anim = FuncAnimation(fig, update, frames=len(frame_t), interval=1000 / fps, blit=False)
    if filename:
        anim.save(filename, writer=PillowWriter(fps=fps), dpi=80)
        print(f"Animation saved to '{filename}'.")
    if show:
        plt.show()
    return anim


class LiveTopView:
    """
    Live top-view window (manual control). Key events of the window are forwarded to
    on_press(key) / on_release(key). The view keeps a fixed scale and re-centres on
    the robot when it gets close to the edge.
    Blitting: the axes background is rendered once (and on resize / re-centring),
    only the moving artists are redrawn at each frame.
    """

    def __init__(self, body, on_press, on_release, half_size=1.0,
                 title='Arrows: drive | Esc or close: quit'):
        # free the arrow keys from matplotlib's default shortcuts (back / forward)
        for name, key in (('keymap.back', 'left'), ('keymap.forward', 'right')):
            if key in plt.rcParams[name]:
                plt.rcParams[name].remove(key)
        from matplotlib.patches import Polygon
        self.body, self.half_size = body, half_size
        self.centre = np.zeros(2)
        self.trail_xy = []

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self._set_limits()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linewidth=0.5, alpha=0.5)
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_title(title)
        self.trail, = self.ax.plot([], [], color='tab:blue', linewidth=1.5, zorder=3,
                                   animated=True)
        self.shape = self.ax.add_patch(Polygon(body.outline(body.pose), closed=True,
                                               facecolor='tab:orange', edgecolor='black',
                                               alpha=0.6, zorder=2, animated=True))
        self.label = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, va='top',
                                  family='monospace', animated=True)
        self.background = None
        canvas = self.fig.canvas
        canvas.mpl_connect('draw_event', self._on_full_draw)
        canvas.mpl_connect('key_press_event', lambda e: on_press(e.key))
        canvas.mpl_connect('key_release_event', lambda e: on_release(e.key))
        plt.show(block=False)

    def _on_full_draw(self, event):
        """After any full redraw (first frame, resize, re-centring): store the background."""
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_moving()

    def _draw_moving(self):
        for artist in (self.trail, self.shape, self.label):
            self.fig.draw_artist(artist)

    def _set_limits(self):
        self.ax.set_xlim(self.centre[0] - self.half_size, self.centre[0] + self.half_size)
        self.ax.set_ylim(self.centre[1] - self.half_size, self.centre[1] + self.half_size)

    @property
    def is_open(self):
        return plt.fignum_exists(self.fig.number)

    def close(self):
        plt.close(self.fig)

    def draw(self, pose, t_ms, speed, steer):
        """Update and redraw the body, trail and text."""
        if np.max(np.abs(pose[:2] - self.centre)) > 0.8 * self.half_size:
            self.centre = pose[:2].copy()
            self._set_limits()
            self.background = None
        self.trail_xy.append(pose[:2])
        xy = np.array(self.trail_xy)
        self.trail.set_data(xy[:, 0], xy[:, 1])
        self.shape.set_xy(self.body.outline(pose))
        mode = 'stop' if speed == 0 and steer == 0 else 'spin' if speed == 0 else 'walk'
        self.label.set_text(f"t = {t_ms / 1000:6.2f} s\n{mode}\nspeed {speed:+.2f}\n"
                            f"steer {steer:+.2f}")
        canvas = self.fig.canvas
        if self.background is None:
            canvas.draw()                  # full redraw, calls _on_full_draw
        else:
            canvas.restore_region(self.background)
            self._draw_moving()
        canvas.blit(self.fig.bbox)

    def wait(self, seconds):
        """Process window events (keys, resize) for the given time."""
        self.fig.canvas.start_event_loop(max(seconds, 0.001))
