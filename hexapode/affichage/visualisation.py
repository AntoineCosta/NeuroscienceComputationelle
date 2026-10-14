"""Toutes les figures et animations (les classes de simulation ne contiennent pas de code matplotlib)."""
import os

import numpy as np
import matplotlib.pyplot as plt
from brian2 import ms, mV

from hexapode.reseau.legs import RAISE, LOWER, FORWARD, BACKWARD, KIND_NAMES

KIND_COLORS = ['green', 'saddlebrown', 'blue', 'red']
# dossier graphs/ à la racine du projet, quel que soit le dossier de lancement
GRAPH_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "graphs"))


def _out_path(filename):
    """Chemin d'un fichier de sortie dans GRAPH_DIR (sous-dossiers compris, créés si besoin)."""
    path = os.path.join(GRAPH_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _finish(filename, show):
    plt.tight_layout()
    if filename:
        filename = _out_path(filename)
        plt.savefig(filename, dpi=200)
        print(f"Figure enregistrée dans '{filename}'.")
    if show:
        plt.show()


def _shade_half_cycles(ax, on_t, on_p, half_ms):
    """Fond coloré pour chaque demi-cycle du CPG (bleu = phase A, rouge = phase B)."""
    for t, p in zip(on_t, on_p):
        ax.axvspan(t, t + half_ms, color='blue' if p == 0 else 'red', alpha=0.06)


def _plot_leg_raster(ax, legs, leg_index, mon_motor, on_t, on_p, end_ms, title):
    """Raster des motoneurones d'une patte (R, D, F, B), fond coloré par demi-cycle du CPG."""
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
    """Temps (ms) et phases des potentiels d'action du CPG, triés : une entrée par demi-cycle."""
    cpg_t, cpg_i = np.asarray(mon_cpg.t / ms), np.asarray(mon_cpg.i[:])
    order = np.argsort(cpg_t)
    return cpg_t[order], cpg_i[order]


def plot_cpg(cpg, mon_state, mon_spikes, filename="cpg_test.png", show=True):
    """Potentiels de membrane et raster du CPG."""
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(mon_state.t / ms, mon_state.v[0] / mV, color='blue', label='A (pacemaker)')
    plt.plot(mon_state.t / ms, mon_state.v[1] / mV, color='red', linestyle='--', label='B (relais)')
    plt.title(f'Potentiel de membrane du CPG (période {float(cpg.period / ms):.0f} ms)')
    plt.ylabel('v (mV)')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(mon_spikes.t / ms, mon_spikes.i, 'k|', markersize=15)
    plt.yticks([0, 1], ['A', 'B'])
    plt.ylim(-0.5, 1.5)
    plt.title("Raster du CPG : un potentiel d'action par demi-cycle")
    plt.xlabel('Temps (ms)')
    _finish(filename, show)


def plot_legs(legs, mon_cpg, mon_motor, mon_v, show_legs, inhibition, duration,
              filename="legs_test.png", show=True):
    """
    Raster du CPG, raster moteur des pattes de show_legs, et zoom sur les potentiels
    du groupe F (mon_v enregistre le groupe F de la dernière patte de show_legs).
    inhibition : {indice_patte: (m_F, m_B)}, utilisé seulement dans les titres.
    """
    n, cpg = legs.n_motor, legs.cpg
    half_ms = float(cpg.period / 2 / ms)
    end_ms = float(duration / ms)
    on_t, on_p = _cpg_onsets(mon_cpg)

    n_rows = 2 + len(show_legs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.2 + 2.4 * n_rows))

    # 1 : CPG
    ax = axes[0]
    _shade_half_cycles(ax, on_t, on_p, half_ms)
    ax.plot(on_t, on_p, 'k|', markersize=12)
    ax.set_yticks([0, 1], ['A', 'B'])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0, end_ms)
    ax.set_title("Potentiels d'action du CPG (demi-cycles colorés : bleu = phase A, rouge = phase B)")

    # 2.. : raster moteur des pattes choisies
    for ax, li in zip(axes[1:-1], show_legs):
        leg = legs.legs[li]
        m_f, m_b = inhibition.get(li, (0, 0))
        _plot_leg_raster(ax, legs, li, mon_motor, on_t, on_p, end_ms,
                         f"Patte {leg.name} (appui {'AB'[leg.phase]}), inhibition de test "
                         f"m_F={m_f}, m_B={m_b}")

    # dernier : zoom sur les potentiels du groupe F autour du deuxième potentiel d'action du CPG
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
    ax.axvline(t0, color='k', linestyle='--', linewidth=0.8, label="Potentiel d'action du CPG (+ inhibition)")
    ax.axvline(t0 + float(legs.exc_delay / ms), color='orange', linestyle='--', linewidth=0.8,
               label='Excitation du CPG')
    ax.set_title(f"Potentiels du groupe F, patte {v_leg.name} (pointillés : seuils, couleur : rang)")
    ax.set_ylabel('v (mV)')
    ax.set_xlabel('Temps (ms)')
    ax.legend(loc='upper right', fontsize=7)
    _finish(filename, show)


def plot_control(layer, mon_cpg, mon_level, mon_motor, command_fn, counts, duration,
                 show_legs=None, filename="control_test.png", show=True):
    """
    Commandes au cours du temps, niveaux des intermédiaires, amplitude d'appui de chaque
    patte par demi-cycle (neurones F - B qui déchargent en appui : > 0 avant, < 0 arrière)
    et raster moteur des pattes de show_legs.
    counts : [demi-cycle, patte, type], sortie de LegSet.recruitment().
    """
    from hexapode.reseau.intermediates import INTER_NAMES
    legs = layer.legs
    n = legs.n_motor
    end_ms = float(duration / ms)
    on_t, on_p = _cpg_onsets(mon_cpg)
    if show_legs is None:
        show_legs = [0, legs.n_legs // 2]

    n_rows = 3 + len(show_legs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.0 + 2.3 * n_rows), sharex=True)

    # 1 : commandes (lues à chaque demi-cycle)
    ax = axes[0]
    cmds = np.array([command_fn(t * ms) for t in on_t])
    ax.step(on_t, cmds[:, 0], where='post', color='black', label='vitesse')
    ax.step(on_t, cmds[:, 1], where='post', color='purple', linestyle='--', label='virage (<0 gauche)')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Commandes du cerveau (lues à chaque demi-cycle du CPG)')
    ax.legend(loc='upper right', fontsize=7)

    # 2 : niveaux des intermédiaires (0 = pas de frein, 1 = frein total)
    ax = axes[1]
    styles = ['-', '--', '-', '--', '-', '--']
    colors = ['tab:blue', 'tab:blue', 'tab:red', 'tab:red', 'tab:green', 'tab:green']
    for k, name in enumerate(INTER_NAMES):
        ax.plot(mon_level.t / ms, mon_level.level[k], linestyle=styles[k],
                color=colors[k], label=name)
    ax.set_ylim(-0.1, 1.2)
    ax.set_title('Niveaux des intermédiaires (0 = pas de frein, 1 = frein total ; tirets = côté droit)')
    ax.legend(loc='upper right', fontsize=7, ncol=3)

    # 3 : amplitude d'appui par patte (seulement les demi-cycles où la patte est en appui)
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
    ax.set_title("Amplitude d'appui par patte (fraction de N, > 0 avant)")
    ax.legend(loc='upper right', fontsize=7, ncol=6)

    # 4.. : rasters moteurs
    for ax, li in zip(axes[3:], show_legs):
        leg = legs.legs[li]
        _plot_leg_raster(ax, legs, li, mon_motor, on_t, on_p, end_ms,
                         f"Patte {leg.name} (appui {'AB'[leg.phase]})")
    axes[-1].set_xlabel('Temps (ms)')
    axes[-1].set_xlim(0, end_ms)
    _finish(filename, show)


def _command_changes(times, command_fn):
    """Indices des temps où la commande change (le premier inclus)."""
    cmds = [command_fn(t * ms) for t in times]
    return [k for k in range(len(times)) if k == 0 or cmds[k] != cmds[k - 1]], cmds


def _top_view_axes(ax, body, poses, margin=None):
    """
    Limites fixes, carrées, à échelle égale, contenant toute la trajectoire (+ marge,
    par défaut une longueur de corps), avec une grille.
    """
    margin = body.length if margin is None else margin
    centre = (poses[:, :2].min(axis=0) + poses[:, :2].max(axis=0)) / 2
    half = np.max(np.ptp(poses[:, :2], axis=0)) / 2 + margin
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')


def _draw_world(ax, world):
    """Murs de l'arène et obstacles (fixes)."""
    from matplotlib.patches import Rectangle
    h = world.size / 2
    ax.add_patch(Rectangle((-h, -h), world.size, world.size, fill=False, edgecolor='black',
                           linewidth=2, zorder=1))
    for x0, y0, x1, y1 in world.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='dimgrey',
                               edgecolor='black', zorder=1))


def _body_color(world, body, pose):
    """Orange, ou rouge quand le corps chevauche un obstacle ou sort de l'arène."""
    if world is not None and world.overlaps(np.vstack([body.outline(pose), pose[:2]])):
        return 'red'
    return 'tab:orange'


def _sensor_artists(ax, sensors, animated=False):
    """Un trait par ligne de visée."""
    return [ax.plot([], [], linewidth=1.5, zorder=2, animated=animated)[0]
            for _ in range(len(sensors.ranges))]


def _update_sensors(lines, sensors, pose):
    """Lignes de visée à une pose, colorées selon l'activité (vert = loin, rouge = proche)."""
    origins, ends, dist = sensors.rays(pose)
    for line, o, e, a in zip(lines, origins, ends, sensors.near(dist)):
        line.set_data([o[0], e[0]], [o[1], e[1]])
        line.set_color(plt.cm.RdYlGn_r(a))


def plot_trajectory(body, times, poses, command_fn, filename="body_trajectory.png", show=True,
                    world=None, title=None):
    """
    Vue de dessus de toute la simulation : flèche de cap à chaque demi-cycle (couleur = temps),
    repères numérotés aux changements de commande (listés dans l'encadré), contour du corps à la fin.
    times, poses : sortie de Body.trajectory() ou Robot.trajectory() ; world : dessiné si fourni.
    """
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(8, 8))
    # avec un monde : marge plus large, pour voir les obstacles proches du trajet
    _top_view_axes(ax, body, poses, None if world is None else 2 * body.length)
    if world is not None:
        _draw_world(ax, world)
    ax.plot(poses[:, 0], poses[:, 1], color='grey', linewidth=0.8, zorder=2)
    arrows = ax.quiver(poses[:, 0], poses[:, 1], np.cos(poses[:, 2]), np.sin(poses[:, 2]),
                       times / 1000, cmap='viridis', scale=25, width=0.004, zorder=3)
    fig.colorbar(arrows, ax=ax, label='Temps (s)', shrink=0.8)
    ax.add_patch(Polygon(body.outline(poses[-1]), closed=True, facecolor='none',
                         edgecolor='black', linestyle='--', zorder=4))

    # Changements de commande numérotés ; ceux au même endroit partagent une étiquette
    changes, cmds = _command_changes(times[:-1], command_fn)
    lines, labels = [], {}
    for n, k in enumerate(changes, start=1):
        speed, steer = cmds[k]
        lines.append(f"{n}: {times[k] / 1000:4.1f} s  vitesse {speed:+.1f}  virage {steer:+.1f}")
        labels.setdefault(tuple(np.round(poses[k, :2], 4)), []).append(str(n))
    for (x, y), numbers in labels.items():
        ax.plot(x, y, 'ko', markersize=4, zorder=5)
        ax.annotate(",".join(numbers), (x, y), textcoords='offset points', xytext=(6, -10),
                    fontsize=9, fontweight='bold', zorder=5)
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va='top', fontsize=8,
            family='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    ax.set_title(title or "Vue de dessus de l'hexapode (flèches : cap à chaque demi-cycle ; tirets : pose finale)")
    _finish(filename, show)


LEGS_LEGEND = 'trépied A bleu, B rouge | pied plein : au sol, creux : levé'


def _leg_artists(ax, body, animated=False):
    """Traits (hanche -> pied) et marqueurs de pied des pattes dessinées, colorés par trépied."""
    colors = ['blue' if leg.phase == 0 else 'red' for leg in body.shown]
    lines = [ax.plot([], [], color=c, linewidth=2, zorder=4, animated=animated)[0] for c in colors]
    marks = [ax.plot([], [], 'o', color=c, markersize=7, zorder=5, animated=animated)[0]
             for c in colors]
    return lines, marks, colors


def _update_legs(artists, feet):
    """Place les tracés des pattes selon (hanches, pieds, levés), sortie de Body.feet()."""
    for line, mark, color, hip, foot, up in zip(*artists, *feet):
        line.set_data([hip[0], foot[0]], [hip[1], foot[1]])
        mark.set_data([foot[0]], [foot[1]])
        mark.set_markerfacecolor('white' if up else color)


def animate_body(body, times, poses, feet_states, command_fn, fps=20,
                 filename="body_animation.gif", show=True, world=None, sensors=None):
    """
    Animation en temps réel vue de dessus : flèche du corps, pattes, traînée, temps et
    commande courante ; avec un monde : obstacles, lignes de visée, corps rouge s'il
    chevauche un obstacle.
    Les poses sont interpolées linéairement dans chaque demi-cycle (la poussée d'appui
    est étalée sur le demi-cycle). Enregistrée en GIF si filename est donné.
    times, poses, feet_states : sortie de Body.trajectory() ou Robot.trajectory().
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon
    frame_t = np.arange(times[0], times[-1] + 1e-9, 1000 / fps)
    frames = np.column_stack([np.interp(frame_t, times, poses[:, k]) for k in range(3)])
    # demi-cycle de chaque image (pour les pieds)
    cycle = np.clip(np.searchsorted(times, frame_t, side='right') - 1, 0, len(feet_states) - 1)

    fig, ax = plt.subplots(figsize=(7, 7))
    # avec un monde : marge plus large, pour voir les obstacles proches du trajet
    _top_view_axes(ax, body, poses, None if world is None else 2 * body.length)
    if world is not None:
        _draw_world(ax, world)
    ax.plot(poses[:, 0], poses[:, 1], color='lightgrey', linewidth=0.8, linestyle=':')
    trail, = ax.plot([], [], color='grey', linewidth=1.5, zorder=2)
    rays = _sensor_artists(ax, sensors) if sensors is not None else []
    shape = ax.add_patch(Polygon(body.outline(frames[0]), closed=True, facecolor='tab:orange',
                                 edgecolor='black', alpha=0.6, zorder=3))
    legs = _leg_artists(ax, body)
    label = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', family='monospace')
    ax.text(0.02, 0.02, LEGS_LEGEND, transform=ax.transAxes, fontsize=8)
    ax.set_title("Vue de dessus de l'hexapode (virage < 0 = gauche)")

    def update(f):
        trail.set_data(frames[:f + 1, 0], frames[:f + 1, 1])
        shape.set_xy(body.outline(frames[f]))
        shape.set_facecolor(_body_color(world, body, frames[f]))
        if rays:
            _update_sensors(rays, sensors, frames[f])
        c = cycle[f]
        _update_legs(legs, body.feet(frames[f], frame_t[f] - times[c], feet_states[c]))
        speed, steer = command_fn(frame_t[f] * ms)
        label.set_text(f"t = {frame_t[f] / 1000:5.2f} s\nvitesse {speed:+.2f}\nvirage {steer:+.2f}")
        return trail, shape, label

    anim = FuncAnimation(fig, update, frames=len(frame_t), interval=1000 / fps, blit=False)
    if filename:
        filename = _out_path(filename)
        anim.save(filename, writer=PillowWriter(fps=fps), dpi=80)
        print(f"Animation enregistrée dans '{filename}'.")
    if show:
        plt.show()
    return anim


def animate_demo(body, times, poses, feet_states, command_fn, on_p, counts, title, filename,
                 fps=15, world=None, sensors=None, leg_map=False, verdict=None):
    """
    GIF de démonstration « commande vs vue de dessus » :
      à gauche, la commande du cerveau (vitesse, virage), l'amplitude d'appui réellement
      produite par les motoneurones de chaque côté (F - B en appui, fraction de N) et,
      si leg_map, la carte de toutes les pattes ; un curseur suit le temps ;
      à droite, la vue de dessus (corps, pattes, capteurs, obstacles).
    times, poses, feet_states : sortie de Robot.trajectory() ;
    on_p, counts : phases et comptes par demi-cycle (LegSet.recruitment), un par poussée.
    leg_map : carte [patte, demi-cycle] de (F - B) / N pour TOUTES les pattes, triées par
              côté puis par phase du CPG (preuve que chaque patte reçoit la bonne commande).
    verdict : texte de vérification affiché sous la carte.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon
    legs = body.legs
    n = legs.n_motor
    t_s = times / 1000
    frame_t = np.arange(times[0], times[-1] + 1e-9, 1000 / fps)
    frames = np.column_stack([np.interp(frame_t, times, poses[:, k]) for k in range(3)])
    cycle = np.clip(np.searchsorted(times, frame_t, side='right') - 1, 0, len(feet_states) - 1)

    n_rows = 3 if leg_map else 2
    fig = plt.figure(figsize=(13, 6.5))
    grid = fig.add_gridspec(n_rows, 2, width_ratios=[1.15, 1], hspace=0.55, wspace=0.18,
                            left=0.11 if leg_map else 0.06, right=0.97, top=0.88,
                            bottom=0.12 if verdict else 0.08)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    left = [fig.add_subplot(grid[r, 0]) for r in range(n_rows)]
    for ax in left[1:]:
        ax.sharex(left[0])
    cursors = [ax.axvline(t_s[0], color='black', linewidth=1.2, alpha=0.7) for ax in left]

    # 1 : commande du cerveau
    ax = left[0]
    cmds = np.array([command_fn(t * ms) for t in times])
    ax.step(t_s, cmds[:, 0], where='post', color='black', linewidth=2, label='vitesse')
    ax.step(t_s, cmds[:, 1], where='post', color='purple', linestyle='--', linewidth=2,
            label='virage (< 0 gauche)')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title('Commande du cerveau (entrée)', fontsize=10, loc='left')
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=8, ncol=2, frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.5)

    # 2 : amplitude d'appui produite, moyenne des pattes en appui de chaque côté
    ax = left[1]
    amp = np.array([body.side_push(c, p) for c, p in zip(counts, on_p)]) / body.stride
    amp = np.vstack([amp, amp[-1:]])
    ax.step(t_s, amp[:, 0], where='post', color='tab:blue', linewidth=2, label='côté gauche')
    ax.step(t_s, amp[:, 1], where='post', color='tab:orange', linestyle='--', linewidth=2,
            label='côté droit')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Sortie motrice : (F - B) / N en appui", fontsize=10, loc='left')
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=8, ncol=2, frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.5)

    # 3 : carte de toutes les pattes (optionnelle)
    if leg_map:
        ax = left[2]
        groups = [(side, phase) for side in (0, 1) for phase in (0, 1)]
        order = sorted(legs.legs, key=lambda leg: (leg.side, leg.phase, leg.rank))
        stroke = (counts[:, :, FORWARD] - counts[:, :, BACKWARD]) / n          # [demi-cycle, patte]
        in_stance = np.array([[p == leg.phase for leg in order] for p in on_p])
        img = np.where(in_stance, stroke[:, [leg.index for leg in order]], np.nan).T
        cmap = plt.get_cmap('coolwarm_r').copy()
        cmap.set_bad('gainsboro')                                       # en vol
        ax.imshow(img, aspect='auto', cmap=cmap, vmin=-1, vmax=1, interpolation='nearest',
                  extent=(t_s[0], t_s[-1], len(order), 0))
        sizes = [sum((leg.side, leg.phase) == g for leg in order) for g in groups]
        edges = np.cumsum([0] + sizes)
        ax.set_yticks((edges[:-1] + edges[1:]) / 2,
                      [f'{"GD"[s]}, phase {"AB"[p]} ({k})' for (s, p), k in zip(groups, sizes)],
                      fontsize=7)
        for k, e in enumerate(edges[1:-1], start=1):
            ax.axhline(e, color='black', linewidth=0.8 if k == 2 else 0.4)
        ax.set_title(f"Les {legs.n_legs} pattes : amplitude d'appui de chacune "
                     f"(bleu : avant, rouge : arrière, gris : en vol)", fontsize=10)
        if verdict:
            fig.text(0.31, 0.012, verdict, ha='center', fontsize=9, fontweight='bold',
                     color='darkgreen' if 'aucune erreur' in verdict else 'red')
    left[-1].set_xlabel('Temps (s)')
    left[0].set_xlim(t_s[0], t_s[-1])
    for ax in left[:-1]:
        ax.tick_params(labelbottom=False)

    # Vue de dessus
    ax = fig.add_subplot(grid[:, 1])
    _top_view_axes(ax, body, poses, None if world is None else 2 * body.length)
    if world is not None:
        _draw_world(ax, world)
    ax.plot(poses[:, 0], poses[:, 1], color='lightgrey', linewidth=0.8, linestyle=':')
    trail, = ax.plot([], [], color='grey', linewidth=1.5, zorder=2)
    rays = _sensor_artists(ax, sensors) if sensors is not None else []
    shape = ax.add_patch(Polygon(body.outline(frames[0]), closed=True, facecolor='tab:orange',
                                 edgecolor='black', alpha=0.6, zorder=3))
    leg_artists = _leg_artists(ax, body)
    label = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', family='monospace',
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='grey'))
    shown = '' if legs.n_legs == 6 else f' (3 pattes dessinées par côté sur {legs.n_legs // 2})'
    ax.text(0.02, 0.02, LEGS_LEGEND + shown, transform=ax.transAxes, fontsize=7)
    ax.set_title('Vue de dessus', fontsize=10)

    def update(f):
        for cursor in cursors:
            cursor.set_xdata([frame_t[f] / 1000] * 2)
        trail.set_data(frames[:f + 1, 0], frames[:f + 1, 1])
        shape.set_xy(body.outline(frames[f]))
        shape.set_facecolor(_body_color(world, body, frames[f]))
        if rays:
            _update_sensors(rays, sensors, frames[f])
        c = cycle[f]
        _update_legs(leg_artists, body.feet(frames[f], frame_t[f] - times[c], feet_states[c]))
        speed, steer = command_fn(frame_t[f] * ms)
        mode = 'arrêt' if speed == 0 and steer == 0 else 'rotation' if speed == 0 else \
            'avance' if speed > 0 else 'recule'
        label.set_text(f"t = {frame_t[f] / 1000:5.2f} s\n{mode}\nvitesse {speed:+.2f}\n"
                       f"virage  {steer:+.2f}\ncap {np.degrees(frames[f, 2]) - 90:+7.1f} deg")

    anim = FuncAnimation(fig, update, frames=len(frame_t), interval=1000 / fps, blit=False)
    filename = _out_path(filename)
    anim.save(filename, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"Animation enregistrée dans '{filename}'.")


class LiveTopView:
    """
    Fenêtre en direct vue de dessus (pilotage manuel). Les touches de la fenêtre sont
    transmises à on_press(key) / on_release(key). La vue garde une échelle fixe et se
    recentre sur le robot quand il approche du bord.
    Pattes : les 6 pattes dessinées du corps, colorées par trépied (bleu = A, rouge = B) ;
    pied plein = au sol, pied creux = levé.
    Monde (optionnel) : obstacles, lignes de visée colorées selon l'activité (vert = loin,
    rouge = proche), corps rouge s'il chevauche un obstacle.
    Blitting : le fond des axes est dessiné une seule fois (et au redimensionnement /
    recentrage), seuls les éléments mobiles sont redessinés à chaque image.
    """

    def __init__(self, body, on_press, on_release, half_size=0.75,
                 title='Flèches : piloter | Échap ou fermer : quitter', world=None, sensors=None):
        # libère les flèches des raccourcis par défaut de matplotlib (précédent / suivant)
        for name, key in (('keymap.back', 'left'), ('keymap.forward', 'right')):
            if key in plt.rcParams[name]:
                plt.rcParams[name].remove(key)
        from matplotlib.patches import Polygon
        self.body, self.half_size = body, half_size
        self.world, self.sensors = world, sensors
        self.centre = body.pose[:2].copy()
        self.trail_xy = []

        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self._set_limits()
        self.ax.set_aspect('equal')
        self.ax.grid(True, linewidth=0.5, alpha=0.5)
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_title(title)
        if world is not None:
            _draw_world(self.ax, world)
        self.trail, = self.ax.plot([], [], color='grey', linewidth=1.5, zorder=2,
                                   animated=True)
        self.rays = _sensor_artists(self.ax, sensors, animated=True) if sensors is not None else []
        self.shape = self.ax.add_patch(Polygon(body.outline(body.pose), closed=True,
                                               facecolor='tab:orange', edgecolor='black',
                                               alpha=0.6, zorder=3, animated=True))
        self.legs = _leg_artists(self.ax, body, animated=True)
        self.label = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, va='top',
                                  family='monospace', animated=True)
        self.ax.text(0.02, 0.02, LEGS_LEGEND, transform=self.ax.transAxes, fontsize=8)
        self.background = None
        canvas = self.fig.canvas
        canvas.mpl_connect('draw_event', self._on_full_draw)
        canvas.mpl_connect('key_press_event', lambda e: on_press(e.key))
        canvas.mpl_connect('key_release_event', lambda e: on_release(e.key))
        plt.show(block=False)

    def _on_full_draw(self, event):
        """Après chaque redessin complet (première image, redimensionnement, recentrage) : mémorise le fond."""
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_moving()

    def _draw_moving(self):
        for artist in (self.trail, *self.rays, self.shape, *self.legs[0], *self.legs[1],
                       self.label):
            self.fig.draw_artist(artist)

    def _set_limits(self):
        self.ax.set_xlim(self.centre[0] - self.half_size, self.centre[0] + self.half_size)
        self.ax.set_ylim(self.centre[1] - self.half_size, self.centre[1] + self.half_size)

    @property
    def is_open(self):
        return plt.fignum_exists(self.fig.number)

    def close(self):
        plt.close(self.fig)

    def draw(self, pose, t_ms, speed, steer, feet):
        """
        Met à jour et redessine le corps, les pattes, les capteurs, la traînée et le texte.
        feet : (hanches, pieds, levés), sortie de Body.feet().
        """
        if np.max(np.abs(pose[:2] - self.centre)) > 0.8 * self.half_size:
            self.centre = pose[:2].copy()
            self._set_limits()
            self.background = None
        self.trail_xy.append(pose[:2])
        xy = np.array(self.trail_xy)
        self.trail.set_data(xy[:, 0], xy[:, 1])
        self.shape.set_xy(self.body.outline(pose))
        self.shape.set_facecolor(_body_color(self.world, self.body, pose))
        if self.rays:
            _update_sensors(self.rays, self.sensors, pose)
        _update_legs(self.legs, feet)
        mode = 'arrêt' if speed == 0 and steer == 0 else 'rotation' if speed == 0 else 'marche'
        self.label.set_text(f"t = {t_ms / 1000:6.2f} s\n{mode}\nvitesse {speed:+.2f}\n"
                            f"virage {steer:+.2f}")
        canvas = self.fig.canvas
        if self.background is None:
            canvas.draw()                  # redessin complet, appelle _on_full_draw
        else:
            canvas.restore_region(self.background)
            self._draw_moving()
        canvas.blit(self.fig.bbox)

    def wait(self, seconds):
        """Traite les événements de la fenêtre (touches, redimensionnement) pendant la durée donnée."""
        self.fig.canvas.start_event_loop(max(seconds, 0.001))
