"""
Figures du rapport / de l'affiche, enregistrées dans graphs/rapport/ (PNG 300 dpi + PDF vectoriel).

  fig1_recrutement   potentiel du groupe F sous inhibition + courbe de recrutement N - m
  fig2_chronologie   raster de ~15 ms autour d'un potentiel d'action du CPG (appui et vol)
  fig3_commande      commandes -> niveaux des intermédiaires -> amplitude d'appui
  fig4_reflexe       approche d'un mur : distance, activités du capteur avant, freins
  fig5_couloir       trajectoire dans un couloir (validation comportementale)

Usage (depuis la racine du projet) : python -m hexapode.affichage.figures_rapport
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle
from brian2 import (EventMonitor, Network, SpikeMonitor, StateMonitor, Synapses, mV, ms)

from hexapode import config
from hexapode.reseau.CPG import CPG
from hexapode.reseau.legs import LegSet, RAISE, LOWER, FORWARD, BACKWARD
from hexapode.reseau.intermediates import IntermediateLayer, I_FWD_L, I_REV_L, INTER_NAMES
from hexapode.simulation.brain import Brain
from hexapode.simulation.robot import Robot
from hexapode.simulation.world import World
from hexapode.reseau.sensors import FRONT
from hexapode.simulation.scenarios import DEFAULT_TIMELINE, timeline_command
from hexapode.affichage.visualisation import GRAPH_DIR

OUT_DIR = os.path.join(GRAPH_DIR, "rapport")

# Palette (3 premières teintes de la palette de référence, validées daltonisme) + encres
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#b5b4ad', '#e5e4e0'
SEQ_BLUE = LinearSegmentedColormap.from_list('bleu', ['#86b6ef', '#2a78d6', '#0d366b'])

plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13, 'legend.fontsize': 11,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'lines.linewidth': 2, 'legend.frameon': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"Figure enregistrée : {os.path.join(OUT_DIR, name)}.png (+ .pdf)")


# --- Figure 1 : recrutement ordonné ---------------------------------------------------

def fig1_recrutement(m_shown=4, step=0.25):
    """
    Un seul potentiel d'action du CPG (A, t = 0) sur des pattes recevant chacune une
    inhibition de test m différente (de 0 à N par pas de `step`), comme dans test_legs().
    (a) potentiel commun du groupe F pour m = m_shown, avec les seuils recrutés ou non ;
    (b) nombre de neurones F recrutés en fonction de m.
    """
    n = config.N_MOTOR
    m_values = np.arange(0, n + 1e-9, step)
    n_legs = len(m_values) + len(m_values) % 2           # nombre de pattes pair
    m_leg = np.append(m_values, [n] * (n_legs - len(m_values)))

    cpg = CPG(period=config.CYCLE_PERIOD)
    legs = LegSet(cpg, n_legs=n_legs, n_motor=n)
    m_test = np.zeros(len(legs.motors))
    for leg in legs.legs:
        m_test[leg.start + 2:leg.start + 2 + n] = m_leg[leg.index]
    test_inh = Synapses(cpg.neurons, legs.motors, model='w : volt', on_pre='v_post -= w')
    test_inh.connect(condition=f'kind_post == {FORWARD}')
    test_inh.w = legs.inhibition_weight(m_test[np.asarray(test_inh.j[:])])

    # le neurone F de rang le plus haut ne décharge jamais (m >= 1) : son potentiel est
    # celui, commun, de tout le groupe
    shown = legs.legs[int(np.argmin(np.abs(m_leg - m_shown)))]
    mon_v = StateMonitor(legs.motors, 'v', record=[shown.start + 2 + n - 1])
    mon_motor = SpikeMonitor(legs.motors)
    Network(*cpg.objects, *legs.objects, test_inh, mon_v, mon_motor).run(20 * ms)

    idx = np.asarray(mon_motor.i[:])
    is_f = np.asarray(legs.motors.kind[:])[idx] == FORWARD
    n_fired = np.bincount(np.asarray(legs.motors.leg[:])[idx[is_f]], minlength=n_legs)

    fig, (ax_v, ax_r) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1.25, 1]})

    # (a) potentiel du groupe F
    t = np.concatenate([[-1.0], np.asarray(mon_v.t / ms)])
    v = np.concatenate([[0.0], np.asarray(mon_v.v[0] / mV)])
    theta = (np.arange(n) + 0.5) * float(legs.u / mV)
    peak = n - m_shown
    for k, th in enumerate(theta):
        rec = th < peak
        ax_v.axhline(th, color=BLUE if rec else MUTED, linewidth=1.6 if rec else 1.0,
                     linestyle='-' if rec else ':', zorder=1)
    ax_v.plot(t, v, color=INK, zorder=3)
    ax_v.axvline(0, color=INK2, linestyle='--', linewidth=1)
    ax_v.axvline(float(legs.exc_delay / ms), color=INK2, linestyle='--', linewidth=1)
    ax_v.annotate(f"inhibition : −m·u\n(m = {m_shown})", xy=(0.5, -m_shown + 0.3), xytext=(2.6, -m_shown + 0.2),
                  arrowprops=dict(arrowstyle='->', color=INK2), fontsize=12, va='center')
    ax_v.annotate(f"excitation du CPG : +N·u\n→ v = {peak:g} mV", xy=(2.1, peak), xytext=(3.2, peak + 2.2),
                  arrowprops=dict(arrowstyle='->', color=INK2), fontsize=12)
    ax_v.text(7.9, theta[0], f"{int(np.sum(theta < peak))} seuils franchis\n(recrutés)",
              color=BLUE, ha='right', va='bottom', fontsize=12, fontweight='bold')
    ax_v.text(7.9, theta[-1] + 0.15, f"{int(np.sum(theta >= peak))} seuils non atteints",
              color=INK2, ha='right', va='bottom', fontsize=12)
    ax_v.set_xlim(-1, 8)
    ax_v.set_ylim(-m_shown - 1.5, n + 1.5)
    ax_v.set_xlabel("Temps après le potentiel d'action du CPG (ms)")
    ax_v.set_ylabel("Potentiel du groupe F (mV)")
    ax_v.set_title("(a) Inhibition puis excitation", loc='left')
    ax_v.grid(False)

    # (b) courbe de recrutement
    m_fine = np.linspace(0, n, 200)
    ax_r.plot(m_fine, n - m_fine, color=MUTED, linestyle='--', linewidth=1.5, label="N − m (idéal)")
    ax_r.step(m_leg[:len(m_values)], n_fired[:len(m_values)], where='mid', color=BLUE,
              label="simulation")
    ax_r.plot(m_leg[:len(m_values)], n_fired[:len(m_values)], 'o', color=BLUE, markersize=4)
    ax_r.set_xlabel("Inhibition m (paliers de seuil)")
    ax_r.set_ylabel("Neurones F recrutés")
    ax_r.set_xticks(range(0, n + 1, 2))
    ax_r.set_yticks(range(0, n + 1, 2))
    ax_r.set_title(f"(b) Recrutement (N = {n})", loc='left')
    ax_r.legend(loc='upper right')
    fig.tight_layout()
    _save(fig, "fig1_recrutement")


# --- Figure 2 : chronologie d'un demi-cycle ------------------------------------------

def _build_network(command_fn):
    cpg = CPG(period=config.CYCLE_PERIOD)
    legs = LegSet(cpg)
    layer = IntermediateLayer(cpg, legs)
    brain = Brain(layer)
    return cpg, legs, layer, brain.driver(command_fn)


def fig2_chronologie(speed=0.6, t_cpg=1000.0):
    """Raster de -1 à 15 ms autour du potentiel d'action A à t_cpg, marche avant à `speed`."""
    n = config.N_MOTOR
    cpg, legs, layer, driver = _build_network(lambda t: (speed, 0.0))
    mon_cpg = SpikeMonitor(cpg.neurons)
    mon_rel = EventMonitor(layer.neurons, 'release')
    mon_motor = SpikeMonitor(legs.motors)
    Network(*cpg.objects, *legs.objects, *layer.objects, driver,
            mon_cpg, mon_rel, mon_motor).run((t_cpg + 30) * ms)

    def window(times):
        rel = np.asarray(times / ms) - t_cpg
        return rel, (rel > -1) & (rel < 15)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2.6, 2.6]})

    # CPG + intermédiaires
    ax = axes[0]
    t_c, w_c = window(mon_cpg.t)
    t_r, w_r = window(mon_rel.t)
    ax.plot(t_c[w_c], np.ones(w_c.sum()), '|', color=INK, markersize=22, markeredgewidth=2.5)
    ax.plot(t_r[w_r], np.zeros(w_r.sum()), '|', color=INK2, markersize=22, markeredgewidth=2.5)
    ax.set_yticks([0, 1], ["Intermédiaires\n(libération)", "CPG (A)"])
    ax.set_ylim(-0.7, 1.75)
    ax.grid(False)

    # une patte en appui (L1, phase A) et une en vol (L2, phase B)
    kind_arr = np.asarray(legs.motors.kind[:])
    rank_arr = np.asarray(legs.motors.rank[:])
    rows = {RAISE: lambda r: 0, LOWER: lambda r: 1,
            FORWARD: lambda r: 3 + r, BACKWARD: lambda r: 4 + n + r}
    t_m, w_m = window(mon_motor.t)
    i_m = np.asarray(mon_motor.i[:])
    for ax, leg, title in ((axes[1], legs.legs[0], "Patte L1 — en appui (poussée)"),
                           (axes[2], legs.legs[1], "Patte L2 — en vol (retour)")):
        sel = w_m & (i_m >= leg.start) & (i_m < leg.start + 2 + 2 * n)
        for kind, color, marker, label in ((FORWARD, BLUE, '|', 'F (avant)'),
                                           (BACKWARD, ORANGE, '|', 'B (arrière)'),
                                           (LOWER, INK, 'v', 'D (poser)'),
                                           (RAISE, INK, '^', 'R (lever)')):
            k = sel & (kind_arr[i_m] == kind)
            y = [rows[kind](r) for r in rank_arr[i_m[k]]]
            ax.plot(t_m[k], y, marker, color=color, markersize=12 if marker == '|' else 9,
                    markeredgewidth=2.5, linestyle='none', label=label)
        for kind, y_mid in ((FORWARD, 3 + (n - 1) / 2), (BACKWARD, 4 + n + (n - 1) / 2)):
            fired = int(np.sum(sel & (kind_arr[i_m] == kind)))
            if fired:
                ax.text(3.0, y_mid, f"{fired} / {n} déchargent", va='center', fontsize=12, color=INK2)
        for y0, y1 in ((3, 3 + n - 1), (4 + n, 4 + 2 * n - 1)):   # rangées des groupes F et B
            ax.axhspan(y0 - 0.5, y1 + 0.5, color=GRID, alpha=0.35, zorder=0, linewidth=0)
        ax.set_yticks([0, 1, 3 + (n - 1) / 2, 4 + n + (n - 1) / 2],
                      ['R', 'D', f'F (×{n})', f'B (×{n})'])
        ax.set_ylim(-1, 4 + 2 * n)
        ax.set_title(title, loc='left')
        ax.grid(False)
    axes[1].legend(loc='upper right', ncol=4)

    t_r_ms = float(legs.exc_delay / ms) + float(legs.lift_delay / ms)
    for x in (0, 0.5, 2, t_r_ms):
        for ax in axes:
            ax.axvline(x, color=INK2, linestyle=':', linewidth=1, zorder=0)
    axes[0].text(0.65, 0, "libération de l'inhibition (+0,5 ms)", fontsize=11, va='center', color=INK2)
    axes[0].text(2.12, 1.6, "excitation du CPG (+2 ms)\n→ F, B, D déchargent", fontsize=11, va='top', color=INK2)
    axes[0].text(t_r_ms + 0.12, 1.6, f"excitation retardée (+{t_r_ms:g} ms)\n→ R (lever)", fontsize=11,
                 va='top', color=INK2)
    axes[-1].set_xlabel("Temps après le potentiel d'action du CPG (ms)")
    axes[-1].set_xlim(-1, 15)
    fig.suptitle(f"Un demi-cycle en marche avant (vitesse {speed:.0%} → {round(speed * n)} neurones sur {n})",
                 x=0.01, ha='left', fontsize=15)
    fig.tight_layout()
    _save(fig, "fig2_chronologie")


# --- Figure 3 : de la commande au recrutement -----------------------------------------

SEGMENTS = ["avant\n100 %", "avant\n50 %", "virage\ngauche", "virage\ndroite", "arrêt",
            "recul\n70 %", "rotation\ngauche", "recul +\nvirage d."]


def fig3_commande(duration=8000 * ms):
    n = config.N_MOTOR
    command_fn = timeline_command(DEFAULT_TIMELINE)
    cpg, legs, layer, driver = _build_network(command_fn)
    mon_cpg = SpikeMonitor(cpg.neurons)
    mon_level = StateMonitor(layer.neurons, 'level', record=True)
    mon_motor = SpikeMonitor(legs.motors)
    run_time = duration + legs.exc_delay + legs.lift_delay + 1 * ms
    Network(*cpg.objects, *legs.objects, *layer.objects, driver,
            mon_cpg, mon_level, mon_motor).run(run_time)
    on_t, on_p, counts, _ = legs.recruitment(mon_cpg, mon_motor)
    t_s = on_t / 1000
    end = float(duration / ms) / 1000

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (a) commandes
    ax = axes[0]
    cmds = np.array([command_fn(t * ms) for t in on_t])
    ax.step(np.append(t_s, end), np.append(cmds[:, 0], cmds[-1, 0]), where='post', color=BLUE, label='vitesse')
    ax.step(np.append(t_s, end), np.append(cmds[:, 1], cmds[-1, 1]), where='post', color=ORANGE,
            linestyle='--', label='virage (< 0 : gauche)')
    for k, text in enumerate(SEGMENTS):
        ax.text(k + 0.5, 1.22, text, ha='center', va='bottom', fontsize=10.5, color=INK2)
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("Commande")
    ax.set_title("(a) Commandes du cerveau", loc='left', pad=48)
    ax.legend(loc='lower left', ncol=2)

    # (b) niveaux des intermédiaires
    ax = axes[1]
    t_lvl = np.asarray(mon_level.t / ms) / 1000
    colors = [BLUE, BLUE, ORANGE, ORANGE, AQUA, AQUA]
    for k, name in enumerate(INTER_NAMES):
        ax.plot(t_lvl, mon_level.level[k], color=colors[k], linestyle='-' if k % 2 == 0 else '--',
                linewidth=2 if k % 2 == 0 else 1.6, label=name[:-2] + (' G' if k % 2 == 0 else ' D'))
    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel("Frein (0 à 1)")
    ax.set_title("(b) Niveaux des intermédiaires (trait plein : gauche, tirets : droite)", loc='left')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # (c) amplitude d'appui de chaque côté
    ax = axes[2]
    for side, style, label in ((0, '-', 'pattes gauches'), (1, '--', 'pattes droites')):
        stroke = []
        for c, p in enumerate(on_p):
            leg = next(l for l in legs.legs if l.side == side and l.phase == p)
            stroke.append((counts[c, leg.index, FORWARD] - counts[c, leg.index, BACKWARD]) / n)
        ax.step(np.append(t_s, end), np.append(stroke, stroke[-1]), where='post', color=INK,
                linestyle=style, linewidth=2 if side == 0 else 1.6, label=label)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylim(-1.15, 1.15)
    ax.set_ylabel("(F − B) / N en appui")
    ax.set_xlabel("Temps (s)")
    ax.set_title("(c) Amplitude d'appui (> 0 : poussée vers l'avant)", loc='left')
    ax.legend(loc='lower left', ncol=2)
    ax.set_xlim(0, end)
    for ax in axes:
        for k in range(1, len(SEGMENTS)):
            ax.axvline(k, color=GRID, linewidth=1.2, zorder=0)
    fig.tight_layout()
    _save(fig, "fig3_commande")


# --- Figure 4 : réflexe de recul ------------------------------------------------------

def fig4_reflexe(duration=8000 * ms):
    world = World([(-0.5, 1.0, 0.5, 1.2)])
    robot = Robot(timeline_command([(0 * ms, 1.0, 0.0)]), world=world)
    mon_s = StateMonitor(robot.sensors.neurons, ['distance', 'near', 'close'], record=[FRONT],
                         dt=config.SENSE_DT)
    mon_i = StateMonitor(robot.layer.neurons, 'level', record=[I_FWD_L, I_REV_L], dt=config.SENSE_DT)
    robot.net.add(mon_s, mon_i)
    robot.run(duration)

    t_s = np.asarray(mon_s.t / ms) / 1000
    t_i = np.asarray(mon_i.t / ms) / 1000
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    ax.axhspan(0, config.BACKOFF_DISTANCE, color=ORANGE, alpha=0.10, linewidth=0)
    ax.axhline(config.FRONT_RANGE, color=INK2, linestyle=':', linewidth=1)
    ax.text(0.05, config.FRONT_RANGE + 0.02, "portée du capteur (0,6 m)", fontsize=11, color=INK2)
    ax.text(0.05, config.BACKOFF_DISTANCE - 0.02, "zone de recul (< 0,2 m)", fontsize=11,
            color=INK2, va='top')
    ax.plot(t_s, mon_s.distance[0], color=BLUE)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Distance (m)")
    ax.set_title("(a) Distance mesurée par le capteur avant", loc='left')

    ax = axes[1]
    ax.plot(t_s, mon_s.near[0], color=BLUE, label='near (obstacle proche)')
    ax.plot(t_s, mon_s.close[0], color=ORANGE, label='close (zone de recul)')
    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel("Activité")
    ax.set_title("(b) Activités du neurone capteur avant", loc='left')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    ax = axes[2]
    ax.plot(t_i, mon_i.level[0], color=BLUE, label='I-Fwd (frein avant)')
    ax.plot(t_i, mon_i.level[1], color=ORANGE, linestyle='--', label='I-Rev (frein arrière)')
    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel("Frein (0 à 1)")
    ax.set_xlabel("Temps (s)")
    ax.set_title("(c) Intermédiaires : close freine l'avance et désinhibe le recul", loc='left')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_xlim(0, float(duration / ms) / 1000)
    fig.suptitle("Pilote : marche avant à 100 % en continu vers un mur", x=0.01, ha='left', fontsize=15)
    fig.tight_layout()
    _save(fig, "fig4_reflexe")


# --- Figure 5 : couloir ---------------------------------------------------------------

def fig5_couloir(duration=10000 * ms, start_x=-0.08):
    world = World([(-0.55, 0.2, -0.25, 2.0), (0.25, 0.2, 0.55, 2.0)])
    robot = Robot(timeline_command([(0 * ms, 1.0, 0.0)]), world=world, start=(start_x, 0.0, np.pi / 2))
    robot.run(duration)
    times, poses, _ = robot.trajectory()

    fig, ax = plt.subplots(figsize=(6.5, 8))
    for x0, y0, x1, y1 in world.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=MUTED, edgecolor=INK2, zorder=1))
    ax.axvline(0, color=INK2, linestyle=':', linewidth=1, zorder=1)
    for k in range(0, len(poses), 8):                      # contour du corps toutes les 2 s
        ax.add_patch(Polygon(robot.body.outline(poses[k]), closed=True, facecolor='none',
                             edgecolor=INK2, linewidth=0.8, alpha=0.6, zorder=2))
    pts = poses[:, :2]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=SEQ_BLUE, linewidth=3, zorder=3)
    lc.set_array(times[:-1] / 1000)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, label="Temps (s)", shrink=0.7)
    ax.plot(*pts[0], 'o', color=INK, zorder=4)
    ax.annotate(f"départ décentré\nde {abs(start_x) * 100:.0f} cm", xy=pts[0], xytext=(-0.55, 0.0),
                fontsize=11, va='center', arrowprops=dict(arrowstyle='->', color=INK2))
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.2, max(1.7, pts[:, 1].max() + 0.3))
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Couloir de 0,5 m : capteurs latéraux seuls", loc='left')
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "fig5_couloir")


if __name__ == '__main__':
    fig1_recrutement()
    fig2_chronologie()
    fig3_commande()
    fig4_reflexe()
    fig5_couloir()
