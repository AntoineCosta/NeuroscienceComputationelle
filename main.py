"""
Hexapode piloté par un réseau de neurones à potentiels d'action (Brian2) - point d'entrée et scénarios de test.

Points d'entrée : main.py (ce fichier : démonstrations et tests), manual.py (pilotage en direct).
Le reste du code est dans le package hexapode/ :
  config.py                 paramètres globaux (pattes, neurones par groupe, rythme, délais, dt)
  reseau/        les neurones (Brian2)
    CPG.py                  générateur de rythme (un potentiel d'action par demi-cycle)
    legs.py                 motoneurones des pattes (groupes R, D, F, B)
    intermediates.py        6 interneurones inhibiteurs gradués entre le cerveau et les pattes
    sensors.py              4 capteurs de distance gradués (réflexes sur les intermédiaires)
  simulation/    le code classique autour du réseau
    brain.py                le « cerveau » : traduit les commandes en entrées des intermédiaires
    body.py                 pose du corps vue de dessus, déplacée par les pattes en appui (comme un char)
    world.py                murs de l'arène et obstacles rectangulaires fixes
    robot.py                la boucle fermée : réseau + corps + monde + capteurs
    scenarios.py            scénarios de commandes (timeline -> command_fn)
  affichage/     les figures
    visualisation.py        figures et animations (sorties dans graphs/)
    figures_rapport.py      figures du rapport (graphs/rapport/)

TODO (feuille de route du projet)
- CPG plus rapide pour réduire la latence des commandes (une commande attend jusqu'à une
  demi-période le prochain potentiel d'action du CPG). Contraintes : TAU_M <= période / 20,
  EXC_DELAY + LIFT_DELAY < période / 2, STRIDE par demi-cycle (la vitesse de marche double
  si la période est divisée par deux).
"""
import numpy as np
from brian2 import Network, SpikeMonitor, StateMonitor, Synapses, defaultclock, ms

from hexapode import config
from hexapode.reseau.CPG import CPG
from hexapode.reseau.legs import LegSet, RAISE, LOWER, FORWARD, BACKWARD, KIND_NAMES
from hexapode.reseau.intermediates import IntermediateLayer
from hexapode.simulation.brain import Brain
from hexapode.simulation.body import Body
from hexapode.simulation.robot import Robot
from hexapode.simulation.scenarios import DEFAULT_TIMELINE, timeline_command
from hexapode.simulation.world import World
from hexapode.reseau.sensors import FRONT, REAR
from hexapode.affichage import visualisation as vis


def test_cpg(n_cycles=4, filename="cpg_test.png", show=True):
    """CPG seul : vérifie la période et l'anti-phase, puis trace la figure."""
    cpg = CPG(period=config.CYCLE_PERIOD)
    mon_state = StateMonitor(cpg.neurons, 'v', record=True)
    mon_spikes = SpikeMonitor(cpg.neurons)
    net = Network(*cpg.objects, mon_state, mon_spikes)
    net.run(n_cycles * cpg.period)

    trains = mon_spikes.spike_trains()
    t_a, t_b = np.asarray(trains[0] / ms), np.asarray(trains[1] / ms)
    print(f"Potentiels d'action phase A (ms) : {t_a.tolist()}")
    print(f"Potentiels d'action phase B (ms) : {t_b.tolist()}")
    print(f"Périodes A : {np.diff(t_a).tolist()}, B : {np.diff(t_b).tolist()} "
          f"(attendu {float(cpg.period / ms)})")
    print(f"Décalage B - A : {(t_b - t_a[:len(t_b)]).tolist()} (attendu {float(cpg.period / 2 / ms)})")

    vis.plot_cpg(cpg, mon_state, mon_spikes, filename=filename, show=show)
    return t_a, t_b


def test_legs(n_legs=config.N_LEGS, n_motor=config.N_MOTOR, tau_m=config.TAU_M,
              inhibition=None, n_cycles=4, show_legs=None, filename="legs_test.png", show=True):
    """
    CPG + pattes, avec une inhibition de test qui remplace la couche intermédiaire :
    un potentiel d'action inhibiteur par potentiel d'action du CPG, livré au moment de
    celui-ci (exc_delay avant l'excitation), de poids legs.inhibition_weight(m).

    inhibition : {indice_patte: (m_F, m_B)}, appliquée dans les deux phases. Par défaut :
                 m_F réparti de 0 à N selon les pattes, m_B = N - m_F.
    Attendu à chaque demi-cycle : N - m_F neurones avant et N - m_B neurones arrière
    déchargent (seuils les plus bas d'abord), D décharge en appui, R en balancement
    (après lift_delay).
    """
    cpg = CPG(period=config.CYCLE_PERIOD)
    legs = LegSet(cpg, n_legs=n_legs, n_motor=n_motor, tau_m=tau_m)
    n = legs.n_motor
    if inhibition is None:
        m_f = [int(np.round(i * n / (n_legs - 1))) for i in range(n_legs)]
        inhibition = {i: (m_f[i], n - m_f[i]) for i in range(n_legs)}
    if show_legs is None:
        show_legs = [0, n_legs // 2]

    # Inhibition de test temporaire (groupes F et B seulement)
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

    # marge pour que le dernier demi-cycle (et son potentiel d'action R retardé) soit complet
    duration = n_cycles * cpg.period + legs.exc_delay + legs.lift_delay + 2 * defaultclock.dt
    net = Network(*cpg.objects, *legs.objects, test_inh, mon_cpg, mon_motor, mon_v)
    print(f"Simulation CPG + {n_legs} pattes : N={n}, tau_m={legs.tau_m}, "
          f"exc_delay={legs.exc_delay}, lift_delay={legs.lift_delay}")
    net.run(duration)

    # --- Vérifications ----------------------------------------------------------
    on_t, on_p, counts, max_rank = legs.recruitment(mon_cpg, mon_motor)
    n_errors = 0
    for leg in legs.legs:
        m_f, m_b = inhibition.get(leg.index, (0, 0))
        parts = []
        for kind, expected in ((FORWARD, n - m_f), (BACKWARD, n - m_b)):
            got = counts[:, leg.index, kind]
            ordered = np.array_equal(max_rank[:, leg.index, kind] + 1, got)
            n_errors += int(np.sum(got != expected)) + (0 if ordered else 1)
            parts.append(f"{KIND_NAMES[kind]} : {got.tolist()} (att. {expected})"
                         + ("" if ordered else " seuils bas PAS en premier"))
        exp_d = (on_p == leg.phase).astype(int)
        rd_ok = (np.array_equal(counts[:, leg.index, LOWER], exp_d)
                 and np.array_equal(counts[:, leg.index, RAISE], 1 - exp_d))
        n_errors += 0 if rd_ok else 1
        print(f"{leg.name} (appui {'AB'[leg.phase]}, m_F={m_f}, m_B={m_b}) | "
              + " | ".join(parts) + f" | R/D {'OK' if rd_ok else 'FAUX'}")
    print("Toutes les vérifications sont passées." if n_errors == 0 else f"{n_errors} vérification(s) en ÉCHEC.")

    vis.plot_legs(legs, mon_cpg, mon_motor, mon_v, show_legs, inhibition, duration,
                  filename=filename, show=show)
    return counts


def expected_recruitment(n, speed, steer, leg, in_stance):
    """
    Nombre attendu de neurones qui déchargent [R, D, F, B] pour une patte sur un demi-cycle.
    En marche : amplitude = |vitesse| * (1 - virage du côté de la patte) * N pour la poussée
    (appui) et le retour (balancement) ; R lève la patte dès que |vitesse| * N > 1/2 ;
    D décharge à chaque début d'appui (jamais freiné).
    Vitesse 0 : rotation sur place, amplitude |virage| * N, le côté vers lequel on tourne recule.
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
        out[LOWER] = 1
        out[FORWARD + direction] = stroke          # poussée : F en avant, B en arrière
    else:
        out[RAISE] = steps
        out[FORWARD + 1 - direction] = stroke      # retour : B en avant, F en arrière
    return out


def test_control(timeline=DEFAULT_TIMELINE, n_legs=config.N_LEGS, n_motor=config.N_MOTOR,
                 duration=8000 * ms, show_legs=None, filename="control_test.png", show=True):
    """
    CPG + pattes + couche intermédiaire + cerveau, pilotés par un scénario de commandes.
    Vérifie le nombre de neurones R, D, F, B qui déchargent pour chaque patte à chaque demi-cycle.
    Renvoie (legs, on_t, on_p, counts) comme LegSet.recruitment().
    """
    cpg = CPG(period=config.CYCLE_PERIOD)
    legs = LegSet(cpg, n_legs=n_legs, n_motor=n_motor)
    layer = IntermediateLayer(cpg, legs)
    brain = Brain(layer)
    command_fn = timeline_command(timeline)

    mon_cpg = SpikeMonitor(cpg.neurons)
    mon_level = StateMonitor(layer.neurons, 'level', record=True)
    mon_motor = SpikeMonitor(legs.motors)
    # marge pour que le dernier demi-cycle (et son potentiel d'action R retardé) soit complet
    run_time = duration + legs.exc_delay + legs.lift_delay + 2 * defaultclock.dt
    net = Network(*cpg.objects, *legs.objects, *layer.objects, brain.driver(command_fn),
                  mon_cpg, mon_level, mon_motor)
    print(f"Simulation CPG + {n_legs} pattes + couche intermédiaire pendant {duration}")
    net.run(run_time)

    # --- Vérifications ----------------------------------------------------------
    on_t, on_p, counts, _ = legs.recruitment(mon_cpg, mon_motor)
    n_errors = 0
    for c, (t, p) in enumerate(zip(on_t, on_p)):
        speed, steer = command_fn(t * ms)
        for leg in legs.legs:
            expected = expected_recruitment(n_motor, speed, steer, leg, in_stance=(p == leg.phase))
            got = counts[c, leg.index]
            if not np.array_equal(got, expected):
                n_errors += 1
                print(f"  t={t:.0f} ms {leg.name} : obtenu R,D,F,B={got.tolist()}, "
                      f"attendu {expected.tolist()} (vitesse={speed}, virage={steer})")
    print(f"{len(on_t)} demi-cycles x {n_legs} pattes vérifiés : "
          + ("toutes les vérifications sont passées." if n_errors == 0 else f"{n_errors} en ÉCHEC."))

    vis.plot_control(layer, mon_cpg, mon_level, mon_motor, command_fn, counts, run_time,
                     show_legs=show_legs, filename=filename, show=show)
    return legs, on_t, on_p, counts


def test_body(timeline=DEFAULT_TIMELINE, duration=8000 * ms, animate=True, show=True):
    """Chaîne complète (test_control), puis vue de dessus du corps déplacé par ses pattes en appui."""
    legs, on_t, on_p, counts = test_control(timeline, duration=duration, show=False)
    body = Body(legs)
    times, poses, feet_states = body.trajectory(on_t, on_p, counts)
    command_fn = timeline_command(timeline)

    print("Pose à chaque changement de commande (x, y en m, cap en degrés, 90 = départ) :")
    for t_start, speed, steer in timeline:
        k = np.searchsorted(times, float(t_start / ms))
        x, y, h = poses[k]
        print(f"  {float(t_start / ms):6.0f} ms  vitesse {speed:+.1f} virage {steer:+.1f} -> "
              f"x={x:+.3f} y={y:+.3f} cap={np.degrees(h):6.1f}")
    x, y, h = poses[-1]
    print(f"  fin {times[-1]:6.0f} ms -> x={x:+.3f} y={y:+.3f} cap={np.degrees(h):6.1f}")

    vis.plot_trajectory(body, times, poses, command_fn, show=show and not animate)
    if animate:
        vis.animate_body(body, times, poses, feet_states, command_fn, show=show)
    return times, poses


def _front_gap(robot, pose):
    """Distance (m) mesurée par le capteur avant à une pose."""
    return float(robot.sensors.rays(pose)[2][FRONT])


def _run_scenario(world, timeline, duration, start=(0.0, 0.0, np.pi / 2)):
    robot = Robot(timeline_command(timeline), world=world, start=start)
    robot.run(duration)
    return robot


def test_sensors(animate=True, show=True):
    """
    Réflexes des capteurs, scénarisés (le robot part de (0, 0) orienté vers le haut) :
      1. mur devant, pleine vitesse avant : freine, puis va-et-vient (recul) sans le toucher
      2. mur derrière, pleine vitesse arrière : pareil, va-et-vient vers l'avant
      3. mur dans la zone de recul, rotation sur place : capteurs coupés, la rotation
         est exactement celle obtenue sans obstacle
      4. mur dans la zone de recul, arrêt : capteurs coupés, le robot ne bouge pas
      5. couloir, départ décentré, pleine vitesse avant : le robot y entre sans toucher les murs
         (il ondule autour du centre : les capteurs latéraux ne voient pas le cap)
    Le scénario 1 est aussi tracé (et animé).
    """
    results = []

    def yoyo(robot, world):
        """(mur touché, nombre de changements de sens selon y)."""
        poses = np.array(robot.poses)
        dy = np.diff(poses[:, 1])
        moves = np.sign(dy[np.abs(dy) > 1e-6])
        hit = any(world.overlaps(np.vstack([robot.body.outline(p), p[:2]])) for p in poses)
        return hit, int(np.sum(moves[1:] != moves[:-1]))

    # 1. mur devant
    world = World([(-0.5, 1.0, 0.5, 1.2)])
    timeline = [(0 * ms, 1.0, 0.0)]
    robot = _run_scenario(world, timeline, 8000 * ms)
    times, poses, feet_states = robot.trajectory()
    gaps = np.array([_front_gap(robot, p) for p in poses])
    hit, reversals = yoyo(robot, world)
    results.append(("mur devant, marche avant : pas de contact, va-et-vient",
                    not hit and reversals >= 4,
                    f"écart min {gaps.min():.3f} m, {reversals} changements de sens"))
    if animate or show:
        vis.plot_trajectory(robot.body, times, poses, timeline_command(timeline),
                            filename="sensors_wall.png", show=False, world=world,
                            title="Capteurs : pleine vitesse vers un mur (va-et-vient de recul)")
    if animate:
        vis.animate_body(robot.body, times, poses, feet_states, timeline_command(timeline),
                         filename="sensors_wall.gif", show=False, world=world,
                         sensors=robot.sensors)

    # 2. mur derrière
    world = World([(-0.5, -1.0, 0.5, -0.8)])
    robot = _run_scenario(world, [(0 * ms, -1.0, 0.0)], 8000 * ms)
    hit, reversals = yoyo(robot, world)
    rear_gap = min(float(robot.sensors.rays(p)[2][REAR]) for p in robot.poses)
    results.append(("mur derrière, marche arrière : pas de contact, va-et-vient", not hit and reversals >= 4,
                    f"écart arrière min {rear_gap:.3f} m, {reversals} changements de sens"))

    # 3. rotation dans la zone de recul (nez à y = 0,15, mur à 0,25), comparée à
    #    la même rotation sans le mur
    world = World([(-0.5, 0.25, 0.5, 0.45)])
    spin = [(0 * ms, 0.0, -1.0)]
    poses = np.array(_run_scenario(world, spin, 2000 * ms).poses)
    free = np.array(_run_scenario(World([]), spin, 2000 * ms).poses)
    ok = len(poses) == len(free) and np.allclose(poses, free, atol=1e-9)
    results.append(("rotation près d'un mur : capteurs coupés", ok,
                    f"rotation de {np.degrees(poses[-1, 2] - poses[0, 2]):.1f} deg "
                    f"(sans mur {np.degrees(free[-1, 2] - free[0, 2]):.1f}), "
                    f"déplacement {np.linalg.norm(poses[-1, :2] - poses[0, :2]):.4f} m"))

    # 4. arrêt dans la zone de recul
    robot = _run_scenario(world, [(0 * ms, 0.0, 0.0)], 2000 * ms)
    poses = np.array(robot.poses)
    moved = float(np.max(np.abs(poses - poses[0])))
    results.append(("arrêt dans la zone de recul : capteurs coupés, ne bouge pas",
                    moved == 0.0, f"écart avant {_front_gap(robot, poses[0]):.3f} m, "
                                  f"plus grand changement de pose {moved:.4f}"))

    # 5. couloir de 0,5 m de large, départ décentré de 8 cm : entre et avance sans contact
    world = World([(-0.55, 0.2, -0.25, 2.0), (0.25, 0.2, 0.55, 2.0)])
    timeline = [(0 * ms, 1.0, 0.0)]
    robot = _run_scenario(world, timeline, 10000 * ms, start=(-0.08, 0.0, np.pi / 2))
    if animate:
        times, poses, feet_states = robot.trajectory()
        vis.animate_body(robot.body, times, poses, feet_states, timeline_command(timeline),
                         filename="sensors_corridor.gif", show=False, world=world,
                         sensors=robot.sensors)
    poses = np.array(robot.poses)
    inside = poses[:, 1] > 0.6                        # bien à l'intérieur du couloir
    hit = any(world.overlaps(np.vstack([robot.body.outline(p), p[:2]])) for p in poses)
    off = np.abs(poses[inside, 0]) if inside.any() else np.array([np.nan])
    results.append(("couloir, marche avant : pas de contact, reste dans le couloir",
                    not hit and inside.any(),
                    f"départ décentré de 0.080 m, ondulation max {off.max():.3f} m, "
                    f"y final {poses[-1, 1]:.2f} m"))

    print("Réflexes des capteurs :")
    for name, ok, detail in results:
        print(f"  [{'OK' if ok else 'ÉCHEC'}] {name} : {detail}")
    print("Toutes les vérifications des capteurs sont passées." if all(ok for _, ok, _ in results)
          else "Certaines vérifications des capteurs ont ÉCHOUÉ.")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return results


# --- Démonstrations : un GIF « commande vs vue de dessus » par fonctionnalité --------------

DEMO_DIR = "demos"     # sous-dossier de graphs/

DEMOS = {  # nom : (titre, scénario, durée, options)
    'avant_arriere': (
        "Marche avant et arrière à vitesse variable",
        [(0 * ms, 1.0, 0.0), (2000 * ms, 0.5, 0.0), (4000 * ms, 0.0, 0.0),
         (5000 * ms, -0.5, 0.0), (7000 * ms, -1.0, 0.0)], 9000 * ms, {}),
    'rotation': (
        "Rotation sur place (vitesse 0) : gauche, arrêt, droite à 50 % puis 100 %",
        [(0 * ms, 0.0, -1.0), (3000 * ms, 0.0, 0.0), (4000 * ms, 0.0, 0.5),
         (6500 * ms, 0.0, 1.0)], 9000 * ms, {}),
    'virages': (
        "Virages d'angle variable en marche (gauche 25, 50, 100 %, puis droite, puis en reculant)",
        [(0 * ms, 1.0, 0.0), (1000 * ms, 1.0, -0.25), (3000 * ms, 1.0, -0.5),
         (5000 * ms, 1.0, -1.0), (6500 * ms, 1.0, 0.5), (8500 * ms, -1.0, 0.5)],
        10500 * ms, {}),
    'tous_mouvements': (
        "Enchaînement de tous les mouvements (scénario de test_control)",
        DEFAULT_TIMELINE, 8000 * ms, {}),
    'capteur_avant': (
        "Capteur avant : commande « avance à fond » vers un mur -> freinage et va-et-vient",
        [(0 * ms, 1.0, 0.0)], 8000 * ms, {'world': [(-0.5, 1.0, 0.5, 1.2)]}),
    'couloir': (
        "Capteurs latéraux : couloir de 0,5 m, départ décentré de 8 cm, sans contact",
        [(0 * ms, 1.0, 0.0)], 10000 * ms,
        {'world': [(-0.55, 0.2, -0.25, 2.0), (0.25, 0.2, 0.55, 2.0)],
         'start': (-0.08, 0.0, np.pi / 2)}),
    '20_paires': (
        "20 paires de pattes (40 pattes), même réseau, même scénario",
        DEFAULT_TIMELINE, 8000 * ms, {'n_legs': 40}),
    '200_paires': (
        "200 paires de pattes (400 pattes), même réseau, même scénario",
        DEFAULT_TIMELINE, 8000 * ms, {'n_legs': 400}),
}


def demo(name, fps=15):
    """
    Lance la démonstration `name` de DEMOS en boucle fermée (Robot) et enregistre
    graphs/demos/<name>.gif. Sans obstacle, vérifie aussi, pour chaque patte et chaque
    demi-cycle, le nombre exact de neurones R, D, F, B qui déchargent (comme test_control).
    """
    title, timeline, duration, opts = DEMOS[name]
    n_legs = opts.get('n_legs', config.N_LEGS)
    world = World(opts['world']) if 'world' in opts else None
    command_fn = timeline_command(timeline)
    print(f"\n=== Démo '{name}' : {n_legs} pattes, {float(duration / ms) / 1000:.1f} s ===")
    robot = Robot(command_fn, world=world, n_legs=n_legs,
                  start=opts.get('start', (0.0, 0.0, np.pi / 2)))
    robot.run(duration)
    times, poses, feet_states = robot.trajectory()
    # un demi-cycle par poussée appliquée au corps
    _, on_p, counts, _ = robot.legs.recruitment(robot.mon_cpg, robot.mon_motor)
    k = len(robot.push_times)
    on_p, counts = on_p[:k], counts[:k]

    verdict = None
    if world is None:      # sans capteurs, chaque patte doit suivre exactement la commande
        n_errors = sum(not np.array_equal(
            counts[c, leg.index],
            expected_recruitment(robot.legs.n_motor, *command_fn(t * ms), leg, on_p[c] == leg.phase))
            for c, t in enumerate(robot.push_times) for leg in robot.legs.legs)
        verdict = (f"{k} demi-cycles x {n_legs} pattes vérifiés (R, D, F, B) : "
                   + ("aucune erreur" if n_errors == 0 else f"{n_errors} ERREUR(S)"))
        print(verdict)
    x, y, h = poses[-1]
    print(f"Pose finale : x={x:+.3f} m, y={y:+.3f} m, cap={np.degrees(h):.1f} deg")
    vis.animate_demo(robot.body, times, poses, feet_states, command_fn, on_p, counts,
                     title, f"{DEMO_DIR}/{name}.gif", fps=fps, world=world,
                     sensors=robot.sensors, leg_map=n_legs > 6, verdict=verdict)
    return robot


def run_demos(names=None):
    """Toutes les démonstrations (ou celles de `names`) : GIF dans graphs/demos/."""
    for name in names or DEMOS:
        demo(name)


if __name__ == '__main__':
    # Tests avec vérifications automatiques (figures dans graphs/)
    # test_cpg()
    # test_legs()
    # test_control()
    # test_body()
    # test_sensors()
    run_demos()
