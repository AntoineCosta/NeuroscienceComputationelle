"""Pilotage manuel : conduire l'hexapode en direct à la manette ou aux flèches du clavier (vue de dessus)."""
import os
import time
import numpy as np
from brian2 import network_operation, ms, second

import config
from robot import Robot
from world import World
import visualisation as vis

KEY_SPEED = 1.0   # haut / bas : vitesse de marche
KEY_STEER = 0.5   # + gauche / droite en marche : amplitude de ce côté divisée par 2 (courbe)
KEY_SPIN = 1.0    # gauche / droite seules : rotation sur place
STICK_DEADZONE = 0.15    # valeurs du joystick en dessous ignorées
PAD_MAX_STEER = 0.8      # joystick droit à fond : les pattes intérieures gardent 20 % (virage serré, sans pivot)
PAD_SPEED_AXIS = 1       # joystick gauche vertical (haut = négatif)
PAD_STEER_AXIS = 2       # joystick droit horizontal
PAD_QUIT_BUTTON = 7      # bouton Start (disposition Xbox, numérotation SDL)


class Keyboard:
    """État des flèches -> commande du cerveau (vitesse, virage). Deux touches opposées s'annulent."""

    ARROWS = ('up', 'down', 'left', 'right')

    def __init__(self, speed=KEY_SPEED, steer=KEY_STEER, spin=KEY_SPIN):
        self.speed, self.steer, self.spin = speed, steer, spin
        self.pressed = set()
        self.quit = False

    def press(self, key):
        if key in self.ARROWS:
            self.pressed.add(key)
        elif key == 'escape':
            self.quit = True

    def release(self, key):
        self.pressed.discard(key)

    def command(self, t=None):
        """(vitesse, virage) pour les touches enfoncées (t inutilisé : même signature qu'un scénario)."""
        forward = ('up' in self.pressed) - ('down' in self.pressed)
        side = ('right' in self.pressed) - ('left' in self.pressed)
        if forward:
            return forward * self.speed, side * self.steer
        return 0.0, side * self.spin           # vitesse 0 + virage = rotation sur place


class Gamepad(Keyboard):
    """
    Manette (disposition Xbox, lue avec pygame) ; le clavier reste actif.
      joystick gauche haut / bas      vitesse (proportionnelle)
      joystick droit gauche / droite  virage en marche (proportionnel, jusqu'à PAD_MAX_STEER)
      croix gauche / droite           rotation sur place
    Joysticks séparés : 100 % de vitesse et 100 % de virage en même temps.
      Start                           quitter
    La manette a la priorité quand elle est utilisée (joystick hors zone morte ou croix
    enfoncée), sinon c'est la commande du clavier qui s'applique. Brancher / débrancher
    la manette en cours de route est géré.
    """

    def __init__(self, deadzone=STICK_DEADZONE, **kwargs):
        super().__init__(**kwargs)
        os.environ.setdefault('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')  # pas de fenêtre pygame
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        import pygame
        self.pygame = pygame
        pygame.init()                  # file d'événements seulement, aucune fenêtre ouverte
        pygame.joystick.init()
        self.deadzone = deadzone
        self.pad = None

    def _poll(self):
        """Traite les événements de la manette : (dé)connexion et bouton pour quitter."""
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.JOYDEVICEADDED and self.pad is None:
                self.pad = pg.joystick.Joystick(event.device_index)
                print(f"Manette connectée : {self.pad.get_name()}")
            elif (event.type == pg.JOYDEVICEREMOVED and self.pad is not None
                  and event.instance_id == self.pad.get_instance_id()):
                self.pad = None
                print("Manette déconnectée (clavier toujours actif)")
            elif event.type == pg.JOYBUTTONDOWN and event.button == PAD_QUIT_BUTTON:
                self.quit = True

    def _axis(self, k):
        """Axe du joystick avec zone morte, remis à l'échelle pour atteindre encore 1."""
        v = float(np.clip(self.pad.get_axis(k), -1, 1))
        return 0.0 if abs(v) < self.deadzone else np.sign(v) * (abs(v) - self.deadzone) / (1 - self.deadzone)

    def command(self, t=None):
        self._poll()
        if self.pad is not None:
            dpad_x = self.pad.get_hat(0)[0] if self.pad.get_numhats() else 0
            if dpad_x:
                return 0.0, dpad_x * self.spin
            speed = -self._axis(PAD_SPEED_AXIS)    # joystick vers le haut = valeur d'axe négative
            if speed:                              # un virage seul ferait tourner sur place : seulement en marche
                return speed, PAD_MAX_STEER * self._axis(PAD_STEER_AXIS)
        return super().command(t)


def run_manual(n_legs=config.N_LEGS, n_motor=config.N_MOTOR, fps=20, view_size=0.75,
               max_duration=3600 * second, keys=None, world=None, start=(0.0, 0.0, np.pi / 2)):
    """
    Pilotage manuel en (quasi) temps réel, dans une fenêtre matplotlib.
    Manette (si pygame est installé, voir Gamepad) et clavier :
      haut / bas                    marche avant / arrière
      haut / bas + gauche / droite  marche en tournant de ce côté
      gauche / droite seules        rotation sur place
      Échap ou fermer la fenêtre    quitter
    La fenêtre doit avoir le focus clavier pour les touches (cliquer dessus une fois).

    Le robot marche dans un monde (par défaut : disposition de World()) et ses capteurs
    de distance agissent comme des réflexes sur les intermédiaires (voir sensors.py).
    Les commandes sont lues à chaque demi-cycle du CPG (comme dans les tests scénarisés).
    Le corps avance d'une poussée par demi-cycle, calculée dès que ses potentiels d'action
    sont arrivés et affichée pendant ce même demi-cycle : 0 à 1/4 s entre touche et mouvement.
    Renvoie le Robot (sa trajectory() contient la simulation).
    keys : source des commandes (par défaut : Gamepad, ou Keyboard si pygame manque) ;
           tout objet avec press(key), release(key), command(t) et un attribut `quit`.
    """
    if keys is None:
        try:
            keys = Gamepad()
        except ImportError:
            print("pygame non installé : clavier seulement.")
            keys = Keyboard()
    robot = Robot(keys.command, world=World() if world is None else world,
                  n_legs=n_legs, n_motor=n_motor, start=start)
    view = vis.LiveTopView(robot.body, keys.press, keys.release, half_size=view_size,
                           world=robot.world, sensors=robot.sensors)
    wall_start = None      # heure réelle correspondant au temps de simulation 0

    @network_operation(dt=1000 / fps * ms, when='end')
    def display(t):
        """Dessine le robot à sa pose courante et maintient le temps réel."""
        nonlocal wall_start
        if keys.quit or not view.is_open:
            robot.net.stop()
            return
        t_ms = float(t / ms)
        robot.odometry(t_ms)
        pose = robot.pose_at(t_ms)
        view.draw(pose, t_ms, *keys.command(t), robot.feet_at(t_ms, pose))

        now = time.perf_counter()
        if wall_start is None or wall_start + t_ms / 1000 < now - 0.2:
            wall_start = now - t_ms / 1000     # (re)synchronise en cas de retard : pas de rattrapage brusque
        view.wait(wall_start + t_ms / 1000 - now)

    robot.net.add(display)
    print("Pilotage manuel : joystick gauche vitesse + joystick droit virage (ou flèches), croix ou "
          "flèches seules pour tourner sur place, Start / Échap / fermer la fenêtre pour quitter "
          "(cliquer sur la fenêtre pour les touches).")
    robot.run(max_duration)
    if view.is_open:
        view.close()
    x, y, heading = robot.body.pose
    print(f"Arrêt à t={float(robot.net.t / second):.2f} s : x={x:+.3f} m, y={y:+.3f} m, "
          f"cap={np.degrees(heading):.1f} deg")
    return robot


if __name__ == '__main__':
    run_manual()
