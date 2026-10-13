"""Manual control: drive the hexapod live with a game controller or the arrow keys (top view)."""
import os
import time
import numpy as np
from brian2 import network_operation, ms, second

import config
from robot import Robot
from world import World
import visualisation as vis

KEY_SPEED = 1.0   # up / down: walking speed
KEY_STEER = 0.5   # + left / right while walking: that side's stroke halved (curve)
KEY_SPIN = 1.0    # left / right alone: turn on the spot
STICK_DEADZONE = 0.15    # controller stick values below this are ignored
PAD_MAX_STEER = 0.8      # full right stick: inner legs keep 20 % (tight turn, no pivot)
PAD_SPEED_AXIS = 1       # left stick vertical (up = negative)
PAD_STEER_AXIS = 2       # right stick horizontal
PAD_QUIT_BUTTON = 7      # Start button (Xbox layout, SDL numbering)


class Keyboard:
    """Arrow-key state -> brain command (speed, steer). Opposite keys cancel out."""

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
        """(speed, steer) for the keys currently held (t unused: same signature as a timeline)."""
        forward = ('up' in self.pressed) - ('down' in self.pressed)
        side = ('right' in self.pressed) - ('left' in self.pressed)
        if forward:
            return forward * self.speed, side * self.steer
        return 0.0, side * self.spin           # speed 0 + steer = turn on the spot


class Gamepad(Keyboard):
    """
    Game controller (Xbox layout, read with pygame); the keyboard stays active.
      left stick up / down      speed (proportional)
      right stick left / right  steer while walking (proportional, up to PAD_MAX_STEER)
      D-pad left / right        turn on the spot
    Separate sticks: 100 % speed and 100 % steer at the same time.
      Start                     quit
    The controller wins while it is used (stick out of the dead zone or D-pad
    pressed), otherwise the keyboard command applies. Plugging / unplugging the
    controller while running is handled.
    """

    def __init__(self, deadzone=STICK_DEADZONE, **kwargs):
        super().__init__(**kwargs)
        os.environ.setdefault('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')  # no pygame window
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        import pygame
        self.pygame = pygame
        pygame.init()                  # event queue only, no window is opened
        pygame.joystick.init()
        self.deadzone = deadzone
        self.pad = None

    def _poll(self):
        """Process controller events: (dis)connection and the quit button."""
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.JOYDEVICEADDED and self.pad is None:
                self.pad = pg.joystick.Joystick(event.device_index)
                print(f"Controller connected: {self.pad.get_name()}")
            elif (event.type == pg.JOYDEVICEREMOVED and self.pad is not None
                  and event.instance_id == self.pad.get_instance_id()):
                self.pad = None
                print("Controller disconnected (keyboard still active)")
            elif event.type == pg.JOYBUTTONDOWN and event.button == PAD_QUIT_BUTTON:
                self.quit = True

    def _axis(self, k):
        """Stick axis with dead zone, rescaled so it still reaches 1."""
        v = float(np.clip(self.pad.get_axis(k), -1, 1))
        return 0.0 if abs(v) < self.deadzone else np.sign(v) * (abs(v) - self.deadzone) / (1 - self.deadzone)

    def command(self, t=None):
        self._poll()
        if self.pad is not None:
            dpad_x = self.pad.get_hat(0)[0] if self.pad.get_numhats() else 0
            if dpad_x:
                return 0.0, dpad_x * self.spin
            speed = -self._axis(PAD_SPEED_AXIS)    # stick up = negative axis value
            if speed:                              # steer alone would spin: walking only
                return speed, PAD_MAX_STEER * self._axis(PAD_STEER_AXIS)
        return super().command(t)


def run_manual(n_legs=config.N_LEGS, n_motor=config.N_MOTOR, fps=20, view_size=0.75,
               max_duration=3600 * second, keys=None, world=None, start=(0.0, 0.0, np.pi / 2)):
    """
    Live manual control in (approximately) real time, in a matplotlib window.
    Game controller (if pygame is installed, see Gamepad) and keyboard:
      up / down                  walk forward / backward
      up / down + left / right   walk while turning towards that side
      left / right alone         turn on the spot
      Esc or close the window    quit
    The window needs the keyboard focus for the keys (click on it once).

    The robot walks in a world (default: World() layout) and its distance sensors
    act as reflexes on the intermediates (see sensors.py).
    Commands are latched at each CPG half-cycle (as in the scripted tests). The body
    moves by one stance push per half-cycle, computed as soon as its spikes are in
    and shown during that same half-cycle: 0 to 1/4 s between key and motion.
    Returns the Robot (its trajectory() holds the run).
    keys: command source (default: Gamepad, or Keyboard if pygame is missing);
          anything with press(key), release(key), command(t) and a `quit` flag.
    """
    if keys is None:
        try:
            keys = Gamepad()
        except ImportError:
            print("pygame not installed: keyboard only.")
            keys = Keyboard()
    robot = Robot(keys.command, world=World() if world is None else world,
                  n_legs=n_legs, n_motor=n_motor, start=start)
    view = vis.LiveTopView(robot.body, keys.press, keys.release, half_size=view_size,
                           world=robot.world, sensors=robot.sensors)
    wall_start = None      # wall-clock time of simulation time 0

    @network_operation(dt=1000 / fps * ms, when='end')
    def display(t):
        """Draw the robot at its current pose and keep real time."""
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
            wall_start = now - t_ms / 1000     # (re)synchronise if late: no catch-up rush
        view.wait(wall_start + t_ms / 1000 - now)

    robot.net.add(display)
    print("Manual control: left stick speed + right stick steer (or arrows), D-pad or arrows "
          "alone to turn on the spot, Start / Esc / close the window to quit "
          "(click on the window for the keys).")
    robot.run(max_duration)
    if view.is_open:
        view.close()
    x, y, heading = robot.body.pose
    print(f"Stopped at t={float(robot.net.t / second):.2f} s: x={x:+.3f} m, y={y:+.3f} m, "
          f"heading={np.degrees(heading):.1f} deg")
    return robot


if __name__ == '__main__':
    run_manual()
