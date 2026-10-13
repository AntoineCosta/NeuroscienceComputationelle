"""The 'brain': regular code turning commands into inputs of the intermediate layer."""
import numpy as np
from brian2 import network_operation

from intermediates import I_FWD_L, I_FWD_R, I_REV_L, I_REV_R, I_STEER_L, I_STEER_R


class Brain:
    """
    Converts a command (speed, steer) into the brain input of the 6 intermediates.
    All inputs mean "brake" (0 = no brake, 1 = full brake). Tonic default = stop.

      speed in [-1, 1]: > 0 forward, < 0 reverse
      steer in [-1, 1]: < 0 left, > 0 right
        - while walking: slows the legs of that side
        - at speed 0: turns on the spot (that side reverse, the other forward)

    | intermediate | stop | forward s | reverse s | steer left x | spin left x (speed 0) |
    | I-Fwd-L      |  1   |  1 - s    |    1      |  unchanged   |          1            |
    | I-Fwd-R      |  1   |  1 - s    |    1      |  unchanged   |        1 - x          |
    | I-Rev-L      |  1   |    1      |  1 - s    |  unchanged   |        1 - x          |
    | I-Rev-R      |  1   |    1      |  1 - s    |  unchanged   |          1            |
    | I-Steer-L    |  0   |    0      |    0      |   |s| * x    |          0            |
    | I-Steer-R    |  0   |    0      |    0      |      0       |          0            |
    -> walking: stroke of each side = |s| * (1 - steer of that side), push and return.
    -> spinning: stroke = |x| on both sides, in opposite directions.
    """

    STOP = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    def __init__(self, layer):
        self.layer = layer
        self.layer.neurons.brain = self.STOP
        self.layer.neurons.gate = 0.0            # stopped: sensors off

    @staticmethod
    def is_spin(speed, steer):
        """Turning on the spot: speed 0 with a steer."""
        return speed == 0 and steer != 0

    @staticmethod
    def outputs(speed, steer=0.0):
        """Brain input of the 6 intermediates for a command."""
        speed = float(np.clip(speed, -1, 1))
        steer = float(np.clip(steer, -1, 1))
        s = abs(speed)
        out = Brain.STOP.copy()
        if Brain.is_spin(speed, steer):         # turn on the spot
            a = abs(steer)
            if steer < 0:
                out[[I_REV_L, I_FWD_R]] = 1 - a  # left legs reverse, right legs forward
            else:
                out[[I_FWD_L, I_REV_R]] = 1 - a
            return out
        if speed > 0:
            out[[I_FWD_L, I_FWD_R]] = 1 - s
        elif speed < 0:
            out[[I_REV_L, I_REV_R]] = 1 - s
        out[I_STEER_L] = s * max(0.0, -steer)
        out[I_STEER_R] = s * max(0.0, steer)
        return out

    def set_command(self, speed, steer=0.0):
        """
        Apply a command (takes effect at the next CPG spike). The sensors are gated on
        only while walking (speed != 0): stopped or turning on the spot, they never
        move the robot.
        """
        self.layer.neurons.brain = self.outputs(speed, steer)
        self.layer.neurons.gate = 1.0 if speed != 0 else 0.0

    def driver(self, command_fn):
        """
        NetworkOperation latching command_fn(t) -> (speed, steer) at the start of every
        half-cycle (just before the CPG spike). Add the result to the Network.
        """
        @network_operation(dt=self.layer.cpg.period / 2, when='start')
        def apply_command(t):
            self.set_command(*command_fn(t))
        return apply_command
