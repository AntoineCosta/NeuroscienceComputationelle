"""Intermediate layer: 6 graded inhibitory interneurons between the brain and the legs."""
from brian2 import *
import numpy as np

from legs import FORWARD

# Intermediate neuron indices
I_FWD_L, I_FWD_R, I_REV_L, I_REV_R, I_STEER_L, I_STEER_R = range(6)
INTER_NAMES = ['I-Fwd-L', 'I-Fwd-R', 'I-Rev-L', 'I-Rev-R', 'I-Steer-L', 'I-Steer-R']
# Intermediate types
T_DRIVE, T_STEER = 0, 1


class IntermediateLayer:
    """
    Six graded (non-spiking) inhibitory interneurons, one per brain command.

    Each intermediate is silent at rest. Its activity is the sum of its excitatory
    inputs, saturating at 1:
        level = clip(brain + sensor, 0, 1)
    brain: set by the Brain (regular code); sensor: sensor neurons (TODO).
    Every input therefore means "brake": a sensor can only slow down or turn the robot.

    At each CPG spike (phase signal), each intermediate releases an inhibition of
    level * N threshold steps onto its targets, arriving before the CPG EPSP:

      I-Fwd-X (side X): forward walking of side X
          F of the legs in stance, B of the legs in swing, R and D
      I-Rev-X (side X): reverse walking of side X
          B of the legs in stance, F of the legs in swing, R and D
      I-Steer-X (side X): F and B of side X, in both phases

    Example, forward at speed s (I-Fwd = 1 - s, I-Rev = 1, I-Steer = 0):
      F in stance : N - (1 - s)N = sN fired    (push)
      B in swing  : N - (1 - s)N = sN fired    (return, mirrors the push)
      B in stance, F in swing : N - N = 0
      R / D (EPSP 2N) : 2N - (1 - s)N - N = sN > 0 -> the leg steps
    With all intermediates at 0 everything would fire: the Brain must send the stop
    pattern by default.

    The intermediates know which legs are in stance through `stance_phase`, written by
    the CPG spike (stands for un-modelled phase-signalling neurons).
    """

    def __init__(self, cpg, legs):
        dt = defaultclock.dt
        if legs.exc_delay < 2 * dt:
            raise ValueError("exc_delay must be at least 2 time steps (intermediate release latency)")
        self.cpg, self.legs = cpg, legs

        self.eqs = '''
        brain : 1                            # input from the brain (regular code)
        sensor : 1                           # input from the sensor neurons (TODO)
        level = clip(brain + sensor, 0, 1) : 1   # graded activity, saturates at 1
        release_now : boolean                # set by the phase signal at each CPG spike
        stance_phase : integer               # CPG phase of the legs in stance
        itype : integer (constant)           # drive / steer
        direction : integer (constant)       # drive: 0 = forward, 1 = reverse
        side : integer (constant)            # 0 = left, 1 = right
        '''
        self.neurons = NeuronGroup(6, model=self.eqs, events={'release': 'release_now'})
        self.neurons.run_on_event('release', 'release_now = False')
        self.neurons.itype = [T_DRIVE] * 4 + [T_STEER] * 2
        self.neurons.direction = [0, 0, 1, 1, 0, 0]
        self.neurons.side = [0, 1, 0, 1, 0, 1]

        # --- Phase signal: each CPG spike triggers a release (one step later) -------
        self.trigger = Synapses(cpg.neurons, self.neurons,
                                on_pre='release_now_post = True; stance_phase_post = i')
        self.trigger.connect()

        # Release one step after the CPG spike -> lead = exc_delay - dt
        ns = {'w_full': legs.inhibition_weight(legs.n_motor, lead=legs.exc_delay - dt)}

        # --- Drive intermediates: own direction's push (stance) and return (swing) ---
        # (Brian parses one statement per line: temporaries keep it readable)
        self.drive = Synapses(self.neurons, legs.motors, on_event='release', namespace=ns, on_pre=f'''
            in_stance = int(kind_post == {FORWARD} + direction_pre) * int(phase_post == stance_phase_pre)
            in_swing = int(kind_post == {FORWARD + 1} - direction_pre) * int(phase_post != stance_phase_pre)
            v_post -= w_full * level_pre * (in_stance + in_swing + int(kind_post < {FORWARD}))''')
        self.drive.connect(condition=f'itype_pre == {T_DRIVE} and side_post == side_pre')
        self.drive.delay = legs.input_delay(self.drive, 0*ms)

        # --- Steering intermediates: F and B of one side, both phases --------------
        self.steer = Synapses(self.neurons, legs.motors, on_event='release', namespace=ns,
                              on_pre='v_post -= w_full * level_pre')
        self.steer.connect(condition=f'itype_pre == {T_STEER} and side_post == side_pre '
                                     f'and kind_post >= {FORWARD}')

    @property
    def objects(self):
        """Brian2 objects to add to a Network (CPG and legs not included)."""
        return [self.neurons, self.trigger, self.drive, self.steer]
