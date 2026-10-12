"""Central Pattern Generator: two-phase rhythm, one spike per half-cycle."""
from brian2 import *
import numpy as np

from config import CYCLE_PERIOD


class CPG:
    """
    Two-phase rhythm generator emitting ONE spike per half-cycle.

    - Neuron 0 (phase A): pacemaker LIF, constant drive tuned so it fires exactly
      once per period.
    - Neuron 1 (phase B): relay LIF, silent at rest, fired by A through an excitatory
      synapse delayed by period/2 -> exact anti-phase.
    """

    def __init__(self, period=CYCLE_PERIOD, tau_pace=CYCLE_PERIOD, tau_relay=20*ms,
                 v_rest=-65*mV, v_th=-50*mV, v_reset=-65*mV, t_ref=2*ms, w_relay=20*mV):
        dt = defaultclock.dt
        self.period = period

        # Drive giving ISI = period (exact LIF solution). Brian records a spike at the
        # start of the step in which v crosses v_th: target the crossing half a step later.
        #   v_inf = (a*v_th - v_reset)/(a - 1),  a = exp((ISI - t_ref)/tau)
        a = np.exp(float((period + dt/2 - t_ref) / tau_pace))
        I_pace = (a * v_th - v_reset) / (a - 1) - v_rest

        self.eqs = '''
        dv/dt = (v_rest - v + I_drive) / tau : volt (unless refractory)
        I_drive : volt (constant)
        tau : second (constant)
        '''
        params = {'v_rest': v_rest, 'v_th': v_th, 'v_reset': v_reset, 'w_relay': w_relay}
        self.neurons = NeuronGroup(2, model=self.eqs, threshold='v > v_th',
                                   reset='v = v_reset', refractory=t_ref,
                                   method='exact', namespace=params)
        self.neurons.I_drive = [float(I_pace), 0.0] * volt
        self.neurons.tau = [float(tau_pace), float(tau_relay)] * second
        # A fires at t = 0, B is at rest
        self.neurons.v = [float(v_th + 1*mV), float(v_rest)] * volt

        # A -> B relay. Latency = delay + 1 step (B crosses threshold on the next step).
        self.relay = Synapses(self.neurons, self.neurons, on_pre='v_post += w_relay',
                              delay=period/2 - dt, namespace=params)
        self.relay.connect(i=0, j=1)

    @property
    def objects(self):
        """Brian2 objects to add to a Network."""
        return [self.neurons, self.relay]
