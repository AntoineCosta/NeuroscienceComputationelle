"""Générateur de rythme (CPG) : deux phases, un potentiel d'action par demi-cycle."""
from brian2 import *
import numpy as np

from config import CYCLE_PERIOD


class CPG:
    """
    Générateur de rythme : 2 neurones, un potentiel d'action par demi-cycle.

    - A (neurone 0) : LIF pacemaker, courant constant réglé pour décharger
      exactement une fois par période.
    - B (neurone 1) : LIF relais, silencieux au repos, déclenché par A via une synapse
      excitatrice retardée de période/2 -> anti-phase exacte.
    """

    def __init__(self, period=CYCLE_PERIOD, tau_pace=CYCLE_PERIOD, tau_relay=20*ms,
                 v_rest=-65*mV, v_th=-50*mV, v_reset=-65*mV, t_ref=2*ms, w_relay=20*mV):
        dt = defaultclock.dt
        self.period = period

        # Courant réglé pour décharger exactement toutes les `period` (solution exacte du LIF).
        # On vise un demi-pas après le seuil, car Brian date le potentiel d'action au début
        # du pas où v franchit v_th.
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
        # A décharge à t = 0, B au repos
        self.neurons.v = [float(v_th + 1*mV), float(v_rest)] * volt

        # Relais A -> B. Latence = délai + 1 pas (B franchit le seuil au pas suivant),
        # d'où le délai de période/2 - dt.
        self.relay = Synapses(self.neurons, self.neurons, on_pre='v_post += w_relay',
                              delay=period/2 - dt, namespace=params)
        self.relay.connect(i=0, j=1)

    @property
    def objects(self):
        """Objets Brian2 à ajouter à un Network."""
        return [self.neurons, self.relay]
