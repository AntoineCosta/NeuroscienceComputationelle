"""Global configuration of the hexapod simulation, shared by all modules."""
from brian2 import ms, prefs, defaultclock

N_LEGS = 6                 # Number of legs: even, 6 <= N_LEGS <= 200
N_MOTOR = 10               # Neurons per forward/backward motor group (stroke resolution)
CYCLE_PERIOD = 500 * ms    # Full CPG cycle (phase A + phase B), fixed rhythm
TAU_M = 20 * ms            # Membrane time constant of the leg motor neurons
EXC_DELAY = 2 * ms         # CPG -> motor excitation delay (inhibition arrives first)
LIFT_DELAY = 10 * ms       # Raise (R) delay after swing onset (other tripod touching down)

prefs.codegen.target = 'numpy'
defaultclock.dt = 0.5 * ms
