"""Global configuration of the hexapod simulation, shared by all modules."""
from brian2 import ms, prefs, defaultclock

N_LEGS = 6                 # Number of legs: even, 6 <= N_LEGS <= 200
N_MOTOR = 10               # Neurons per forward/backward motor group (stroke resolution)
CYCLE_PERIOD = 500 * ms    # Full CPG cycle (phase A + phase B), fixed rhythm
TAU_M = 20 * ms            # Membrane time constant of the leg motor neurons
EXC_DELAY = 2 * ms         # CPG -> motor excitation delay (inhibition arrives first)
LIFT_DELAY = 10 * ms       # Raise (R) delay after swing onset (other tripod touching down)

# Body geometry, top view (metres) - kinematics and visualisation only
BODY_LENGTH = 0.30
BODY_WIDTH = 0.15
TRACK_WIDTH = 0.25         # Distance between the left and right feet
STRIDE = 0.06              # Body displacement of a full stance stroke (N neurons fired)

# World and distance sensors (metres) - world.py, sensors.py
ARENA_SIZE = 4.0           # Square arena, walls at +-ARENA_SIZE / 2
SENSOR_ANGLE = 20          # Degrees: front sensors at +-angle, rear ones at 180 -+ angle
FRONT_RANGE = 0.6          # Front sensors brake forward walking from this distance
REAR_RANGE = 0.4           # Rear sensors brake reverse walking from this distance
SIDE_RANGE = 0.3           # Side sensors (+-90 deg) steer away from this distance
CONTACT_MARGIN = 0.05      # Brakes are full at this distance (not only at contact)
BACKOFF_DISTANCE = 0.2     # Front obstacle closer than this: walk backward (reflex)
BACKOFF_RAMP = 0.02        # The back-off goes from 0 to full over this distance
STEER_GAIN = 1.0           # Turn-away strength (difference between the front sensors)
SIDE_STEER_GAIN = 1.0      # Turn-away strength (difference between the side sensors)
SENSE_DT = 10 * ms         # Sensor and odometry update period

prefs.codegen.target = 'numpy'
defaultclock.dt = 0.5 * ms
