"""Configuration globale de la simulation de l'hexapode, partagée par tous les modules."""
from brian2 import ms, prefs, defaultclock

N_LEGS = 6                 # Nombre de pattes : pair, 6 <= N_LEGS <= 400 (200 paires)
N_MOTOR = 10               # Neurones par groupe moteur avant / arrière (résolution du pas)
CYCLE_PERIOD = 500 * ms    # Cycle complet du CPG (phase A + phase B), rythme fixe
TAU_M = 20 * ms            # Constante de temps membranaire des motoneurones
EXC_DELAY = 2 * ms         # Délai d'excitation CPG -> motoneurones (l'inhibition arrive avant)
LIFT_DELAY = 10 * ms       # Retard du lever (R) après le début du balancement (l'autre trépied se pose d'abord)

# Géométrie du corps, vue de dessus (mètres) - cinématique et affichage uniquement
BODY_LENGTH = 0.30
BODY_WIDTH = 0.15
TRACK_WIDTH = 0.25         # Écart entre les pieds gauches et droits
STRIDE = 0.06              # Déplacement du corps pour un appui complet (N neurones déchargent)

# Monde et capteurs de distance (mètres) - world.py, sensors.py
ARENA_SIZE = 4.0           # Arène carrée, murs à +-ARENA_SIZE / 2
FRONT_RANGE = 0.6          # Le capteur avant freine la marche avant à partir de cette distance
REAR_RANGE = 0.4           # Le capteur arrière freine la marche arrière à partir de cette distance
SIDE_RANGE = 0.3           # Les capteurs latéraux (+-90 deg) font tourner à partir de cette distance
CONTACT_MARGIN = 0.05      # Freinage maximal dès cette distance (pas seulement au contact)
BACKOFF_DISTANCE = 0.2     # Obstacle devant plus proche que ça : marche arrière (réflexe)
BACKOFF_RAMP = 0.02        # Le recul passe de 0 au maximum sur cette distance
SIDE_STEER_GAIN = 1.0      # Force du virage d'évitement (différence entre capteurs latéraux)
SENSE_DT = 10 * ms         # Période de mise à jour des capteurs et de l'odométrie

prefs.codegen.target = 'numpy'
defaultclock.dt = 0.5 * ms
