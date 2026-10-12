from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Classe : CPG (Central Pattern Generator) - Oscillateur pour rythme des pattes
# =============================================================================
class CPG:
    def __init__(self, tau_m=20*ms, tau_w=200*ms, b_adapt=4.0*mV):
        """
        Générateur de rythme basé sur un modèle simplifié avec adaptation (proche AdEx).
        """
        # Équations du neurone (Intégration et décharge avec adaptation)
        self.eqs = '''
        dv/dt = -(v + w)/tau_m + I_drive/tau_m : volt (unless refractory)
        dw/dt = -w/tau_w : volt
        I_drive : volt
        '''
        
        # Groupe de 2 neurones maîtres (Phase A et Phase B)
        self.neurons = NeuronGroup(2, model=self.eqs, threshold='v > 10*mV', 
                                   reset='v = -50*mV; w += b_adapt', refractory=10*ms, method='euler')
        
        # Initialisation asymétrique pour forcer l'alternance
        self.neurons.v = [-50, -50] * mV
        self.neurons.w = [0, 2] * mV
        self.neurons.I_drive = 0 * mV # Le courant sera fourni par les commandes descendantes
        
        # Inhibition mutuelle (A inhibe B, B inhibe A) pour créer l'alternance
        self.synapses = Synapses(self.neurons, self.neurons, on_pre='v_post -= 25*mV')
        self.synapses.connect(i=0, j=1)
        self.synapses.connect(i=1, j=0)

# =============================================================================
# Classe : Leg - Neurones moteurs des pattes
# =============================================================================
class Leg:
    def __init__(self, leg_id, phase):
        """
        Représente les motoneurones d'une patte (Élévation Z et Propulsion X).
        """
        self.id = leg_id
        self.phase = phase # 0 (Phase A) ou 1 (Phase B)
        
        # Équations des motoneurones (Filtres passe-bas intégrant les spikes)
        self.eqs = '''
        dz/dt = -z / (30*ms) : 1
        dx/dt = -x / (30*ms) : 1
        '''
        self.neurons = NeuronGroup(1, model=self.eqs, method='euler')

# =============================================================================
# Classe : DescendingCommand - L'Interface Cerveau/Joystick
# =============================================================================
class DescendingCommand:
    def __init__(self):
        """
        Génère des commandes descendantes sous forme de trains de spikes (Rate Coding).
        """
        # Groupes de Poisson pour générer les spikes de commande
        self.cmd_fwd = PoissonGroup(1, rates=0*Hz)
        self.cmd_bwd = PoissonGroup(1, rates=0*Hz)
        self.cmd_rot_L = PoissonGroup(1, rates=0*Hz)
        self.cmd_rot_R = PoissonGroup(1, rates=0*Hz)

    def set_joystick(self, y_axis, x_axis):
        """
        Convertit les axes analogiques (-1 à 1) en taux de décharge (Hz).
        """
        max_rate = 100 * Hz
        
        # Axe Y : Vitesse (Avancer ou Reculer)
        if y_axis > 0:
            self.cmd_fwd.rates = y_axis * max_rate
            self.cmd_bwd.rates = 0 * Hz
        else:
            self.cmd_fwd.rates = 0 * Hz
            self.cmd_bwd.rates = abs(y_axis) * max_rate
            
        # Axe X : Rotation (Inhibition unilatérale)
        if x_axis < 0: # Tourner à gauche
            self.cmd_rot_L.rates = abs(x_axis) * max_rate
            self.cmd_rot_R.rates = 0 * Hz
        else:          # Tourner à droite
            self.cmd_rot_L.rates = 0 * Hz
            self.cmd_rot_R.rates = x_axis * max_rate

# =============================================================================
# Classe : Sensor - Capteur/Neurone pour détection d'obstacles
# =============================================================================
class Sensor:
    def __init__(self, max_distance=5.0, max_rate=150*Hz):
        """
        Modélise 4 capteurs de distance (Avant, Arrière, Gauche, Droite)
        en utilisant l'encodage par taux de décharge (Processus de Poisson).
        
        Args:
            max_distance: La distance maximale de détection (en mètres).
            max_rate: La fréquence de décharge maximale quand l'obstacle est touché (distance = 0).
        """
        self.max_distance = max_distance
        self.max_rate = max_rate
        
        # 4 Groupes de Poisson pour chaque direction
        # Les taux de décharge (rates) seront mis à jour dynamiquement
        self.front = PoissonGroup(1, rates=0*Hz)
        self.back  = PoissonGroup(1, rates=0*Hz)
        self.left  = PoissonGroup(1, rates=0*Hz)
        self.right = PoissonGroup(1, rates=0*Hz)
        
    def update_distances(self, dist_fwd, dist_bwd, dist_left, dist_right):
        """
        Convertit les distances mesurées en taux de décharge.
        Plus la distance est courte (proche de 0), plus le taux est élevé.
        Si la distance dépasse max_distance, le taux est de 0 Hz (pas de décharge).
        """
        # Fonction de transduction (Inversement proportionnel à la distance)
        def distance_to_rate(d):
            if d >= self.max_distance:
                return 0 * Hz
            elif d <= 0:
                return self.max_rate
            else:
                # Relation linéaire inverse : (1 - d/d_max) * rate_max
                return (1.0 - (d / self.max_distance)) * self.max_rate

        # Mise à jour des taux de décharge pour chaque capteur
        self.front.rates = distance_to_rate(dist_fwd)
        self.back.rates  = distance_to_rate(dist_bwd)
        self.left.rates  = distance_to_rate(dist_left)
        self.right.rates = distance_to_rate(dist_right)

# =============================================================================
# Classe : HexapodController - Le Contrôleur Global
# =============================================================================
class HexapodController:
    def __init__(self, n_legs=6):
        # Configuration de base
        prefs.codegen.target = 'numpy'
        defaultclock.dt = 0.5 * ms
        self.n_legs = n_legs
        
        # Instanciation des sous-composants
        self.cpg = CPG()
        self.command = DescendingCommand()
        
        # Création des pattes (Trépied alterné : [0, 1, 0, 1, 0, 1])
        phases = [i % 2 for i in range(self.n_legs)]
        self.legs = [Leg(i, phases[i]) for i in range(self.n_legs)]
        
        # Pour simplifier l'accès global aux neurones des pattes
        self.all_legs = sum([leg.neurons for leg in self.legs], start=NeuronGroup(0, 'dz/dt=0:1; dx/dt=0:1'))
        # Astuce Brian2: On recrée un NeuronGroup global pour faciliter les connexions synaptiques
        eqs_all_legs = '''
        dz/dt = -z / (30*ms) : 1
        dx/dt = -x / (30*ms) : 1
        '''
        self.legs_group = NeuronGroup(self.n_legs, model=eqs_all_legs, method='euler')
        
        self.setup_network()

    def setup_network(self):
        """Construit les synapses entre les différentes classes."""
        
        # 1. Le CPG reçoit un courant de base fixe pour l'exemple
        # (Dans un modèle plus avancé, `cmd_fwd` exciterait directement le CPG)
        self.cpg.neurons.I_drive = 45 * mV
        
        # 2. Synchronisation CPG -> Pattes (Élévation Z)
        # Le CPG dicte le rythme de base à chaque patte selon sa phase
        self.syn_cpg_legs = Synapses(self.cpg.neurons, self.legs_group, on_pre='z += 1.0')
        for i, leg in enumerate(self.legs):
            self.syn_cpg_legs.connect(i=leg.phase, j=i)
            
        # 3. Commandes Descendantes -> Motoneurones (Propulsion X)
        w_cmd = 0.1 # Poids synaptique de base
        
        # Marche avant : Excite toutes les pattes
        self.syn_fwd = Synapses(self.command.cmd_fwd, self.legs_group, on_pre=f'x += {w_cmd}')
        self.syn_fwd.connect()
        
        # Marche arrière : Inhibe toutes les pattes (les fait reculer)
        self.syn_bwd = Synapses(self.command.cmd_bwd, self.legs_group, on_pre=f'x -= {w_cmd}')
        self.syn_bwd.connect()
        
        # Identifier les pattes gauches (0 à N/2-1) et droites (N/2 à N-1)
        gauche = np.arange(0, self.n_legs // 2)
        droite = np.arange(self.n_legs // 2, self.n_legs)
        
        # Rotation Gauche : Excite droite, Inhibe gauche
        self.syn_rot_L_exc = Synapses(self.command.cmd_rot_L, self.legs_group, on_pre=f'x += {w_cmd}')
        self.syn_rot_L_exc.connect(j=droite)
        self.syn_rot_L_inh = Synapses(self.command.cmd_rot_L, self.legs_group, on_pre=f'x -= {w_cmd}')
        self.syn_rot_L_inh.connect(j=gauche)
        
        # Rotation Droite : Excite gauche, Inhibe droite
        self.syn_rot_R_exc = Synapses(self.command.cmd_rot_R, self.legs_group, on_pre=f'x += {w_cmd}')
        self.syn_rot_R_exc.connect(j=gauche)
        self.syn_rot_R_inh = Synapses(self.command.cmd_rot_R, self.legs_group, on_pre=f'x -= {w_cmd}')
        self.syn_rot_R_inh.connect(j=droite)
        
        # Configuration des moniteurs
        self.mon_cpg = SpikeMonitor(self.cpg.neurons)
        self.mon_legs = StateMonitor(self.legs_group, ['z', 'x'], record=True)

    def run_simulation(self, duration_s=1.5):
        """Lance la simulation."""
        # On regroupe tous les objets Brian pour créer le réseau
        net = Network(self.cpg.neurons, self.cpg.synapses, self.legs_group, self.syn_cpg_legs,
                      self.command.cmd_fwd, self.command.cmd_bwd, self.command.cmd_rot_L, self.command.cmd_rot_R,
                      self.syn_fwd, self.syn_bwd, self.syn_rot_L_exc, self.syn_rot_L_inh, self.syn_rot_R_exc, self.syn_rot_R_inh,
                      self.mon_cpg, self.mon_legs)
        net.run(duration_s * second)

    def plot_results(self):
        """Affiche les résultats."""
        plt.ioff()
        plt.figure(figsize=(12, 6))
        
        # Potentiel CPG
        plt.subplot(2, 1, 1)
        plt.plot(self.mon_legs.t/second, self.mon_legs.z[0], label="Patte L1 (Phase A) - Élévation Z")
        plt.plot(self.mon_legs.t/second, self.mon_legs.z[1], label="Patte L2 (Phase B) - Élévation Z")
        plt.ylabel("Élévation (Z)")
        plt.legend(loc='upper right')
        plt.title("Coordination CPG (Opposition de phase)")

        # Propulsion
        plt.subplot(2, 1, 2)
        plt.plot(self.mon_legs.t/second, self.mon_legs.x[0], label="Patte Gauche L1 (X)")
        plt.plot(self.mon_legs.t/second, self.mon_legs.x[-1], label="Patte Droite R3 (X)", linestyle="--")
        plt.ylabel("Propulsion (X)")
        plt.xlabel("Temps (s)")
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig("hexapod_oop.png", dpi=300)
        print("Image sauvegardée : hexapod_oop.png")

# =============================================================================
# EXÉCUTION PRINCIPALE (Exemple en classe)
# =============================================================================
if __name__ == '__main__':
    # 1. Créer le robot
    robot = HexapodController(n_legs=6)
    
    # 2. Simuler une entrée utilisateur (Joystick)
    # Exemple : On veut avancer légèrement (Y=0.5) tout en tournant fortement à gauche (X=-1.0)
    robot.command.set_joystick(y_axis=0.5, x_axis=-1.0)
    
    # 3. Lancer
    print("Simulation en cours...")
    robot.run_simulation(duration_s=1.5)
    robot.plot_results()