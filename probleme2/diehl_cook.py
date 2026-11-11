"""
Partie II - Réseau de Diehl & Cook (2015) pour MNIST, appris sans supervision par STDP.

Architecture :  784 entrées Poisson --(STDP)--> N excitateurs <--> N inhibiteurs
  - Entrée    : codage en fréquence, pixel (0..1) -> 0..63.75 Hz (Poisson).
  - Excitateurs : LIF à conductances + seuil adaptatif theta (homéostasie).
  - E -> I    : un-à-un (proba p_ei), chaque excitateur active "son" inhibiteur.
  - I -> E    : vers tous les AUTRES excitateurs (proba p_ie) => winner-take-all.
  - STDP      : équation (3) de l'article, appliquée à chaque décharge post :
                    dw = eta * (xpre - x_tar) * (w_max - w)^mu
                xpre = trace présynaptique (mise à 1 à chaque décharge pré, décroît
                avec tau_pre). Si l'entrée a déchargé récemment (xpre > x_tar) le
                poids augmente, sinon il diminue -> le neurone "copie" l'image.
  - Normalisation divisive (Goodhill & Barrow, optionnelle) : avant chaque présentation,
    la somme des poids entrant dans chaque neurone est ramenée à 0.1 x (nb d'entrées) (=78).

Lecture (hors ligne, sur les décomptes de décharges enregistrés) : voir `lecture`.

Auteurs : Antoine Costa
"""
import gzip
import os
import time
import urllib.request

import numpy as np
from brian2 import (NeuronGroup, PoissonGroup, Synapses, SpikeMonitor, Network,
                    prefs, defaultclock, ms, mV, nS, Hz)

prefs.codegen.target = 'numpy'   # 'cython' testé : plus lent ici (surcoût Python par pas de temps)
defaultclock.dt = 0.5 * ms

ICI = os.path.dirname(os.path.abspath(__file__))
DOSSIER_RES = os.path.join(ICI, 'resultats')

# Configuration par défaut (une expérience = un dictionnaire de modifications)
DEFAUT = dict(
    n_neurones=100,      # neurones excitateurs (= nb d'inhibiteurs)
    n_images=7000,       # sous-ensemble de MNIST : 5000 train / 1000 val / 1000 test
    epoques=1,           # passages sur l'ensemble d'entraînement
    n_eval_train=1000,   # images d'entraînement utilisées pour étiqueter les neurones
    homeo=True,          # seuil adaptatif theta activé ?
    normalisation=True,  # normalisation divisive des poids activée ?
    p_in=1.0,            # proba de connexion entrée -> excitateurs
    p_ei=1.0,            # proba de connexion excitateur i -> inhibiteur i
    p_ie=1.0,            # proba de connexion inhibiteur i -> excitateur j (j != i)
    eta=0.01, x_tar=0.4, w_max=1.0, mu=1.0, tau_pre=20.0,   # STDP éq. (3) (tau en ms)
    theta_plus=0.2,      # mV ajoutés au seuil à chaque décharge (si homeo) ; article : 0.05,
                         # trop lent pour 5000 images (-3 points, expérience theta_0.05)
    t_image=350.0,       # ms de présentation d'une image
    seed=0,
)


# --- Données -------------------------------------------------------------------
def charger_mnist(n_images, seed):
    """Tire n_images parmi les 70 000 de MNIST et les découpe en train/val/test
    (5/7, 1/7, 1/7) de façon stratifiée : chaque classe garde la même proportion
    dans les trois sous-ensembles."""
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    def lire(nom, decalage):
        chemin = os.path.join(ICI, 'data', nom)
        if not os.path.exists(chemin):
            os.makedirs(os.path.dirname(chemin), exist_ok=True)
            urllib.request.urlretrieve(url + nom, chemin)
        with gzip.open(chemin, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=decalage)
    X = np.concatenate([lire('train-images-idx3-ubyte.gz', 16),
                        lire('t10k-images-idx3-ubyte.gz', 16)]).reshape(-1, 784) / 255.
    y = np.concatenate([lire('train-labels-idx1-ubyte.gz', 8),
                        lire('t10k-labels-idx1-ubyte.gz', 8)])
    rng = np.random.default_rng(seed)
    choix = rng.permutation(len(y))[:n_images]
    parts = ([], [], [])
    for c in range(10):
        ic = choix[y[choix] == c]
        a, b = round(len(ic) * 5 / 7), round(len(ic) * 6 / 7)
        for p, morceau in zip(parts, (ic[:a], ic[a:b], ic[b:])):
            p.extend(morceau)
    parts = [rng.permutation(p) for p in parts]          # mélange les classes
    return [(X[p], y[p]) for p in parts]                 # train, val, test


# --- Réseau --------------------------------------------------------------------
MODELE_NEURONE = '''
dv/dt = ((v_rest - v) + (I_synE + I_synI) / nS) / tau : volt (unless refractory)
I_synE = ge * nS * -v                                  : amp   # E_exc = 0 mV
I_synI = gi * nS * (E_inh - v)                         : amp
dge/dt = -ge / (1 * ms)                                : 1
dgi/dt = -gi / (2 * ms)                                : 1
dtheta/dt = -theta / (1e7 * ms)                        : volt  # seuil adaptatif
plastique                                              : 1 (shared)
'''


class Reseau:
    def __init__(self, cfg, W=None, theta=None):
        c, N = cfg, cfg['n_neurones']
        self.cfg = c
        self.entree = PoissonGroup(784, rates=0 * Hz)

        # Homéostasie : à chaque décharge, le seuil monte de theta_plus (et redescend
        # en 1e7 ms) -> un neurone qui gagne trop souvent devient plus dur à activer.
        self.exc = NeuronGroup(
            N, MODELE_NEURONE, method='euler', refractory=5 * ms,
            threshold='v > v_thresh + theta',
            reset='v = v_reset; theta += theta_plus * plastique',
            namespace=dict(v_rest=-65 * mV, v_reset=-65 * mV, v_thresh=-52 * mV,
                           tau=100 * ms, E_inh=-100 * mV,
                           theta_plus=c['theta_plus'] * mV * c['homeo']))
        self.inh = NeuronGroup(
            N, MODELE_NEURONE, method='euler', refractory=2 * ms,
            threshold='v > v_thresh', reset='v = v_reset',
            namespace=dict(v_rest=-60 * mV, v_reset=-45 * mV, v_thresh=-40 * mV,
                           tau=10 * ms, E_inh=-85 * mV))
        self.exc.v, self.inh.v = -65 * mV, -60 * mV
        if theta is not None:
            self.exc.theta = theta * mV

        # Synapses d'entrée plastiques (STDP éq. 3)
        self.S_in = Synapses(
            self.entree, self.exc,
            '''w : 1
               dxpre/dt = -xpre / tau_pre : 1 (event-driven)''',
            on_pre='ge_post += w; xpre = 1',
            on_post='w = clip(w + plastique_post * eta * (xpre - x_tar) * (w_max - w)**mu, 0, w_max)',
            namespace=dict(eta=c['eta'], x_tar=c['x_tar'], w_max=c['w_max'],
                           mu=c['mu'], tau_pre=c['tau_pre'] * ms))
        self.S_in.connect(p=c['p_in'])
        self.S_in.delay = 'rand() * 10 * ms'
        self.S_in.w = 'rand() * 0.3' if W is None else W[self.S_in.i[:], self.S_in.j[:]]
        self.nb_entrees = np.bincount(self.S_in.j[:], minlength=N)   # pour la normalisation

        # Inhibition latérale : E_i -> I_i -> tous les E_j (j != i)
        self.S_ei = Synapses(self.exc, self.inh, on_pre='ge_post += 10.4')
        self.S_ei.connect(condition='i == j', p=c['p_ei'])
        self.S_ie = Synapses(self.inh, self.exc, on_pre='gi_post += 17.0')
        self.S_ie.connect(condition='i != j', p=c['p_ie'])

        self.mon = SpikeMonitor(self.exc, record=False)
        self.net = Network(self.entree, self.exc, self.inh,
                           self.S_in, self.S_ei, self.S_ie, self.mon)

    def poids(self):
        """Matrice 784 x N des poids d'entrée (0 là où il n'y a pas de synapse)."""
        W = np.zeros((784, self.cfg['n_neurones']), np.float32)
        W[self.S_in.i[:], self.S_in.j[:]] = self.S_in.w[:]
        return W

    def normaliser(self):
        """Normalisation divisive : somme des poids de chaque neurone = 0.1 x nb entrées."""
        j, w = self.S_in.j[:], self.S_in.w[:]
        somme = np.bincount(j, w, minlength=len(self.nb_entrees))
        self.S_in.w = w * 0.1 * self.nb_entrees[j] / np.maximum(somme[j], 1e-9)

    def presenter(self, x, apprend):
        """Présente l'image x (784 valeurs dans [0, 1]) et renvoie le nombre de
        décharges de chaque excitateur. Comme Diehl & Cook, si le réseau décharge
        moins de 5 fois, on recommence avec une intensité d'entrée plus forte."""
        self.exc.plastique = float(apprend)
        for k in range(5):
            if apprend and self.cfg['normalisation']:
                self.normaliser()
            self.entree.rates = x * (63.75 + 32 * k) * Hz
            avant = self.mon.count[:].copy()
            self.net.run(self.cfg['t_image'] * ms)
            compte = self.mon.count[:] - avant
            # Retour au repos (remplace les 150 ms de repos du code à trous)
            self.exc.v, self.inh.v = -65 * mV, -60 * mV
            self.exc.ge = self.exc.gi = self.inh.ge = self.inh.gi = 0
            self.S_in.xpre = 0
            if compte.sum() >= 5:
                break
        return compte

    def compter(self, X, apprend=False):
        """Décomptes de décharges (n_images x N) pour un ensemble d'images."""
        return np.array([self.presenter(x, apprend) for x in X])


# --- Lecture (couche de sortie) --------------------------------------------------
def lecture(C_ref, y_ref, C, methode='vote'):
    """Prédit la classe des images dont les décomptes sont C (n x N), à partir des
    réponses C_ref à des images d'étiquettes y_ref connues.
      - 'meilleur' : code à trous, 1 seul neurone (le plus actif) par classe ;
      - 'vote'     : Diehl & Cook, chaque neurone est assigné à la classe à laquelle
                     il répond le plus ; on prend la classe dont les neurones
                     assignés déchargent le plus en moyenne ;
      - 'pondere'  : chaque neurone vote pour toutes les classes, en proportion de
                     sa réponse moyenne à chacune."""
    R = np.array([C_ref[y_ref == k].mean(0) for k in range(10)])   # 10 x N
    if methode == 'meilleur':
        return C[:, R.argmax(1)].argmax(1)
    if methode == 'vote':
        classe = np.where(R.max(0) > 0, R.argmax(0), -1)   # neurones muets : pas de vote
        score = np.stack([C[:, classe == k].mean(1) if (classe == k).any()
                          else np.zeros(len(C)) for k in range(10)], 1)
        return score.argmax(1)
    return (C @ (R / (R.sum(0) + 1e-9)).T).argmax(1)


METHODES = ('meilleur', 'vote', 'pondere')


# --- Expérience complète -----------------------------------------------------------
def experience(nom, **modifs):
    """Entraîne un réseau, l'évalue après chaque époque et sauvegarde tout dans
    resultats/<nom>.npz. Renvoie un dictionnaire de résultats (une ligne de table)."""
    cfg = {**DEFAUT, **modifs}
    np.random.seed(cfg['seed'])                       # graine de Brian2 (rand(), Poisson)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = charger_mnist(cfg['n_images'], cfg['seed'])
    r = Reseau(cfg)
    n_tot = cfg['epoques'] * len(Xtr)
    instants = {round(f * (n_tot - 1)) for f in (0, .25, .5, .75, 1)}   # copies des poids
    W_hist, comptes, ligne = [], {}, dict(nom=nom, **modifs)
    ligne['synapses'] = len(r.S_in) + len(r.S_ei) + len(r.S_ie)
    t0, it = time.time(), 0
    for e in range(cfg['epoques']):
        for x in Xtr:
            if it in instants:
                W_hist.append(r.poids())
            r.presenter(x, apprend=True)
            it += 1
            if it % 500 == 0:
                print(f'[{nom}] {it}/{n_tot} images  {time.time() - t0:.0f} s', flush=True)
        ligne['duree_entrainement_s'] = round(time.time() - t0)
        # Évaluation sans plasticité : étiquetage sur une partie du train, puis val/test
        Xet, yet = Xtr[:cfg['n_eval_train']], ytr[:cfg['n_eval_train']]
        C = {'train': r.compter(Xet), 'val': r.compter(Xva), 'test': r.compter(Xte)}
        Y = {'train': yet, 'val': yva, 'test': yte}
        for ens in C:
            comptes[f'C_{ens}_ep{e + 1}'] = C[ens]
            for m in METHODES:
                acc = (lecture(C['train'], yet, C[ens], m) == Y[ens]).mean()
                ligne[f'ep{e + 1}_{ens}_{m}'] = round(acc, 4)
        print(f'[{nom}] époque {e + 1} :',
              {k: v for k, v in ligne.items() if k.startswith(f'ep{e + 1}_') and 'vote' in k},
              flush=True)
    ligne['duree_totale_s'] = round(time.time() - t0)
    os.makedirs(DOSSIER_RES, exist_ok=True)
    np.savez_compressed(os.path.join(DOSSIER_RES, nom + '.npz'),
                        W_hist=np.array(W_hist), theta=r.exc.theta[:] / mV,
                        y_train=yet, y_val=yva, y_test=yte, X_val=Xva[:20],
                        cfg=str(cfg), **comptes)
    return ligne


if __name__ == '__main__':
    # Test rapide : petit réseau, peu d'images
    print(experience('test_rapide', n_neurones=25, n_images=280, n_eval_train=100))
