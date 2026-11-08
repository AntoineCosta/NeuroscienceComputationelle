"""
Réseau de neurones à impulsions (SNN) pour MNIST, entraîné par rétropropagation.

Architecture : 784 pixels -> 400 neurones LIF cachés -> 10 neurones LIF de sortie

- Neurones : Leaky Integrate-and-Fire (LIF). C'est le modèle bio-inspiré le plus
  simple qui garde l'essentiel (intégration, fuite, seuil, spike, reset) et sa
  courbe fréquence/courant quasi linéaire au-dessus du seuil se prête bien à la
  rétropropagation.
- Codage : en fréquence (rate coding). L'intensité des pixels est injectée comme
  courant dans la couche cachée ; la couche cachée excite la sortie par de vrais
  spikes (synapses Brian2). La classe prédite = neurone de sortie qui décharge le plus.
- Synapses exponentielles : chaque spike injecte un courant qui décroît avec
  tau_syn, comme un courant post-synaptique réel (au lieu d'un saut instantané).
- Inhibition latérale en sortie : chaque neurone de sortie inhibe les 9 autres
  (motif « winner-take-all » du cortex), ce qui départage les classes.
- Fréquences réalistes : une pénalité dans la perte empêche les neurones de
  dépasser R_MAX (quelques dizaines de Hz, comme dans le cortex). L'information
  est portée par une population de neurones peu actifs (codage par population)
  plutôt que par quelques neurones saturés.
- Apprentissage : la passe avant est simulée dans Brian2. La passe arrière se fait
  en numpy sur les taux de décharge mesurés, avec une dérivée de substitution
  (surrogate gradient) pour la non-linéarité du LIF.

Pour aller vite, un mini-batch de B images est simulé d'un coup : B copies du
réseau en parallèle dans les mêmes NeuronGroup (blocs indépendants).
"""
import gzip
import os
import urllib.request

import numpy as np
from brian2 import (NeuronGroup, Synapses, SpikeMonitor, Network, prefs,
                    defaultclock, ms, start_scope)

prefs.codegen.target = 'numpy'
defaultclock.dt = 0.5 * ms

# --- Hyperparamètres ---------------------------------------------------------
N_IN, N_HID, N_OUT = 784, 400, 10
BATCH = 100
T_SIM = 100 * ms         # durée de présentation d'une image
TAU = 10 * ms            # constante de membrane
TAU_SYN = 5 * ms         # constante de décroissance du courant synaptique
W_INH = 0.5              # force de l'inhibition latérale en sortie
R_MAX = 50.0             # Hz, fréquence au-delà de laquelle on pénalise
LAMBDA_RATE = 0.003      # poids de la pénalité de fréquence
LR = 0.01                # Adam, décroît en cosinus jusqu'à 0
LOGIT_SCALE = 3.0        # logits = LOGIT_SCALE * tau * taux de sortie
EPOCHS = 3
N_TRAIN = 60000
N_TEST = 10000
SEED = 0

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


# --- Données -----------------------------------------------------------------
def load_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    def get(name, offset):
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f'Téléchargement de {name}...')
            urllib.request.urlretrieve(URL + name, path)
        with gzip.open(path, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=offset)
    x_tr = get('train-images-idx3-ubyte.gz', 16).reshape(-1, 784) / 255.0
    y_tr = get('train-labels-idx1-ubyte.gz', 8)
    x_te = get('t10k-images-idx3-ubyte.gz', 16).reshape(-1, 784) / 255.0
    y_te = get('t10k-labels-idx1-ubyte.gz', 8)
    return x_tr, y_tr, x_te, y_te


# --- Réseau Brian2 -----------------------------------------------------------
class SpikingNet:
    """B copies du réseau 784-N_HID-10 simulées en parallèle."""

    def __init__(self, batch):
        self.B = batch
        start_scope()
        ns = {'tau': TAU, 'tau_syn': TAU_SYN}
        # LIF : dv/dt = (I - v)/tau, seuil 1, reset 0.
        # Couche cachée : I = courant d'entrée (W1 · pixels), fixé avant chaque run.
        self.hid = NeuronGroup(batch * N_HID, 'dv/dt = (I - v)/tau : 1\nI : 1',
                               threshold='v > 1', reset='v = 0',
                               refractory=2 * ms, method='exact', namespace=ns)
        # Couche de sortie : courant synaptique I_syn alimenté par les spikes cachés.
        self.out = NeuronGroup(batch * N_OUT,
                               '''dv/dt = (I_syn - v)/tau : 1
                                  dI_syn/dt = -I_syn/tau_syn : 1''',
                               threshold='v > 1', reset='v = 0',
                               refractory=2 * ms, method='exact', namespace=ns)
        # Connexions tout-à-tout à l'intérieur de chaque bloc (une image = un bloc)
        b, h, o = np.meshgrid(np.arange(batch), np.arange(N_HID), np.arange(N_OUT),
                              indexing='ij')
        self.syn = Synapses(self.hid, self.out, 'w : 1', on_pre='I_syn_post += w')
        self.syn.connect(i=(b * N_HID + h).ravel(), j=(b * N_OUT + o).ravel())
        # Inhibition latérale : chaque sortie inhibe les autres sorties de son bloc
        b, i, j = np.meshgrid(np.arange(batch), np.arange(N_OUT), np.arange(N_OUT),
                              indexing='ij')
        keep = (i != j).ravel()
        self.inh = Synapses(self.out, self.out, on_pre='I_syn_post -= w_inh',
                            namespace={'w_inh': W_INH})
        self.inh.connect(i=(b * N_OUT + i).ravel()[keep],
                         j=(b * N_OUT + j).ravel()[keep])
        self.mon_h = SpikeMonitor(self.hid, record=False)
        self.mon_o = SpikeMonitor(self.out, record=False)
        self.net = Network(self.hid, self.out, self.syn, self.inh,
                           self.mon_h, self.mon_o)
        self.net.store()

    def run(self, I_hid, W2):
        """Simule le batch et renvoie les taux de décharge (Hz) cachés et de sortie."""
        self.net.restore()
        self.hid.I = I_hid.ravel()
        # Ordre des synapses = (b, h, o) -> on répète W2 (h, o) pour chaque image
        self.syn.w = np.tile(W2.ravel(), self.B)
        self.net.run(T_SIM)
        T = float(T_SIM)
        r_h = np.array(self.mon_h.count).reshape(self.B, N_HID) / T
        r_o = np.array(self.mon_o.count).reshape(self.B, N_OUT) / T
        return r_h, r_o


# --- Entraînement ------------------------------------------------------------
def surrogate(I):
    """Dérivée de substitution du LIF (fast sigmoid centrée sur le seuil 1).
    Non nulle même sous le seuil : un neurone silencieux peut encore apprendre."""
    return 1.0 / (1.0 + 2.0 * np.abs(I - 1.0)) ** 2


def rate_penalty_grad(r):
    """Gradient de LAMBDA * mean_batch(sum_neurones ((r - R_MAX)+ / R_MAX)^2)."""
    return 2 * LAMBDA_RATE * np.maximum(r - R_MAX, 0) / R_MAX ** 2 / len(r)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Adam:
    """Optimiseur Adam : pas adapté à chaque poids, bien plus stable que la
    descente de gradient simple sur des gradients bruités par les spikes."""

    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def evaluate(net, W1, W2, x, y):
    correct, rates = 0, []
    for k in range(0, len(x), BATCH):
        r_h, r_o = net.run(x[k:k + BATCH] @ W1, W2)
        correct += np.sum(r_o.argmax(1) == y[k:k + BATCH])
        rates.append((r_h.mean(), r_o.mean()))
    return correct / len(x), np.mean(rates, axis=0)


def main():
    rng = np.random.default_rng(SEED)
    x_tr, y_tr, x_te, y_te = load_mnist()
    x_tr, y_tr = x_tr[:N_TRAIN], y_tr[:N_TRAIN]
    x_te, y_te = x_te[:N_TEST], y_te[:N_TEST]

    # Initialisation mise à l'échelle par le nombre d'entrées (fan-in)
    W1 = rng.normal(0, 6 / np.sqrt(N_IN), (N_IN, N_HID))    # pixels -> courant caché
    W2 = rng.normal(0, 2 / np.sqrt(N_HID), (N_HID, N_OUT))    # caché -> sortie
    net = SpikingNet(BATCH)
    opt = Adam([W1, W2], LR)
    n_steps = EPOCHS * int(np.ceil(len(x_tr) / BATCH))

    tau, tau_syn = float(TAU), float(TAU_SYN)
    for epoch in range(EPOCHS):
        perm = rng.permutation(len(x_tr))
        for step, k in enumerate(range(0, len(x_tr), BATCH)):
            idx = perm[k:k + BATCH]
            x, y = x_tr[idx], y_tr[idx]

            # Passe avant spiking (Brian2)
            I_h = x @ W1
            r_h, r_o = net.run(I_h, W2)

            # Passe arrière (numpy) sur les taux de décharge.
            # Perte = entropie croisée sur les taux de sortie + pénalité de fréquence.
            p = softmax(r_o * tau * LOGIT_SCALE)
            g_ro = (p - np.eye(N_OUT)[y]) / len(x) * tau * LOGIT_SCALE
            g_ro += rate_penalty_grad(r_o)
            # Courant synaptique moyen reçu par la sortie (excitation - inhibition),
            # la pente du LIF est approchée par surrogate(I) / tau.
            I_o = tau_syn * (r_h @ W2 - W_INH * (r_o.sum(1, keepdims=True) - r_o))
            d_o = g_ro * surrogate(I_o) / tau
            gW2 = (tau_syn * r_h).T @ d_o
            g_rh = d_o @ (tau_syn * W2).T + rate_penalty_grad(r_h)
            d_h = g_rh * surrogate(I_h) / tau
            gW1 = x.T @ d_h

            # Décroissance en cosinus du taux d'apprentissage
            opt.lr = LR * 0.5 * (1 + np.cos(np.pi * opt.t / n_steps))
            opt.step([gW1, gW2])

            if step % 100 == 0:
                acc = np.mean(r_o.argmax(1) == y)
                loss = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))
                print(f'epoch {epoch} step {step:4d}  loss {loss:.3f}  '
                      f'acc batch {acc:.2f}  cachés {r_h.mean():.0f} Hz  '
                      f'sortie {r_o.mean():.0f} Hz')

        acc, (f_h, f_o) = evaluate(net, W1, W2, x_te, y_te)
        print(f'=== epoch {epoch} : précision test {acc:.3f}  '
              f'(cachés {f_h:.0f} Hz, sortie {f_o:.0f} Hz) ===')

    np.savez(os.path.join(DATA_DIR, 'poids.npz'), W1=W1, W2=W2)
    return acc


if __name__ == '__main__':
    main()
