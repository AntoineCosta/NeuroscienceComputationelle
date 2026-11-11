"""
Partie I - Formes de STDP (Caporale & Dan 2008, fig. 1) sur le réseau de Song,
Miller & Abbott (2000) : 1000 entrées Poisson excitatrices -> 1 neurone LIF.

Un seul noyau STDP paramétré couvre les 4 formes demandées :
    dt = t_post - t_pre > 0 :  dw = A1 exp(-dt/t1) + A2 exp(-dt/t2)
    dt < 0                  :  dw = A3 exp( dt/t3)
Chaque exponentielle = une trace (a1, a2 côté pré, b côté post), donc chaque
forme n'est qu'un dictionnaire de paramètres (FORMES, amplitudes en fraction de gmax).

Expériences : 4 formes x 3 fréquences (6, 10, 15 Hz), avec la STDP standard puis
avec la STDP "robuste" : traces saturantes + mise à l'échelle synaptique
homéostatique (taux post ramené vers 10 Hz quelle que soit la fréquence d'entrée).
Sorties : resultats/partie1/*.png et partie1.csv

Auteurs : Antoine Costa
"""
import os
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import (NeuronGroup, PoissonGroup, Synapses, SpikeMonitor, StateMonitor,
                    Network, network_operation, prefs, defaultclock, ms, mV, Hz, second)

prefs.codegen.target = 'numpy'
defaultclock.dt = 0.5 * ms
SORTIE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resultats', 'partie1')

N, GMAX, DUREE = 1000, 0.01, 200 * second
R_CIBLE, GAMMA = 10, 0.1     # STDP robuste : taux post visé (Hz) et vitesse de l'homéostasie
FREQUENCES = (6, 10, 15)
COULEURS = {6: 'tab:blue', 10: 'tab:orange', 15: 'tab:green'}
FORMES = {   # amplitudes (fraction de gmax) et constantes de temps (ms)
    # a) excitateur -> excitateur
    'I_EE':  dict(A1=+0.010, t1=20, A2=0.0,    t2=1,   A3=-0.0105, t3=20),  # hebbien classique (Song)
    'II_EE': dict(A1=+0.015, t1=15, A2=-0.005, t2=60,  A3=-0.0105, t3=15),  # LTP courte puis creux LTD ~50 ms
    # b) excitateur -> inhibiteur
    'I_EI':  dict(A1=-0.012, t1=25, A2=+0.0015, t2=200, A3=+0.0100, t3=40), # anti-hebbien : LTD pour 0 < dt < ~60 ms, LTP pour dt < 0
    'II_EI': dict(A1=-0.010, t1=10, A2=0.0,    t2=1,   A3=0.0,     t3=1),   # LTD seulement pour 0 < dt < ~20 ms
}


def noyau(dt, A1, t1, A2, t2, A3, t3):
    """Fenêtre STDP théorique dw(dt) (en fraction de gmax)."""
    dt = np.asarray(dt, float)
    return np.where(dt > 0, A1 * np.exp(-dt / t1) + A2 * np.exp(-dt / t2),
                    A3 * np.exp(np.minimum(dt, 0) / t3))


def simuler(forme, freq, robuste=False, seed=0):
    """Simule DUREE secondes et renvoie l'évolution des poids et du taux post."""
    np.random.seed(seed)
    p = FORMES[forme]
    neurone = NeuronGroup(1, '''dv/dt = (ge * (0*mV - v) + (-74*mV) - v) / (10*ms) : volt
                                dge/dt = -ge / (5*ms) : 1''',
                          threshold='v > -54*mV', reset='v = -60*mV', method='euler')
    neurone.v = -74 * mV
    entree = PoissonGroup(N, rates=freq * Hz)
    # Traces saturantes (robuste) : "a = A" au lieu de "a += A" -> seule la paire de
    # décharges la plus proche compte, donc moins de dépendance au taux de décharge.
    op = '=' if robuste else '+='
    S = Synapses(entree, neurone,
                 '''w : 1
                    da1/dt = -a1 / t1 : 1 (event-driven)
                    da2/dt = -a2 / t2 : 1 (event-driven)
                    db/dt  = -b  / t3 : 1 (event-driven)''',
                 on_pre=f'''ge_post += w
                            a1 {op} A1 * gmax
                            a2 {op} A2 * gmax
                            w = clip(w + b, 0, gmax)''',
                 on_post=f'''b {op} A3 * gmax
                             w = clip(w + a1 + a2, 0, gmax)''',
                 namespace=dict(gmax=GMAX, A1=p['A1'], A2=p['A2'], A3=p['A3'],
                                t1=p['t1'] * ms, t2=p['t2'] * ms, t3=p['t3'] * ms))
    S.connect()
    S.w = 'rand() * gmax'
    mon_w = StateMonitor(S, 'w', record=True, dt=1 * second)
    mon_s = SpikeMonitor(neurone)
    deja = [0]

    @network_operation(dt=1 * second)
    def homeostasie():
        # Mise à l'échelle synaptique (Turrigiano) : toutes les secondes, on multiplie
        # les poids pour ramener le taux post vers R_CIBLE. Multiplicatif -> les
        # rapports entre poids (ce qu'a appris la STDP) sont conservés.
        r = mon_s.num_spikes - deja[0]          # décharges pendant la dernière seconde
        deja[0] = mon_s.num_spikes
        if robuste:
            facteur = np.clip(1 + GAMMA * (R_CIBLE - r) / R_CIBLE, 0.5, 1.5)
            S.w = np.clip(S.w[:] * facteur, 0, GMAX)

    Network(neurone, entree, S, homeostasie, mon_w, mon_s).run(DUREE)
    t_spk = mon_s.t / second
    taux = np.histogram(t_spk, bins=int(DUREE / second), range=(0, DUREE / second))[0]
    return dict(forme=forme, freq=freq, robuste=robuste,
                w=mon_w.w[:] / GMAX, taux=taux)   # w : (N, temps) en fraction de gmax


def resume(r):
    """Une ligne de la table de synthèse."""
    w_fin = r['w'][:, -1]
    # Stabilisation : premier instant à partir duquel moyenne ET écart-type des poids
    # restent à moins de 0.02 gmax de leur valeur finale.
    traj = np.stack([r['w'].mean(0), r['w'].std(0)])
    instable = np.abs(traj - traj[:, -1:]).max(0) > 0.02
    t_stab = len(instable) - np.argmax(instable[::-1]) if instable.any() else 0
    return dict(forme=r['forme'], freq_Hz=r['freq'], robuste=r['robuste'],
                w_moyen=round(w_fin.mean(), 3), w_ecart_type=round(w_fin.std(), 3),
                frac_w_haut=round((w_fin > 0.9).mean(), 3),
                frac_w_bas=round((w_fin < 0.1).mean(), 3),
                taux_post_Hz=round(r['taux'][-20:].mean(), 1), t_stabilisation_s=int(t_stab))


TITRES = {'I_EE': 'Excitateur vers excitateur I', 'II_EE': 'Excitateur vers excitateur II',
          'I_EI': 'Excitateur vers inhibiteur I', 'II_EI': 'Excitateur vers inhibiteur II'}


def fig_noyaux():
    """Les 4 fenêtres STDP (grille 2x2), normalisées par leur amplitude maximale."""
    os.makedirs(SORTIE, exist_ok=True)
    dt = np.concatenate([np.linspace(-100, -0.01, 500), np.linspace(0.01, 100, 500)])
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (nom, p) in zip(axes.flat, FORMES.items()):
        dw = noyau(dt, **p)
        ax.plot(dt, dw / np.abs(dw).max(), c='k')
        ax.axhline(0, c='grey', lw=.5)
        ax.set(title=TITRES[nom], xlabel='Δt = t_post − t_pre (ms)', ylabel='Variation relative')
    fig.tight_layout(); fig.savefig(os.path.join(SORTIE, 'noyaux.png'), dpi=120)


def figures(res):
    # 1. Les 4 fenêtres STDP
    fig_noyaux()
    # 2. Histogrammes finaux et évolution du poids moyen, pour chaque variante
    for robuste in (False, True):
        sel = [r for r in res if r['robuste'] == robuste]
        fig, axes = plt.subplots(3, 4, figsize=(15, 9))
        for k, nom in enumerate(FORMES):
            for r in (r for r in sel if r['forme'] == nom):
                # 6 Hz en trait épais : visible même quand 10 Hz donne exactement la même courbe
                st = dict(color=COULEURS[r['freq']], lw=3 if r['freq'] == 6 else 1.3)
                muet = ' (muet)' if r['taux'].sum() == 0 else ''
                axes[0, k].hist(r['w'][:, -1], bins=40, range=(0, 1), histtype='step',
                                label=f"{r['freq']} Hz{muet}", **st)
                axes[1, k].plot(r['w'].mean(0), **st)
                axes[1, k].plot(r['w'].std(0), ls=':', **st)
                lisse = np.convolve(r['taux'], np.ones(10) / 10, mode='valid')   # moyenne sur 10 s
                axes[2, k].plot(np.arange(10, 10 + len(lisse)), lisse, **st)
            axes[0, k].set(title=nom, xlabel='w final / gmax')
            axes[0, k].legend(fontsize=8)
            axes[1, k].set(ylim=(0, 1))
            axes[2, k].set(xlabel='temps (s)', ylim=(0, None))
            if robuste:
                axes[2, k].axhline(R_CIBLE, c='grey', ls='--', lw=1)
        axes[0, 0].set_ylabel('nb de synapses')
        axes[1, 0].set_ylabel('w / gmax : moyenne (—), écart-type (···)')
        axes[2, 0].set_ylabel('taux post (Hz, moyenne sur 10 s)')
        fig.suptitle('STDP ' + ('robuste (homéostasie + traces saturantes)' if robuste else 'standard'))
        fig.tight_layout()
        fig.savefig(os.path.join(SORTIE, f"poids_{'robuste' if robuste else 'standard'}.png"), dpi=120)


def _job(args):
    return simuler(*args)


if __name__ == '__main__':
    jobs = [(f, fr, rob) for rob in (False, True) for f in FORMES for fr in FREQUENCES]
    with Pool(max(1, min(len(jobs), os.cpu_count() - 2))) as pool:
        res = pool.map(_job, jobs)
    figures(res)
    import csv
    lignes = [resume(r) for r in res]
    with open(os.path.join(SORTIE, 'partie1.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=lignes[0]); w.writeheader(); w.writerows(lignes)
    for l in lignes:
        print(l)
