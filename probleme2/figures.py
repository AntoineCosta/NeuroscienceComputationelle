"""
Parties III et IV - Figures, à partir des fichiers produits par experiences.py
(aucun réentraînement : seule la carte d'activation re-simule UNE image).

  precisions.png         précision val/test par expérience et par méthode de lecture
  iterations.png         précision train/val/test en fonction du nb d'époques (exp. 'base')
  champs_<exp>.png       champs récepteurs (poids d'entrée remis en 28x28) de chaque neurone
  evolution_<exp>.png    champs récepteurs de 2 neurones au cours de l'apprentissage
  accord_<exp>.png       courbes d'accord : réponse moyenne de quelques neurones à chaque classe
  activation_<exp>.png   carte d'activation (1 pixel = 1 neurone) au cours de la présentation

Usage : python figures.py [exp1 exp2 ...]   (par défaut : base sans_homeo p_ie_0.1)

Auteurs : Antoine Costa
"""
import ast
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diehl_cook import DOSSIER_RES, METHODES, Reseau, ms

SORTIE = os.path.join(DOSSIER_RES, 'figures')


def sauver(fig, nom):
    fig.tight_layout()
    fig.savefig(os.path.join(SORTIE, nom), dpi=110)
    plt.close(fig)


def grille(W, titre=''):
    """Assemble les colonnes de W (784 x N) en une seule image de champs récepteurs."""
    n = int(np.ceil(np.sqrt(W.shape[1])))
    img = np.full((n * 29, n * 29), np.nan)
    for k in range(W.shape[1]):
        r, c = divmod(k, n)
        img[r * 29:r * 29 + 28, c * 29:c * 29 + 28] = W[:, k].reshape(28, 28)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap='hot_r'); ax.axis('off'); ax.set_title(titre)
    return fig


def fig_precisions():
    lignes = list(csv.DictReader(open(os.path.join(DOSSIER_RES, 'resultats.csv'), encoding='utf-8')))
    noms = [l['nom'] for l in lignes]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, ens in zip(axes, ('val', 'test')):
        for k, m in enumerate(METHODES):
            acc = [float(l[f'ep1_{ens}_{m}']) for l in lignes]
            ax.bar(np.arange(len(noms)) + 0.27 * (k - 1), acc, 0.27, label=m)
        ax.set_xticks(range(len(noms)), noms, rotation=45, ha='right')
        ax.set(title=f'Précision ({ens}, après 1 époque)', ylim=(0, 1))
    axes[0].legend(title='lecture')
    sauver(fig, 'precisions.png')
    base = next((l for l in lignes if l['nom'] == 'base'), None)
    if base:
        ep = list(range(1, int(base['epoques']) + 1))
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for ens in ('train', 'val', 'test'):
            ax.plot(ep, [float(base[f'ep{e}_{ens}_pondere']) for e in ep], 'o-', label=ens)
        ax.set(xlabel="nombre d'époques", ylabel='précision (lecture pondere)', xticks=ep)
        ax.legend()
        sauver(fig, 'iterations.png')


def figs_experience(nom):
    d = np.load(os.path.join(DOSSIER_RES, nom + '.npz'))
    cfg = ast.literal_eval(str(d['cfg']))
    W_hist = d['W_hist']                           # (instants, 784, N)
    ep = max(int(k.split('ep')[1]) for k in d.files if k.startswith('C_val'))
    C_val, y_val = d[f'C_val_ep{ep}'], d['y_val']

    # 1. Champs récepteurs finaux de tous les neurones
    sauver(grille(W_hist[-1], f'Champs récepteurs - {nom}'), f'champs_{nom}.png')

    # 2. Évolution des champs récepteurs des 2 neurones les plus actifs
    actifs = np.argsort(C_val.sum(0))[::-1][:2]
    fig, axes = plt.subplots(2, len(W_hist), figsize=(2 * len(W_hist), 4.3))
    for r, n in enumerate(actifs):
        for c, W in enumerate(W_hist):
            axes[r, c].imshow(W[:, n].reshape(28, 28), cmap='hot_r'); axes[r, c].axis('off')
            axes[0, c].set_title(f'{100 * c // (len(W_hist) - 1)} % appris')
        axes[r, 0].text(-5, 14, f'n{n}', ha='right', va='center')
    sauver(fig, f'evolution_{nom}.png')

    # 3. Courbes d'accord : réponse moyenne (validation) de 6 neurones à chaque classe
    R = np.array([C_val[y_val == k].mean(0) for k in range(10)])     # 10 x N
    choix = np.argsort(R.max(0))[::-1][:6]
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8), sharey=True)
    for ax, n in zip(axes, choix):
        ax.bar(range(10), R[:, n]); ax.set(title=f'neurone {n}', xticks=range(10), xlabel='classe')
        ins = ax.inset_axes([0.6, 0.55, 0.4, 0.45])
        ins.imshow(W_hist[-1][:, n].reshape(28, 28), cmap='hot_r'); ins.axis('off')
    axes[0].set_ylabel('décharges / image')
    sauver(fig, f'accord_{nom}.png')

    # 4. Carte d'activation : on re-simule 1 image de validation et on découpe la
    #    présentation en 5 fenêtres ; chaque pixel = nb de décharges d'un neurone.
    from brian2 import SpikeMonitor
    r = Reseau(cfg, W=W_hist[-1], theta=d['theta'])
    mon = SpikeMonitor(r.exc); r.net.add(mon)
    x = d['X_val'][0]
    r.presenter(x, apprend=False)
    N, n = cfg['n_neurones'], int(np.ceil(np.sqrt(cfg['n_neurones'])))
    bords = np.linspace(0, float(mon.t[-1] / ms) + 1 if len(mon.t) else 1, 6)
    fig, axes = plt.subplots(1, 6, figsize=(15, 2.8))
    axes[0].imshow(x.reshape(28, 28), cmap='gray_r'); axes[0].set_title(f'entrée ({y_val[0]})')
    for k, ax in enumerate(axes[1:]):
        dans = (mon.t / ms >= bords[k]) & (mon.t / ms < bords[k + 1])
        carte = np.zeros(n * n); carte[:N] = np.bincount(mon.i[dans], minlength=N)
        ax.imshow(carte.reshape(n, n), cmap='viridis')
        ax.set_title(f'{bords[k]:.0f}-{bords[k + 1]:.0f} ms')
    for ax in axes: ax.axis('off')
    sauver(fig, f'activation_{nom}.png')


if __name__ == '__main__':
    os.makedirs(SORTIE, exist_ok=True)
    if os.path.exists(os.path.join(DOSSIER_RES, 'resultats.csv')):
        fig_precisions()
    for nom in sys.argv[1:] or ['base', 'sans_homeo', 'p_ie_0.1']:
        if os.path.exists(os.path.join(DOSSIER_RES, nom + '.npz')):
            figs_experience(nom)
            print('Figures faites :', nom)
