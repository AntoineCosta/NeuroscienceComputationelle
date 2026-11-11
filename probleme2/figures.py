"""
Parties III et IV - Figures, à partir des fichiers produits par experiences.py
(aucun réentraînement : seule la figure d'activation re-simule quelques images).

  precisions.png         précision val/test par expérience et par méthode de lecture
  iterations.png         précision train/val/test en fonction du nb d'époques (exp. 'base')
  champs_<exp>.png       champs récepteurs (poids d'entrée remis en 28x28) de chaque neurone
  evolution_<exp>.png    champs récepteurs de 2 neurones au cours de l'apprentissage
  accord_<exp>.png       courbes d'accord : réponse moyenne de quelques neurones à chaque classe
  activation_<exp>.png   décharges des neurones actifs pour 3 images de validation (EXEMPLES)

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

from brian2 import SpikeMonitor, seed

from diehl_cook import DOSSIER_RES, METHODES, Reseau, charger_mnist, ms

SORTIE = os.path.join(DOSSIER_RES, 'figures')
EXEMPLES = (2, 6, 63)   # images de validation : réussite nette, réussite de justesse, échec


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

    # 4. Activité : on re-simule quelques images de validation (graine fixe) et on
    #    trace une ligne par neurone qui a déchargé (les autres restent muets), avec
    #    son champ récepteur, ses décharges dans le temps, sa part de vote par classe
    #    (lecture pondere) et le score par classe, découpé selon le neurone qui vote.
    W, (C_tr, y_tr) = W_hist[-1], (d[f'C_train_ep{ep}'], d['y_train'])
    R = np.array([C_tr[y_tr == k].mean(0) for k in range(10)])        # 10 x N
    classe, P = R.argmax(0), R / (R.sum(0) + 1e-9)
    X_val = charger_mnist(cfg['n_images'], cfg['seed'])[1][0]
    sims = []
    for k in EXEMPLES:                       # réseau neuf par image : rien ne déborde d'une image à l'autre
        seed(cfg['seed']); r = Reseau(cfg, W=W, theta=d['theta'])
        mon = SpikeMonitor(r.exc); r.net.add(mon)
        seed(k); compte = r.presenter(X_val[k], apprend=False)
        sims.append((k, compte, mon.t / ms, mon.i[:]))
    coul = plt.cm.tab10(np.arange(10))
    fig = plt.figure(figsize=(14, sum(0.75 * (c > 0).sum() + 1.2 for _, c, _, _ in sims)))
    sfigs = fig.subfigures(len(sims), 1, height_ratios=[0.75 * (c > 0).sum() + 1.2 for _, c, _, _ in sims])
    for sf, (k, compte, t, i) in zip(np.atleast_1d(sfigs), sims):
        actifs = np.flatnonzero(compte)
        actifs = actifs[np.lexsort((-compte[actifs], classe[actifs]))]   # triés par classe
        score = compte @ P.T
        t_max = max(t.max() if len(t) else 0, cfg['t_image']) + 5
        g = sf.add_gridspec(max(len(actifs), 1), 5, width_ratios=[2.2, 0.6, 6, 1.4, 3.2], hspace=0.15)
        ax = sf.add_subplot(g[:, 0]); ax.imshow(X_val[k].reshape(28, 28), cmap='gray_r'); ax.axis('off')
        ok = score.argmax() == y_val[k]
        ax.set_title(f'image {k} : vrai {y_val[k]}, prédit {score.argmax()}', color='k' if ok else 'r')
        for row, n in enumerate(actifs):
            a = sf.add_subplot(g[row, 1]); a.imshow(W[:, n].reshape(28, 28), cmap='hot_r')
            a.set(xticks=[], yticks=[])
            for s in a.spines.values(): s.set(color=coul[classe[n]], linewidth=3)
            a = sf.add_subplot(g[row, 2])
            a.axhspan(0, 1, color=coul[classe[n]], alpha=0.12, lw=0)
            a.vlines(t[i == n], 0.15, 0.85, color=coul[classe[n]], lw=2.5)
            for e in range(1, int(t.max() // cfg['t_image']) + 1):    # nouvel essai à +32 Hz
                a.axvline(e * cfg['t_image'], color='k', ls='--', lw=0.8)
            a.set(xlim=(0, t_max), ylim=(0, 1), yticks=[])
            a.set_ylabel(f'n{n}\nclasse {classe[n]}', rotation=0, ha='right', va='center',
                         color=coul[classe[n]], fontsize=9)
            parts = ', '.join(f'{c} : {P[c, n]:.0%}' for c in np.argsort(P[:, n])[::-1][:3] if P[c, n] >= 0.05)
            a.text(1.02, 0.5, f'{compte[n]} déch.\nvote {parts}', transform=a.transAxes, va='center', fontsize=8)
            if row == 0: a.set_title(f'{len(actifs)} neurones actifs sur {len(compte)}')
            if row < len(actifs) - 1: a.set_xticklabels([])
            else: a.set_xlabel('temps (ms)')
        a, gauche = sf.add_subplot(g[:, 4]), np.zeros(10)
        for n in actifs:                     # 1 segment = la contribution d'un neurone actif
            a.barh(range(10), compte[n] * P[:, n], left=gauche, color=coul[classe[n]], edgecolor='w', lw=0.5)
            gauche += compte[n] * P[:, n]
        a.invert_yaxis()
        a.set(yticks=range(10), xticks=[], ylabel='classe', title='score pondere (couleur = neurone)')
        sf.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.15, wspace=0.35)
    fig.savefig(os.path.join(SORTIE, f'activation_{nom}.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(SORTIE, exist_ok=True)
    if os.path.exists(os.path.join(DOSSIER_RES, 'resultats.csv')):
        fig_precisions()
    for nom in sys.argv[1:] or ['base', 'sans_homeo', 'p_ie_0.1']:
        if os.path.exists(os.path.join(DOSSIER_RES, nom + '.npz')):
            figs_experience(nom)
            print('Figures faites :', nom)
