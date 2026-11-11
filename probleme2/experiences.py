"""
Partie III - Expériences MNIST. Chaque expérience = la configuration par défaut de
diehl_cook.DEFAUT modifiée sur un seul paramètre (sauf best_* et n800, qui combinent
les leviers favorables). Les expériences tournent en
parallèle (un processus par expérience) ; chaque ligne de résultats est ajoutée à
resultats/resultats.csv et les données brutes vont dans resultats/<nom>.npz.

Usage :  python experiences.py            (toutes les expériences)
         python experiences.py base n400  (seulement celles nommées)

Auteurs : Antoine Costa
"""
import csv
import os
import sys
import traceback
from multiprocessing import Pool

from diehl_cook import experience, DOSSIER_RES

EXPERIENCES = {
    'base':          dict(epoques=3),          # impact du nombre d'itérations (éval. après chaque époque)
    'n25':           dict(n_neurones=25),      # impact du nombre de neurones
    'n50':           dict(n_neurones=50),
    'n200':          dict(n_neurones=200),
    'n400':          dict(n_neurones=400),
    'sans_homeo':    dict(homeo=False),        # impact de l'homéostasie
    'theta_0.05':    dict(theta_plus=0.05),    # valeur de l'article
    'sans_norm':     dict(normalisation=False),
    'p_ie_0.5':      dict(p_ie=0.5),           # densité des connexions
    'p_ie_0.1':      dict(p_ie=0.1),
    'p_ei_0.5':      dict(p_ei=0.5),
    'p_in_0.5':      dict(p_in=0.5),
    # Meilleure configuration : on combine les leviers favorables trouvés ci-dessus,
    # et on étiquette les neurones sur les 5000 images d'entraînement (au lieu de 1000)
    'best_norm':      dict(n_neurones=400, epoques=3, n_eval_train=5000),
    'best_sans_norm': dict(n_neurones=400, epoques=3, n_eval_train=5000, normalisation=False),
    'n800':           dict(n_neurones=800, n_eval_train=5000),
}


def lancer(nom):
    try:
        return experience(nom, **EXPERIENCES[nom])
    except Exception:                            # une expérience qui plante n'arrête pas les autres
        print(f'[{nom}] ERREUR :\n{traceback.format_exc()}', flush=True)
        return None


if __name__ == '__main__':
    noms = sys.argv[1:] or list(EXPERIENCES)
    os.makedirs(DOSSIER_RES, exist_ok=True)
    fichier = os.path.join(DOSSIER_RES, 'resultats.csv')
    with Pool(max(1, min(len(noms), os.cpu_count() - 1))) as pool:
        for ligne in pool.imap_unordered(lancer, noms):
            if ligne is None:
                continue
            # Les colonnes varient (époques, paramètres modifiés) : on réécrit tout le CSV
            lignes = list(csv.DictReader(open(fichier, encoding='utf-8'))) if os.path.exists(fichier) else []
            lignes = [l for l in lignes if l['nom'] != ligne['nom']] + [ligne]
            colonnes = list(dict.fromkeys(k for l in lignes for k in l))
            with open(fichier, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=colonnes, restval='')
                w.writeheader()
                w.writerows(lignes)
            print('Terminé :', ligne['nom'], flush=True)
