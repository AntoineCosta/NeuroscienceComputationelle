"""
Partie III - Mesure propre des durées d'exécution. Les durées de resultats.csv sont
faussées par le parallélisme (12 simulations à la fois) ; ici on chronomètre une
configuration SEULE sur quelques images, puis on extrapole à l'expérience complète.

Usage :  python duree.py best_sans_norm [nb_images]     (à lancer machine au repos)
Sortie : une ligne ajoutée à resultats/durees.csv

Auteurs : Antoine Costa
"""
import csv
import os
import sys
import time

from diehl_cook import DEFAUT, DOSSIER_RES, Reseau, charger_mnist
from experiences import EXPERIENCES

nom = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
cfg = {**DEFAUT, **EXPERIENCES[nom]}
(Xtr, _), (Xva, _), (Xte, _) = charger_mnist(cfg['n_images'], cfg['seed'])
r = Reseau(cfg)

t0 = time.time()
for x in Xtr[:n]:
    r.presenter(x, apprend=True)
s_app = (time.time() - t0) / n                  # secondes par image, avec STDP
t0 = time.time()
r.compter(Xva[:n])
s_eval = (time.time() - t0) / n                 # secondes par image, sans plasticité

# Extrapolation : chaque époque = entraînement + évaluation (étiquetage, val, test)
n_eval = cfg['n_eval_train'] + len(Xva) + len(Xte)
total_h = cfg['epoques'] * (len(Xtr) * s_app + n_eval * s_eval) / 3600
ligne = dict(nom=nom, n_neurones=cfg['n_neurones'], epoques=cfg['epoques'],
             s_par_image_apprentissage=round(s_app, 3), s_par_image_evaluation=round(s_eval, 3),
             duree_totale_estimee_h=round(total_h, 2), images_mesurees=n)
print(ligne)

fichier = os.path.join(DOSSIER_RES, 'durees.csv')
nouveau = not os.path.exists(fichier)
with open(fichier, 'a', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=ligne)
    if nouveau:
        w.writeheader()
    w.writerow(ligne)
