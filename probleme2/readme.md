# GEI723 – Problème 2 : STDP

Auteurs : Antoine Costa

Le code est écrit pour être court et lisible. Chaque script fait une seule partie du problème, et
toutes les analyses se font **hors ligne** sur des résultats sauvegardés : on simule une fois, puis
on trace autant de figures qu'on veut sans réentraîner.

| Fichier | Partie | Rôle |
|---|---|---|
| `partie1_stdp.py` | I | Les 4 formes de STDP (Caporale & Dan) sur le réseau de Song et al. (2000), à 6, 10 et 15 Hz, en version standard et robuste |
| `diehl_cook.py` | II | Réseau de Diehl & Cook : entrée Poisson, couches E/I, STDP éq. (3), homéostasie, normalisation, lectures |
| `experiences.py` | III | Grille d'expériences lancée en parallèle, écrit `resultats/resultats.csv` |
| `figures.py` | III-IV | Précisions, champs récepteurs, évolution, courbes d'accord, cartes d'activation |
| `duree.py` | III | Chronomètre une configuration seule sur 300 images, écrit `resultats/durees.csv` |

## Installation

Python 3.11, `brian2`, `numpy`, `matplotlib` (aucune autre dépendance). MNIST est téléchargé
automatiquement dans `data/` au premier lancement.

Brian2 utilise la cible `numpy` (environ 0.45 s par image seul sur le processeur). La cible
`cython` a été testée et est plus lente ici (0.55 s par image) : le temps est dominé par le
surcoût Python de chaque pas de temps, pas par les calculs eux-mêmes.

## Reproduire les résultats

```bash
python partie1_stdp.py     # ~3 min  -> resultats/partie1/
python experiences.py      # ~7 h sur 12 cœurs (une expérience par processus) -> resultats/
python figures.py          # -> resultats/figures/
python duree.py best_sans_norm   # ~3 min, machine au repos -> resultats/durees.csv
```

Toutes les graines aléatoires sont fixées (`seed=0`). Une seule expérience se relance avec
`python experiences.py base`.

## Choix d'implémentation

### Partie I
- **Noyau unique** : Δt > 0 → `A1·e^(−Δt/τ1) + A2·e^(−Δt/τ2)`, Δt < 0 → `A3·e^(Δt/τ3)`. Il y a
  une trace par exponentielle, et chaque forme n'est qu'un dictionnaire de paramètres (`FORMES`).
- **STDP robuste** (question 2b) :
  1. traces saturantes (`a = A` au lieu de `a += A`, on ne garde que la paire de décharges la plus proche) ;
  2. mise à l'échelle synaptique multiplicative qui ramène le taux post vers 10 Hz. Elle conserve
     les rapports entre poids, donc ce que la STDP a appris.

### Partie II : différences avec le code à trous
- La STDP suit **directement l'équation (3)** : `Δw = η (x_pre − x_tar)(w_max − w)^μ` à chaque
  décharge post. Il n'y a qu'une seule trace, au lieu de la règle à triplets du GitHub des auteurs.
- **Homéostasie** : `theta += theta_plus` à chaque décharge (désactivable avec `homeo=False`).
  `theta` est figé pendant l'évaluation.
- **Normalisation divisive** (Goodhill & Barrow) en 2 lignes vectorisées : la somme des poids de
  chaque neurone est ramenée à 0.1 × (nb d'entrées), soit 78.4 en connexion complète.
- **Repos** : les 150 ms de repos sont remplacées par une remise à zéro directe des variables
  (environ 30 % de temps de calcul en moins).
- **Lecture** : on enregistre les décomptes de décharges, puis on compare 3 lectures en numpy :
  `meilleur` (code à trous), `vote` (Diehl & Cook) et `pondere`.
- **Données** : 7000 images tirées au hasard parmi les 70 000, découpées de façon stratifiée en
  5000 entraînement / 1000 validation / 1000 test. L'étiquetage des neurones utilise
  1000 images d'entraînement présentées sans plasticité (5000 pour les expériences `best_*` et
  `n800`).

## Paramètres du réseau (Partie II)

| Paramètre | Valeur | | Paramètre | Valeur |
|---|---|---|---|---|
| dt | 0.5 ms | | Présentation d'une image | 350 ms |
| Taux d'entrée max | 63.75 Hz (+32 Hz si < 5 décharges) | | Délais entrée → E | U(0, 10) ms |
| E : v_repos / v_reset / v_seuil | −65 / −65 / −52 mV | | I : v_repos / v_reset / v_seuil | −60 / −45 / −40 mV |
| E : τ_m / réfractaire | 100 ms / 5 ms | | I : τ_m / réfractaire | 10 ms / 2 ms |
| τ_ge / τ_gi | 1 ms / 2 ms | | E_inh (E / I) | −100 / −85 mV |
| theta_plus / τ_theta | 0.2 mV (article : 0.05) / 10⁷ ms | | Poids E→I / I→E | 10.4 / 17.0 |
| η / x_tar / w_max / μ | 0.01 / 0.4 / 1 / 1 | | τ_pre | 20 ms |
| Poids initiaux | U(0, 0.3) | | Somme des poids par neurone | 0.1 × nb entrées |

## Résultats

### Partie I (`resultats/partie1/partie1.csv`)

| Forme | STDP standard | STDP robuste (homéostasie + traces saturantes) |
|---|---|---|
| I E→E (hebbienne) | 6 et 10 Hz : neurone post muet, aucun apprentissage. 15 Hz : distribution bimodale (22 % des poids en haut, 21 % en bas), post à 53 Hz | 10 et 15 Hz : bimodale, post ramené à ~10 Hz |
| II E→E (creux LTD) | 15 Hz : dépression globale, neurone muet | 10 et 15 Hz : bimodale très marquée, post à ~7.5 Hz |
| I E→I (anti-hebbienne) | 15 Hz : emballement (aire positive), tous les poids à gmax, post à 380 Hz | 10 et 15 Hz : poids unimodaux, post à ~12 Hz |
| II E→I (LTD seule) | 15 Hz : dépression, neurone muet | 10 et 15 Hz : bimodale, post à ~8 Hz |

La STDP standard dépend fortement de la fréquence d'entrée : le même noyau ne fait rien, ou
s'emballe, selon le taux. La version robuste rend les 4 formes stables à 10 et 15 Hz.

Limite : à 6 Hz, même avec tous les poids à gmax, le neurone post ne décharge qu'à 0.6 Hz.
L'homéostasie ne peut donc rien faire, et il faudrait plus d'entrées ou un gmax plus grand.

### Partie III (`resultats/resultats.csv`, précision test après 1 époque, 100 neurones sauf indication)

| Expérience | vote | pondere | meilleur |
|---|---|---|---|
| n400 | 79.6 % | **81.3 %** | 26.3 % |
| sans_norm | 77.7 % | 78.6 % | 34.2 % |
| n200 | 74.8 % | 78.5 % | 32.4 % |
| base | 69.3 % | 73.2 % | 38.9 % |
| p_in_0.5 | 66.9 % | 72.6 % | 33.3 % |
| p_ei_0.5 | 68.3 % | 71.8 % | 47.1 % |
| theta_0.05 (valeur de l'article) | 65.7 % | 70.2 % | 36.7 % |
| p_ie_0.5 | 62.9 % | 66.8 % | 47.8 % |
| n50 | 61.0 % | 66.0 % | 39.1 % |
| n25 | 57.0 % | 63.1 % | 39.2 % |
| p_ie_0.1 | 56.0 % | 59.0 % | 49.1 % |
| sans_homeo | 18.1 % | 19.8 % | 19.6 % |

- **Itérations** (`iterations.png`, lecture pondere, base) : test 73.2 → 76.2 → 76.9 % après
  1, 2 et 3 époques. Le gain se fait surtout à la 2e époque, et l'écart train/test reste faible
  (pas de surapprentissage).
- **Neurones** : c'est le paramètre le plus important (63 % à 25 neurones, 81 % à 400). Plus il y a
  de neurones, plus il y a de prototypes, donc plus de variantes de chaque chiffre couvertes.
- **Homéostasie** : indispensable. Sans elle, deux neurones gagnent toutes les compétitions et le
  réseau tombe au hasard (20 %). Avec la valeur de l'article (0.05 mV), θ monte trop lentement pour
  5000 images, et une partie des neurones n'a encore rien appris.
- **Connexions** : affaiblir l'inhibition latérale (p_ie) dégrade le plus : les neurones ne sont
  plus en compétition et apprennent des mélanges de chiffres (`champs_p_ie_0.1.png`). Retirer la
  moitié des connexions entrée→E ou E→I coûte peu (−1 à −2 points).
- **Normalisation** : sans elle, la précision est *meilleure* après 1 époque. La règle (3) se
  régule déjà en partie toute seule : le terme −x_tar déprime les entrées inactives. La somme des
  poids chute de 118 à 26, puis remonte vers 58. Chaque neurone apprend alors un exemplaire très
  net (`champs_sans_norm.png`) plutôt qu'une moyenne floue. Ça reste stable sur 3 époques à
  400 neurones (voir ci-dessous).
- **Lectures** : `pondere` > `vote` partout, et `meilleur` (celle du code à trous) est la pire.
  `meilleur` ne garde qu'un neurone représentant par classe (celui qui y répond le plus) et jette
  tous les autres. Avec le winner-take-all, seuls 1 à 3 neurones déchargent par image, et ce n'est
  souvent pas le représentant. Elle profite donc d'une inhibition faible (plus de neurones actifs)
  et souffre quand il y a beaucoup de neurones.
- **Durées** : celles de `resultats.csv` sont des temps réels sous charge variable (jusqu'à
  12 processus en parallèle, priorité abaissée une partie du temps), donc non comparables entre
  expériences. La colonne `pause_s` indique la suspension manuelle de 3 h déjà retirée. Les durées
  fiables sont mesurées seules avec `duree.py` (voir ci-dessous).

### Meilleure configuration (`best_*`, `n800`)

En combinant les leviers favorables trouvés ci-dessus (plus de neurones, plusieurs époques,
étiquetage sur 5000 images), avec et sans normalisation. Le choix se fait sur la **validation**,
le test ne sert qu'à mesurer la généralisation (lecture pondere).

| Expérience | Neurones | Époque | Validation | Test |
|---|---|---|---|---|
| n800 | 800 | 1 | 84.4 % | 84.4 % |
| best_norm | 400 | 1 / 2 / 3 | 82.8 / 83.1 / 82.7 % | 82.8 / 83.6 / 83.6 % |
| **best_sans_norm** | 400 | **1** / 2 / 3 | **87.7** / 87.3 / 86.5 % | **85.9** / 87.4 / 85.4 % |

- **Configuration retenue** : 400 neurones sans normalisation, 1 époque. Elle donne **87.7 %** en
  validation et **85.9 % en test**, contre 73.2 % pour `base`.
- **Normalisation** : sans normalisation, on gagne environ 5 points à 400 neurones aussi, et c'est
  stable sur 3 époques (pas d'emballement des poids).
- **Époques** : à 400 neurones, les époques supplémentaires n'aident plus en validation.
- **Étiquetage** : étiqueter les neurones avec 5000 images au lieu de 1000 supprime l'écart
  train/validation de `n400` (85 % contre 76 % en vote). Cet écart venait d'étiquettes bruitées,
  pas d'un surapprentissage.

**Durée mesurée seule** (`duree.py`, 300 images, `resultats/durees.csv`) : 0.28 s par image en
apprentissage comme en évaluation, à 400 neurones. La configuration retenue (5000 images
d'apprentissage et 7000 d'évaluation) prend donc environ **56 min**, et 2 h 50 sur 3 époques.
C'est environ 2.3 fois moins que le temps observé en parallèle.

### Partie IV (`resultats/figures/`)

- **Champs récepteurs** (`champs_*.png`) : chaque neurone devient un prototype de chiffre. Le
  réseau projette l'image sur une base de 100 filtres appris sans étiquette, comme les centres
  d'un k-moyennes, et la lecture ne fait qu'étiqueter ces filtres.
- **Évolution** (`evolution_*.png`) : le champ aléatoire initial est remplacé dès le premier
  quart de l'apprentissage. Ensuite il ne fait que s'affiner (bruit retiré autour du trait). Un
  neurone peut aussi changer de chiffre en cours de route s'il perd ses compétitions.
- **Courbes d'accord** (`accord_*.png`) : très sélectives, avec presque toute la réponse sur une
  classe. Les neurones du « 1 » répondent un peu aux 8 et aux 2, qui partagent un trait vertical.
  Sur le test (base, 3 époques), les erreurs les plus fréquentes sont visuellement logiques :
  9→4 (34), 4→9 (19), 2→1 (18), 8→1 (16), 8→3 (15) et 5→3 (14).
- **Cartes d'activation** (`activation_*.png`) : l'activité est très éparse, avec 1 à 3 neurones
  actifs par fenêtre de temps à cause de l'inhibition latérale (winner-take-all). Il faut parfois
  une 2e présentation plus intense de l'image avant que le réseau réponde.
