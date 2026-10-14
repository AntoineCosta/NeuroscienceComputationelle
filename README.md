# Hexapode piloté par un réseau de neurones impulsionnels (Brian2)

Simulation d'un robot hexapode dont la marche est **entièrement produite par un réseau de
neurones** écrit avec [Brian2](https://brian2.readthedocs.io). On peut le piloter en direct
(manette ou clavier) et il évite seul les obstacles grâce à des capteurs de distance.

**En bref :**
- un **CPG** (2 neurones) donne le rythme : une marche en tripode, 500 ms par cycle ;
- des **motoneurones** à seuils étagés codent l'amplitude du pas par le *nombre* de neurones qui déchargent ;
- 6 **interneurones inhibiteurs** traduisent les commandes (avancer, reculer, tourner) en freinage de ces motoneurones ;
- 4 **capteurs** ajoutent des réflexes (freiner, reculer, s'écarter d'un mur) en passant par ces mêmes interneurones.

Le projet est uniquement en simulation (pas de matériel). Le nombre de pattes est réglable (6 à 400, soit 200 paires).

---

## Lancer le projet

```bash
conda create -n brian2_env -c conda-forge brian2 matplotlib pillow
conda activate brian2_env
pip install pygame        # optionnel : seulement pour la manette

python manual.py          # pilotage en direct
python main.py            # démonstrations (GIF dans graphs/demos/)
```

| Commande | Manette (Xbox) | Clavier |
|---|---|---|
| Avancer / reculer | stick gauche | ↑ / ↓ |
| Tourner en marchant | stick droit | ↑ ou ↓ + ← / → |
| Tourner sur place | croix gauche / droite | ← / → seules |
| Quitter | Start | Échap |

Cliquer une fois sur la fenêtre pour qu'elle reçoive le clavier. À l'écran : tripode A en bleu,
B en rouge (pied creux = levé) ; rayons des capteurs du vert (loin) au rouge (proche).

---

## Architecture

![Schéma du réseau](graphs/Graph_neurons.jpg)

```
 Commande (manette / clavier / scénario)
      │ (vitesse, virage)
      ▼
 Cerveau (code) ──► 6 INTERMÉDIAIRES inhibiteurs ◄── 4 CAPTEURS de distance
                          │ inhibition                    ▲
 CPG (rythme) ──────────► MOTONEURONES des pattes         │
                          │ potentiels d'action           │
                          ▼                               │
                     Corps (odométrie) ──► Monde (lancer de rayons)
```

Le **cerveau** décide *quoi faire*, le **CPG** décide *quand*, les **intermédiaires** décident
*combien* de motoneurones sont recrutés.

<details>
<summary>Correspondance entre les noms du schéma et ceux du code</summary>

| Schéma | Code |
|---|---|
| AV 1..N / RE 1..N | groupes F (avant) / B (arrière) d'une patte |
| UP / DWN | R (lever) / D (poser) |
| AVG, AVD, REG, RED int | I-Fwd-L, I-Fwd-R, I-Rev-L, I-Rev-R |
| G, D int | I-Steer-L, I-Steer-R |
| SeH, SeB, SeG, SeD | capteurs avant, arrière, côtés gauche (SL) et droit (SR) |

</details>

### 1. CPG : le rythme

Deux neurones LIF. **A** est un pacemaker qui décharge toutes les 500 ms ; **B** est un relais
déclenché par A avec un retard de 250 ms. On obtient un potentiel d'action par demi-cycle,
alternativement A puis B. Chaque patte suit A ou B selon sa position, ce qui forme deux
tripodes qui alternent appui et vol.

### 2. Motoneurones : le nombre code l'amplitude

Par patte : **R** (lever), **D** (poser), et deux groupes de N neurones **F** (avant) et **B** (arrière).
Les seuils sont régulièrement espacés : θ_k = (k + ½)·u. À chaque potentiel d'action du CPG,
les neurones d'un groupe reçoivent tous la même excitation N·u ; **une inhibition de m·u fait taire exactement les m seuils
les plus hauts**. Le nombre de neurones qui déchargent (N − m) est donc l'amplitude du pas.

L'inhibition arrive 2 ms avant l'excitation. R est retardé de 10 ms pour que le tripode en vol
ne se lève qu'une fois l'autre posé.

### 3. Intermédiaires : 6 neurones de « frein »

Neurones gradués (sans potentiel d'action), d'activité `level = clip(brain + gate·sensor, 0, 1)` :
0 = pas de frein, 1 = frein total. À chaque potentiel d'action du CPG, chacun inhibe son côté de
`level·N` paliers :

| Intermédiaire | Pattes en appui | Pattes en vol |
|---|---|---|
| **I-Fwd-X** (avancer) | freine F (poussée) | freine B (retour) + R |
| **I-Rev-X** (reculer) | freine B (poussée) | freine F (retour) + R |
| **I-Steer-X** (virage) | freine F et B | freine F et B |

*Exemple, marche avant à la vitesse s :* le cerveau envoie I-Fwd = 1 − s et I-Rev = 1 ;
s·N neurones F poussent en appui, s·N neurones B ramènent la patte en vol.
Tourner revient à freiner davantage un côté.

### 4. Capteurs : 4 neurones de réflexe

Chaque capteur est une ligne de visée qui produit `near` (obstacle proche, graduel) et
`close` (zone de recul, < 0,2 m). Ils agissent sur l'entrée `sensor` des intermédiaires :

| Capteur | Position | Effet |
|---|---|---|
| Avant | nez, dans l'axe | freine la marche avant, puis déclenche le recul |
| Arrière | queue, dans l'axe | freine la marche arrière, puis déclenche le retour en avant |
| SL, SR | flancs, ±90° | font tourner à l'opposé du mur (le robot suit un couloir sans le toucher) |

Trop près (< 0,2 m), le capteur **désinhibe la direction opposée** : le robot recule jusqu'à
sortir de la zone, puis la commande reprend la main (va-et-vient). À l'arrêt ou en rotation
sur place, les capteurs sont coupés (`gate = 0`).

---

## Code non neuronal

Le code Python classique ne touche le réseau que de trois façons : il **écrit** des variables
de neurones (`brain`, `gate`, `distance`), il **lit** les potentiels d'action (`SpikeMonitor`),
et il tourne sous forme de `network_operation` Brian2 à intervalles fixes.

| Code | Quand | Rôle |
|---|---|---|
| Cerveau (`brain.py`) | début de chaque demi-cycle | (vitesse, virage) → `brain` et `gate` des intermédiaires |
| Odométrie (`robot.py`, `body.py`) | toutes les 10 ms | potentiels d'action F − B des pattes en appui → déplacement du corps |
| Capteurs (`sensors.py`, `world.py`) | toutes les 10 ms | lancer de rayons → `distance` des neurones capteurs |
| Affichage (`manual.py`, `visualisation.py`) | 20 images/s | dessin en temps réel |

<details>
<summary>Chronologie d'un demi-cycle</summary>

| Temps après le potentiel d'action du CPG | Événement |
|---|---|
| 0 (juste avant) | le cerveau fixe la commande (`brain`, `gate`) |
| 0 | potentiel d'action du CPG ; les intermédiaires apprennent quel tripode est en appui |
| + 0,5 ms | les intermédiaires libèrent leur inhibition |
| + 2 ms | l'excitation du CPG atteint F, B, D : les neurones non freinés déchargent |
| + 12 ms | R décharge : le tripode en vol se lève |
| ≈ + 13 ms | l'odométrie compte les potentiels d'action et déplace le corps |

Une commande met donc 0 à ¼ s à prendre effet (attente du prochain potentiel d'action du CPG).

</details>

---

## Fichiers

Seuls les deux points d'entrée sont à la racine ; tout le reste est dans le package `hexapode/`.

```
main.py, manual.py         points d'entrée (démos et tests / pilotage en direct)
hexapode/
  config.py                tous les paramètres (pattes, rythme, délais, géométrie, capteurs), commentés
  reseau/                  les neurones (Brian2)
    CPG.py, legs.py, intermediates.py, sensors.py
  simulation/              le code classique autour du réseau
    brain.py, body.py, world.py, robot.py, scenarios.py
  affichage/               les figures
    visualisation.py       figures et animations
    figures_rapport.py     figures du rapport
graphs/                    figures générées, schéma du réseau ; demos/ (GIF), rapport/
docs/                      ancien résumé du code
```

Les figures du rapport se régénèrent depuis la racine avec
`python -m hexapode.affichage.figures_rapport` (sorties dans `graphs/rapport/`).
Les sorties vont toujours dans `graphs/` du projet, quel que soit le dossier de lancement.

## Démonstrations

`python main.py` produit un GIF par fonctionnalité dans `graphs/demos/` (environ 10 min).
À gauche, la commande du cerveau et l'amplitude d'appui réellement produite par chaque côté ;
à droite, la vue de dessus. Sans obstacle, chaque démo vérifie aussi le nombre exact de
neurones R, D, F, B qui déchargent, pour chaque patte et chaque demi-cycle.

| GIF | Montre |
|---|---|
| `avant_arriere` | marche avant / arrière à 100 % et 50 %, arrêt |
| `rotation` | rotation sur place à gauche, à droite, à 50 % et 100 % |
| `virages` | virages de plus en plus serrés (25, 50, 100 %), à droite, en reculant |
| `tous_mouvements` | enchaînement complet (scénario de `test_control`) |
| `capteur_avant` | commande « avance » constante vers un mur : freinage puis va-et-vient |
| `couloir` | traversée d'un couloir de 0,5 m sans contact |
| `20_paires`, `200_paires` | même réseau avec 40 et 400 pattes, carte de toutes les pattes |

Une seule démo : `python -c "import main; main.demo('couloir')"`.

## Tests

Les tests sont à décommenter en bas de `main.py`.
Chacun vérifie automatiquement ses résultats et enregistre ses figures dans `graphs/`.

| Test | Vérifie |
|---|---|
| `test_cpg()` | période de 500 ms et décalage de 250 ms entre A et B |
| `test_legs()` | recrutement de N − m neurones sous inhibition, seuils bas d'abord |
| `test_control()` | réseau complet sur 8 s : nombre exact de R, D, F, B par patte et par demi-cycle |
| `test_body()` | idem, plus la trajectoire du corps (figure + animation) |
| `test_sensors()` | 5 réflexes : mur devant, mur derrière, rotation et arrêt près d'un mur, traversée d'un couloir sans contact |

`test_control(n_legs=400)` fonctionne aussi.

## Limites connues

- **Latence :** jusqu'à ¼ s entre une commande et le mouvement (attente du CPG).
- **Ondulation dans un couloir :** les capteurs latéraux voient la distance au mur mais pas le
  cap, donc le virage n'est pas amorti : le robot oscille autour du centre (±7 cm) sans toucher.
- **À basse vitesse :** le frein progressif arrête le robot avant la zone de recul (pas de va-et-vient).
- **Collisions :** pas de physique, le corps traverse les obstacles (affiché en rouge).
