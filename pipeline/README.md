# 📂 Dossier `pipeline/` — Tes scripts ML/SIG

C'est **ICI** que tu dois copier tous tes scripts du projet.

## ⚠️ Petite modification à faire dans tes scripts

Tes scripts ont des chemins **codés en dur** au début. Pour qu'ils soient
configurables depuis l'interface web, remplace ces variables par des lectures
de variables d'environnement.

### Exemple — `model_materiau.py` AVANT :

```python
INPUT_FILE = "BDD_Pretraitee_Finale_plastique_divisé.csv"
OUTPUT_DIR = "/home/user-ia/ML2/resultats_comparatif_materiau"
```

### Exemple — `model_materiau.py` APRÈS :

```python
import os

INPUT_FILE = os.environ.get("INPUT_FILE", "BDD_Pretraitee_Finale_plastique_divisé.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/user-ia/ML2/resultats_comparatif_materiau")
```

C'est tout. Les valeurs par défaut restent les mêmes (le script tourne toujours
en standalone), mais l'interface peut maintenant injecter d'autres valeurs.

## 📋 Liste des scripts attendus

Le fichier `backend/pipeline_steps.py` déclare les variables d'environnement
attendues pour chaque étape. Voici le récapitulatif :

| Script                              | Variables à exposer                                                |
|-------------------------------------|--------------------------------------------------------------------|
| `pretraitement_donnees_latest.py`   | `PATH_BASE`, `PATH_EXCEL_MAPPING`, `OUTPUT_FILE`, `OUTPUT_GPKG`     |
| `calcul_spacial_altitude.py`        | `PATH_SHP`, `PATH_DALLES`, `PATH_RASTERS`                          |
| `update_sig.py`                     | `PATH_BASE`, `PATH_EXCEL_MAPPING`, `FILE_TREE`, `FILE_IMPER`       |
| `ajout_autres_vars.py`              | `INPUT_GPKG`, `INPUT_LAYER`, `OUTPUT_GPKG`, `OUTPUT_LAYER`, `ARGILES_SHP`, `NAPPES_DIR` |
| `model_materiau.py`                 | `INPUT_FILE`, `OUTPUT_DIR`                                         |
| `completion_BDD_materiau_sup_70.py` | `INPUT_FILE`, `MODEL_DIR`, `BEST_MODEL`, `OUTPUT_FILE`, `OUTPUT_GPKG` |
| `model_classe_age.py`               | `INPUT_FILE`, `OUTPUT_DIR`                                         |
| `completion_BDD_age.py`             | `INPUT_FILE`, `MODEL_DIR`, `OUTPUT_FILE`, `OUTPUT_GPKG`            |
| `comparatif_modeles_eds.py`         | `INPUT_FILE`, `INPUT_LAYER`, `OUTPUT_DIR_BASE`                     |
| `completion_BDD_eds.py`             | `INPUT_FILE`, `INPUT_LAYER`, `MODELS_DIR`, `TARGET_NAME`, `MODEL_FILENAME`, `OUTPUT_DIR` |
| `analyse_statistique_eds4_j1.py`    | `INPUT_GPKG`, `INPUT_LAYER`, `OUTPUT_DIR`                          |

## 🆕 Ajouter une nouvelle étape

1. Place ton script ici dans `pipeline/`
2. Ajoute un bloc dans `backend/pipeline_steps.py` (copier-coller un existant)
3. Recharge la page → l'étape apparaît dans le menu

C'est tout — pas besoin de redémarrer le serveur (sauf si Celery tourne, dans ce cas relance avec `./stop.sh && ./start.sh`).
