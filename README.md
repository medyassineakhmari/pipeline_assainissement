# Pipeline Assainissement — Plateforme web

Plateforme locale pour orchestrer le pipeline ML/SIG de gestion patrimoniale
des réseaux d'assainissement (Cabinet MERLIN — Stage PFE 2026).

## 📁 Structure du projet

```
pipeline-platform/
├── README.md                  ← Ce fichier
├── requirements.txt           ← Dépendances Python additionnelles
├── start.sh / stop.sh         ← Lancement / arrêt
│
├── backend/                   ← API FastAPI + Celery
│   ├── main.py                ← Point d'entrée FastAPI
│   ├── celery_app.py          ← Config Celery
│   ├── tasks.py               ← Exécution des scripts en subprocess
│   ├── pipeline_steps.py      ← ⭐ Définition des étapes du pipeline
│   ├── config.py              ← Chemins, settings
│   └── routes/
│       ├── pipeline.py        ← /api/pipeline/...
│       ├── jobs.py            ← /api/jobs/... (statut + logs SSE)
│       └── files.py           ← /api/files/... (browse / upload)
│
├── pipeline/                  ← ⭐ TES SCRIPTS VONT ICI
│   ├── README.md              ← Instructions pour l'adaptation
│   └── (scripts à copier)
│
├── frontend/                  ← Interface web (HTML/JS pur, pas de build)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
└── data/                      ← Stockage local
    ├── inputs/                ← Fichiers uploadés
    ├── outputs/               ← Résultats
    └── logs/                  ← Logs des jobs (.log)
```

## 🚀 Installation (PC Linux avec env conda `rapids_gpu`)

### 1. Installation de Redis

```bash
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
redis-cli ping       # Doit répondre "PONG"
```

### 2. Activation de l'env conda et installation des dépendances API

```bash
conda activate rapids_gpu
cd pipeline-platform
pip install -r requirements.txt
```

Note : les libs ML/SIG (geopandas, sklearn, xgboost, etc.) sont déjà dans
l'env `rapids_gpu` — `requirements.txt` n'installe que les briques web
(FastAPI, Celery, Redis client).

### 3. Copie de tes scripts

```bash
cp /chemin/vers/tes/scripts/*.py pipeline-platform/pipeline/
```

Puis lis `pipeline/README.md` pour le petit ajustement à faire dans chaque
script (remplacer les chemins codés en dur par `os.environ.get(...)`).

### 4. Configuration des chemins autorisés

Édite `backend/config.py` et ajoute dans `ALLOWED_ROOTS` les dossiers où tu
veux pouvoir lire/écrire (ex: ton montage SMB Merlin) :

```python
ALLOWED_ROOTS = [
    str(DATA_DIR.resolve()),
    str(PIPELINE_DIR.resolve()),
    "/run/user/1000/gvfs/smb-share:server=172.16.39.24,share=groupe-merlin",
    "/home/user-ia/ML2",
]
```

### 5. Lancement

```bash
./start.sh
```

Puis ouvre **http://localhost:8000** dans ton navigateur.

## 🎯 Comment ça marche

1. **Tu choisis une étape** dans le menu de gauche (Phase A, B, C, ou stats)
2. **Tu remplis le formulaire** au centre avec les chemins des fichiers d'entrée
   et les emplacements de sortie
3. **Tu cliques sur Lancer** → un job Celery démarre en arrière-plan
4. **Les logs s'affichent en temps réel** dans le panneau de droite
5. Le statut passe de `STARTED` → `SUCCESS` ou `FAILURE`

## 🔧 Points clés de l'architecture

### Indépendance des étapes
Chaque étape est lancée séparément. Tu n'es pas obligé de respecter l'ordre du
pipeline — tu peux par exemple relancer uniquement la complétion EDS sans
refaire le prétraitement.

### Streaming des logs
Les logs sont poussés dans Redis ligne par ligne par le subprocess, et streamés
via SSE (Server-Sent Events) au navigateur. Tu vois les logs apparaître en
temps réel comme dans un terminal.

### Aucune modification structurelle de tes scripts
Tes scripts continuent à fonctionner en standalone (avec leurs valeurs par
défaut). La seule modification est de remplacer
`MA_VAR = "/chemin/dur"` par
`MA_VAR = os.environ.get("MA_VAR", "/chemin/dur")`.

### Sécurité (limitée — usage local)
L'API n'autorise la lecture/écriture que sous les chemins déclarés dans
`ALLOWED_ROOTS`. Pas d'authentification — c'est volontaire pour un usage local.
**À ajouter avant déploiement réseau.**

## 📡 API REST (pour intégration future)

| Méthode | Endpoint                       | Description                              |
|---------|--------------------------------|------------------------------------------|
| `GET`   | `/api/pipeline/steps`          | Liste des étapes                         |
| `GET`   | `/api/pipeline/steps/{id}`     | Détail d'une étape                       |
| `POST`  | `/api/pipeline/run/{id}`       | Lance un job (retourne `job_id`)         |
| `GET`   | `/api/jobs/{job_id}/status`    | Statut Celery                            |
| `GET`   | `/api/jobs/{job_id}/logs`      | Logs accumulés (polling)                 |
| `GET`   | `/api/jobs/{job_id}/stream`    | Streaming SSE des logs                   |
| `POST`  | `/api/jobs/{job_id}/cancel`    | Annule un job                            |
| `GET`   | `/api/files/browse?path=...`   | Liste un dossier                         |
| `POST`  | `/api/files/upload`            | Upload un fichier                        |
| `GET`   | `/api/files/download?path=...` | Télécharge un fichier                    |

Documentation interactive Swagger : **http://localhost:8000/docs**

## 🚢 Déploiement futur

Quand tu seras prêt pour la prod :

1. **Conteneuriser** : Docker Compose avec 4 services (api, worker, redis, nginx)
2. **Authentification** : ajouter un middleware FastAPI (OAuth2 ou simple token)
3. **HTTPS** : reverse proxy nginx avec Let's Encrypt
4. **Stockage** : monter un volume persistant pour `data/`
5. **Multi-utilisateurs** : associer chaque job à un user_id (ajout d'une table en BDD)

L'architecture actuelle est déjà compatible : workers séparés de l'API,
broker Redis externalisable, frontend statique servable derrière n'importe
quel reverse proxy.

## 📝 TODO / améliorations possibles

- [ ] Persistance des jobs dans une vraie DB (SQLite suffirait pour commencer)
- [ ] Dashboard d'historique des jobs (page séparée)
- [ ] Visualisation des résultats (preview des PNG, des CSV)
- [ ] Notifications (email / Slack) en fin de job long
- [ ] Authentification utilisateur
- [ ] Mode "pipeline complet" : enchaîner automatiquement A → B → C
