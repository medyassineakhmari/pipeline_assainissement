"""
Configuration centrale de la plateforme.
Tous les chemins sont calculés depuis la racine du projet.
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Racines
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent     # pipeline-platform/
PIPELINE_DIR = ROOT_DIR / "pipeline"                       # scripts utilisateur
DATA_DIR     = ROOT_DIR / "data"
INPUTS_DIR   = DATA_DIR / "inputs"
OUTPUTS_DIR  = DATA_DIR / "outputs"
LOGS_DIR     = DATA_DIR / "logs"

# Création automatique des dossiers
for d in (INPUTS_DIR, OUTPUTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Redis / Celery
# ─────────────────────────────────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# ─────────────────────────────────────────────────────────────────────────────
# Exécution des scripts
# ─────────────────────────────────────────────────────────────────────────────
# Interpréteur Python à utiliser pour exécuter les scripts du pipeline.
# Par défaut : le même que celui qui lance Celery (donc l'env conda actif).
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")

# Timeout maximum d'une tâche (en secondes). Mettre à None pour pas de timeout.
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", "21600"))   # 6h par défaut

# Sécurité : racines autorisées pour la lecture/écriture de fichiers.
# Évite que l'API accède à n'importe où sur le disque.
ALLOWED_ROOTS = [
    "/",
    str(DATA_DIR.resolve()),
    str(PIPELINE_DIR.resolve()),
    # Ajouter ici les chemins SMB/NAS si nécessaire :
    "/run/user/1000/gvfs/smb-share:server=172.16.39.24,share=groupe-merlin/Lyon-Siege/Hors_Affaires/HYDRAU/THEMES/02-Assainissement/05-Documentation/06- GPASS/15- Modele predictif/",
    "/home/user-ia/M-yassine",
]
