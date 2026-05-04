"""
Tâche Celery générique : exécute un script du dossier pipeline/ comme subprocess
Injecte les dossiers de sortie, compresse le résultat en .zip et le renvoie au front.
"""
import os
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import redis

from .celery_app import celery_app
from . import config
from .pipeline_steps import get_step_by_id

_redis = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)

LOG_KEY_PREFIX = "pipeline:logs:"
LOG_TTL        = 60 * 60 * 24    # 24h

def _log(job_id: str, line: str):
    key = LOG_KEY_PREFIX + job_id
    _redis.rpush(key, line)
    _redis.expire(key, LOG_TTL)

@celery_app.task(bind=True, name="pipeline.run_step")
def run_step(self, step_id: str, params: dict):
    job_id = self.request.id
    step   = get_step_by_id(step_id)

    if step is None:
        msg = f"[ERREUR] Étape inconnue : {step_id}"
        _log(job_id, msg)
        raise ValueError(msg)

    script_path = config.PIPELINE_DIR / step["script"]
    if not script_path.exists():
        msg = f"[ERREUR] Script introuvable : {script_path}"
        _log(job_id, msg)
        raise FileNotFoundError(msg)

    # 1. CRÉATION DU DOSSIER DE SORTIE UNIQUE POUR CE JOB
    job_out_dir = config.OUTPUTS_DIR / job_id
    job_out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for k, v in (params or {}).items():
        if v is not None and str(v).strip():
            env[k] = str(v)

    # 2. INJECTION INVISIBLE DES CHEMINS DE SORTIE
    env["PIPELINE_JOB_ID"]  = job_id
    env["PYTHONUNBUFFERED"] = "1"

    # N'injecter QUE si l'utilisateur n'a rien fourni (setdefault = ne pas écraser)
    env.setdefault("OUTPUT_DIR",      str(job_out_dir))
    env.setdefault("OUTPUT_DIR_BASE", str(job_out_dir))
    env.setdefault("OUTPUT_FILE",     str(job_out_dir / "Resultats_Export.csv"))
    env.setdefault("OUTPUT_GPKG",     str(job_out_dir / "Resultats_Export.gpkg"))
    env.setdefault("PATH_OUTPUT",     str(job_out_dir / "Resultats_Export.shp"))
    env.setdefault("OUTPUT_PLOT",     str(job_out_dir / "Stats_Completude_Pretraitement.png"))

    _log(job_id, "=" * 70)
    _log(job_id, f"[{datetime.now().strftime('%H:%M:%S')}] Démarrage : {step['name']}")
    _log(job_id, f"  Sauvegarde auto dans : data/outputs/{job_id}/")
    _log(job_id, "=" * 70)

    process = subprocess.Popen(
        [config.PYTHON_BIN, "-u", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(config.PIPELINE_DIR),
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    nb_lines = 0
    if process.stdout is not None:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                _log(job_id, line)
                nb_lines += 1

    returncode = process.wait()
    zip_path = None

    _log(job_id, "=" * 70)
    if returncode == 0:
        _log(job_id, f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Terminé avec succès. Compression des résultats...")
        
        # 3. COMPRESSION DU DOSSIER EN .ZIP
        zip_path = config.OUTPUTS_DIR / f"{job_id}.zip"
        shutil.make_archive(str(config.OUTPUTS_DIR / job_id), 'zip', str(job_out_dir))
        
        _log(job_id, f"[{datetime.now().strftime('%H:%M:%S')}] 📥 Archive ZIP prête pour le téléchargement !")
    else:
        _log(job_id, f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Échec — code retour {returncode}")
    _log(job_id, "=" * 70)

    log_file = config.LOGS_DIR / f"{job_id}.log"
    try:
        all_lines = _redis.lrange(LOG_KEY_PREFIX + job_id, 0, -1)
        log_file.write_text("\n".join(all_lines), encoding="utf-8")
    except Exception as e:
        pass

    if returncode != 0:
        raise RuntimeError(f"Script {step_id} a échoué (code {returncode}).")

    return {
        "returncode": returncode,
        "log_file":   str(log_file),
        "n_lines":    nb_lines,
        "zip_file":   str(zip_path) if zip_path and zip_path.exists() else None
    }

