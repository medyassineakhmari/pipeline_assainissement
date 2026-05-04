"""
Routes API : liste des étapes et lancement d'un job.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..pipeline_steps import PIPELINE_STEPS, get_step_by_id
from ..tasks import run_step

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get("/steps")
def list_steps():
    """Retourne la liste de toutes les étapes du pipeline."""
    return {"steps": PIPELINE_STEPS}


@router.get("/steps/{step_id}")
def get_step(step_id: str):
    """Retourne la définition d'une étape donnée."""
    step = get_step_by_id(step_id)
    if step is None:
        raise HTTPException(404, f"Étape inconnue : {step_id}")
    return step


class RunRequest(BaseModel):
    params: dict[str, Optional[str]] = {}


@router.post("/run/{step_id}")
def run_pipeline_step(step_id: str, body: RunRequest):
    """
    Lance une étape du pipeline en tâche de fond.
    Retourne immédiatement avec un job_id à utiliser pour suivre l'avancement.
    """
    step = get_step_by_id(step_id)
    if step is None:
        raise HTTPException(404, f"Étape inconnue : {step_id}")

    # Validation des paramètres requis
    missing = []
    for p in step["params"]:
        if p["required"]:
            val = body.params.get(p["env_var"])
            if not val or not str(val).strip():
                missing.append(p["env_var"])
    if missing:
        raise HTTPException(
            400,
            f"Paramètres requis manquants : {', '.join(missing)}",
        )

    # Soumission Celery
    async_result = run_step.delay(step_id, body.params)

    return {
        "job_id":  async_result.id,
        "step_id": step_id,
        "status":  "PENDING",
    }
