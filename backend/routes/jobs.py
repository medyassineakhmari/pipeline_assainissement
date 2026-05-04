"""
Routes API : suivi des jobs (statut + streaming des logs).
"""
import asyncio
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
import redis

from ..celery_app import celery_app
from ..tasks import LOG_KEY_PREFIX
from .. import config

router = APIRouter(prefix="/jobs", tags=["jobs"])

_redis = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)


@router.get("/{job_id}/status")
def get_status(job_id: str):
    """Retourne le statut Celery d'un job (PENDING/STARTED/SUCCESS/FAILURE)."""
    result = celery_app.AsyncResult(job_id)
    info = {
        "job_id":   job_id,
        "status":   result.status,
        "ready":    result.ready(),
        "success":  result.successful() if result.ready() else None,
    }
    if result.ready() and result.successful():
        info["result"] = result.result
    elif result.ready() and result.failed():
        info["error"] = str(result.result)
    return info


@router.get("/{job_id}/logs")
def get_logs(job_id: str, since: int = 0):
    """Retourne les logs accumulés pour un job (à partir de l'index `since`)."""
    key = LOG_KEY_PREFIX + job_id
    total = _redis.llen(key)
    if since >= total:
        return {"lines": [], "next_index": total}
    lines = _redis.lrange(key, since, -1)
    return {"lines": lines, "next_index": total}


@router.get("/{job_id}/stream")
async def stream_logs(job_id: str):
    """Streaming SSE des logs en temps réel."""
    async def event_generator():
        last_index = 0
        while True:
            key   = LOG_KEY_PREFIX + job_id
            total = _redis.llen(key)
            if total > last_index:
                lines = _redis.lrange(key, last_index, -1)
                for line in lines:
                    yield {"event": "log", "data": line}
                last_index = total

            # Vérifier l'état du job
            result = celery_app.AsyncResult(job_id)
            if result.ready():
                yield {
                    "event": "done",
                    "data":  result.status,
                }
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    """Annule un job en cours d'exécution."""
    celery_app.control.revoke(job_id, terminate=True)
    return {"job_id": job_id, "status": "CANCELLED"}
