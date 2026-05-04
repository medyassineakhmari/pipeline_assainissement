"""
Configuration Celery — broker Redis pour exécuter les scripts en tâche de fond.
"""
from celery import Celery
from . import config

celery_app = Celery(
    "pipeline_platform",
    broker=config.REDIS_URL,
    backend=config.REDIS_URL,
    include=["backend.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Paris",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=config.TASK_TIMEOUT,
    task_soft_time_limit=config.TASK_TIMEOUT - 60 if config.TASK_TIMEOUT else None,
    worker_prefetch_multiplier=1,    # Une tâche à la fois (scripts longs)
    worker_max_tasks_per_child=1,    # Recycle après chaque tâche (évite les fuites mémoire)
)
