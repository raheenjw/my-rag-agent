"""Celery application configuration."""

from celery import Celery
from config import get_settings

settings = get_settings()

celery = Celery(
    "rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["tasks"],  # Auto-discover tasks from tasks.py
)

celery.conf.update(
    task_track_started=True,       # STARTED state visible in status endpoint
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,           # results expire after 1 hour
)
