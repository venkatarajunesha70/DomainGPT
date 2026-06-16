"""
Celery application factory.
"""
from celery import Celery
from apps.api.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "domaingpt",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "apps.workers.ingest_worker",
        "apps.workers.embedding_worker",
        "apps.workers.indexing_worker",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # process one task at a time per worker
)
