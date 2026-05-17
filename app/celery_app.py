from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "bling_blue_team",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.evidence.trail_builder"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_soft_time_limit=900,   # 15 minutes soft limit
    task_time_limit=1200,       # 20 minutes hard limit
)
