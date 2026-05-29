from celery import Celery
from kombu import Queue
from app.core.config import settings

celery_app = Celery(
    "bling_blue_team",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.evidence.trail_builder",
        "app.detection.tier3.shap_explainer",
        "app.graph.tasks",
    ],
)

celery_app.config_from_object("celeryconfig")

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_soft_time_limit=900,
    task_time_limit=1200,
    task_queues=(
        Queue("default"),
        Queue("evidence"),
        Queue("graph"),
        Queue("compliance"),
        Queue("dlq_evidence"),
    ),
    task_default_queue="default",
)
