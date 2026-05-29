"""
app/graph/tasks.py — Celery task wrappers for scheduled graph operations (P0-2).

Replaces APScheduler. All tasks registered here are scheduled via celeryconfig.py beat_schedule.

SLA: nightly_batch must complete within 3600s. SLA miss → CRITICAL log + Slack alert.
"""
from __future__ import annotations

import time

import structlog

from app.celery_app import celery_app

log = structlog.get_logger()

NIGHTLY_BATCH_SLA_SECONDS = 3600  # 1 hour — must complete before 4am UTC


@celery_app.task(name="app.graph.tasks.run_node2vec_task")
def run_node2vec_task():
    """
    Node2Vec 32-dim embedding computation — runs at 4:30am UTC (after nightly batch).
    Writes emb:{account} keys to Redis. (P2-6)
    """
    try:
        from app.graph.precompute.node2vec_runner import run_node2vec_embedding
        count = run_node2vec_embedding()
        log.info("node2vec_task_complete", embeddings=count)
    except Exception as exc:
        log.error("node2vec_task_failed", error=str(exc))


@celery_app.task(name="app.graph.tasks.run_nightly_batch_task", bind=True, max_retries=1)
def run_nightly_batch_task(self):
    """
    Nightly graph feature computation — runs at 3am UTC.
    Computes all ~35+ features per account and writes to Redis feat:{account}.
    SLA: must complete by 4am (1h window). Alerts on miss.
    """
    from app.graph.precompute.nightly_batch import run_nightly_feature_computation
    start = time.time()
    try:
        run_nightly_feature_computation()
    except Exception as exc:
        _record_nightly_failure(exc)
        raise self.retry(exc=exc, countdown=300)  # retry once after 5 min
    finally:
        duration = time.time() - start
        _record_nightly_duration(duration)


@celery_app.task(name="app.graph.tasks.update_betweenness_task")
def update_betweenness_task():
    """
    Approximate betweenness centrality update — every 2 hours.
    Uses k=500 approximation via NetworkX. Updates ONLY betweenness_centrality
    field in each feat:{account} hash — does NOT overwrite other fields.
    """
    try:
        from app.graph.precompute.nightly_batch import update_betweenness_only
        update_betweenness_only()
    except AttributeError:
        # update_betweenness_only not yet implemented — silently skip
        log.debug("betweenness_update_skipped", reason="not_yet_implemented")
    except Exception as exc:
        log.error("betweenness_update_failed", error=str(exc))


@celery_app.task(name="app.graph.tasks.run_micro_batch_task")
def run_micro_batch_task():
    """
    Fast-changing feature micro-batch — every 5 minutes.
    Updates: degree_centrality, temporal_acceleration, sink_score.
    Uses partial Redis hash update — never overwrites the full feat:{account}.
    """
    try:
        from app.graph.precompute.nightly_batch import update_micro_batch_features
        update_micro_batch_features()
    except AttributeError:
        log.debug("micro_batch_skipped", reason="not_yet_implemented")
    except Exception as exc:
        log.error("micro_batch_failed", error=str(exc))


@celery_app.task(name="app.graph.tasks.check_dlq_depth")
def check_dlq_depth():
    """
    DLQ depth monitor — every 15 minutes (P1-2).
    Checks Redis LLEN on dlq_evidence list. Alerts if depth > 5.
    NOTE: Uses LLEN not inspector.reserved() — reserved only shows in-flight tasks.
    """
    try:
        from app.utils.redis_client import get_redis
        r = get_redis()
        depth = r.llen("dlq_evidence")
        if depth > 5:
            log.critical(
                "dlq_depth_exceeded",
                depth=int(depth),
                queue="dlq_evidence",
                alert="Evidence DLQ backing up — check trail_builder failures",
            )
            _send_sla_alert("dlq_evidence", float(depth))
        else:
            log.debug("dlq_depth_ok", depth=int(depth))
    except Exception as exc:
        log.error("dlq_monitor_failed", error=str(exc))


@celery_app.task(name="app.graph.tasks.refresh_gnn_embeddings_task")
def refresh_gnn_embeddings_task():
    """
    GNN embedding refresh — every 5 minutes.
    Updates emb:{account} in Redis for accounts active in the last hour.
    Scorer B uses these embeddings; stale embeddings degrade to missing_flag=True.
    Sets scorer_b:emb_refresh_at key with 7200s TTL on success.
    """
    try:
        from app.utils.redis_client import get_redis
        from app.graph.precompute.node2vec_runner import refresh_recent_embeddings
        r = get_redis()
        updated = refresh_recent_embeddings(lookback_minutes=60)
        import time
        r.setex("scorer_b:emb_refresh_at", 7200, int(time.time()))
        log.info("gnn_embedding_refresh_complete", accounts_updated=updated)
    except AttributeError:
        log.debug("gnn_refresh_skipped", reason="refresh_recent_embeddings_not_implemented")
    except Exception as exc:
        log.error("gnn_embedding_refresh_failed", error=str(exc))


def _record_nightly_duration(duration: float) -> None:
    """Log duration and SLA miss for nightly batch."""
    try:
        from app.utils.metrics import nightly_batch_duration_seconds
        nightly_batch_duration_seconds.observe(duration)
    except Exception:
        pass

    if duration > NIGHTLY_BATCH_SLA_SECONDS:
        log.critical(
            "nightly_batch_sla_miss",
            duration_seconds=round(duration, 1),
            sla_seconds=NIGHTLY_BATCH_SLA_SECONDS,
            alert="Nightly batch exceeded 1-hour SLA window",
        )
        _send_sla_alert("nightly_batch", duration)
    else:
        log.info("nightly_batch_completed", duration_seconds=round(duration, 1))


def _record_nightly_failure(exc: Exception) -> None:
    """Log and alert on nightly batch failure."""
    try:
        from app.utils.metrics import nightly_batch_failure_total
        nightly_batch_failure_total.inc()
    except Exception:
        pass
    log.critical("nightly_batch_failed", error=str(exc), error_type=type(exc).__name__)
    _send_sla_alert("nightly_batch_failure", 0)


def _send_sla_alert(task_name: str, duration: float) -> None:
    """Send Slack/log alert for SLA misses. Best-effort — never raises."""
    try:
        import os
        import httpx
        webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        if not webhook:
            return
        msg = {
            "text": (
                f":rotating_light: BLING Blue Team — `{task_name}` SLA miss. "
                f"Duration: {duration:.0f}s (SLA: {NIGHTLY_BATCH_SLA_SECONDS}s). "
                "Check Celery worker logs."
            )
        }
        httpx.post(webhook, json=msg, timeout=5.0)
    except Exception:
        pass
