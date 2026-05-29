"""
celeryconfig.py — Celery Beat schedule for BLING Blue Team (P0-2).
Replaces APScheduler. All recurring tasks defined here.

Start worker: celery -A app.celery_app worker --loglevel=info
Start beat:   celery -A app.celery_app beat --loglevel=info --scheduler=celery.beat:PersistentScheduler
"""
from celery.schedules import crontab

# ── Task routes ───────────────────────────────────────────────────────────────
task_routes = {
    "app.evidence.trail_builder.reconstruct_fund_trail":    {"queue": "evidence"},
    "evidence.compute_shap":                                {"queue": "evidence"},
    "app.graph.tasks.run_nightly_batch_task":               {"queue": "graph"},
    "app.graph.tasks.run_node2vec_task":                    {"queue": "graph"},
    "app.graph.tasks.update_betweenness_task":              {"queue": "graph"},
    "app.graph.tasks.run_micro_batch_task":                 {"queue": "graph"},
    "app.graph.tasks.refresh_gnn_embeddings_task":          {"queue": "graph"},
    "app.compliance.psi_monitor.run_psi_check":             {"queue": "compliance"},
    "app.integrations.sanctions_client.sync_sanctions":     {"queue": "compliance"},
}

# ── Dead-letter queue config ─────────────────────────────────────────────────
task_acks_late = True
task_reject_on_worker_lost = True

# ── Beat schedule ─────────────────────────────────────────────────────────────
beat_schedule = {
    # Nightly graph feature computation — runs at 3am UTC
    # SLA: must complete by 4am (1h budget). See run_nightly_batch_with_sla().
    "nightly-graph-batch": {
        "task": "app.graph.tasks.run_nightly_batch_task",
        "schedule": crontab(hour=3, minute=0),
        "options": {"queue": "graph"},
    },
    # Approximate betweenness centrality — updated every 2 hours
    # Uses k=500 approximation via NetworkX. Updates only betweenness_centrality
    # field in feat:{account} — does NOT overwrite the full hash.
    "2h-betweenness": {
        "task": "app.graph.tasks.update_betweenness_task",
        "schedule": crontab(minute=0, hour="*/2"),
        "options": {"queue": "graph"},
    },
    # Fast-changing features micro-batch — degree, temporal_acceleration, sink_score
    "5min-micro-batch": {
        "task": "app.graph.tasks.run_micro_batch_task",
        "schedule": 300,  # every 300 seconds
        "options": {"queue": "graph"},
    },
    # Node2Vec embedding computation — runs at 4:30am UTC (after nightly batch, P2-6)
    "nightly-node2vec": {
        "task": "app.graph.tasks.run_node2vec_task",
        "schedule": crontab(hour=4, minute=30),
        "options": {"queue": "graph"},
    },
    # PSI drift monitoring — weekly (Phase 4, stubbed until P4-7)
    "weekly-psi-check": {
        "task": "app.compliance.psi_monitor.run_psi_check",
        "schedule": crontab(day_of_week="monday", hour=6, minute=0),
        "options": {"queue": "compliance"},
    },
    # Sanctions list sync — daily at 2:30am UTC (Phase 5, stubbed until P5-6)
    "daily-sanctions-sync": {
        "task": "app.integrations.sanctions_client.sync_sanctions",
        "schedule": crontab(hour=2, minute=30),
        "options": {"queue": "compliance"},
    },
    # GNN embedding refresh — every 5 minutes for recently-active accounts (Scorer B)
    # Updates emb:{account} for accounts active in the last hour.
    "5min-gnn-refresh": {
        "task": "app.graph.tasks.refresh_gnn_embeddings_task",
        "schedule": 300,
        "options": {"queue": "graph"},
    },
    # DLQ depth monitor — every 15 minutes
    "dlq-monitor": {
        "task": "app.graph.tasks.check_dlq_depth",
        "schedule": 900,  # every 900 seconds
        "options": {"queue": "graph"},
    },
    # DPDP retention cleanup — daily (Phase 5, stubbed until P5-3)
    "daily-dpdp-cleanup": {
        "task": "app.compliance.dpdp.cleanup_expired_features",
        "schedule": crontab(hour=4, minute=0),
        "options": {"queue": "compliance"},
    },
}
