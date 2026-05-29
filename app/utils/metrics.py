from prometheus_client import Counter, Histogram, Gauge

scoring_requests_total = Counter(
    "bling_scoring_requests_total",
    "Total scoring requests",
    ["action", "gate_fired"],
)

scoring_latency_ms = Histogram(
    "bling_scoring_latency_ms",
    "Scoring pipeline latency in milliseconds",
    buckets=[5, 10, 20, 30, 50, 75, 90, 100, 150, 200, 500],
)

alerts_created_total = Counter(
    "bling_alerts_created_total",
    "Total alerts created",
    ["action"],
)

feedback_received_total = Counter(
    "bling_feedback_received_total",
    "Total investigator feedback events",
    ["confirmed_fraud"],
)

neo4j_query_latency_ms = Histogram(
    "bling_neo4j_query_latency_ms",
    "Neo4j query latency in milliseconds",
    ["query_type"],
    buckets=[1, 5, 10, 20, 50, 100, 200, 500],
)

model_version_info = Gauge(
    "bling_model_version_info",
    "Current model version",
    ["version"],
)

novelty_flags_total = Counter(
    "bling_novelty_flags_total",
    "Total transactions flagged as structurally novel by Isolation Forest",
)

novelty_escalations_total = Counter(
    "bling_novelty_escalations_total",
    "Novel patterns escalated to Red Team (fingerprint seen 10+ times in 7 days)",
)

# Phase 0 — P0-2 Celery Beat metrics
nightly_batch_duration_seconds = Histogram(
    "bling_nightly_batch_duration_seconds",
    "Nightly batch computation duration in seconds",
    buckets=[60, 300, 600, 1200, 1800, 2700, 3600, 5400],
)

nightly_batch_failure_total = Counter(
    "bling_nightly_batch_failure_total",
    "Total nightly batch failures (triggers SLA alert)",
)

# Phase 1 — P1-1 Neo4j circuit breaker
graph_fallback_total = Counter(
    "bling_graph_fallback_total",
    "Times Neo4j was unavailable and scoring fell back to stale Redis cache",
)
