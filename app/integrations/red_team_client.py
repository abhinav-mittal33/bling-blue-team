from __future__ import annotations
"""
Red Team notification client.
Fires on confirmed_fraud=True feedback so Red Team can add the pattern to attack corpus.
Best-effort — failure does NOT block the feedback response.

P7-1 (D-03): Added density_override to DNA payload.
Allows Red Team sandbox to replay the pattern with configurable network densities
for evasion testing (e.g., density_override=0.3 = sparse network, density_override=0.9 = dense).
"""
import structlog
from app.core.config import settings

logger = structlog.get_logger()


def notify_confirmed_fraud(
    alert_id: str,
    fraud_type: str | None,
    transaction_id: str,
    graph_features: dict | None = None,
    density_override: float | None = None,
) -> bool:
    """
    Notify Red Team of a confirmed fraud case with fraud DNA payload.
    density_override (P7-1): optional bipartite density for sandbox replay.
      0.0-1.0 range. None = use actual graph density from graph_features.
    Returns True on success, False on any failure.
    """
    if not settings.red_team_endpoint:
        logger.warning("red_team_endpoint_not_configured")
        return False

    # Compute density from graph_features if not overridden
    actual_density = None
    if graph_features:
        actual_density = float(graph_features.get("bipartite_score") or 0)
    sandbox_density = density_override if density_override is not None else actual_density

    try:
        import httpx
        response = httpx.post(
            settings.red_team_endpoint,
            json={
                "alert_id": alert_id,
                "transaction_id": transaction_id,
                "fraud_type": fraud_type,
                "source": "blue_team_feedback",
                # P7-1: D-03 sandbox density override
                "density_override": sandbox_density,
                "fraud_dna": {
                    "pagerank_fraud_seeded": float(graph_features.get("pagerank_fraud_seeded") or 0) if graph_features else None,
                    "community_fraud_ratio": float(graph_features.get("community_fraud_ratio") or 0) if graph_features else None,
                    "cycle_membership": float(graph_features.get("cycle_membership") or 0) if graph_features else None,
                    "sink_score": float(graph_features.get("sink_score") or 0) if graph_features else None,
                },
            },
            headers={"X-API-Key": settings.red_team_api_key or ""},
            timeout=5.0,
        )
        response.raise_for_status()
        logger.info("red_team_notified", alert_id=alert_id, fraud_type=fraud_type, density=sandbox_density)
        return True
    except Exception as exc:
        logger.error("red_team_notification_failed", alert_id=alert_id, error=str(exc))
        return False
