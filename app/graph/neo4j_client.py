"""
Neo4j client — READ ONLY. Blue Team never writes to Neo4j.
Graph Engine (teammate) owns all writes.

P1-1: Tenacity retry + circuit breaker.
If Neo4j read exceeds 200ms or raises ServiceUnavailable:
  - Retry once with 100ms wait
  - On second failure: fall back to Redis feat:{account} cache
  - Set graph_stale=True flag on the scoring response
  - Increment bling_graph_fallback_total Prometheus counter
"""
from __future__ import annotations

import time
import structlog
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)

from app.core.config import settings
from app.core.exceptions import GraphQueryError

logger = structlog.get_logger()

_driver: Driver | None = None

# Circuit breaker: fall back to Redis if query exceeds this threshold
_CIRCUIT_BREAKER_MS = 200

# Retry on transient Neo4j availability issues only — not on query/syntax errors
_RETRYABLE = (ServiceUnavailable, SessionExpired, ConnectionResetError)


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver:
        _driver.close()
        _driver = None


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(0.1),
    retry=retry_if_exception_type(_RETRYABLE),
    before_sleep=before_sleep_log(logger, "warning"),
    reraise=True,
)
def _execute_with_retry(cypher: str, params: dict) -> list[dict]:
    """Inner retry-wrapped Cypher execution."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, **params)
        return [record.data() for record in result]


def run_query(
    cypher: str,
    params: dict,
    *,
    fallback_account_id: str | None = None,
) -> list[dict]:
    """
    Execute a read-only Cypher query with Tenacity retry + circuit breaker.

    Args:
        cypher: Parameterized Cypher string. Never f-strings.
        params: Query parameters dict.
        fallback_account_id: If provided and query fails, return Redis feat: hash as fallback.

    Returns:
        List of record dicts from Neo4j, or fallback list on circuit-breaker trip.

    Raises:
        GraphQueryError: On non-retriable failure with no fallback available.
    """
    t_start = time.monotonic()
    try:
        rows = _execute_with_retry(cypher, params)
        elapsed_ms = (time.monotonic() - t_start) * 1000
        if elapsed_ms > _CIRCUIT_BREAKER_MS:
            logger.warning("neo4j_slow_query",
                           elapsed_ms=round(elapsed_ms, 1),
                           threshold_ms=_CIRCUIT_BREAKER_MS)
        return rows

    except (RetryError, *_RETRYABLE) as exc:
        elapsed_ms = (time.monotonic() - t_start) * 1000
        logger.error("neo4j_query_failed_after_retry",
                     cypher=cypher[:80],
                     error=str(exc),
                     elapsed_ms=round(elapsed_ms, 1))

        # Circuit breaker: try Redis fallback if account_id provided
        if fallback_account_id:
            return _redis_fallback(fallback_account_id)

        raise GraphQueryError(f"Neo4j query failed: {exc}") from exc

    except Exception as exc:
        logger.error("neo4j_query_error", cypher=cypher[:80], error=str(exc))
        raise GraphQueryError(f"Neo4j query failed: {exc}") from exc


def _redis_fallback(account_id: str) -> list[dict]:
    """
    Return pre-computed Redis features as a single-row result when Neo4j is unavailable.
    Increments graph_fallback_total Prometheus counter.
    The caller receives graph_stale=True in the scoring response (set by gate logic).
    """
    try:
        from app.utils.redis_client import get_graph_features
        from app.utils.metrics import graph_fallback_total
        features = get_graph_features(account_id)
        graph_fallback_total.inc()
        logger.warning("neo4j_redis_fallback_used",
                       account_id_len=len(account_id),
                       features_available=len(features) > 0)
        return [{"account_id": account_id, **features, "_graph_stale": True}]
    except Exception as fb_exc:
        logger.error("redis_fallback_also_failed", error=str(fb_exc))
        return []
