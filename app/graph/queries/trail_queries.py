"""
Cypher queries for fund trail reconstruction.
Used by trail_builder.py (Celery async task only — never in API path).
"""
from app.graph.neo4j_client import run_query

FORWARD_TRAIL_QUERY = """
MATCH path = (start:Account {id: $account_id})-[:SENT*1..10]->(end:Account)
WHERE all(r in relationships(path) WHERE r.timestamp >= $seed_timestamp)
RETURN [n in nodes(path) | n.id] AS account_ids,
       [r in relationships(path) | r.amount] AS amounts,
       [r in relationships(path) | r.channel] AS channels,
       [r in relationships(path) | toString(r.timestamp)] AS timestamps,
       length(path) AS hop_count
ORDER BY length(path) DESC
LIMIT 20
"""

BACKWARD_TRAIL_QUERY = """
MATCH path = (start:Account)-[:SENT*1..10]->(end:Account {id: $account_id})
WHERE all(r in relationships(path) WHERE r.timestamp <= $seed_timestamp)
RETURN [n in nodes(path) | n.id] AS account_ids,
       [r in relationships(path) | r.amount] AS amounts,
       [r in relationships(path) | r.channel] AS channels,
       [r in relationships(path) | toString(r.timestamp)] AS timestamps,
       length(path) AS hop_count
ORDER BY length(path) DESC
LIMIT 20
"""

NODE_DETAILS_QUERY = """
MATCH (a:Account)
WHERE a.id IN $account_ids
RETURN a.id AS id,
       a.account_type AS account_type,
       a.kyc_occupation AS kyc_occupation,
       a.is_merchant AS is_merchant,
       a.account_age_days AS account_age_days
"""


def trace_forward(account_id: str, seed_timestamp: str) -> list[dict]:
    return run_query(FORWARD_TRAIL_QUERY, {
        "account_id": account_id,
        "seed_timestamp": seed_timestamp,
    })


def trace_backward(account_id: str, seed_timestamp: str) -> list[dict]:
    return run_query(BACKWARD_TRAIL_QUERY, {
        "account_id": account_id,
        "seed_timestamp": seed_timestamp,
    })


def get_node_details(account_ids: list[str]) -> list[dict]:
    return run_query(NODE_DETAILS_QUERY, {"account_ids": account_ids})
