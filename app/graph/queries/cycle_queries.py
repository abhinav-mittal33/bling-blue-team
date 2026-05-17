"""
Cypher queries for cycle (round-trip) detection.
Uses pre-computed nightly attributes + delta only — never full traversal.
"""
from app.graph.neo4j_client import run_query

# Only check transactions from last 7 days — pre-computed community labels
# narrow the search space before traversal.
CYCLE_DETECT_QUERY = """
MATCH path = (start:Account {id: $account_id})-[:SENT*2..8]->(start)
WHERE all(r in relationships(path) WHERE r.timestamp > datetime() - duration('P7D'))
RETURN path,
       length(path) AS hops,
       [r in relationships(path) | r.amount] AS amounts,
       [r in relationships(path) | r.timestamp] AS timestamps,
       [n in nodes(path) | n.id] AS node_ids
LIMIT 5
"""

CYCLE_NODE_TYPES_QUERY = """
MATCH (a:Account {id: $account_id})
RETURN a.account_type AS account_type,
       a.kyc_occupation AS kyc_occupation
"""

KYC_RELATIONSHIP_QUERY = """
MATCH (a:Account {id: $account_a})-[:KYC_RELATED]-(b:Account {id: $account_b})
RETURN count(*) AS relationship_count
"""

CYCLE_PATH_NODES_QUERY = """
MATCH (a:Account)
WHERE a.id IN $node_ids
RETURN a.id AS id,
       a.account_type AS account_type,
       a.kyc_occupation AS kyc_occupation,
       a.is_merchant AS is_merchant
"""


def find_cycles(account_id: str) -> list[dict]:
    return run_query(CYCLE_DETECT_QUERY, {"account_id": account_id})


def get_account_type(account_id: str) -> dict:
    results = run_query(CYCLE_NODE_TYPES_QUERY, {"account_id": account_id})
    return results[0] if results else {}


def check_kyc_relationship(account_a: str, account_b: str) -> bool:
    results = run_query(KYC_RELATIONSHIP_QUERY, {"account_a": account_a, "account_b": account_b})
    return bool(results and results[0].get("relationship_count", 0) > 0)


def get_cycle_node_details(node_ids: list[str]) -> list[dict]:
    return run_query(CYCLE_PATH_NODES_QUERY, {"node_ids": node_ids})
