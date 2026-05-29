"""
P2-2: Heterogeneous Neo4j schema setup + seed from PostgreSQL.

Creates the hetero graph:
  (:Account)  -[:SENT]->   (:Account)     with amount, timestamp, channel
  (:Account)  -[:USES]->   (:Device)      with device_id, first_seen, last_seen
  (:Account)  -[:HAS_VPA]-> (:VPA)        with vpa_address, created_at, age_days

This makes Tier 2 gates capable of:
  - Cycle detection (SENT paths)
  - SIM-swap / account takeover (shared Device across multiple Accounts)
  - Burner VPA networks (many Accounts sharing a new VPA node)

Run: python scripts/setup_neo4j_schema.py
After seed, restart the blue team API so neo4j_client reloads.
"""
import os
import sys
from datetime import datetime, timezone

import structlog

log = structlog.get_logger()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "blingblue123")
POSTGRES_URL = os.environ.get("POSTGRES_URL", "postgresql://bling_user:trust@localhost:5432/bling_blue")


def get_neo4j_driver():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def create_constraints_and_indexes(session):
    """Create uniqueness constraints + indexes for the hetero schema."""
    cmds = [
        # Uniqueness constraints (also creates indexes)
        "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT device_id_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT vpa_address_unique IF NOT EXISTS FOR (v:VPA) REQUIRE v.address IS UNIQUE",
        # Property indexes for common filter patterns
        "CREATE INDEX account_type_idx IF NOT EXISTS FOR (a:Account) ON (a.account_type)",
        "CREATE INDEX account_fraud_confirmed_idx IF NOT EXISTS FOR (a:Account) ON (a.fraud_confirmed)",
        "CREATE INDEX vpa_age_idx IF NOT EXISTS FOR (v:VPA) ON (v.age_days)",
        "CREATE INDEX sent_timestamp_idx IF NOT EXISTS FOR ()-[r:SENT]-() ON (r.timestamp)",
    ]
    for cmd in cmds:
        try:
            session.run(cmd)
        except Exception as exc:
            print(f"  Schema: {exc} (may already exist)")

    print("Schema constraints and indexes created.")


def seed_from_postgres(session, pg_conn):
    """Read PostgreSQL transactions and populate hetero Neo4j graph."""
    cursor = pg_conn.cursor()

    # ── Accounts ────────────────────────────────────────────────────────────────
    cursor.execute("""
        SELECT id, account_type, kyc_occupation, kyc_age, account_age_days,
               kyc_completeness_score, is_merchant, created_at
        FROM accounts
        LIMIT 50000
    """)
    accounts = cursor.fetchall()
    print(f"Seeding {len(accounts)} accounts...")

    for batch_start in range(0, len(accounts), 500):
        batch = accounts[batch_start:batch_start + 500]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:Account {id: row.id})
            SET a.account_type = row.account_type,
                a.kyc_occupation = row.kyc_occupation,
                a.kyc_age = row.kyc_age,
                a.account_age_days = row.account_age_days,
                a.kyc_completeness_score = row.kyc_completeness_score,
                a.is_merchant = row.is_merchant,
                a.fraud_confirmed = false,
                a.active = true
            """,
            rows=[{
                "id": str(r[0]),
                "account_type": r[1] or "SAVINGS",
                "kyc_occupation": r[2],
                "kyc_age": r[3],
                "account_age_days": r[4] or 0,
                "kyc_completeness_score": float(r[5] or 0),
                "is_merchant": bool(r[6]),
            } for r in batch]
        )

    # ── SENT relationships (30-day window) ─────────────────────────────────────
    cursor.execute("""
        SELECT account_id, payee_account_id, amount, timestamp, channel, id
        FROM transactions
        WHERE timestamp > NOW() - INTERVAL '30 days'
          AND payee_account_id IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 200000
    """)
    txns = cursor.fetchall()
    print(f"Seeding {len(txns)} SENT relationships (30-day window)...")

    for batch_start in range(0, len(txns), 1000):
        batch = txns[batch_start:batch_start + 1000]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:Account {id: row.sender})
            MERGE (b:Account {id: row.payee})
            CREATE (a)-[r:SENT {
                transaction_id: row.txn_id,
                amount: row.amount,
                timestamp: datetime(row.ts),
                channel: row.channel
            }]->(b)
            """,
            rows=[{
                "sender": str(r[0]),
                "payee": str(r[1]),
                "amount": float(r[2]),
                "ts": r[3].isoformat() if r[3] else datetime.now(timezone.utc).isoformat(),
                "channel": r[4] or "UPI",
                "txn_id": str(r[5]),
            } for r in batch]
        )

    # ── Device nodes + USES relationships ──────────────────────────────────────
    # P2-2: Device heterogeneous nodes for SIM-swap / account takeover detection
    cursor.execute("""
        SELECT DISTINCT account_id, device_id, MIN(timestamp) as first_seen, MAX(timestamp) as last_seen
        FROM transactions
        WHERE device_id IS NOT NULL AND device_id != ''
        GROUP BY account_id, device_id
        LIMIT 100000
    """)
    devices = cursor.fetchall()
    print(f"Seeding {len(devices)} Device nodes + USES relationships...")

    for batch_start in range(0, len(devices), 500):
        batch = devices[batch_start:batch_start + 500]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:Account {id: row.account_id})
            MERGE (d:Device {id: row.device_id})
            ON CREATE SET d.first_seen = datetime(row.first_seen)
            SET d.last_seen = datetime(row.last_seen)
            MERGE (a)-[:USES]->(d)
            """,
            rows=[{
                "account_id": str(r[0]),
                "device_id": str(r[1]),
                "first_seen": r[2].isoformat() if r[2] else datetime.now(timezone.utc).isoformat(),
                "last_seen": r[3].isoformat() if r[3] else datetime.now(timezone.utc).isoformat(),
            } for r in batch]
        )

    # ── VPA nodes + HAS_VPA relationships ──────────────────────────────────────
    # P2-2: VPA heterogeneous nodes for burner VPA detection
    cursor.execute("""
        SELECT DISTINCT account_id, payee_vpa, payee_vpa_created_at
        FROM transactions
        WHERE payee_vpa IS NOT NULL AND payee_vpa != ''
        LIMIT 100000
    """)
    vpas = cursor.fetchall()
    print(f"Seeding {len(vpas)} VPA nodes + HAS_VPA relationships...")

    now = datetime.now(timezone.utc)
    for batch_start in range(0, len(vpas), 500):
        batch = vpas[batch_start:batch_start + 500]
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:Account {id: row.account_id})
            MERGE (v:VPA {address: row.vpa})
            ON CREATE SET
                v.created_at = CASE WHEN row.vpa_created IS NOT NULL THEN datetime(row.vpa_created) ELSE null END,
                v.age_days = row.age_days
            MERGE (a)-[:HAS_VPA]->(v)
            """,
            rows=[{
                "account_id": str(r[0]),
                "vpa": str(r[1]),
                "vpa_created": r[2].isoformat() if r[2] else None,
                "age_days": (now - r[2].replace(tzinfo=timezone.utc)).days if r[2] else 0,
            } for r in batch]
        )

    cursor.close()
    print("Seed complete.")


def mark_confirmed_fraud(session):
    """Mark accounts with confirmed fraud from alerts table."""
    try:
        import psycopg2
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT t.account_id
            FROM alerts a
            JOIN transactions t ON t.id = a.transaction_id
            WHERE a.investigator_decision = true
        """)
        fraud_accounts = [str(r[0]) for r in cursor.fetchall()]
        cursor.close()
        conn.close()

        if fraud_accounts:
            session.run(
                "UNWIND $ids AS id MATCH (a:Account {id: id}) SET a.fraud_confirmed = true",
                ids=fraud_accounts
            )
            print(f"Marked {len(fraud_accounts)} confirmed fraud accounts.")
    except Exception as exc:
        print(f"Could not mark fraud accounts: {exc}")


def print_graph_stats(session):
    result = session.run("""
        MATCH (a:Account) WITH count(a) AS accounts
        MATCH ()-[r:SENT]->() WITH accounts, count(r) AS sent
        MATCH ()-[r:USES]->() WITH accounts, sent, count(r) AS uses
        MATCH ()-[r:HAS_VPA]->() WITH accounts, sent, uses, count(r) AS vpas
        RETURN accounts, sent, uses, vpas
    """).single()
    if result:
        print(f"\nNeo4j hetero graph stats:")
        print(f"  Account nodes:   {result['accounts']}")
        print(f"  SENT edges:      {result['sent']}")
        print(f"  USES edges:      {result['uses']}")
        print(f"  HAS_VPA edges:   {result['vpas']}")


def main():
    import psycopg2

    print("=== Neo4j Heterogeneous Schema Setup (P2-2) ===\n")

    print("Connecting to Neo4j...")
    driver = get_neo4j_driver()

    print("Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(POSTGRES_URL)

    with driver.session() as session:
        print("\n1. Creating constraints and indexes...")
        create_constraints_and_indexes(session)

        print("\n2. Seeding from PostgreSQL transactions...")
        seed_from_postgres(session, pg_conn)

        print("\n3. Marking confirmed fraud nodes...")
        mark_confirmed_fraud(session)

        print("\n4. Graph statistics:")
        print_graph_stats(session)

    pg_conn.close()
    driver.close()
    print("\n=== Setup complete. Restart blue team API to pick up Neo4j connection. ===")


if __name__ == "__main__":
    main()
