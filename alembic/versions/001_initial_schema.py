"""Initial schema — all 7 BLING Blue Team tables

Revision ID: 001
Revises:
Create Date: 2026-05-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, INET

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── accounts ─────────────────────────────────────────────────────────────
    op.create_table(
        "accounts",
        sa.Column("id", sa.String(20), primary_key=True),
        sa.Column("account_type", sa.String(20), nullable=False),
        sa.Column("kyc_occupation", sa.String(100)),
        sa.Column("kyc_declared_income_monthly", sa.Numeric(15, 2)),
        sa.Column("kyc_age", sa.Integer),
        sa.Column("kyc_home_state", sa.String(50)),
        sa.Column("kyc_completeness_score", sa.Float, server_default="0.0"),
        sa.Column("account_age_days", sa.Integer),
        sa.Column("is_merchant", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )

    # ── transactions (append-only) ────────────────────────────────────────────
    op.create_table(
        "transactions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("account_id", sa.String(20), sa.ForeignKey("accounts.id"), nullable=False),
        sa.Column("payee_account_id", sa.String(20)),
        sa.Column("amount", sa.Numeric(15, 2), nullable=False),
        sa.Column("channel", sa.String(10), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("payee_vpa", sa.String(100)),
        sa.Column("payee_vpa_created_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("merchant_terminal_id", sa.String(50)),
        sa.Column("merchant_mcc", sa.String(10)),
        sa.Column("device_id", sa.String(100)),
        sa.Column("ip_address", INET),
        sa.Column("geo_city", sa.String(50)),
        sa.Column("geo_state", sa.String(50)),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_transactions_account_timestamp", "transactions", ["account_id", sa.text("timestamp DESC")])
    op.create_index("idx_transactions_payee", "transactions", ["payee_account_id"])
    op.create_index("idx_transactions_timestamp", "transactions", [sa.text("timestamp DESC")])

    # ── fraud_scores ──────────────────────────────────────────────────────────
    op.create_table(
        "fraud_scores",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("transaction_id", sa.String(36), sa.ForeignKey("transactions.id"), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("gate_fired", sa.String(50)),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("tier1_flags", JSONB),
        sa.Column("tier2_gate", sa.String(50)),
        sa.Column("tier3_score", sa.Float),
        sa.Column("feature_vector", JSONB),
        sa.Column("shap_values", JSONB),
        sa.Column("indian_context_applied", JSONB),
        sa.Column("model_version", sa.String(20)),
        sa.Column("processing_ms", sa.Integer),
        sa.Column("scored_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # ── alerts ────────────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("transaction_id", sa.String(36), sa.ForeignKey("transactions.id"), nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("gate", sa.String(50)),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="OPEN"),
        sa.Column("trail_status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("evidence_package", JSONB),
        sa.Column("fraud_type", sa.String(100)),
        sa.Column("investigator_id", sa.String(100)),
        sa.Column("investigator_decision", sa.Boolean),
        sa.Column("investigator_notes", sa.Text),
        sa.Column("blockchain_sealed", sa.Boolean, server_default="false"),
        sa.Column("red_team_notified", sa.Boolean, server_default="false"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # ── feedback_log ──────────────────────────────────────────────────────────
    op.create_table(
        "feedback_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("alert_id", sa.String(36), sa.ForeignKey("alerts.id"), nullable=False),
        sa.Column("transaction_id", sa.String(36), nullable=False),
        sa.Column("label", sa.Integer, nullable=False),
        sa.Column("investigator_id", sa.String(100)),
        sa.Column("model_version_at_feedback", sa.String(20)),
        sa.Column("features_at_scoring", JSONB),
        sa.Column("feedback_timestamp", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # ── graph_features_cache ──────────────────────────────────────────────────
    op.create_table(
        "graph_features_cache",
        sa.Column("account_id", sa.String(20), sa.ForeignKey("accounts.id"), primary_key=True),
        sa.Column("degree_centrality", sa.Float),
        sa.Column("betweenness_centrality", sa.Float),
        sa.Column("clustering_coefficient", sa.Float),
        sa.Column("pagerank_fraud_seeded", sa.Float),
        sa.Column("community_id", sa.Integer),
        sa.Column("community_fraud_ratio", sa.Float),
        sa.Column("shortest_path_to_fraud", sa.Integer),
        sa.Column("cycle_membership", sa.Boolean, server_default="false"),
        sa.Column("sink_score", sa.Float),
        sa.Column("bipartite_score", sa.Float),
        sa.Column("fan_out_ratio", sa.Float),
        sa.Column("computed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # ── model_audit (IMMUTABLE — INSERT ONLY) ─────────────────────────────────
    op.create_table(
        "model_audit",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("transaction_id", sa.String(36)),
        sa.Column("model_version", sa.String(20), nullable=False),
        sa.Column("event_data", JSONB, nullable=False),
        sa.Column("event_timestamp", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # DB-level immutability — required for RBI PMLA Section 12 compliance
    op.execute("""
        CREATE RULE no_update_model_audit AS
            ON UPDATE TO model_audit DO INSTEAD NOTHING;
    """)
    op.execute("""
        CREATE RULE no_delete_model_audit AS
            ON DELETE TO model_audit DO INSTEAD NOTHING;
    """)


def downgrade() -> None:
    # Drop rules before dropping table
    op.execute("DROP RULE IF EXISTS no_update_model_audit ON model_audit;")
    op.execute("DROP RULE IF EXISTS no_delete_model_audit ON model_audit;")

    op.drop_table("model_audit")
    op.drop_table("graph_features_cache")
    op.drop_table("feedback_log")
    op.drop_table("alerts")
    op.drop_table("fraud_scores")
    op.drop_index("idx_transactions_timestamp", table_name="transactions")
    op.drop_index("idx_transactions_payee", table_name="transactions")
    op.drop_index("idx_transactions_account_timestamp", table_name="transactions")
    op.drop_table("transactions")
    op.drop_table("accounts")
