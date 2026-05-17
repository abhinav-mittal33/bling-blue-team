"""Add novelty_queue table for Isolation Forest findings

Revision ID: 002
Revises: 001
Create Date: 2026-05-17
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "novelty_queue",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        # Transaction that was flagged as structurally novel
        sa.Column("transaction_id", sa.String(36), nullable=False, unique=True),
        sa.Column("account_id", sa.String(20), nullable=False),
        # Scores — for developer context only, NOT for fraud classification
        sa.Column("anomaly_score", sa.Float, nullable=False),
        sa.Column("fraud_score", sa.Float, nullable=False),
        sa.Column("fraud_action", sa.String(20), nullable=False),
        sa.Column("gate_fired", sa.String(50), nullable=True),
        # Fingerprint groups similar novelty patterns together
        sa.Column("novelty_fingerprint", sa.String(16), nullable=False),
        sa.Column("fingerprint_occurrences", sa.Integer, nullable=False, server_default="1"),
        # Full graph feature snapshot — developer inspects this to understand the pattern
        sa.Column("graph_features_snapshot", JSONB, nullable=True),
        # Escalation: True when same fingerprint seen 10+ times in 7 days
        sa.Column("requires_escalation", sa.Boolean, nullable=False, server_default="false"),
        # Developer review workflow
        # Status values:
        #   PENDING_REVIEW    — not yet reviewed
        #   REVIEWED_NORMAL   — confirmed legitimate unusual behavior
        #   REVIEWED_NEW_FRAUD — confirmed new fraud pattern
        #   NEW_GATE_ADDED    — detection gate written for this pattern
        sa.Column("status", sa.String(30), nullable=False, server_default="'PENDING_REVIEW'"),
        sa.Column("developer_notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index("idx_novelty_status", "novelty_queue", ["status"])
    op.create_index("idx_novelty_fingerprint", "novelty_queue", ["novelty_fingerprint"])
    op.create_index("idx_novelty_escalation", "novelty_queue", ["requires_escalation"])
    op.create_index("idx_novelty_created", "novelty_queue", ["created_at"])
    # Note: this table is NOT immutable — developers update status and notes.
    # The model_audit table (from 001) is the immutable legal record.


def downgrade():
    op.drop_table("novelty_queue")
