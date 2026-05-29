"""Feedback queue tables to replace River FTRL (Phase 3 — Tier 3 upgrade)

Revision ID: 006
Revises: 005
Create Date: 2026-05-29

Creates:
  - curated_dataset_queue: labeled examples for next scheduled retrain
  - prototype_injection_candidates: confirmed-fraud cases flagged for prototype injection
  - reviewed_novelty_registry: dedup registry for anomaly discovery pipeline

These three tables collectively replace the River FTRL feedback path.
False positive feedback → curated_dataset_queue (label=0) + reviewed_novelty_registry
Confirmed fraud feedback → prototype_injection_candidates (developer reviews before injecting)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Labeled dataset queue — feeds next XGBoost retrain
    op.create_table(
        "curated_dataset_queue",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("transaction_id", sa.String(36), nullable=False),
        sa.Column("alert_id", sa.String(36), sa.ForeignKey("alerts.id")),
        sa.Column("label", sa.Integer, nullable=False),  # 0=benign, 1=fraud
        sa.Column("label_source", sa.String(50), nullable=False),
        sa.Column("feature_vector", JSONB),
        sa.Column("investigator_id_hash", sa.String(64)),
        sa.Column("added_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("batch_exported", sa.Boolean, server_default="false"),
        sa.Column("batch_exported_at", sa.TIMESTAMP(timezone=True)),
        sa.CheckConstraint("label IN (0, 1)", name="chk_curated_label"),
    )
    op.create_index("idx_curated_label", "curated_dataset_queue", ["label"])
    op.create_index("idx_curated_exported", "curated_dataset_queue", ["batch_exported"])

    # Prototype injection candidates — developer reviews before injecting into vault
    op.create_table(
        "prototype_injection_candidates",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("transaction_id", sa.String(36), nullable=False),
        sa.Column("alert_id", sa.String(36), sa.ForeignKey("alerts.id")),
        sa.Column("fraud_type", sa.String(100)),
        sa.Column("feature_vector", JSONB, nullable=False),
        sa.Column("investigator_id_hash", sa.String(64)),
        sa.Column("investigator_notes", sa.Text),
        sa.Column("status", sa.String(30), nullable=False, server_default="'PENDING_REVIEW'"),
        sa.Column("developer_notes", sa.Text),
        sa.Column("submitted_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("reviewed_at", sa.TIMESTAMP(timezone=True)),
    )
    op.create_index("idx_proto_status", "prototype_injection_candidates", ["status"])
    op.create_index("idx_proto_submitted", "prototype_injection_candidates", ["submitted_at"])

    # Reviewed novelty registry — dedup for anomaly discovery pipeline
    op.create_table(
        "reviewed_novelty_registry",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("fingerprint", sa.String(64), nullable=False, unique=True),
        sa.Column("centroid_features", JSONB, nullable=False),
        sa.Column("label", sa.Integer, nullable=False),  # 0=benign, 1=confirmed_novel_fraud
        sa.Column("source_transaction_id", sa.String(36)),
        sa.Column("registered_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("registered_by", sa.String(64)),
        sa.CheckConstraint("label IN (0, 1)", name="chk_registry_label"),
    )
    op.create_index("idx_registry_fingerprint", "reviewed_novelty_registry", ["fingerprint"])
    op.create_index("idx_registry_label", "reviewed_novelty_registry", ["label"])


def downgrade() -> None:
    op.drop_index("idx_registry_label", table_name="reviewed_novelty_registry")
    op.drop_index("idx_registry_fingerprint", table_name="reviewed_novelty_registry")
    op.drop_table("reviewed_novelty_registry")

    op.drop_index("idx_proto_submitted", table_name="prototype_injection_candidates")
    op.drop_index("idx_proto_status", table_name="prototype_injection_candidates")
    op.drop_table("prototype_injection_candidates")

    op.drop_index("idx_curated_exported", table_name="curated_dataset_queue")
    op.drop_index("idx_curated_label", table_name="curated_dataset_queue")
    op.drop_table("curated_dataset_queue")
