"""Meta-learner versioning table (Phase 2 — Tier 3 upgrade)

Revision ID: 007
Revises: 006
Create Date: 2026-05-29

Creates meta_learner_versions to track trained stacking meta-learner versions.
Enforces at most one active meta-learner via partial unique index.
"""
from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "meta_learner_versions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("version", sa.String(20), nullable=False, unique=True),
        sa.Column("algorithm", sa.String(50), nullable=False),  # 'logistic_regression' | 'gbm_depth3'
        sa.Column("training_sample_count", sa.Integer, nullable=False),
        sa.Column("pr_auc_validation", sa.Float),
        sa.Column("trained_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("deployed_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("model_path", sa.String(200)),
        sa.Column("is_active", sa.Boolean, server_default="false"),
    )
    # At most one active meta-learner: partial unique index on is_active WHERE is_active = true
    op.execute("""
        CREATE UNIQUE INDEX idx_meta_one_active
        ON meta_learner_versions(is_active)
        WHERE is_active = true
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_meta_one_active")
    op.drop_table("meta_learner_versions")
