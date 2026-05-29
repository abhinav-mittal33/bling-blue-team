"""Committee shadow scoring table (Phase 0 — Tier 3 upgrade)

Revision ID: 005
Revises: 004
Create Date: 2026-05-29

Creates shadow_score_committee to hold all 5-scorer outputs alongside the
live single-XGBoost decision. NOT immutable — meta-learner training reads and
writes back meta_score + specialist_override during Phase 2.
"""
from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shadow_score_committee",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("transaction_id", sa.String(36), sa.ForeignKey("transactions.id"), nullable=False),
        sa.Column("scored_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        # Scorer A — upgraded GBM
        sa.Column("scorer_a_score", sa.Float),
        sa.Column("scorer_a_confidence", sa.Float),
        sa.Column("scorer_a_missing_flag", sa.Boolean, server_default="false"),
        # Scorer B — graph embeddings
        sa.Column("scorer_b_score", sa.Float),
        sa.Column("scorer_b_confidence", sa.Float),
        sa.Column("scorer_b_missing_flag", sa.Boolean, server_default="false"),
        # Scorer C — prototype bank
        sa.Column("scorer_c_score", sa.Float),
        sa.Column("scorer_c_confidence", sa.Float),
        sa.Column("scorer_c_missing_flag", sa.Boolean, server_default="false"),
        # Scorer D — sequence / set-based
        sa.Column("scorer_d_score", sa.Float),
        sa.Column("scorer_d_confidence", sa.Float),
        sa.Column("scorer_d_missing_flag", sa.Boolean, server_default="false"),
        # Scorer F — remark screener
        sa.Column("scorer_f_score", sa.Float),
        sa.Column("scorer_f_confidence", sa.Float),
        sa.Column("scorer_f_missing_flag", sa.Boolean, server_default="false"),
        # Meta-decision (populated in Phase 2)
        sa.Column("meta_score", sa.Float),
        sa.Column("specialist_override", sa.Boolean, server_default="false"),
        sa.Column("final_committee_score", sa.Float),
        # Conformal calibration intervals (Phase 2)
        sa.Column("mapie_lower", sa.Float),
        sa.Column("mapie_upper", sa.Float),
        # Ground truth from existing pipeline (for validation)
        sa.Column("live_score", sa.Float, nullable=False),
        sa.Column("live_action", sa.String(20), nullable=False),
    )
    op.create_index("idx_shadow_txn", "shadow_score_committee", ["transaction_id"])
    op.create_index("idx_shadow_scored_at", "shadow_score_committee", ["scored_at"])


def downgrade() -> None:
    op.drop_index("idx_shadow_scored_at", table_name="shadow_score_committee")
    op.drop_index("idx_shadow_txn", table_name="shadow_score_committee")
    op.drop_table("shadow_score_committee")
