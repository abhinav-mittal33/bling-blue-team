"""shap_access_log — INSERT-only audit table for SHAP value access (P1-6)

Revision ID: 003
Revises: 002
Create Date: 2026-05-28
"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shap_access_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("alert_id", sa.String(36), nullable=False),
        # Pseudonymized with HMAC-SHA256 — never store raw investigator ID
        sa.Column("investigator_id_hash", sa.String(64), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("accessed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_shap_access_alert", "shap_access_log", ["alert_id"])
    op.create_index("idx_shap_access_investigator", "shap_access_log", ["investigator_id_hash"])

    # INSERT-only: role-gated SHAP data requires tamper-proof access log
    op.execute("""
        CREATE RULE shap_access_log_no_update AS
            ON UPDATE TO shap_access_log DO INSTEAD NOTHING;
        CREATE RULE shap_access_log_no_delete AS
            ON DELETE TO shap_access_log DO INSTEAD NOTHING;
    """)


def downgrade() -> None:
    op.execute("DROP RULE IF EXISTS shap_access_log_no_update ON shap_access_log")
    op.execute("DROP RULE IF EXISTS shap_access_log_no_delete ON shap_access_log")
    op.drop_table("shap_access_log")
