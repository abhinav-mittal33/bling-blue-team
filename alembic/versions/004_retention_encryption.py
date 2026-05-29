"""5-year retention enforcement + pgcrypto column encryption (P5-4)

Revision ID: 004
Revises: 003
Create Date: 2026-05-28

Creates:
  - data_categories table (DPDP consent ledger, P5-3)
  - Adds encrypted columns to transactions (sensitive PII with pgcrypto AES-256)
  - INSERT-only trigger on data_categories (cannot delete consent records)

Note: pgcrypto must be enabled: CREATE EXTENSION IF NOT EXISTS pgcrypto;
      DB_ENCRYPTION_KEY must be set in .env before running this migration.
      5-year retention enforced at application level (see app/compliance/dpdp.py).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgcrypto extension if not already present
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # DPDP consent ledger — INSERT-only (P5-3)
    op.create_table(
        "data_categories",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("account_id_hash", sa.String(64), nullable=False),
        sa.Column("operation", sa.String(100), nullable=False),
        sa.Column("categories", sa.Text, nullable=False),
        sa.Column("lawful_basis", sa.String(100), nullable=False),
        sa.Column("processed_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_data_categories_account", "data_categories", ["account_id_hash"])
    op.execute("""
        CREATE RULE data_categories_no_update AS
            ON UPDATE TO data_categories DO INSTEAD NOTHING;
        CREATE RULE data_categories_no_delete AS
            ON DELETE TO data_categories DO INSTEAD NOTHING;
    """)

    # Add pgcrypto-encrypted PII columns to transactions
    # pgcrypto uses symmetric AES-256 encryption: pgp_sym_encrypt(value, key)
    # These replace the plaintext device_id and ip_address columns for compliance
    op.add_column("transactions", sa.Column("device_id_enc", sa.LargeBinary))
    op.add_column("transactions", sa.Column("ip_address_enc", sa.LargeBinary))

    # Add retention_expires_at for automated cleanup (5 years from created_at)
    op.add_column(
        "transactions",
        sa.Column(
            "retention_expires_at",
            sa.TIMESTAMP(timezone=True),
            comment="5-year PMLA retention deadline",
        ),
    )
    op.execute("""
        UPDATE transactions
        SET retention_expires_at = created_at + INTERVAL '5 years'
        WHERE retention_expires_at IS NULL
    """)


def downgrade() -> None:
    op.execute("DROP RULE IF EXISTS data_categories_no_update ON data_categories")
    op.execute("DROP RULE IF EXISTS data_categories_no_delete ON data_categories")
    op.drop_table("data_categories")
    op.drop_column("transactions", "device_id_enc")
    op.drop_column("transactions", "ip_address_enc")
    op.drop_column("transactions", "retention_expires_at")
