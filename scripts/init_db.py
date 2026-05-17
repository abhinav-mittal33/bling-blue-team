"""
Initialize PostgreSQL database. Drops and recreates all tables.
WILL DESTROY ALL DATA. Never run against production.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import settings
from app.models.database import Base
from app.utils.postgres_client import engine

PROD_KEYWORDS = ("prod", "production", "bling_prod", "live")


def check_not_production() -> None:
    url = settings.postgres_url.lower()
    for keyword in PROD_KEYWORDS:
        if keyword in url:
            print(f"ERROR: POSTGRES_URL contains '{keyword}' — refusing to drop production database.")
            sys.exit(1)


def init_db() -> None:
    check_not_production()
    print(f"Dropping all tables on: {settings.postgres_url}")
    Base.metadata.drop_all(bind=engine)
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)
    print("Running immutability rules for model_audit...")
    with engine.connect() as conn:
        conn.execute(
            __import__("sqlalchemy").text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_rules
                        WHERE tablename = 'model_audit' AND rulename = 'no_update_model_audit'
                    ) THEN
                        CREATE RULE no_update_model_audit AS
                            ON UPDATE TO model_audit DO INSTEAD NOTHING;
                    END IF;
                END$$;
            """)
        )
        conn.execute(
            __import__("sqlalchemy").text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_rules
                        WHERE tablename = 'model_audit' AND rulename = 'no_delete_model_audit'
                    ) THEN
                        CREATE RULE no_delete_model_audit AS
                            ON DELETE TO model_audit DO INSTEAD NOTHING;
                    END IF;
                END$$;
            """)
        )
        conn.commit()
    print("Database initialized successfully.")
    print("Next steps:")
    print("  alembic stamp 001  (mark migration as applied)")
    print("  python ml/train.py")
    print("  python scripts/generate_test_data.py && python scripts/load_sample_data.py")


if __name__ == "__main__":
    init_db()
