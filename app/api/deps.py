from sqlalchemy.orm import Session
from app.utils.postgres_client import get_db

# Re-export for use in routers via FastAPI Depends
__all__ = ["get_db"]
