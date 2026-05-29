"""
Isolated conftest for committee tests.

Overrides the root conftest to avoid importing the full FastAPI app — many existing
files use `X | None` union type syntax (Python 3.10+) that fails on 3.9.6.
Committee tests only need mocks, not the live app.
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Patch settings to avoid .env requirement."""
    import sys
    import os

    # Set required env vars before settings are instantiated
    os.environ.setdefault("POSTGRES_URL", "postgresql://test:test@localhost/test")
    os.environ.setdefault("NEO4J_PASSWORD", "test")
    os.environ.setdefault("GRAPH_ENGINE_API_KEY", "test-graph-key")
    os.environ.setdefault("INVESTIGATOR_API_KEY", "test-investigator-key")
    os.environ.setdefault("SALT", "test-salt")


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_redis():
    with patch("app.utils.redis_client.get_redis") as mock:
        r = MagicMock()
        r.get.return_value = None
        r.hgetall.return_value = {}
        mock.return_value = r
        yield r
