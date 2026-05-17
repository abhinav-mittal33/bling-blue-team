"""
Shared pytest fixtures for all test modules.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.core.config import settings


@pytest.fixture
def client():
    """FastAPI test client with internal API key."""
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": settings.internal_api_key or "test-internal-key"})
        yield c


@pytest.fixture
def graph_engine_client():
    """Test client simulating Graph Engine caller."""
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": settings.graph_engine_api_key})
        yield c


@pytest.fixture
def investigator_client():
    """Test client simulating Investigator Dashboard caller."""
    with TestClient(app) as c:
        c.headers.update({"X-API-Key": settings.investigator_api_key})
        yield c


@pytest.fixture
def mock_redis():
    with patch("app.utils.redis_client.get_redis") as mock:
        r = MagicMock()
        r.get.return_value = None
        r.hgetall.return_value = {}
        r.ping.return_value = True
        mock.return_value = r
        yield r


@pytest.fixture
def mock_neo4j():
    with patch("app.graph.neo4j_client.get_driver") as mock:
        driver = MagicMock()
        mock.return_value = driver
        yield driver
