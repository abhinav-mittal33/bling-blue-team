from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # PostgreSQL
    postgres_url: str

    # Neo4j (teammate's service — read-only)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Redis
    redis_url: str = "redis://localhost:6379"

    # API Keys
    graph_engine_api_key: str
    investigator_api_key: str
    internal_api_key: str = ""

    # Integration endpoints
    blockchain_service_url: str = "http://localhost:8001"
    red_team_service_url: str = "http://localhost:8002"
    investigator_dashboard_url: str = "http://localhost:8003"

    # Integration client settings (optional — best-effort)
    blockchain_endpoint: str = ""
    blockchain_api_key: str = ""
    red_team_endpoint: str = ""
    red_team_api_key: str = ""
    investigator_webhook_url: str = ""
    investigator_webhook_key: str = ""

    # Online model persistence
    online_model_path: str = "ml/river_ftrl.json"

    # PII pseudonymization
    salt: str

    # Model
    model_version: str = "v1.0"

    # Score thresholds
    threshold_log: float = 0.38
    threshold_review: float = 0.62
    threshold_high_risk: float = 0.83

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # App
    debug: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

    @property
    def valid_api_keys(self) -> set[str]:
        keys = {self.graph_engine_api_key, self.investigator_api_key}
        if self.internal_api_key:
            keys.add(self.internal_api_key)
        return keys

    @property
    def graph_engine_keys(self) -> set[str]:
        keys = {self.graph_engine_api_key}
        if self.internal_api_key:
            keys.add(self.internal_api_key)
        return keys

    @property
    def investigator_keys(self) -> set[str]:
        keys = {self.investigator_api_key}
        if self.internal_api_key:
            keys.add(self.internal_api_key)
        return keys


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
