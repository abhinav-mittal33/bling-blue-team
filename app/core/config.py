from pydantic_settings import BaseSettings
from pydantic import ConfigDict, model_validator
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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

    # PII pseudonymization
    salt: str
    pseudonymization_key: str = ""  # 32-byte hex for HMAC (P1-8). Falls back to sha256+salt if empty.

    # Model
    model_version: str = "v1.0"

    # Score thresholds (updated after Phase 4 calibration — see docs/IMPLEMENTATION_PLAN.md)
    threshold_log: float = 0.38
    threshold_review: float = 0.62
    threshold_high_risk: float = 0.83

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # App
    debug: bool = False
    log_level: str = "INFO"

    # JWT RS256 (P1-7) — PEM strings from .env, empty = JWT disabled (X-API-Key only)
    jwt_public_key: str = ""
    jwt_private_key: str = ""
    jwt_expiry_seconds: int = 900

    # Phase 1 additions
    ensemble_alpha: float = 0.65          # P4-3 ensemble XGBoost weight

    # Phase 2/3 additions
    gate0_live: bool = False              # P3-2: Gate 0 LOG-ONLY until pilot complete
    leiden_deployed: bool = False         # P2-1: set True after Leiden deploys

    # Phase 5 stubs
    finnet_live: bool = False
    npci_live: bool = False
    dpdp_live: bool = False

    # ── Committee engine (Tier 3 upgrade) ────────────────────────────────────
    # Shadow mode: scorers run and write to shadow table, no live decision impact
    committee_shadow_mode: bool = True
    # Live mode: committee replaces single XGBoost (Phase 5 go-live)
    committee_live_mode: bool = False
    committee_model_version: str = "committee_v1"

    # Scorer A — upgraded GBM
    scorer_a_model_path: str = "ml/models/scorer_a_v1.joblib"

    # Scorer B — graph embedding classifier
    scorer_b_model_path: str = "ml/models/scorer_b_v1.joblib"
    scorer_b_embedding_dim: int = 32

    # Scorer C — prototype bank (FAISS)
    scorer_c_faiss_index_path: str = "ml/models/prototype_faiss.index"
    scorer_c_prototype_meta_path: str = "ml/models/prototype_meta.joblib"
    scorer_c_max_prototypes: int = 512

    # Scorer D — sequence / set-based (Mamba limited mode until labeled sequences available)
    mamba_limited_mode: bool = True

    # Scorer F — multilingual remark screener
    scorer_f_phrase_dict_path: str = "ml/models/upi_fraud_phrases.json"

    # Specialist override thresholds (Track B) — calibrated empirically in Phase 2
    specialist_override_threshold_a: float = 0.92
    specialist_override_threshold_b: float = 0.90
    specialist_override_threshold_c: float = 0.88
    specialist_override_threshold_d: float = 0.90
    specialist_override_threshold_f: float = 0.85

    # Meta-learner
    meta_learner_model_path: str = "ml/models/meta_learner_v1.joblib"
    meta_learner_min_samples: int = 10_000

    # Anomaly discovery pipeline (separate from scoring — runs on PASS stream only)
    discovery_pipeline_enabled: bool = True
    discovery_ecod_model_path: str = "ml/models/ecod_v1.joblib"
    discovery_deep_svdd_model_path: str = "ml/models/deep_svdd_v1.pt"

    # GNN embedding refresh job
    gnn_embedding_refresh_enabled: bool = True

    # Committee-derived thresholds (set via env after derive_committee_thresholds.py runs)
    threshold_log_committee: float = 0.38
    threshold_review_committee: float = 0.62
    threshold_high_risk_committee: float = 0.83

    @model_validator(mode="after")
    def committee_modes_exclusive(self) -> "Settings":
        if self.committee_shadow_mode and self.committee_live_mode:
            raise ValueError(
                "committee_shadow_mode and committee_live_mode cannot both be True. "
                "Shadow mode runs the committee without affecting decisions; "
                "live mode makes committee decisions authoritative. Set exactly one."
            )
        return self

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
