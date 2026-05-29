import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.security import limiter
from app.api.v1 import health, score, alert, feedback, novelty, model_admin, analyze_graph, developer_queue

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.detection.novelty.isolation_forest import novelty_detector
    logger.info("BLING Blue Team API starting", version=settings.model_version)

    # Model integrity verification — P0-3
    # Verify each model artifact against its stored SHA-256 hash before loading.
    # On mismatch: log error but allow startup (model may not exist yet on first run).
    _verify_models_on_startup()

    # Novelty detection (Isolation Forest) — kept for backward compat; score.py now
    # uses discovery_ensemble which loads the same artifact plus ECOD/DeepSVDD.
    novelty_loaded = novelty_detector.load()
    if novelty_loaded:
        logger.info("novelty_detection_enabled")
    else:
        logger.warning("novelty_detection_disabled", fix="Run: python ml/train_isolation_forest.py")

    # Discovery ensemble (Phase 3: replaces novelty_router — PASS stream only)
    from app.detection.novelty.discovery_ensemble import discovery_ensemble
    disc_loaded = discovery_ensemble.load()
    if disc_loaded:
        logger.info("discovery_ensemble_enabled")
    else:
        logger.warning("discovery_ensemble_disabled", fix="Run: python ml/train_isolation_forest.py")

    # Prototype vault (Scorer C) — load FAISS index into memory at startup
    from app.detection.tier3.prototype_vault import prototype_vault
    vault_loaded = prototype_vault.load(
        settings.scorer_c_faiss_index_path,
        settings.scorer_c_prototype_meta_path,
    )
    if vault_loaded:
        logger.info("prototype_vault_enabled", size=prototype_vault.size)
    else:
        logger.warning(
            "prototype_vault_disabled",
            fix="python ml/scripts/build_initial_prototypes.py",
        )

    # APScheduler removed — nightly batch now runs via Celery Beat (P0-2).
    # Start beat scheduler separately: celery -A app.celery_app beat
    yield

    logger.info("BLING Blue Team API shutting down")


def _verify_models_on_startup() -> None:
    """Verify model integrity hashes. Logs error but does not crash startup."""
    from pathlib import Path
    try:
        from app.utils.model_integrity import verify_model_hash
        for model_file in ["ml/models/xgboost_v1.json",
                           "ml/models/isolation_forest_v1.joblib"]:
            path = Path(model_file)
            if not path.exists():
                continue  # not trained yet — OK on first run
            try:
                verify_model_hash(path)
            except RuntimeError as e:
                logger.error("model_integrity_check_failed",
                             model=model_file, error=str(e))
    except Exception as exc:
        logger.error("model_integrity_startup_error", error=str(exc))


app = FastAPI(
    title="BLING Blue Team Detection Engine",
    description="Post-transaction forensic fraud detection API",
    version=settings.model_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Routers
app.include_router(health.router, tags=["health"])
app.include_router(score.router, tags=["detection"])
app.include_router(alert.router, tags=["investigation"])
app.include_router(feedback.router, tags=["investigation"])
app.include_router(novelty.router, tags=["novelty"])
app.include_router(model_admin.router, tags=["internal"])
app.include_router(analyze_graph.router, tags=["tgie-integration"])
app.include_router(developer_queue.router, tags=["internal"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"data": None, "error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred"}, "meta": {}},
    )
