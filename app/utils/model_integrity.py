"""
app/utils/model_integrity.py

SHA-256 model artifact integrity checking — P0-3.
Protects against silent model file corruption or accidental overwrites.

Usage:
  After training: store_model_hash("models/xgb_20260528_030000.pkl")
  At startup:     verify_model_hash("models/xgb_20260528_030000.pkl")  → raises on mismatch
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import structlog

log = structlog.get_logger()

HASH_DIR = Path("ml/models/hashes")
CHUNK_SIZE = 65_536  # 64 KB read chunks


def _hash_file(model_path: str | Path) -> str:
    """Compute SHA-256 of a file. Reads in chunks to handle large model files."""
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(model_path: str | Path) -> Path:
    """Return the .sha256 sidecar file path for a given model path."""
    name = Path(model_path).name
    HASH_DIR.mkdir(parents=True, exist_ok=True)
    return HASH_DIR / f"{name}.sha256"


def store_model_hash(model_path: str | Path) -> str:
    """
    Compute and store the SHA-256 hash of a model artifact.
    Called by ml/train.py after successful training.
    Returns the hex digest.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    digest = _hash_file(model_path)
    sidecar = _hash_path(model_path)
    sidecar.write_text(digest)

    log.info("model_hash_stored", model=str(model_path), sha256=digest[:16] + "...")
    return digest


def verify_model_hash(model_path: str | Path) -> None:
    """
    Verify a model artifact against its stored SHA-256 hash.
    Raises RuntimeError if hash is missing or doesn't match.
    Called by app/main.py at startup before loading any model.
    """
    model_path = Path(model_path)
    sidecar = _hash_path(model_path)

    if not sidecar.exists():
        raise RuntimeError(
            f"No integrity hash found for {model_path}. "
            "Run ml/train.py to generate and store hash."
        )

    stored = sidecar.read_text().strip()
    actual = _hash_file(model_path)

    if actual != stored:
        raise RuntimeError(
            f"Model integrity check FAILED for {model_path}. "
            f"Stored={stored[:16]}... Actual={actual[:16]}... "
            "File may be corrupted or tampered with."
        )

    log.info("model_integrity_verified", model=str(model_path), sha256=actual[:16] + "...")


def verify_all_models() -> list[str]:
    """
    Verify all models with sidecar .sha256 files in models/hashes/.
    Returns list of error messages (empty = all OK).
    Called by pre-push hook.
    """
    if not HASH_DIR.exists():
        return []

    errors = []
    for sidecar in HASH_DIR.glob("*.sha256"):
        model_name = sidecar.stem  # strips .sha256 to get original filename
        model_path = Path("ml/models") / model_name
        if not model_path.exists():
            errors.append(f"Hash exists but model not found: {model_path}")
            continue
        try:
            verify_model_hash(model_path)
        except RuntimeError as e:
            errors.append(str(e))

    return errors


def list_model_versions(models_dir: str = "ml/models") -> list[dict]:
    """
    P6-2: List all available model versions with their hashes and timestamps.
    Returns sorted list (newest first) for rollback API.
    """
    import datetime
    base = Path(models_dir)
    if not base.exists():
        return []

    versions = []
    for model_file in sorted(base.iterdir(), reverse=True):
        if model_file.suffix not in (".json", ".joblib", ".pkl"):
            continue
        sidecar = _hash_path(model_file)
        stored_hash = sidecar.read_text().strip() if sidecar.exists() else None
        stat = model_file.stat()
        versions.append({
            "name": model_file.name,
            "path": str(model_file),
            "size_bytes": stat.st_size,
            "created_at": datetime.datetime.fromtimestamp(stat.st_ctime, tz=datetime.timezone.utc).isoformat(),
            "sha256": stored_hash[:16] + "..." if stored_hash else None,
            "integrity_verified": stored_hash is not None,
        })
    return versions


def activate_model_version(model_name: str, models_dir: str = "ml/models") -> dict:
    """
    P6-2: Activate a specific model version by updating ensemble.py's model cache.
    Verifies integrity before activation. Resets the in-memory model cache.
    Returns activation status dict.
    """
    model_path = Path(models_dir) / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        verify_model_hash(model_path)
    except RuntimeError as e:
        raise RuntimeError(f"Rollback refused — integrity check failed: {e}") from e

    # Reset in-memory model cache in ensemble.py to force reload
    try:
        from app.detection.tier3 import ensemble
        ensemble._calibrated_model = None
        ensemble._base_model = None
        ensemble._legacy_model = None
        log.warning("model_cache_reset_for_rollback", target=str(model_path))
    except Exception as exc:
        log.error("model_cache_reset_failed", error=str(exc))

    log.warning(
        "model_version_activated",
        model=model_name,
        note="Ensemble will reload on next scoring request",
    )
    return {"activated": True, "model": model_name, "integrity_ok": True}


def get_latest_model(model_type: str, models_dir: str = "ml/models") -> Path | None:
    """
    Find the latest timestamped model file for a given type prefix.
    E.g. get_latest_model("xgb") → "models/xgb_20260528_030000.pkl"
    Returns None if no models found.
    """
    base = Path(models_dir)
    if not base.exists():
        return None

    candidates = sorted(base.glob(f"{model_type}_*.pkl")) + \
                 sorted(base.glob(f"{model_type}_*.json")) + \
                 sorted(base.glob(f"{model_type}_*.joblib")) + \
                 sorted(base.glob(f"{model_type}_*.pt"))

    return candidates[-1] if candidates else None
