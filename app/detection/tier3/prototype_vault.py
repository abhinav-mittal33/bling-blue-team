"""
Tier 3 Committee — Prototype Vault (Scorer C backend)

Singleton FAISS index of labelled fraud prototypes. k=5 ANN search; fraud score
is weighted ratio of fraud neighbors (weight = 1/(1+distance)).

Security contract:
  - score() returns ScorerOutput only. Raw feature vectors are NEVER returned.
  - inject_prototype() is only callable from developer_queue.py (INTERNAL_KEY gated).
    It is not on any scoring code path.
  - _validate_vector() rejects malformed or degenerate inputs before touching the index.

Index size cap: settings.scorer_c_max_prototypes (default 512). Oldest prototypes
evicted when cap exceeded — FAISS flat index rebuilt on each inject call (small index).
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import structlog

from app.detection.tier3.committee_types import ScorerOutput
from app.core.config import settings

logger = structlog.get_logger()

# Expected input dimension = len(FEATURE_NAMES) (currently V2 = 69).
# The vault stores vectors at whatever dim it was built with; _validate_vector
# checks against the stored dim at runtime.
_NAN_FRACTION_LIMIT = 0.30   # reject vector if >30% NaN
_MIN_K = 1
_K = 5


class PrototypeVault:
    """
    Singleton FAISS flat L2 index.

    load() must be called once from app startup (main.py lifespan).
    After load() returns True, score() is safe to call from any thread.
    """

    def __init__(self) -> None:
        self._index: Optional[object] = None   # faiss.IndexFlatL2
        self._labels: Optional[np.ndarray] = None   # parallel array: 0=benign, 1=fraud
        self._dim: int = 0
        self._loaded: bool = False

    def load(self, faiss_index_path: str, meta_path: str) -> bool:
        """
        Load FAISS index and prototype label metadata from disk.
        Returns True on success, False if files absent (vault operates in unavailable mode).
        """
        faiss_path_abs = os.path.abspath(faiss_index_path)
        meta_path_abs = os.path.abspath(meta_path)

        if not os.path.exists(faiss_path_abs) or not os.path.exists(meta_path_abs):
            logger.warning(
                "prototype_vault_files_missing",
                faiss=faiss_path_abs,
                meta=meta_path_abs,
                fix="python ml/scripts/build_initial_prototypes.py",
            )
            return False

        try:
            import faiss
            import joblib

            self._index = faiss.read_index(faiss_path_abs)
            meta = joblib.load(meta_path_abs)
            self._labels = np.array(meta["labels"], dtype=np.int32)
            self._dim = self._index.d
            self._loaded = True
            logger.info("prototype_vault_loaded", n_prototypes=self._index.ntotal, dim=self._dim)
            return True
        except Exception as exc:
            logger.error("prototype_vault_load_failed", error=str(exc))
            return False

    def score(self, feature_vector: np.ndarray) -> ScorerOutput:
        """
        ANN search: k=5 nearest prototypes. Fraud score = weighted fraud ratio.

        Weights = 1/(1+distance) so closer neighbors dominate.
        Returns ScorerOutput.unavailable("C") if vault not loaded or vector invalid.
        Raw prototypes are NEVER returned.
        """
        if not self._loaded or self._index is None:
            return ScorerOutput.unavailable("C")

        if not self._validate_vector(feature_vector):
            return ScorerOutput.unavailable("C")

        try:
            # Replace NaN with 0.0 for FAISS (L2 distance handles zeros; NaN would corrupt results)
            clean_vec = np.where(np.isnan(feature_vector), 0.0, feature_vector).astype(np.float32)
            query = clean_vec.reshape(1, -1)

            k = min(_K, self._index.ntotal)
            if k < _MIN_K:
                return ScorerOutput.unavailable("C")

            distances, indices = self._index.search(query, k)
            dists = distances[0]
            idxs = indices[0]

            # Filter out invalid FAISS indices (-1 = no result)
            valid = [(d, i) for d, i in zip(dists, idxs) if i >= 0]
            if not valid:
                return ScorerOutput.unavailable("C")

            weights = np.array([1.0 / (1.0 + d) for d, _ in valid])
            fraud_flags = np.array([float(self._labels[i]) for _, i in valid])

            fraud_score = float(np.dot(weights, fraud_flags) / weights.sum())
            # Confidence: inverse of mean normalized distance (closer = more confident)
            mean_dist = float(np.mean([d for d, _ in valid]))
            confidence = float(1.0 / (1.0 + mean_dist))

            return ScorerOutput(
                score=fraud_score,
                confidence=confidence,
                missing_flag=False,
                scorer_id="C",
            )
        except Exception as exc:
            logger.warning("prototype_vault_score_failed", error=str(exc))
            return ScorerOutput.unavailable("C")

    def inject_prototype(
        self,
        feature_vector: np.ndarray,
        label: int,
        fraud_type: str,
        source_transaction_id: str,
    ) -> bool:
        """
        Add new prototype to the index. Rebuilds flat index in-place.

        ONLY callable from developer_queue.py (INTERNAL_KEY gated).
        NOT on any scoring or alert path.
        """
        if not self._loaded or self._index is None:
            logger.error("prototype_vault_inject_vault_not_loaded")
            return False

        if label not in (0, 1):
            logger.error("prototype_vault_inject_invalid_label", label=label)
            return False

        if not self._validate_vector(feature_vector):
            logger.error("prototype_vault_inject_invalid_vector")
            return False

        try:
            import faiss
            import joblib

            clean_vec = np.where(np.isnan(feature_vector), 0.0, feature_vector).astype(np.float32)

            # Evict oldest if at cap
            if self._index.ntotal >= settings.scorer_c_max_prototypes:
                self._evict_oldest()

            self._index.add(clean_vec.reshape(1, -1))
            self._labels = np.append(self._labels, label)

            # Persist updated index to disk
            faiss.write_index(
                self._index,
                os.path.abspath(settings.scorer_c_faiss_index_path),
            )
            meta = {"labels": self._labels.tolist()}
            joblib.dump(meta, os.path.abspath(settings.scorer_c_prototype_meta_path))

            # Fingerprint for dedup check (sha256 of vector, no raw vector logged)
            fingerprint = hashlib.sha256(clean_vec.tobytes()).hexdigest()[:16]
            logger.info(
                "prototype_vault_injected",
                label=label,
                fraud_type=fraud_type,
                n_prototypes=self._index.ntotal,
                fingerprint=fingerprint,
            )
            return True
        except Exception as exc:
            logger.error("prototype_vault_inject_failed", error=str(exc))
            return False

    def _validate_vector(self, v: np.ndarray) -> bool:
        """Reject degenerate inputs before touching FAISS index."""
        if v is None or len(v) == 0:
            return False
        if self._dim > 0 and v.shape[0] != self._dim:
            return False
        nan_fraction = float(np.isnan(v).mean())
        if nan_fraction > _NAN_FRACTION_LIMIT:
            return False
        clean = np.where(np.isnan(v), 0.0, v)
        if np.all(clean == 0.0):
            return False
        return True

    def _evict_oldest(self) -> None:
        """Rebuild flat index without the oldest entry (FIFO eviction)."""
        import faiss
        if self._index.ntotal < 2:
            return
        # Retrieve all existing vectors, drop first, rebuild
        all_vecs = np.zeros((self._index.ntotal, self._dim), dtype=np.float32)
        self._index.reconstruct_n(0, self._index.ntotal, all_vecs)
        self._labels = self._labels[1:]
        new_index = faiss.IndexFlatL2(self._dim)
        new_index.add(all_vecs[1:])
        self._index = new_index

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)


# Module-level singleton — load() called once from main.py lifespan
prototype_vault = PrototypeVault()
