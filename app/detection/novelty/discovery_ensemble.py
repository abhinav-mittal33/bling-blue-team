"""
app/detection/novelty/discovery_ensemble.py

Multi-model anomaly discovery ensemble for cleared (PASS) transactions only.

Models:
  1. IsolationForest — existing trained artifact (isolation_forest_v1.joblib)
  2. ECOD (Empirical Cumulative distribution functions Outlier Detection) — Phase 3
  3. DeepSVDD — optional, skipped if torch absent or model file missing

CRITICAL:
  - This module is ONLY called when action == "PASS" (see score.py gate).
  - Anomaly scores NEVER enter fraud_score or investigator alerts.
  - Scores go to novelty_queue (developer review only) and optionally Red Team.

Degrades gracefully: missing models skip silently; at least IsoForest must be
present for the ensemble to be "available".
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from app.core.config import settings

log = structlog.get_logger()

NOVELTY_THRESHOLD = -0.20   # IsoForest threshold (unchanged from isolation_forest.py)


class DiscoveryEnsemble:
    """
    Multi-model discovery ensemble. Loaded once at startup via load().
    Thread-safe (read-only after load).
    """

    def __init__(self) -> None:
        self._iso_model = None
        self._iso_scaler = None
        self._iso_features: list[str] = []
        self._ecod = None
        self._deep_svdd = None
        self._available = False

    def load(self, iso_model_path: str = "ml/models/isolation_forest_v1.joblib") -> bool:
        """
        Load all available anomaly models. Returns True if at least IsoForest loaded.
        ECOD and DeepSVDD failures are logged but do not fail startup.
        """
        iso_ok = self._load_isoforest(iso_model_path)
        self._load_ecod()
        self._load_deep_svdd()
        self._available = iso_ok
        return iso_ok

    def _load_isoforest(self, path: str) -> bool:
        path_obj = Path(path)
        if not path_obj.exists():
            log.warning("discovery_isoforest_missing", path=str(path_obj))
            return False
        try:
            import joblib
            artifact = joblib.load(path_obj)
            self._iso_model = artifact["model"]
            self._iso_scaler = artifact["scaler"]
            self._iso_features = artifact["features"]
            log.info("discovery_isoforest_loaded", n_features=len(self._iso_features))
            return True
        except Exception as exc:
            log.error("discovery_isoforest_load_failed", error=str(exc))
            return False

    def _load_ecod(self) -> None:
        ecod_path = Path(settings.discovery_ecod_model_path)
        if not ecod_path.exists():
            return
        try:
            import joblib
            self._ecod = joblib.load(ecod_path)
            log.info("discovery_ecod_loaded", path=str(ecod_path))
        except Exception as exc:
            log.warning("discovery_ecod_load_failed", error=str(exc))

    def _load_deep_svdd(self) -> None:
        svdd_path = Path(settings.discovery_deep_svdd_model_path)
        if not svdd_path.exists():
            return
        try:
            import torch
            self._deep_svdd = torch.load(svdd_path, map_location="cpu")
            log.info("discovery_deep_svdd_loaded", path=str(svdd_path))
        except ImportError:
            log.debug("discovery_deep_svdd_skipped", reason="torch_not_installed")
        except Exception as exc:
            log.warning("discovery_deep_svdd_load_failed", error=str(exc))

    @property
    def available(self) -> bool:
        return self._available

    def score(self, graph_features: dict) -> Optional[float]:
        """
        Compute ensemble anomaly score. Returns the most extreme (lowest) IsoForest
        score as the primary signal; ECOD/DeepSVDD votes are advisory only.

        Returns None if ensemble not available or scoring fails.
        NEVER raises — failure is degradation, not error.
        """
        if not self._available:
            return None

        try:
            vec = np.array(
                [float(graph_features.get(f) or 0.0) for f in self._iso_features],
                dtype=np.float32,
            ).reshape(1, -1)
            vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
            vec_scaled = self._iso_scaler.transform(vec)
            iso_score = float(self._iso_model.decision_function(vec_scaled)[0])
            return iso_score
        except Exception as exc:
            log.error("discovery_score_failed", error=str(exc))
            return None

    def is_novel(self, anomaly_score: Optional[float]) -> bool:
        """True if score crosses novelty threshold. Safe with None input."""
        if anomaly_score is None:
            return False
        return anomaly_score < NOVELTY_THRESHOLD


# Singleton — loaded once at startup from main.py lifespan
discovery_ensemble = DiscoveryEnsemble()
