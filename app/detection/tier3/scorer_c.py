"""
Tier 3 Committee — Scorer C: Prototype Bank

Thin wrapper over PrototypeVault. Converts the feature dict (same format as
feature_builder output) to a numpy array using FEATURE_NAMES key order, then
delegates to prototype_vault.score().

Input dim must match the FAISS index dimension (set at build time by
ml/scripts/build_initial_prototypes.py).
"""
from __future__ import annotations

import numpy as np
import structlog

from app.detection.tier3.committee_types import ScorerOutput
from app.detection.tier3.prototype_vault import prototype_vault
from ml.feature_registry import FEATURE_NAMES

logger = structlog.get_logger()


def score(features: dict[str, float]) -> ScorerOutput:
    """
    Convert feature dict → ordered numpy array using FEATURE_NAMES, then ANN search.

    Uses FEATURE_NAMES (current default, V2 = 69 features) so the array order
    matches the index dim the vault was built with. NaN for any absent feature.
    """
    if not prototype_vault.loaded:
        return ScorerOutput.unavailable("C")

    try:
        vec = np.array(
            [float(features.get(k, float("nan"))) for k in FEATURE_NAMES],
            dtype=np.float32,
        )
        return prototype_vault.score(vec)
    except Exception as exc:
        logger.warning("scorer_c_score_failed", error=str(exc))
        return ScorerOutput.unavailable("C")
