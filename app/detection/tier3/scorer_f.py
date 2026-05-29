"""
Tier 3 Committee — Scorer F: Multilingual Remark Screener

Encodes the UPI transaction remark with paraphrase-multilingual-MiniLM-L12-v2
(384-dim, CPU-fast, handles Hindi/Hinglish/English) then computes max cosine
similarity to 7 pre-computed cluster centroids stored in upi_fraud_phrases.json.

Performance contract: <5ms per call on CPU (MiniLM, not a large model).
  - Remark is None or empty → missing_flag=True, score=0.5 (neutral)
  - Model absent → missing_flag=True, score=0.5
  - Any exception → ScorerOutput.unavailable("F")

Phrase clusters (7):
  digital_arrest_hindi, investment_fraud_hindi, otp_social_eng_hindi,
  otp_social_eng_english, romance_scam_english, lottery_fraud_hindi,
  sim_swap_indicators
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import structlog

from app.detection.tier3.committee_types import ScorerOutput
from app.core.config import settings

logger = structlog.get_logger()

_model: Optional[object] = None         # SentenceTransformer
_cluster_centroids: Optional[dict] = None   # {cluster_name: np.ndarray (384-dim)}
_load_attempted: bool = False


def _load_scorer_f() -> None:
    """Lazy-load SentenceTransformer + phrase centroids. Thread-safe via GIL."""
    global _model, _cluster_centroids, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    phrase_dict_path = os.path.abspath(settings.scorer_f_phrase_dict_path)

    if not os.path.exists(phrase_dict_path):
        logger.warning(
            "scorer_f_phrase_dict_missing",
            path=phrase_dict_path,
            fix="python ml/scripts/build_phrase_dict.py",
        )
        return

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logger.info("scorer_f_model_loaded")
    except Exception as exc:
        logger.warning("scorer_f_model_load_failed", error=str(exc))
        return

    try:
        with open(phrase_dict_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # raw format: {cluster_name: [[float, ...], ...]} — list of phrase embeddings
        # Pre-compute centroid for each cluster
        _cluster_centroids = {}
        for cluster_name, embeddings in raw.items():
            arr = np.array(embeddings, dtype=np.float32)
            centroid = arr.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm   # L2-normalize centroid
            _cluster_centroids[cluster_name] = centroid
        logger.info("scorer_f_clusters_loaded", n_clusters=len(_cluster_centroids))
    except Exception as exc:
        logger.warning("scorer_f_phrase_dict_load_failed", path=phrase_dict_path, error=str(exc))
        _cluster_centroids = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def score(txn_remark: Optional[str]) -> ScorerOutput:
    """
    Screen UPI remark for fraud-indicative language.

    Returns max cosine similarity to any cluster centroid as the fraud score.
    High similarity to a fraud cluster → high score.

    Remark absent or empty → missing_flag=True (not an error; many txns have no remark).
    """
    _load_scorer_f()

    if not txn_remark or not txn_remark.strip():
        return ScorerOutput(score=0.5, confidence=0.0, missing_flag=True, scorer_id="F")

    if _model is None or _cluster_centroids is None:
        return ScorerOutput.unavailable("F")

    try:
        remark_clean = txn_remark.strip()[:512]   # cap at 512 chars to avoid encoding latency spikes

        embedding = _model.encode(
            [remark_clean],
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize for cosine sim
            show_progress_bar=False,
        )[0]

        max_sim = 0.0
        for centroid in _cluster_centroids.values():
            sim = _cosine_similarity(embedding, centroid)
            if sim > max_sim:
                max_sim = sim

        # Similarity is in [-1, 1]; map to [0, 1] — shift+scale
        fraud_score = float((max_sim + 1.0) / 2.0)

        # Confidence: how much the remark fired above baseline (0.5 = no signal)
        confidence = float(min(max(max_sim * 2.0, 0.0), 1.0))

        return ScorerOutput(
            score=fraud_score,
            confidence=confidence,
            missing_flag=False,
            scorer_id="F",
        )
    except Exception as exc:
        logger.warning("scorer_f_score_failed", error=str(exc))
        return ScorerOutput.unavailable("F")
