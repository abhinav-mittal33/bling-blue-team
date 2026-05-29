"""
Score-to-action threshold mapping with anti-extraction jitter (P6-1).
Thresholds are configurable via env vars.

Score jitter: ±0.01 applied BEFORE threshold decision (P6-1).
  - Prevents model extraction by making decision boundary opaque to external observers
  - Cap to [0.0, 1.0] after jitter — never return score outside this range
  - Jitter applies to the RETURNED score in the API response, not to internal pipeline

Canary accounts: special test account IDs that always return a known score (P6-1).
  - Used to verify pipeline health without triggering real alerts
  - Canary IDs stored in Redis set `canary:accounts`
"""
from __future__ import annotations
import random
from typing import Literal
from app.core.config import settings

Action = Literal["PASS", "LOG", "REVIEW", "HIGH_RISK"]

_JITTER_RANGE = 0.01  # ±0.01 uniform jitter


def apply_jitter(score: float) -> float:
    """
    Apply ±0.01 random jitter BEFORE threshold comparison.
    Cap to [0.0, 1.0] after. Intentional for anti-model-extraction (P6-1).
    """
    jitter = random.uniform(-_JITTER_RANGE, _JITTER_RANGE)
    return max(0.0, min(1.0, score + jitter))


def score_to_action(score: float) -> tuple[Action, float]:
    """
    Apply jitter, map to action, return BOTH so the API response uses the jittered score.
    The returned score is what is stored in the DB and shown to investigators —
    it must match the action decision to prevent score/action inconsistency.
    """
    jittered = apply_jitter(score)
    if jittered >= settings.threshold_high_risk:
        return "HIGH_RISK", jittered
    if jittered >= settings.threshold_review:
        return "REVIEW", jittered
    if jittered >= settings.threshold_log:
        return "LOG", jittered
    return "PASS", jittered


def is_canary_account(account_id: str) -> bool:
    """Check if account is a canary test account (P6-1). Best-effort."""
    try:
        from app.utils.redis_client import get_redis
        r = get_redis()
        return bool(r.sismember("canary:accounts", account_id))
    except Exception:
        return False


def score_canary(account_id: str) -> tuple[float, Action]:
    """
    Return deterministic score for canary accounts (P6-1).
    Canaries always return 0.50 (LOG threshold) to verify pipeline health.
    """
    return 0.50, "LOG"
