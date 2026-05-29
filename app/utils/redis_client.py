from __future__ import annotations
import time
import uuid
import redis
from redis.connection import ConnectionPool
from app.core.config import settings

_pool: ConnectionPool | None = None


def get_redis() -> redis.Redis:
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=50,
        )
    return redis.Redis(connection_pool=_pool)


# ── Sliding-window velocity (ZSET — P0-1 fix) ────────────────────────────────
# Old fixed-window keys (vel:1h:, vel:24h:) reset on TTL expiry regardless of
# when individual transactions occurred. A burst straddling an hour boundary
# could be split across two windows, undercounting velocity.
# ZSET keys (velz:1h:, velz:24h:) trim by wall-clock position — true sliding window.

def velocity_1h(account_id: str) -> int:
    """True sliding-window transaction count: last 60 minutes."""
    r = get_redis()
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - 3_600_000
    key = f"velz:1h:{account_id}"
    # Remove expired members then count remaining
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, cutoff_ms)
    pipe.zcard(key)
    _, count = pipe.execute()
    return int(count)


def velocity_24h(account_id: str) -> int:
    """True sliding-window transaction count: last 24 hours."""
    r = get_redis()
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - 86_400_000
    key = f"velz:24h:{account_id}"
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, cutoff_ms)
    pipe.zcard(key)
    _, count = pipe.execute()
    return int(count)


def velocity_volume_1h(account_id: str) -> float:
    """Sum of transaction amounts in the last 60 minutes (sliding window)."""
    r = get_redis()
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - 3_600_000
    key = f"volz:1h:{account_id}"
    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, cutoff_ms)
    pipe.zrangebyscore(key, cutoff_ms, now_ms, withscores=False)
    _, members = pipe.execute()
    # Members stored as "amount:uuid" — parse the amount prefix
    total = 0.0
    for m in members:
        try:
            total += float(m.split(":")[0])
        except (ValueError, IndexError):
            pass
    return total


def increment_velocity(account_id: str, amount: float) -> None:
    """Record a new transaction in the ZSET sliding-window counters."""
    r = get_redis()
    now_ms = int(time.time() * 1000)
    txn_member = str(uuid.uuid4())
    amount_member = f"{amount}:{txn_member}"

    pipe = r.pipeline()

    # 1h sliding window — count
    k1h = f"velz:1h:{account_id}"
    pipe.zadd(k1h, {txn_member: now_ms})
    pipe.expire(k1h, 3_700)  # 1h + 100s buffer for the cleanup lag

    # 24h sliding window — count
    k24h = f"velz:24h:{account_id}"
    pipe.zadd(k24h, {txn_member: now_ms})
    pipe.expire(k24h, 86_500)

    # 1h sliding window — volume (member encodes amount so we can sum)
    kv1h = f"volz:1h:{account_id}"
    pipe.zadd(kv1h, {amount_member: now_ms})
    pipe.expire(kv1h, 3_700)

    pipe.execute()


def get_graph_features(account_id: str) -> dict:
    """Fetch pre-computed graph features from Redis cache."""
    r = get_redis()
    raw = r.hgetall(f"feat:{account_id}")
    if not raw:
        return {}
    def _cast(v: str):
        if v == "True":
            return True
        if v == "False":
            return False
        if v == "None":
            return None
        try:
            return float(v)
        except ValueError:
            return v  # keep as string (e.g. account_type, kyc_occupation)
    return {k: _cast(v) for k, v in raw.items()}


def set_graph_features(account_id: str, features: dict) -> None:
    """Store pre-computed graph features. Called by nightly batch only."""
    r = get_redis()
    serialized = {k: str(v) for k, v in features.items() if v is not None}
    if serialized:
        pipe = r.pipeline()
        pipe.hset(f"feat:{account_id}", mapping=serialized)
        pipe.expire(f"feat:{account_id}", 86400 + 7200)  # 26h — nightly batch has 2h window
        pipe.execute()
