"""
One-time script: Flush all FTRL per-investigator rate cap keys from Redis.

Run once in each environment after deploying Phase 3 (FTRL removal).
These keys (ftrl_count:*) are orphaned after feedback.py stops using River FTRL.

Run: python scripts/flush_ftrl_redis.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def flush_ftrl_keys() -> int:
    from app.utils.redis_client import get_redis
    r = get_redis()
    keys = list(r.scan_iter("ftrl_count:*"))
    if not keys:
        print("No ftrl_count:* keys found. Already clean.")
        return 0
    deleted = r.delete(*keys)
    print(f"Deleted {deleted} ftrl_count:* key(s).")
    return deleted


if __name__ == "__main__":
    count = flush_ftrl_keys()
    print("Done." if count == 0 else f"Done. {count} key(s) removed.")
