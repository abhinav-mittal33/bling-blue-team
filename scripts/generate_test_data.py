"""
Generate 10K synthetic transactions (270 fraud, 9730 legit) across 8 fraud archetypes.
Output: test_data.json in project root.
Run: python scripts/generate_test_data.py
"""
from __future__ import annotations
import json
import random
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

random.seed(42)

FRAUD_ARCHETYPES = [
    "rapid_layering",
    "low_slow_mule",
    "digital_arrest",
    "ghost_node_cash",
    "structuring",
    "bipartite_mule",
    "cycle_round_trip",
    "merchant_terminal",
]

CHANNELS = ["UPI", "IMPS", "NEFT", "CASH", "RTGS"]
KYC_OCCUPATIONS = ["SALARIED", "BUSINESS", "RETIRED", "STUDENT", "FARMER", "GIG_WORKER", None]


def _random_ts(days_back: int = 30, hour_bias: int | None = None) -> str:
    base = datetime.now(timezone.utc) - timedelta(days=random.randint(0, days_back))
    h = hour_bias if hour_bias is not None else random.randint(0, 23)
    return base.replace(hour=h, minute=random.randint(0, 59)).isoformat()


def _legit_txn(account_id: str, idx: int) -> dict:
    return {
        "id": f"txn_legit_{idx:06d}",
        "account_id": account_id,
        "amount": round(random.uniform(500, 50000), 2),
        "channel": random.choice(CHANNELS[:3]),  # UPI/IMPS/NEFT only
        "timestamp": _random_ts(30),
        "payee_account_id": f"ACC_{random.randint(1000, 9999):04d}",
        "label": 0,
        "archetype": None,
    }


def _fraud_txn(account_id: str, idx: int, archetype: str) -> dict:
    configs = {
        "rapid_layering": {"amount": random.uniform(80000, 99000), "channel": "UPI", "hour": 2},
        "low_slow_mule": {"amount": random.uniform(150000, 200000), "channel": "UPI", "hour": 3},
        "digital_arrest": {"amount": random.uniform(200000, 500000), "channel": "UPI", "hour": 1},
        "ghost_node_cash": {"amount": random.uniform(100000, 130000), "channel": "CASH", "hour": random.randint(7, 22)},
        "structuring": {"amount": random.uniform(93000, 97000), "channel": "IMPS", "hour": random.randint(9, 18)},
        "bipartite_mule": {"amount": random.uniform(20000, 60000), "channel": "UPI", "hour": random.randint(10, 20)},
        "cycle_round_trip": {"amount": random.uniform(50000, 95000), "channel": "UPI", "hour": random.randint(0, 5)},
        "merchant_terminal": {"amount": random.uniform(1000, 10000), "channel": "UPI", "hour": random.randint(9, 21)},
    }
    cfg = configs[archetype]
    return {
        "id": f"txn_fraud_{idx:06d}",
        "account_id": account_id,
        "amount": round(cfg["amount"], 2),
        "channel": cfg["channel"],
        "timestamp": _random_ts(30, hour_bias=cfg["hour"]),
        "payee_account_id": f"ACC_FRAUD_{random.randint(100, 999):03d}",
        "label": 1,
        "archetype": archetype,
    }


def generate(n_total: int = 10000, n_fraud: int = 270) -> dict:
    accounts = [f"ACC_{i:06d}" for i in range(1, 2001)]
    fraud_accounts = [f"ACC_FRAUD_{i:03d}" for i in range(1, 100)]

    transactions = []

    # Fraud transactions — evenly distribute across 8 archetypes
    per_archetype = n_fraud // len(FRAUD_ARCHETYPES)
    for i, archetype in enumerate(FRAUD_ARCHETYPES):
        for j in range(per_archetype):
            acct = random.choice(fraud_accounts)
            transactions.append(_fraud_txn(acct, i * per_archetype + j, archetype))

    # Remainder fraud
    for k in range(n_fraud - len(FRAUD_ARCHETYPES) * per_archetype):
        transactions.append(_fraud_txn(random.choice(fraud_accounts), 9000 + k, random.choice(FRAUD_ARCHETYPES)))

    # Legit transactions
    for i in range(n_total - n_fraud):
        transactions.append(_legit_txn(random.choice(accounts), i))

    random.shuffle(transactions)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(transactions),
        "fraud_count": sum(1 for t in transactions if t["label"] == 1),
        "legit_count": sum(1 for t in transactions if t["label"] == 0),
        "transactions": transactions,
    }


if __name__ == "__main__":
    data = generate()
    out = Path("test_data.json")
    out.write_text(json.dumps(data, indent=2))
    print(f"Generated {data['total']} transactions ({data['fraud_count']} fraud) → {out}")
