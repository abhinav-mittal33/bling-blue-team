"""
Locust load test for BLING Blue Team API (P5-7).

Tests:
  POST /api/v1/score    — main scoring endpoint (target: p99 < 100ms at 500 RPS)
  GET  /api/v1/alerts/{id} — investigator alert fetch (target: p99 < 200ms)
  POST /api/v1/feedback — investigator feedback (target: p99 < 50ms)

Run:
  locust -f tests/load/locustfile.py --host http://localhost:8000 --users 500 --spawn-rate 50
  or headless:
  locust -f tests/load/locustfile.py --host http://localhost:8000 --users 500 --spawn-rate 50 --run-time 5m --headless --csv=load_results

Performance targets (from spec):
  POST /score:    p50 < 55ms (timing pad target), p99 < 100ms
  GET  /alerts:   p99 < 200ms
  POST /feedback: p99 < 50ms
  Error rate:     < 0.1%
"""
import random
import uuid
from datetime import datetime, timezone

from locust import HttpUser, between, task


# Test API key (must match INTERNAL_API_KEY in .env for test environment)
_API_KEY = "test-internal-key"
_HEADERS = {"X-API-Key": _API_KEY, "Content-Type": "application/json"}


def _random_txn():
    return {
        "transaction_id": str(uuid.uuid4()),
        "account_id": f"ACC_{random.randint(1000, 9999)}",
        "amount": round(random.uniform(100, 200000), 2),
        "channel": random.choice(["UPI", "IMPS", "NEFT"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payee_account_id": f"PAY_{random.randint(1000, 9999)}",
        "payee_vpa": f"test{random.randint(1000, 9999)}@upi",
        "payee_vpa_created_at": None,
        "merchant_terminal_id": None,
        "merchant_mcc": None,
        "device_id": f"device_{random.randint(1000, 9999)}",
        "ip_address": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
        "geo_city": random.choice(["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]),
        "geo_state": random.choice(["MH", "DL", "KA", "TN", "WB"]),
    }


class BlingBlueTeamUser(HttpUser):
    wait_time = between(0.1, 0.5)
    alert_ids: list[str] = []

    @task(10)
    def score_transaction(self):
        with self.client.post(
            "/api/v1/score",
            json=_random_txn(),
            headers=_HEADERS,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json().get("data", {})
                alert_id = data.get("alert_id")
                if alert_id:
                    BlingBlueTeamUser.alert_ids.append(alert_id)
                    if len(BlingBlueTeamUser.alert_ids) > 100:
                        BlingBlueTeamUser.alert_ids = BlingBlueTeamUser.alert_ids[-100:]
                response.success()
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Unexpected status {response.status_code}")

    @task(3)
    def get_alert(self):
        if not BlingBlueTeamUser.alert_ids:
            return
        alert_id = random.choice(BlingBlueTeamUser.alert_ids)
        with self.client.get(
            f"/api/v1/alerts/{alert_id}",
            headers=_HEADERS,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"Unexpected status {response.status_code}")

    @task(1)
    def submit_feedback(self):
        if not BlingBlueTeamUser.alert_ids:
            return
        alert_id = random.choice(BlingBlueTeamUser.alert_ids)
        with self.client.post(
            "/api/v1/feedback",
            json={
                "alert_id": alert_id,
                "transaction_id": str(uuid.uuid4()),
                "label": random.choice([0, 1]),
                "investigator_id": f"inv_{random.randint(1, 20)}",
                "notes": "load test feedback",
            },
            headers=_HEADERS,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"Unexpected status {response.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health", headers=_HEADERS)
