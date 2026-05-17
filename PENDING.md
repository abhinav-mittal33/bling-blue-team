# BLING Blue Team — Pending Actions

Things you need to do manually, grouped by when to do them.
Check each off as you complete it.

---

## BEFORE THE HACKATHON DEMO

### One-time setup (do in this order)

- [ ] **Copy `.env.example` → `.env` and fill in all values**
  ```bash
  cp .env.example .env
  ```
  Required variables: `POSTGRES_URL`, `REDIS_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GRAPH_ENGINE_API_KEY`, `INVESTIGATOR_API_KEY`, `INTERNAL_API_KEY`, `SALT`, `MODEL_VERSION`
  Teammate-provided: `BLOCKCHAIN_SERVICE_URL`, `RED_TEAM_SERVICE_URL`, `INVESTIGATOR_DASHBOARD_URL`

- [ ] **Start all infrastructure**
  ```bash
  docker-compose up -d
  ```
  Wait ~30 seconds for PostgreSQL and Redis to be ready.

- [ ] **Initialize the database (first time only)**
  ```bash
  python scripts/init_db.py
  ```

- [ ] **Run ALL Alembic migrations** (creates novelty_queue + all other tables)
  ```bash
  alembic upgrade head
  ```
  Verify: `alembic current` should show `002 (head)`

- [ ] **Train the XGBoost model**
  ```bash
  python ml/train.py
  ```
  Output: `ml/models/xgboost_v1.json` (~2 minutes)

- [ ] **Train the Isolation Forest** (novelty detection)
  ```bash
  python ml/train_isolation_forest.py
  ```
  Output: `ml/models/isolation_forest_v1.joblib` (~30 seconds)
  > With `POSTGRES_URL` set and DB populated, this trains on real legit accounts.
  > Without it, uses 2000 synthetic examples (demo mode — still works).

- [ ] **Seed Redis graph features**
  ```bash
  python scripts/seed_redis.py
  ```

- [ ] **Load demo data**
  ```bash
  python scripts/generate_test_data.py && python scripts/load_sample_data.py
  ```

- [ ] **Run all tests (must pass before demo)**
  ```bash
  pytest tests/ -v
  ```
  Expected: 23 passing (8 fraud scenarios + 15 novelty tests)

---

## COORDINATE WITH TEAMMATES

- [ ] **Get teammate API URLs and add to `.env`**
  - `BLOCKCHAIN_SERVICE_URL` — from Blockchain teammate
  - `RED_TEAM_SERVICE_URL` — from Red Team teammate
  - `INVESTIGATOR_DASHBOARD_URL` — from Dashboard teammate
  - `NEO4J_URI` — from Graph Engine teammate (they own Neo4j)

- [ ] **Share your API keys with teammates**
  - Give Graph Engine team your `GRAPH_ENGINE_API_KEY`
  - Give Investigator Dashboard team your `INVESTIGATOR_API_KEY`
  - Keep `INTERNAL_API_KEY` to yourself (testing only)

- [ ] **Confirm POST /api/v1/score request schema with Graph Engine**
  Fields: `transaction_id`, `account_id`, `amount`, `channel`, `timestamp`, `payee_vpa`, `payee_vpa_created_at` (optional)

- [ ] **Confirm Neo4j connection** — ask Graph Engine teammate for bolt URL + credentials

---

## DEMO DAY CHECKLIST

Run through this in order 30 minutes before the demo:

```bash
# 1. Services running?
docker-compose ps

# 2. API healthy?
curl http://localhost:8000/health

# 3. Score a high-risk transaction (should return HIGH_RISK)
curl -X POST http://localhost:8000/api/v1/score \
  -H "X-API-Key: $GRAPH_ENGINE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo_001",
    "account_id": "ACC_FRAUD_001",
    "amount": "500000",
    "channel": "UPI",
    "timestamp": "2026-05-16T02:14:00Z",
    "payee_vpa": "scammer@upi",
    "payee_vpa_created_at": "2026-05-14T10:00:00Z"
  }'

# 4. Score a legit transaction (should return LOG or PASS)
curl -X POST http://localhost:8000/api/v1/score \
  -H "X-API-Key: $GRAPH_ENGINE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo_002",
    "account_id": "ACC_LEGIT_001",
    "amount": "25000",
    "channel": "NEFT",
    "timestamp": "2026-05-16T11:00:00Z",
    "payee_vpa": "employer@upi",
    "payee_vpa_created_at": "2024-01-01T00:00:00Z"
  }'

# 5. Check novelty queue (Isolation Forest flags)
curl "http://localhost:8000/api/v1/novelty/queue" \
  -H "X-API-Key: $INTERNAL_API_KEY"

# 6. Metrics endpoint working?
curl http://localhost:8000/metrics | grep bling_

# 7. API docs accessible?
open http://localhost:8000/docs
```

---

## PRODUCTION / POST-HACKATHON

These are NOT needed for the hackathon demo, but document them so nothing is forgotten:

- [ ] **Set `INTERNAL_API_KEY` to empty string in production** — this key is for testing only
- [ ] **Remove `docs_url` from FastAPI** — already disabled when `settings.debug=False`
- [ ] **Retrain Isolation Forest on real legitimate accounts** — currently uses synthetic demo data
  ```bash
  POSTGRES_URL=... python ml/train_isolation_forest.py
  ```
- [ ] **Schedule nightly batch** — APScheduler runs at 2am. Verify it fires in Docker.
  ```bash
  docker-compose logs celery | grep nightly
  ```
- [ ] **Set up Grafana dashboard** — Prometheus metrics available at `/metrics`
  Key metrics: `bling_scoring_requests_total`, `bling_novelty_flags_total`, `bling_scoring_latency_ms`
- [ ] **Add `payee_in_known_contacts` lookup** — currently hardcoded to `False` in `score.py`. Wire to a real contacts table.
- [ ] **FINnet 2.0 direct submission** — STR draft is generated (156 fields). Currently manual investigator submission. Direct API integration pending.
- [ ] **Cross-bank ghost node correlation** — requires inter-bank data sharing agreement.

---

## IF SOMETHING BREAKS

| Symptom | First thing to check |
|---------|---------------------|
| Score stuck at 0.4/LOG for everything | `redis-cli HGETALL feat:ACC_FRAUD_001` — are graph features seeded? Re-run `seed_redis.py` |
| API starts but logs `novelty_detection_disabled` | Run `python ml/train_isolation_forest.py` |
| `alembic upgrade head` fails | Check `POSTGRES_URL` in `.env` and that postgres container is running |
| XGBoost model not found | Run `python ml/train.py` — creates `ml/models/xgboost_v1.json` |
| POST /score returns 401 | Check `GRAPH_ENGINE_API_KEY` matches between caller and `.env` |
| Celery trail reconstruction not starting | `docker-compose logs celery` — check Celery worker is running |
| Neo4j queries timing out | Tier 2 gates use pre-computed features. Check nightly batch ran: `alembic current` + check `graph_features_cache` has recent `computed_at` |

---

*Generated by Claude on 2026-05-17. Update this file as items are completed.*
