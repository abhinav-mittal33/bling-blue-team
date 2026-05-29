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
  Required variables:
  - `POSTGRES_URL`, `REDIS_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - `GRAPH_ENGINE_API_KEY`, `INVESTIGATOR_API_KEY`, `INTERNAL_API_KEY`
  - `SALT`, `PSEUDONYMIZATION_KEY` (32-byte hex), `DB_ENCRYPTION_KEY`
  - `MODEL_VERSION`, `ENSEMBLE_ALPHA`

  For JWT auth (optional in dev, required in production):
  ```bash
  openssl genrsa -out jwt_private.pem 2048
  openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem
  ```
  Then set `JWT_PRIVATE_KEY` and `JWT_PUBLIC_KEY` in `.env` (PEM content, one-liner with `\n`).

  Teammate-provided: `BLOCKCHAIN_SERVICE_URL`, `RED_TEAM_SERVICE_URL`, `INVESTIGATOR_DASHBOARD_URL`

- [ ] **Start all infrastructure**
  ```bash
  docker-compose up -d
  ```
  Wait ~30 seconds for PostgreSQL and Redis to be ready.

- [ ] **Run ALL Alembic migrations** (creates all 7 migration sets including committee tables)
  ```bash
  alembic upgrade head
  ```
  Verify: `alembic current` should show `007 (head)`

- [ ] **Train the XGBoost model**
  ```bash
  python ml/train.py
  ```
  Output: `ml/models/xgboost_v1.json` + calibrated `xgboost_calibrated_v2.joblib` (~2 minutes)

- [ ] **Build committee engine assets (Scorer F phrase dict + Scorer C FAISS vault)**
  ```bash
  python ml/scripts/build_phrase_dict.py
  python ml/scripts/build_initial_prototypes.py
  ```
  These must run before the API starts — otherwise `prototype_vault_disabled` warning appears at startup.

- [ ] **Train the discovery ensemble** (anomaly detection — PASS stream only)
  ```bash
  python ml/train_isolation_forest.py
  ```
  Output: `ml/models/isolation_forest_v1.joblib` (~30 seconds)

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
  Expected: 102+ passing. All 8 fraud scenario integration tests must pass.

- [ ] **Start Celery worker + beat scheduler** (separate terminal or Docker)
  ```bash
  celery -A app.celery_app worker -l info -Q default,evidence,graph
  celery -A app.celery_app beat -l info
  ```

- [ ] **Start the API**
  ```bash
  uvicorn app.main:app --reload --port 8000
  ```

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
  - Keep `INTERNAL_API_KEY` to yourself (model admin + developer queue + testing only)

- [ ] **Confirm POST /api/v1/score request schema with Graph Engine**
  Fields: `transaction_id`, `account_id`, `amount`, `channel`, `timestamp`, `payee_vpa`, `payee_vpa_created_at` (optional)

- [ ] **Confirm Neo4j connection** — ask Graph Engine teammate for bolt URL + credentials

- [ ] **Confirm Neo4j schema includes `payee_vpa_created_at` on Transaction nodes** (required by D-01 gate)

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
# Expect: action="HIGH_RISK", score >= 0.83

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
# Expect: action="LOG" or "PASS", score < 0.62

# 5. Check developer queue (novel fraud candidates)
curl "http://localhost:8000/api/v1/developer-queue/prototype-candidates" \
  -H "X-API-Key: $INTERNAL_API_KEY"

# 6. Metrics endpoint working?
curl http://localhost:8000/metrics | grep bling_

# 7. API docs accessible?
open http://localhost:8000/docs

# 8. Run the 8 integration tests one more time
pytest tests/test_integration/test_fraud_scenarios.py -v
```

---

## PRODUCTION / POST-HACKATHON

These are NOT needed for the hackathon demo:

- [ ] **Set `INTERNAL_API_KEY` to empty string in production** — this key is for testing only
- [ ] **Remove `docs_url` from FastAPI** — already disabled when `settings.debug=False`
- [ ] **Derive post-Phase-4 thresholds** once ≥50K shadow rows collected:
  ```bash
  python ml/derive_committee_thresholds.py
  ```
  Update `LOG_THRESHOLD`, `REVIEW_THRESHOLD`, `HIGH_RISK_THRESHOLD` in `.env`.

- [ ] **Train committee meta-learner** once ≥10K shadow rows collected:
  ```bash
  python ml/train_meta_learner.py
  ```

- [ ] **Gate 0 (rapid relay) go-live** — after 2-week pilot log review:
  Set `GATE0_LIVE=true` in `.env` and restart API.

- [ ] **Retrain discovery ensemble on real PASS-stream data** once shadow rows accumulate:
  ```bash
  python ml/train_ecod.py
  ```

- [ ] **P2-2 (Heterogeneous schema)** — requires teammate to add Device + VPA nodes in Neo4j. Then: train HGT ensemble component.

- [ ] **Set up Grafana dashboard** — Prometheus metrics at `/metrics`
  Key metrics: `bling_scoring_requests_total`, `bling_alerts_created_total`, `bling_scoring_latency_ms`

- [ ] **Wire `payee_in_known_contacts` lookup** — currently hardcoded `False` in `score.py`. Wire to a real contacts table.

- [ ] **FINnet 2.0 live submission** — STR draft generated (156 fields). Set `FINNET_LIVE=true` and configure `FINNET_API_KEY`.

- [ ] **NPCI pre-settlement** — set `NPCI_LIVE=true` when integration ready.

- [ ] **DPDP Act erasure endpoint** — set `DPDP_LIVE=true` to expose the right-to-erasure API.

---

## IF SOMETHING BREAKS

| Symptom | First thing to check |
|---------|---------------------|
| Score stuck at 0.4/LOG for everything | `redis-cli HGETALL feat:ACC_FRAUD_001` — are graph features seeded? Re-run `seed_redis.py` |
| `prototype_vault_disabled` at startup | Run `python ml/scripts/build_initial_prototypes.py` |
| `novelty_detection_disabled` at startup | Run `python ml/train_isolation_forest.py` |
| `alembic upgrade head` fails | Check `POSTGRES_URL` in `.env` and that postgres container is running |
| XGBoost model not found | Run `python ml/train.py` — creates `ml/models/xgboost_v1.json` |
| POST /score returns 401 | Check `GRAPH_ENGINE_API_KEY` matches between caller and `.env` |
| JWT 401 "not configured" | Set `JWT_PUBLIC_KEY` in `.env`. Use X-API-Key for now. |
| Celery trail reconstruction not starting | `docker-compose logs celery` — worker must be running |
| D-01 gate fires on old accounts | Verify `days_since_last_send` key exists in `feat:{account}` Redis hash. If absent, D-01 correctly does NOT fire (missing data ≠ confirmed dormant). |
| Leiden flag not set after nightly batch | Check `redis-cli GET leiden:deployed`. If absent, Leiden may have returned empty community map — check nightly batch logs. |
| Shadow committee not writing rows | Check `committee_shadow_mode=true` in env and that `shadow_score_committee` table exists (`alembic current` must show 007). |
| `model_integrity_check_failed` at startup | Run `ml/train.py` then `store_model_hash()` — or ignore on first run (model not yet trained). |

---

*Updated 2026-05-29. Update as items are completed.*
