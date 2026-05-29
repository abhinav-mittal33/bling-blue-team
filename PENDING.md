# BLING Blue Team — Full Pending List

Everything that still needs to be done, in priority order. Nothing omitted.

---

## BEFORE DEMO — MUST DO

### Setup (do once, in order)

- [ ] Copy `.env.example` → `.env` and fill ALL values
  ```bash
  cp .env.example .env
  ```
  Required: `POSTGRES_URL`, `REDIS_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`,
  `GRAPH_ENGINE_API_KEY`, `INVESTIGATOR_API_KEY`, `INTERNAL_API_KEY`,
  `SALT`, `PSEUDONYMIZATION_KEY` (32-byte hex), `MODEL_VERSION`

- [ ] Generate JWT keys (optional in dev, required in production)
  ```bash
  openssl genrsa -out jwt_private.pem 2048
  openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem
  ```
  Set `JWT_PRIVATE_KEY` and `JWT_PUBLIC_KEY` in `.env`

- [ ] Start infrastructure
  ```bash
  docker-compose up -d
  ```

- [ ] Run all migrations (001 → 007)
  ```bash
  alembic upgrade head
  # verify: alembic current → should show 007 (head)
  ```

- [ ] Train XGBoost model
  ```bash
  python ml/train.py
  # output: ml/models/xgboost_v1.json + xgboost_calibrated_v2.joblib
  ```

- [ ] Build Scorer F phrase dictionary (committee engine)
  ```bash
  python ml/scripts/build_phrase_dict.py
  # output: ml/models/upi_fraud_phrases.json
  ```

- [ ] Build Scorer C initial FAISS prototype vault
  ```bash
  python ml/scripts/build_initial_prototypes.py
  # output: ml/models/prototype_faiss.index + prototype_meta.joblib
  ```

- [ ] Train discovery ensemble (anomaly detection, PASS stream only)
  ```bash
  python ml/train_isolation_forest.py
  # output: ml/models/isolation_forest_v1.joblib
  ```

- [ ] Seed Redis and load demo data
  ```bash
  python scripts/seed_redis.py
  python scripts/generate_test_data.py && python scripts/load_sample_data.py
  ```

- [ ] Run all tests — must pass before demo
  ```bash
  pytest tests/ -v
  # expect: 102+ passing
  ```

- [ ] Start Celery worker + beat (separate terminal)
  ```bash
  celery -A app.celery_app worker -l info -Q default,evidence,graph
  celery -A app.celery_app beat -l info
  ```

---

## TEAMMATE COORDINATION

- [ ] Get Neo4j bolt URL + credentials from Graph Engine teammate → set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- [ ] Get `BLOCKCHAIN_SERVICE_URL` from Blockchain teammate
- [ ] Get `RED_TEAM_SERVICE_URL` from Red Team teammate
- [ ] Get `INVESTIGATOR_DASHBOARD_URL` from Dashboard teammate
- [ ] Share `GRAPH_ENGINE_API_KEY` with Graph Engine teammate
- [ ] Share `INVESTIGATOR_API_KEY` with Investigator Dashboard teammate
- [ ] Confirm POST /api/v1/score request schema with Graph Engine teammate
- [ ] Confirm Neo4j schema includes `payee_vpa_created_at` on Transaction nodes (needed by D-01 gate)

### BLOCKED ON TEAMMATE — GNN (most important future item)
- [ ] **Graph Engine teammate must add Device + VPA nodes to Neo4j schema (P2-2)**
  Current schema: `Account --[SENT]--> Account` only
  Required: `Account --[USED]--> Device`, `Account --[HAS_VPA]--> VPA`
  Nothing can be done on our side until this happens

- [ ] After P2-2 done: add `torch>=2.1.0` + `torch-geometric` back to `requirements.txt`
- [ ] After P2-2 done: write and run `ml/train_hgt.py` (HGT model — 3 node types, 3 edge types)
- [ ] After P2-2 done: update Scorer B to use HGT embeddings instead of Node2Vec
- [ ] After P2-2 done: retrain ensemble (P4-3 — thresholds change again after HGT added)

---

## FUNCTIONS THAT EXIST IN CODE BUT ARE NOT IMPLEMENTED YET

These are silently skipped right now via `AttributeError` catch — they will never crash, but the feature doesn't run.

- [ ] `update_betweenness_only()` in `nightly_batch.py`
  Celery Beat calls it every 2h but function doesn't exist yet. Implement: approximate betweenness (k=500) for recently-active accounts only, partial Redis update.

- [ ] `update_micro_batch_features()` in `nightly_batch.py`
  Celery Beat calls it every 5min but function doesn't exist. Implement: update `degree_centrality`, `temporal_acceleration`, `sink_score` from Neo4j micro-batch query.

- [ ] `refresh_recent_embeddings(lookback_minutes=60)` in `node2vec_runner.py`
  Called by `refresh_gnn_embeddings_task` every 5min but function doesn't exist. Implement: re-embed only accounts active in last 60min rather than full nightly re-run.

- [ ] `payee_in_known_contacts` in `score.py` line 260
  Currently hardcoded `False`. Wire to a real known-contacts table lookup.

---

## MODELS THAT STILL NEED TRAINING

- [ ] Scorer A model (`scorer_a_v1.joblib`) — currently falls back to existing XGBoost
  ```bash
  python ml/train_scorer_a.py   # needs UPI session features in transaction schema first
  ```

- [ ] Scorer B model (`scorer_b_v1.joblib`) — embedding + structural context MLP
  ```bash
  python ml/train_scorer_b.py   # needs Node2Vec embeddings in Redis first
  ```

- [ ] ECOD discovery model (`ecod_v1.joblib`) — second anomaly detector
  ```bash
  python ml/train_ecod.py       # train on PASS-stream feature vectors
  ```

- [ ] XGBOD model (`xgbod_v1.joblib`) — P4-6 second novelty layer
  Needs `ml/train_xgbod.py` to be written, then run

- [ ] Meta-learner (`meta_learner_v1.joblib`) — committee stacking model
  ```bash
  python ml/train_meta_learner.py   # auto-aborts if fewer than 10,000 shadow rows
  ```
  Will not train until 10K SUSPICIOUS transactions have been scored in shadow mode

- [ ] HGT model — blocked on teammate (see above)

---

## COMMITTEE ENGINE — PHASE GATES

- [ ] **Accumulate shadow rows** — committee is in shadow mode, every SUSPICIOUS transaction writes one row. Need 10,000 rows before meta-learner can be trained.
  Monitor: `SELECT COUNT(*) FROM shadow_score_committee`

- [ ] **Train meta-learner** after 10K shadow rows (see Models above)

- [ ] **Accumulate 50,000 shadow rows** before deriving new thresholds and going live

- [ ] **Derive committee thresholds** after 50K shadow rows
  ```bash
  python ml/derive_committee_thresholds.py
  # updates ml/models/committee_thresholds.json
  # update LOG_THRESHOLD / REVIEW_THRESHOLD / HIGH_RISK_THRESHOLD in .env
  ```
  Warning fires if LOG threshold shifts more than 0.05 from current 0.38 — manual review required

- [ ] **Validate committee performance** — meta-learner PR-AUC must beat existing XGBoost PR-AUC on held-out shadow validation

- [ ] **Staging 48h run** — run `COMMITTEE_LIVE_MODE=true` on staging for 48h with zero incidents before touching production

- [ ] **Flip committee live** — only after all above gates pass
  Set `COMMITTEE_LIVE_MODE=true`, `COMMITTEE_SHADOW_MODE=false` in `.env`, restart

---

## GATE 0 — RAPID RELAY PILOT

- [ ] Gate 0 is LOG-ONLY right now (`GATE0_LIVE=false`)
- [ ] After 2 weeks of live traffic: review pilot logs — are the LOG flags legitimate relays or false positives?
  ```bash
  # check Gate 0 pilot logs
  SELECT * FROM model_audit WHERE event_data->>'gate' = 'rapid_relay' LIMIT 100;
  ```
- [ ] If pilot looks good: set `GATE0_LIVE=true` in `.env` and restart to promote to REVIEW

---

## COMPLIANCE STUBS — NEED LIVE CREDENTIALS

These exist in code but do nothing until credentials are provided.

- [ ] **FINnet 2.0** — set `FINNET_LIVE=true` and `FINNET_API_KEY` in `.env`
  STR drafts (156 fields) are generated and ready — just not submitted

- [ ] **NPCI pre-settlement** — set `NPCI_LIVE=true` and `NPCI_API_KEY` in `.env`

- [ ] **DPDP Act erasure** — set `DPDP_LIVE=true` to expose right-to-erasure endpoint
  Erasure fields: `kyc_occupation`, `kyc_home_state`, `kyc_phone`, `kyc_email`

- [ ] **OFAC / UN / MHA sanctions sync** — set live URLs in `.env`:
  `OFAC_SDN_URL`, `UN_SANCTIONS_URL`, `MHA_SANCTIONS_URL`
  Sync runs daily at 2:30am UTC — currently fetches empty lists because URLs are blank
  Also: implement actual XML/CSV parsing in `sanctions_client._fetch_list()`

- [ ] **CISO email alerts** — set in `.env`:
  `CISO_EMAIL`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`
  Fires when fraud amount exceeds ₹10L (HIGH_RISK). Currently logs only.

---

## SCORER D — MAMBA UPGRADE (future, not urgent)

- [ ] Scorer D runs in limited mode (Random Forest over set features)
- [ ] Full Mamba (state-space sequence model) needs ordered transaction history as input sequences
- [ ] When ready: set `MAMBA_LIMITED_MODE=false` in `.env` and provide `ml/models/scorer_d_mamba_v1.pt`
- [ ] Requires labeled sequence training data — not available yet

---

## INFRASTRUCTURE (production hardening)

- [ ] **Run pip-audit before production deploy**
  ```bash
  pip-audit -r requirements.txt
  ```

- [ ] **Set up Grafana dashboard** — Prometheus metrics available at `/metrics`
  Key metrics: `bling_scoring_requests_total`, `bling_alerts_created_total`, `bling_scoring_latency_ms`

- [ ] **ML model storage** — currently volume-mounted from local disk. For cloud deployment, models need to be pulled from S3/GCS at container startup or baked into image at build time.

- [ ] **PostgreSQL backups** — set up automated daily backups before production

- [ ] **Redis persistence verification** — AOF is enabled. Verify `appendonly yes` is active in production Redis.

- [ ] **Run load test** — locustfile exists at `tests/load/locustfile.py`
  ```bash
  locust -f tests/load/locustfile.py --headless -u 1000 -r 50 --run-time 5m
  ```
  Target: P99 latency ≤ 85ms under 1000 concurrent users

- [ ] **PSI drift monitoring baseline** — P4-7. Run after initial production data accumulates to set baseline for population stability index alerts.

- [ ] **Set `INTERNAL_API_KEY` to empty string in production** — dev/testing key only

- [ ] **Flush FTRL Redis keys** (one-time, all environments)
  ```bash
  python scripts/flush_ftrl_redis.py
  ```

---

## DEMO DAY CHECKLIST

Run 30 minutes before demo:

```bash
# Services running?
docker-compose ps

# API healthy?
curl http://localhost:8000/health

# Score a high-risk transaction — expect HIGH_RISK, score >= 0.83
curl -X POST http://localhost:8000/api/v1/score \
  -H "X-API-Key: $GRAPH_ENGINE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo_001",
    "account_id": "ACC_FRAUD_001",
    "amount": "500000",
    "channel": "UPI",
    "timestamp": "2026-05-30T02:14:00Z",
    "payee_vpa": "scammer@upi",
    "payee_vpa_created_at": "2026-05-28T10:00:00Z"
  }'

# Score a legit transaction — expect LOG or PASS
curl -X POST http://localhost:8000/api/v1/score \
  -H "X-API-Key: $GRAPH_ENGINE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "demo_002",
    "account_id": "ACC_LEGIT_001",
    "amount": "25000",
    "channel": "NEFT",
    "timestamp": "2026-05-30T11:00:00Z",
    "payee_vpa": "employer@upi",
    "payee_vpa_created_at": "2024-01-01T00:00:00Z"
  }'

# Developer queue (novel fraud candidates)
curl http://localhost:8000/api/v1/developer-queue/prototype-candidates \
  -H "X-API-Key: $INTERNAL_API_KEY"

# Metrics working?
curl http://localhost:8000/metrics | grep bling_

# Docs accessible?
open http://localhost:8000/docs

# All 8 integration scenarios pass?
pytest tests/test_integration/test_fraud_scenarios.py -v
```

---

## TROUBLESHOOTING

| Symptom | Fix |
|---------|-----|
| Score stuck at 0.4 for everything | `redis-cli HGETALL feat:ACC_FRAUD_001` — re-run `seed_redis.py` |
| `prototype_vault_disabled` at startup | `python ml/scripts/build_initial_prototypes.py` |
| `novelty_detection_disabled` at startup | `python ml/train_isolation_forest.py` |
| XGBoost model not found | `python ml/train.py` |
| POST /score returns 401 | `GRAPH_ENGINE_API_KEY` mismatch between caller and `.env` |
| JWT 401 "not configured" | Set `JWT_PUBLIC_KEY` in `.env` or use X-API-Key |
| D-01 gate fires on old accounts | Check `days_since_last_send` in `feat:{account}` Redis hash. Missing key = gate correctly stays silent. |
| `leiden:deployed` not set | Leiden returned empty community map — check nightly batch logs |
| Shadow committee not writing | Check `alembic current` = 007 and `COMMITTEE_SHADOW_MODE=true` |
| Fund trail never arrives | Celery worker must use `-Q default,evidence,graph` — check `docker-compose.yml` |
| `betweenness_update_skipped` in logs | Normal — `update_betweenness_only()` not yet implemented |
| `micro_batch_skipped` in logs | Normal — `update_micro_batch_features()` not yet implemented |
| `gnn_refresh_skipped` in logs | Normal — `refresh_recent_embeddings()` not yet implemented |

---

*Updated 2026-05-30. Check off items as completed.*
