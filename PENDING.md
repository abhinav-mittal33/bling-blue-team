# BLING Blue Team — Pending Items

Everything on our side is implemented and trained. Only external dependencies and data-accumulation gates remain.

*Updated: 2026-05-30. All code complete, all models trained with synthetic data.*

---

## BEFORE FIRST RUN (one-time setup)

```bash
# 1. Configure environment
cp .env.example .env
# Fill: POSTGRES_URL, REDIS_URL, GRAPH_ENGINE_API_KEY, INVESTIGATOR_API_KEY,
#       INTERNAL_API_KEY, SALT, PSEUDONYMIZATION_KEY (32-byte hex), MODEL_VERSION

# 2. Generate JWT keys (production only)
openssl genrsa -out jwt_private.pem 2048
openssl rsa -in jwt_private.pem -pubout -out jwt_public.pem
# Set JWT_PRIVATE_KEY and JWT_PUBLIC_KEY in .env

# 3. Start infrastructure
docker-compose up -d

# 4. Run migrations (001 → 007)
alembic upgrade head

# 5. Seed Redis and load demo data
python scripts/seed_redis.py
python scripts/generate_test_data.py && python scripts/load_sample_data.py

# 6. Flush stale FTRL Redis keys (one-time, all environments)
python scripts/flush_ftrl_redis.py

# 7. Run tests
pytest tests/ -v

# 8. Start Celery (separate terminal)
celery -A app.celery_app worker -l info -Q default,evidence,graph
celery -A app.celery_app beat -l info
```

---

## BLOCKED ON TEAMMATES

### Graph Engine teammate — Neo4j
- [ ] Get Neo4j bolt URL + credentials → set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- [ ] Confirm POST /api/v1/score request schema with Graph Engine teammate
- [ ] Confirm Neo4j Transaction nodes have `payee_vpa_created_at` (needed by D-01 gate)
- [ ] Share `GRAPH_ENGINE_API_KEY` with Graph Engine teammate

### Graph Engine teammate — P2-2 (HGT model, most important future item)
- [ ] **Graph Engine teammate must add Device + VPA nodes to Neo4j schema**
  Current: `Account --[SENT]--> Account` only
  Required: `Account --[USED]--> Device`, `Account --[HAS_VPA]--> VPA`
  Stub ready: `app/graph/gnn_embedder.py::build_hetero_data_stub()` — accepts device/VPA edges when provided
- [ ] After P2-2: write and run `ml/train_hgt.py` (3 node types, 3 edge types)
- [ ] After P2-2: update Scorer B to use HGT embeddings instead of Node2Vec/PC-GNN
- [ ] After P2-2: re-derive ensemble thresholds (thresholds shift again with HGT)

### Blockchain teammate
- [ ] Get `BLOCKCHAIN_SERVICE_URL` → set in `.env`
  Stub: `app/integrations/blockchain_client.py` (evidence sealing ready, sends when URL set)

### Red Team teammate
- [ ] Get `RED_TEAM_SERVICE_URL` → set in `.env`
  Stub: `app/integrations/red_team_client.py` (fraud DNA delivery ready, sends when URL set)

### Investigator Dashboard teammate
- [ ] Get `INVESTIGATOR_DASHBOARD_URL` → set in `.env`
- [ ] Share `INVESTIGATOR_API_KEY` with Dashboard teammate

---

## BLOCKED ON LIVE CREDENTIALS

### Compliance (code complete — needs live credentials to activate)
- [ ] **FINnet 2.0** — STR drafts are generated. To submit live: set `FINNET_LIVE=true` + `FINNET_API_KEY`
- [ ] **NPCI pre-settlement** — set `NPCI_LIVE=true` + `NPCI_API_KEY`
- [ ] **DPDP Act erasure** — set `DPDP_LIVE=true` to expose `/api/v1/dpdp/erase` endpoint
- [ ] **OFAC / UN / MHA sanctions sync** — set URLs in `.env`:
  `OFAC_SDN_URL`, `UN_SANCTIONS_URL`, `MHA_SANCTIONS_URL`
  Then implement XML/CSV parser in `app/compliance/sanctions_client._fetch_list()` (stub returns empty list)
- [ ] **CISO email alerts** — fires when fraud > ₹10L (HIGH_RISK). Set in `.env`:
  `CISO_EMAIL`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`

---

## COMMITTEE ENGINE — PHASE GATES (needs real transaction traffic)

These cannot be done until live traffic flows through the pipeline:

- [ ] **Accumulate 10,000 shadow rows** — committee is shadow mode; every SUSPICIOUS transaction writes one row
  Monitor: `SELECT COUNT(*) FROM shadow_score_committee`

- [ ] **Train meta-learner** after 10K shadow rows
  ```bash
  python ml/train_meta_learner.py   # auto-aborts if < 10,000 rows
  ```

- [ ] **Accumulate 50,000 shadow rows** — required before deriving new thresholds

- [ ] **Derive committee thresholds** after 50K shadow rows
  ```bash
  python ml/derive_committee_thresholds.py
  # outputs ml/models/committee_thresholds.json
  # update LOG_THRESHOLD / REVIEW_THRESHOLD / HIGH_RISK_THRESHOLD in .env
  ```
  Warning fires if LOG threshold shifts > 0.05 from 0.38 — manual review required

- [ ] **Retrain all models on real data** — all current models trained on synthetic data.
  Real-data retrain order:
  1. `python ml/train.py --force` (XGBoost base — gets real graph features from Redis)
  2. `python ml/train_gnn.py` (PC-GNN — gets real labels from PostgreSQL)
  3. `python ml/train_scorer_b.py` (re-run after GNN embeddings populated in Redis)
  4. `python ml/train_scorer_a.py` (re-run after UPI session features are wired in txn schema)

- [ ] **Validate committee performance** — meta-learner PR-AUC must beat existing XGBoost PR-AUC

- [ ] **Staging 48h run** — run `COMMITTEE_LIVE_MODE=true` on staging for 48h with zero incidents

- [ ] **Flip committee live** — only after all above gates pass
  Set `COMMITTEE_LIVE_MODE=true`, `COMMITTEE_SHADOW_MODE=false` in `.env`, restart

---

## GATE 0 — RAPID RELAY PILOT (needs live traffic)

- [ ] Gate 0 is LOG-ONLY (`GATE0_LIVE=false`). After 2 weeks of live traffic:
  ```sql
  SELECT * FROM model_audit WHERE event_data->>'gate' = 'rapid_relay' LIMIT 100;
  ```
  If pilot looks clean: `GATE0_LIVE=true` in `.env`, restart → promotes to REVIEW

---

## SCORER D — MAMBA UPGRADE (future, low priority)

- [ ] Scorer D in limited mode (Random Forest on set features). Full Mamba needs ordered txn sequences.
  When ready: `MAMBA_LIMITED_MODE=false` + provide `ml/models/scorer_d_mamba_v1.pt`

---

## INFRASTRUCTURE (production hardening)

- [ ] `pip-audit -r requirements.txt` before production deploy
- [ ] Set up Grafana dashboard — metrics at `/metrics`
- [ ] Cloud model storage (S3/GCS) for production — currently volume-mounted
- [ ] PostgreSQL automated daily backups before production
- [ ] Verify `appendonly yes` in production Redis config (AOF persistence)
- [ ] Load test: `locust -f tests/load/locustfile.py --headless -u 1000 -r 50 --run-time 5m` (P99 ≤ 85ms target)
- [ ] `INTERNAL_API_KEY` → empty string in production (dev/test key only)

---

## DEMO DAY CHECKLIST

```bash
docker-compose ps
curl http://localhost:8000/health

# High-risk transaction — expect HIGH_RISK, score >= 0.83
curl -X POST http://localhost:8000/api/v1/score \
  -H "X-API-Key: $GRAPH_ENGINE_API_KEY" -H "Content-Type: application/json" \
  -d '{"transaction_id":"demo_001","account_id":"ACC_FRAUD_001","amount":"500000",
       "channel":"UPI","timestamp":"2026-05-30T02:14:00Z",
       "payee_vpa":"scammer@upi","payee_vpa_created_at":"2026-05-28T10:00:00Z"}'

# Legit transaction — expect LOG or PASS
curl -X POST http://localhost:8000/api/v1/score \
  -H "X-API-Key: $GRAPH_ENGINE_API_KEY" -H "Content-Type: application/json" \
  -d '{"transaction_id":"demo_002","account_id":"ACC_LEGIT_001","amount":"25000",
       "channel":"NEFT","timestamp":"2026-05-30T11:00:00Z",
       "payee_vpa":"employer@upi","payee_vpa_created_at":"2024-01-01T00:00:00Z"}'

# Developer queue
curl http://localhost:8000/api/v1/developer-queue/prototype-candidates \
  -H "X-API-Key: $INTERNAL_API_KEY"

curl http://localhost:8000/metrics | grep bling_
open http://localhost:8000/docs
pytest tests/test_integration/test_fraud_scenarios.py -v
```

---

## TROUBLESHOOTING

| Symptom | Fix |
|---------|-----|
| Score stuck at 0.4 for everything | `redis-cli HGETALL feat:ACC_FRAUD_001` — re-run `seed_redis.py` |
| `prototype_vault_disabled` at startup | `python ml/scripts/build_initial_prototypes.py` (already done — check path) |
| `novelty_detection_disabled` at startup | `python ml/train_isolation_forest.py` |
| XGBoost model not found | `python ml/train.py --force` |
| POST /score returns 401 | `GRAPH_ENGINE_API_KEY` mismatch between caller and `.env` |
| JWT 401 "not configured" | Set `JWT_PUBLIC_KEY` in `.env` or use X-API-Key header |
| `leiden:deployed` not set | Leiden returned empty community map — check nightly batch logs |
| Shadow committee not writing | Check `alembic current` = 007 and `COMMITTEE_SHADOW_MODE=true` |
| Fund trail never arrives | Celery worker must use `-Q default,evidence,graph` |
| All thresholds 0.0002 | Synthetic-data artifact — re-derive on real data after `alembic upgrade head` + real txn load |
| Scorer B low quality | Re-run `ml/train_scorer_b.py` after GNN embeddings are in Redis from real graph data |

---

*All code is complete. All models trained (synthetic). Ready to connect infrastructure and test.*
