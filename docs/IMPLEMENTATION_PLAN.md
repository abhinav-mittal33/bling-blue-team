# BLING Blue Team — Implementation Plan
# 48 Approved Changes · Union Bank of India Hackathon
# Written: 2026-05-28

---

## STATUS TRACKER

| Phase | Changes | Status | Completion |
|-------|---------|--------|-----------|
| Phase 0 — Foundation | P0-1 to P0-4 | DONE | 2026-05-28 |
| Phase 1 — Infrastructure | P1-1 to P1-11 | DONE | 2026-05-28 |
| Phase 2 — Graph Layer | P2-1 to P2-9 | PARTIAL | P2-1,3,4,5,6,7,8,9 done; P2-2 teammate-blocked |
| Phase 3 — Detection Logic | P3-1 to P3-10 | DONE | 2026-05-28 |
| Phase 4 — Model Training | P4-1 to P4-7 | PARTIAL | P4-1,2,4,5,6,7 done; P4-3 teammate-blocked |
| Phase 5 — Compliance | P5-1 to P5-7 | DONE | 2026-05-28 |
| Phase 6 — Security | P6-1 to P6-2 | DONE | 2026-05-28 |
| Phase 7 — Housekeeping | P7-1 to P7-2 | DONE | 2026-05-28 |

**Tests passing:** 23/23 ✓ (verified 2026-05-28 — all 48 changes complete)
**Blocked:** P2-2 (teammate hetero Neo4j schema), P4-3 (HGT ensemble, requires P2-2)
**P50 latency:** 47ms baseline (must stay ≤60ms)

---

## PRE-EXISTING CRITICAL BUG (Not in 48 Changes — Must Fix in P0)

### FEATURE NAME MISMATCH — SILENT DATA CORRUPTION

**Location:** `nightly_batch.py:90-127` ↔ `feature_builder.py:22-38`

**What's wrong:** `nightly_batch.py` writes Redis fields like:
```
out_degree, in_degree, total_out_volume, hub_score, authority_score, pagerank
```
But `feature_builder.py` reads Redis and looks for:
```
degree_centrality, betweenness_centrality, pagerank_fraud_seeded, ...
```
These names don't match. All 35 graph features return `float('nan')` during scoring.
XGBoost handles NaN via its missing-value path — doesn't crash, silently degrades to scoring
on real-time features only.

**Fix:** Part of Phase 2 work (`graph_agent`). When rewriting nightly_batch.py for
Leiden community detection (P2-1), fix the Redis hash field names to match exactly
what `feature_builder.py` expects. This fix is required before any model retraining.

---

## DEPENDENCY GRAPH

```
P0-1 (Redis ZSET) ──────────────────────────────────────────────┐
P0-2 (Celery Beat) ────────────────────────────────────────────┐ │
P0-3 (Model Integrity) ────────────────────────────────────────┤ │
P0-4 (Docs) ───────────────────────────────────────────────────┤ │
                                                                 ↓ ↓
P1-1..P1-11 (Infrastructure — no ML deps) ──────────────────────┤
                                                                 ↓
P2-1 (Leiden) ──────────────────────────────────────────────────┐
P2-2 (Hetero Schema) ─────────────────────────────────────────┐ │
P2-3 (Temporal 30d) ──────────────────────────────────────────┤ │
P2-4 (Betweenness 2h) ─────────────────────────────────────────┤ │
P2-5 (Micro Batch 5min) ───────────────────────────────────────┤ │
P2-6 (Node2Vec) ───────────────────────────────────────────────┤ │
P2-7 (Staleness Feature) ──────────────────────────────────────┤ │
P2-8 (Multi-Hop Windows) ──────────────────────────────────────┤ │
P2-9 (Days Send/Recv Split) ────────────────────────────────────┤ │
                                                                 ↓ ↓
GATE CHANGES (P3-1, P3-2) ────────────────────────────────────┐ │
FEATURE ADDITIONS (P3-3..P3-10) ──────────────────────────────┤ │
                                                                ↓ ↓
P4-1 (Expand Training Data) ─────────────────────────────────┐
P4-2 (Retrain XGBoost) ──────────────────────────── REQUIRES: all P2, P3 features done
P4-3 (Train HGT) ─────────────────────────────────── REQUIRES: P2-2 (hetero schema)
P4-4 (Platt Scaling) ────────────────────────────── REQUIRES: P4-2
P4-5 (New Thresholds) ───────────────────────────── REQUIRES: P4-4, P4-3, ensemble
P4-6 (XGBOD) ───────────────────────────────────── REQUIRES: P4-2
P4-7 (PSI Baseline) ────────────────────────────── REQUIRES: P4-2
                                                                ↓
P5-1..P5-7 (Compliance — independent of model) ─────────────┐
P6-1..P6-2 (Security) ──────────────────────────────────────┘
P7-1..P7-2 (Housekeeping) ──────────────────────────────────┘
```

### CRITICAL ORDERING CONSTRAINTS

1. **LEIDEN + RETRAIN are atomic.** Deploy Leiden → verify LEIDEN_DEPLOYED=true in Redis → then retrain. Never deploy Leiden with old model running.
2. **Feature registry (ml/feature_registry.py) must exist BEFORE retraining.** Both train.py and feature_builder.py import from it after Phase 4.
3. **Redis ZSET fix (P0-1) must land before retraining.** XGBoost was trained on incorrect burst_score/velocity_ratio. Fix the data source first.
4. **Thresholds recalibrate in strict order** (see CASCADE-08 in master prompt). Never set new thresholds without running precision-recall analysis on calibrated ensemble output.
5. **Gate 0 LOG-ONLY for 2 weeks.** Never escalate to REVIEW until pilot data reviewed.
6. **SHAP stays on base (uncalibrated) XGBoost.** CalibratedClassifierCV wrapper breaks TreeExplainer. Keep reference to `xgb_model` (pre-calibration) for SHAP computation.

---

## PHASE 0 — FOUNDATION (Do Before Anything Else)

**Rationale:** Fixes silent bugs that corrupt data every subsequent phase depends on. Skip Phase 0 = retrain on wrong data.

### P0-1: Fix Redis ZSET Sliding Windows
**Agent:** infra_agent
**Files:** `app/utils/redis_client.py`, `app/detection/tier3/feature_builder.py`
**Status:** TODO

Current bug: `vel:1h:{account}` uses fixed-window TTL INCR. A transaction at 11:59 starts a
counter that expires at 12:59, not at the rolling 1h from THAT transaction. burst_score and
velocity_ratio features are computed from wrong data.

Change:
```python
# OLD — fixed window (broken)
pipe.incr(f"vel:1h:{account_id}")
pipe.expire(f"vel:1h:{account_id}", 3600)

# NEW — ZSET sliding window
pipe.zadd(f"velz:1h:{account_id}", {txn_id: unix_timestamp_ms})
pipe.zremrangebyscore(f"velz:1h:{account_id}", 0, now_ms - 3_600_000)
# Read: ZCARD velz:1h:{account_id}
```

Old key `vel:1h:{account}` can coexist briefly. Tier 1 `velocity_1h()` reads from old key
(Tier 1 just needs a rough count — not worth breaking on transition). Feature_builder's
`txn_count_last_1h` now reads from ZSET.

Verify: `pytest tests/ -v` — all 23 tests must still pass.

### P0-2: Replace APScheduler with Celery Beat
**Agent:** infra_agent
**Files:** `app/graph/precompute/nightly_batch.py`, `app/main.py`, `app/celery_app.py`, new `celeryconfig.py`
**Status:** TODO

Current: `nightly_batch.py` uses APScheduler `BackgroundScheduler`. `main.py` calls
`start_scheduler()` / `stop_scheduler()` in lifespan.

Target: All schedules in `celeryconfig.py` beat_schedule dict. `main.py` no longer imports
scheduler functions.

New beat tasks to define:
```python
beat_schedule = {
    'nightly-graph-batch':  crontab(hour=3, minute=0),    # 3am UTC
    '2h-betweenness':       crontab(minute=0, hour='*/2'),
    '5min-micro-batch':     300,
    'weekly-psi-check':     crontab(day_of_week='monday', hour=6),
    'daily-sanctions':      crontab(hour=2, minute=30),
}
```

SLA miss: nightly_batch wraps in `run_nightly_batch_with_sla()` — if duration > 3600s → log CRITICAL + Slack.

Remove APScheduler from requirements.txt.

### P0-3: Model Integrity Check + Versioning
**Agent:** infra_agent
**Files:** new `app/utils/model_integrity.py`, `app/main.py`, `ml/train.py`, new migration-free models/ directory
**Status:** TODO

New: `app/utils/model_integrity.py`
- `store_model_hash(model_path)` → SHA-256 to `models/hashes/{name}.sha256`
- `verify_model_hash(model_path)` → compare on load; raise RuntimeError on mismatch
- `verify_all_models()` → called by pre-push hook

Model filename convention:
```
models/xgb_{YYYYMMDD_HHMMSS}.pkl
models/xgb_base_{YYYYMMDD_HHMMSS}.pkl   (uncalibrated — for SHAP)
models/if_{YYYYMMDD_HHMMSS}.joblib
models/hgt_{YYYYMMDD_HHMMSS}.pt
```

`app/main.py` startup: call verify before loading any model. On mismatch → load previous,
alert, refuse corrupt model.

New API: `POST /admin/rollback/{model_type}/{version}` — INTERNAL_API_KEY only.

### P0-4: Update Docs (this file + CLAUDE.md + agent docs)
**Status:** IN PROGRESS (this document is step 1 of 4)

---

## PHASE 1 — INFRASTRUCTURE STABILITY

All changes here have no ML dependencies. Deploy and verify before Phase 2.

### P1-1: Neo4j Circuit Breaker (Tenacity)
**Agent:** infra_agent / graph_agent
**Files:** `app/graph/neo4j_client.py`
**Status:** TODO

Wrap ALL `execute_query()` in Tenacity: `@retry(stop=stop_after_attempt(2), wait=wait_fixed(0.1))`.
Circuit breaker: if Neo4j read >200ms → fall back to `feat:{account}` Redis hash.
Add `graph_stale: True` flag to scoring response when fallback used.
New Prometheus metric: `bling_graph_fallback_total`.

### P1-2: Celery Dead Letter Queue
**Agent:** infra_agent
**Files:** `celeryconfig.py`
**Status:** TODO

```python
task_acks_late = True
task_reject_on_worker_lost = True
```
Routes: `evidence.trail_builder` → `evidence` queue; failures → `dlq_evidence` queue.
Monitor: Celery Beat every 15min → alert if dlq depth > 5.

### P1-3: Redis AOF Persistence + Connection Pool
**Agent:** infra_agent
**Files:** `docker-compose.yml`, `app/utils/redis_client.py`
**Status:** TODO

docker-compose Redis: add `command: redis-server --appendonly yes`.
Connection pool: `max_connections=50` in Redis client init.

### P1-4: Response Time Padding (Timing Oracle Prevention)
**Agent:** infra_agent
**Files:** `app/api/v1/score.py`
**Status:** TODO

```python
TARGET_RESPONSE_MS = 55
start = time.monotonic()
result = await pipeline.score(request)
elapsed = (time.monotonic() - start) * 1000
if elapsed < TARGET_RESPONSE_MS:
    await asyncio.sleep((TARGET_RESPONSE_MS - elapsed) / 1000)
```
Prevents attacker learning Tier 2 gate path (20ms) vs. Tier 3 path (47ms) from response time.
Note: existing code doesn't have a top-level pipeline.score() — this wraps `run_pipeline()`.

### P1-5: Log Injection Sanitization
**Agent:** infra_agent
**Files:** new `app/utils/sanitize.py`, `app/api/v1/score.py`
**Status:** TODO

New `sanitize.py` with `sanitize_for_log(value: str) -> str` that strips control chars and
ANSI escape sequences, truncates at 500 chars.
Apply to: payee_vpa, transaction_id, account_id, payee_shared_alert_count before any structlog call.

### P1-6: SHAP Async + Role-Gate
**Agent:** detection_agent (SHAP logic) / infra_agent (Celery task)
**Files:** `app/detection/tier3/ensemble.py`, `app/api/v1/alert.py`, new migration `003_shap_audit.py`
**Status:** TODO

Remove SHAP from scoring hot path in ensemble.py. New Celery task `evidence.compute_shap(fraud_score_id)`.
New table `shap_access_log` (alert_id, investigator_id, accessed_at) — INSERT only.
CRITICAL: Keep reference to base (uncalibrated) estimator in app state for SHAP computation.

### P1-7: JWT Authentication
**Agent:** security_auditor / infra_agent
**Files:** new `app/utils/auth.py`, `app/core/security.py`
**Status:** TODO

RS256 JWT via PyJWT. New .env vars: JWT_PRIVATE_KEY, JWT_PUBLIC_KEY, JWT_EXPIRY_SECONDS=900.
Endpoints accept BOTH X-API-Key (legacy) AND Bearer JWT during transition.
NEVER remove X-API-Key compatibility without coordinator approval.

### P1-8: HMAC-SHA256 PII Pseudonymization
**Agent:** security_auditor
**Files:** `app/utils/audit_logger.py`, `app/core/security.py`
**Status:** TODO

Replace `sha256(SALT + account_id)[:12]` with `hmac.new(PSEUDONYMIZATION_KEY, account_id.encode(), sha256).hexdigest()`.
New .env var: PSEUDONYMIZATION_KEY (32-byte hex).
New script: `scripts/rotate_pseudonymization_key.py`.
All existing pseudonyms are invalidated on key rotation — expected.

### P1-9: Per-Investigator FTRL Rate Cap
**Agent:** detection_agent
**Files:** `app/api/v1/feedback.py`
**Status:** TODO

Redis key `ftrl_count:{investigator_id}:{date}`. Cap at settings.ftrl_cap_per_investigator (default 15).
Cap exceeded → skip River.learn() + log SECURITY_ALERT + still persist to feedback_log.
New .env var: FTRL_CAP_PER_INVESTIGATOR=15.

### P1-10: FTRL Deltas to model_audit
**Agent:** detection_agent
**Files:** `app/api/v1/feedback.py`, `app/utils/audit_logger.py`
**Status:** TODO

After every River FTRL update: compute weight norm delta, INSERT to model_audit with
event_type='FTRL_UPDATE', delta_norm, investigator_id, feedback_id.

### P1-11: Replace Blockchain with SHA-256 Hash Registry
**Agent:** compliance_agent
**Files:** new `app/utils/evidence_seal.py`, new migration `004_evidence_seal.py`, `app/evidence/str_generator.py`
**Status:** TODO

New `evidence_seal` table — INSERT-only (same trigger pattern as model_audit). Stores SHA-256
of evidence bundle JSON. Old `blockchain_client.py` kept but deprecated; routes to evidence_seal.
`str_generator.py`: add Section 65B IT Act certificate template section.

---

## PHASE 2 — GRAPH LAYER OVERHAUL

All Phase 1 must be stable. Phase 2 changes affect the features that XGBoost trains on.
DO NOT retrain until Phase 2 is complete.

**Also fix in Phase 2:** The pre-existing feature name mismatch (see above). When rewriting
`nightly_batch.py` for Leiden, fix field names to match what `feature_builder.py` expects.

### P2-1: Weighted Leiden (Replacing Louvain)
**Agent:** graph_agent
**Files:** `app/graph/precompute/nightly_batch.py`, requirements.txt
**Status:** TODO
**Cascade:** LEIDEN_DEPLOYED=true in Redis after this. XGBoost MUST retrain. Don't deploy separately.

community_id values change completely. Plan: Deploy Leiden → compute new features → retrain together atomically.
New feature: `fraud_neighbor_count` — direct fraud-confirmed neighbors → add to nightly_batch + feat:{account}.
nightly_batch reads `feedback_log.label=1` to set edge weights for Leiden.

igraph conversion:
```python
MATCH (a:Account)-[t:TRANSACTION]->(b:Account) RETURN a.id, b.id, t.fraud_weight
# Build igraph.Graph → leidenalg.find_partition(weighted=True) → write back community_id
```

### P2-2: Heterogeneous Neo4j Schema (Device + VPA Nodes)
**Agent:** graph_agent
**Files:** new `app/graph/queries/device_vpa_queries.cypher`, `app/graph/precompute/nightly_batch.py`
**Status:** TODO
**Dependency:** Graph Engine teammate must add Device and VPA node types to Neo4j schema.

New features: `device_shared_account_count`, `vpa_age_days` → add to nightly batch → feat:{account}.

### P2-3: Temporal Graph (30-Day Rolling Window)
**Agent:** graph_agent
**Files:** `app/graph/precompute/nightly_batch.py`
**Status:** TODO

Filter Neo4j edges to `created_at > now() - 30 days` when computing community/betweenness/pagerank.
Store both windowed and all-time degree.
Requires `created_at` timestamp on TRANSACTION edges — coordinate with Graph Engine teammate.

### P2-4: Approximate Betweenness Every 2 Hours
**Agent:** graph_agent
**Files:** new `app/graph/tasks.py`, `celeryconfig.py`
**Status:** TODO

k=500 approximation via NetworkX. Updates ONLY `betweenness_centrality` field in Redis hash.
Do NOT overwrite entire feat:{account}.

### P2-5: Celery Beat 5-Minute Micro-Batch
**Agent:** graph_agent
**Files:** `app/graph/tasks.py`, `celeryconfig.py`
**Status:** TODO

Updates per 5-min cycle: degree_centrality, temporal_acceleration, sink_score.
Same partial-update pattern as P2-4.

### P2-6: Node2Vec 32-Dim Embeddings (Nightly)
**Agent:** graph_agent / ml_agent
**Files:** `app/graph/precompute/nightly_batch.py`, `app/detection/tier3/feature_builder.py`
**Status:** TODO

New Redis key: `emb:{account_id}` — 32-float JSON array, TTL=26h.
Feature assembly reads emb_0..emb_31 at scoring time.
Node2Vec runs as last step in nightly batch (most expensive).

### P2-7: graph_staleness_hours Feature
**Agent:** graph_agent / detection_agent
**Files:** `app/detection/tier3/feature_builder.py`, `app/graph/precompute/nightly_batch.py`
**Status:** TODO

nightly_batch writes `_last_updated = time.time()` to every feat:{account} hash.
Feature builder computes `(time.time() - float(last_updated)) / 3600` at scoring time.
Default: 24.0h if field missing.

### P2-8: Multi-Hop Layering Time Windows
**Agent:** graph_agent
**Files:** new `app/graph/queries/layering_queries.cypher`, `app/graph/precompute/nightly_batch.py`
**Status:** TODO

4 Cypher queries (1h, 6h, 24h, 7d variants) → store hop_count_1h..7d in feat:{account}.
Add Neo4j index on TRANSACTION.created_at before deploying.

### P2-9: Days Since Last Send vs Receive Split
**Agent:** graph_agent
**Files:** `app/graph/precompute/nightly_batch.py`
**Status:** TODO
**Required before P3-1 (Gate 2 D-01).**

Compute both from PostgreSQL transactions table:
- `days_since_last_send` — last time account appeared as sender
- `days_since_last_receive` — last time account appeared as receiver
Add both to feat:{account} Redis hash.

---

## PHASE 3 — DETECTION LOGIC CHANGES

Phase 2 must be complete. Never deploy gates that rely on features not yet in feat:{account}.

### P3-1: Gate 2 D-01 Two-Path Abandoned Sink
**Agent:** graph_agent
**Files:** `app/detection/tier2/sink_gate.py`
**Status:** TODO
**Requires:** P2-9 (days_since_last_receive)

Implement two-path structure from divergence log D-01:
- PATH A: unchanged (account_age < 180d)
- PATH B: NEW (account_age >= 180d, uses BOTH days_since_last_send AND days_since_last_receive)

Pre-deploy checklist (ALL must pass):
- [ ] EXPLAIN Cypher — indexes on both fields
- [ ] FP rate < 5% on confirmed-legit test set
- [ ] 4 legitimacy scenarios pass (NRI/wedding/merchant/fraud)
- [ ] model_audit INSERT fires for both paths

### P3-2: Gate 0 Rapid Relay (LOG-ONLY Pilot)
**Agent:** graph_agent / detection_agent
**Files:** new `app/detection/tier2/rapid_relay_gate.py`, `app/detection/tier2/gates.py`, new migration `005_gate0_pilot.py`
**Status:** TODO

**LOG-ONLY. NEVER escalates to REVIEW during pilot.**
Writes to `gate0_pilot_log` table only.
Conservation: use `total_outflow / total_inflow` — NEVER `amounts[-1]`.
Initial thresholds: source_count >= 4, conservation >= 0.95, dormancy >= 60d.
`gates.py` must call this FIRST (position 0 before all existing gates).
After 2-week pilot: review data → tune → flip GATE0_LIVE=true in .env → switch to REVIEW mode.
New .env var: GATE0_LIVE=false.

### P3-3: Granular Festival Multipliers
**Agent:** detection_agent
**Files:** `app/detection/context/indian_adjuster.py`
**Status:** TODO

Replace blanket ×0.70 with 3-branch logic:
- Night + new VPA + high counterparty_novelty → factor=1.0 (no reduction)
- Daytime + known counterparty → factor=0.70
- All other festival → factor=0.85

Test: `test_digital_arrest` must still score ≥0.80 even during festival. Write this test.

### P3-4: Shell Company Detection
**Agent:** detection_agent
**Files:** `app/detection/tier2/legitimacy_filter.py`
**Status:** TODO

Before granting merchant/retailer exemption, check p2p_ratio. If >0.70 and no terminal
activity: exemption denied. Also add `shell_company_risk` boolean feature for XGBoost.

### P3-5: Micro Test Payment Flag
**Agent:** detection_agent
**Files:** `app/detection/tier3/feature_builder.py`, migration or DB index
**Status:** TODO

Parameterized SQL query at scoring time. Add DB index on `(sender_account_id, receiver_vpa, created_at)`.
Measure latency. If P50 added > 5ms → cache in Redis with 1h TTL.

### P3-6: Benford's Law Supporting Feature
**Agent:** detection_agent
**Files:** `app/detection/tier3/feature_builder.py`, new `app/utils/stats.py`
**Status:** TODO

New `utils/stats.py` with `compute_benford_chi_square(amounts: list) -> float`.
WEAK signal — regularization prevents dominance in XGBoost.
100-row LIMIT on the amounts query to cap latency.

### P3-7: New Archetypes (Hawala, Crypto On-Ramp, Benami)
**Agent:** ml_agent
**Files:** `ml/train.py`
**Status:** TODO

Add three generators: `generate_hawala_adjacent()`, `generate_crypto_onramp()`, `generate_benami_property()`.
Add test cases for all three in `tests/test_integration/test_fraud_scenarios.py`.

### P3-8: Multi-Hop Layering Detection
**Agent:** ml_agent
**Files:** `app/detection/tier3/feature_builder.py`
**Status:** TODO
**Requires:** P2-8 (features in Redis)

Feature assembly reads hop_count_1h..7d from feat:{account}. XGBoost learns weights in retraining.

### P3-9: Fan-In Sender Z-Score
**Agent:** detection_agent
**Files:** `app/detection/tier2/bipartite_gate.py`, `app/detection/tier2/rapid_relay_gate.py`, new migration `006_account_profile_stats.py`
**Status:** TODO

New table `account_profile_stats` — computed nightly.
Z-score > 3.0 = anomalous regardless of absolute count.
Hard floor thresholds remain: ≥4 for Gate 0, ≥7 for Gate 3 (bipartite).

### P3-10: Staleness Penalty Multiplier
**Agent:** detection_agent
**Files:** `app/detection/context/indian_adjuster.py`
**Status:** TODO
**Requires:** P2-7 (graph_staleness_hours feature)

After Indian Context Adjustments:
```python
if features['graph_staleness_hours'] > 20:
    if features['burst_score'] > 0.5 or txn_amount > 50000:
        score = min(score * 1.4, 1.0)
```
Log `staleness_penalty_applied` in fraud_scores.

---

## PHASE 4 — MODEL TRAINING

**ALL must be true before starting:**
- [ ] P0, P1, P2, P3 complete
- [ ] LEIDEN_DEPLOYED=true in Redis
- [ ] All new features computing correctly in feat:{account}
- [ ] IEEE-CIS + ADBench data downloaded
- [ ] feature_registry.py created (ml_agent owns this)

### P4-1: Expand Training Data
**Agent:** ml_agent
**Files:** new `ml/ieee_cis_bridge.py`, new `ml/adbench_bridge.py`, new `ml/archetype_blender.py`
**Status:** TODO

Target: ~400K+ rows (from 310K). Recalculate scale_pos_weight on combined dataset.
Print and update in CLAUDE.md: `scale_pos_weight = clean_count / fraud_count`.

### P4-2: Retrain XGBoost with All New Features
**Agent:** ml_agent
**Files:** `ml/train.py`, new `ml/feature_registry.py`
**Status:** TODO
**Critical:** `feature_registry.py` is the SINGLE source of truth for feature names/order.
Both `train.py` and `detection/tier3/feature_builder.py` MUST import from here.

Hard negative mining:
- 3 hard negatives per 1 easy negative per 1 fraud sample
- Hard negatives: clean nodes with similar community_id and degree_centrality to fraud nodes

### P4-3: Train HGT
**Agent:** ml_agent
**Files:** new `ml/train_hgt.py`
**Status:** TODO
**Requires:** P2-2 (Device + VPA nodes in Neo4j)

2-layer HGT, 128 hidden channels.
pos_weight = clean_count / fraud_count (same ratio as scale_pos_weight).
Save as `models/hgt_{timestamp}.pt` + SHA-256 hash.

### P4-4: Platt Scaling Calibration
**Agent:** ml_agent
**Files:** `ml/train.py`, `app/detection/tier3/ensemble.py`
**Status:** TODO
**Cascade effects:**
- Old thresholds (0.38/0.62/0.83) are for raw scores. Calibrated scores need NEW thresholds.
- SHAP runs on base estimator, NOT CalibratedClassifierCV. Keep separate reference.
- Indian Context multipliers can push calibrated score >1.0. `min(score, 1.0)` cap already exists in `indian_adjuster.py`. Verify it's still applied.
- River FTRL unaffected (runs on its own feature space, not XGBoost internals). Verify.
- model_audit must log calibration method + parameters + date.

Store both: `models/xgb_calibrated_{ts}.pkl` and `models/xgb_base_{ts}.pkl`.

### P4-5: Derive New Thresholds
**Agent:** ml_agent
**Files:** `app/detection/scoring/thresholds.py`, `app/core/config.py`, CLAUDE.md
**Status:** TODO

On held-out TEST set (NOT validation — validation used for calibration):
```python
precision, recall, thresholds = precision_recall_curve(y_test, ensemble_scores)
# LOG threshold: recall = 0.95
# REVIEW threshold: recall = 0.80, precision ≥ 0.60
# HIGH_RISK threshold: precision = 0.90
```
Print all three. Update config.py defaults. Update CLAUDE.md.
Run all 8 fraud scenario tests with new thresholds — all must pass.

### P4-6: Train XGBOD (Second Novelty Layer)
**Agent:** ml_agent
**Files:** new `ml/train_xgbod.py`, new migration `007_novelty_source.py`
**Status:** TODO

XGBOD result NEVER enters fraud_score — invariant is absolute.
New `source` column in novelty_queue: 'isolation_forest' | 'xgbod'.

### P4-7: PSI Drift Monitoring Baseline
**Agent:** ml_agent / infra_agent
**Files:** new `app/compliance/psi_monitor.py`, `celeryconfig.py`
**Status:** TODO

Capture percentile snapshot of top 10 features after training.
Store as `models/feature_baselines_{timestamp}.json`.
Weekly Celery Beat task: compute PSI → alert if any feature PSI > PSI_ALERT_THRESHOLD (default 0.2).
New .env var: PSI_ALERT_THRESHOLD=0.2.

---

## PHASE 5 — COMPLIANCE AND REGULATORY

No model changes. Pure compliance additions.

### P5-1: FINnet 2.0 Gateway Stub
**Files:** new `app/integrations/finnet_client.py`, `app/evidence/str_generator.py`
**Status:** TODO

FINNET_LIVE=false → log only. When live → real API call.
Route str_generator.py output through finnet_client.py.
New .env var: FINNET_LIVE=false.

### P5-2: NPCI Pre-Settlement Stub
**Files:** new `app/integrations/npci_client.py`
**Status:** TODO

NPCI_LIVE=false → returns `{"mode": "stub", "action": "ALLOW"}`.
Never blocks in stub mode.
New .env var: NPCI_LIVE=false.

### P5-3: DPDP Act 2023 Compliance Layer
**Files:** new migration `008_dpdp_compliance.py`, new `app/api/v1/data_principal.py`
**Status:** TODO

New table `data_categories` — maps feature_name to legal_basis, retention_days.
New Celery Beat task (daily): delete graph_features_cache rows older than 30 days.
Stub endpoints: `GET /api/v1/data-principal/{account_id}`, `DELETE /api/v1/data-principal/{account_id}`.
New .env var: DPDP_LIVE=false.

### P5-4: 5-Year Retention + pgcrypto Encryption
**Files:** new migration `009_retention_encryption.py`
**Status:** TODO

INSERT-only triggers on fraud_scores and alerts.
pgcrypto encryption on account_id, payee_vpa in both tables.
New .env var: DB_ENCRYPTION_KEY.

### P5-5: CISO Notification Workflow
**Files:** `app/api/v1/feedback.py`, new `app/compliance/ciso_notifier.py`
**Status:** TODO

Triggered when `confirmed_fraud=True AND txn_amount > 1_000_000`.
New .env vars: CISO_EMAIL, SLACK_WEBHOOK_URL.

### P5-6: OFAC + UN Sanctions List
**Files:** new `app/integrations/sanctions_client.py`, new migration `010_sanctions_list.py`, new `scripts/sync_sanctions_lists.py`
**Status:** TODO

Daily Celery Beat sync at 2:30am.
Real-time feature `payee_sanctions_flag` at scoring time.
IMPORTANT: Store only as boolean flag — never log actual sanctions list data.

### P5-7: Locust Load Test
**Files:** new `tests/load/locustfile.py`, new `docs/LOAD_TEST_RESULTS.md`
**Status:** TODO

Target: 1000 TPS for 5 minutes. Record P50/P95/P99.
Must run AFTER Phase 4 (new model) is deployed.

---

## PHASE 6 — SECURITY HARDENING

### P6-1: Score Jitter + Canary Accounts
**Files:** `app/api/v1/score.py`, new `scripts/seed_canary_accounts.py`
**Status:** TODO

±0.01 uniform noise applied BEFORE threshold decision (intentional for anti-extraction).
Cap final_score to [0.0, 1.0] after jitter.
50 synthetic canary accounts. Detection logs to `suspicious_probe_log`.

### P6-2: Model Versioning Rollback API
**Files:** `app/main.py`, `app/utils/model_integrity.py`
**Status:** TODO
Already largely covered by P0-3. Ensure MODEL_VERSION env var controls active model.

---

## PHASE 7 — SANDBOX HOUSEKEEPING

### P7-1: D-03 Sandbox Density Override
**Files:** `red_team/sandbox/blue_clone.py`
**Status:** TODO

Add `density_override` field to account fixtures.

### P7-2: D-05 Known Issue Cleanup
**Files:** `.claude/CLAUDE.md`, divergence log
**Status:** TODO

Remove "Known Issue #2" from CLAUDE.md. Update D-05 to RESOLVED.

---

## NEW ENVIRONMENT VARIABLES (.env.example additions)

```bash
JWT_PRIVATE_KEY=              # RS256 private key (PEM)
JWT_PUBLIC_KEY=               # RS256 public key (PEM)
JWT_EXPIRY_SECONDS=900
PSEUDONYMIZATION_KEY=         # 32-byte hex for HMAC PII masking
DB_ENCRYPTION_KEY=            # pgcrypto column encryption
FINNET_LIVE=false
NPCI_LIVE=false
DPDP_LIVE=false
GATE0_LIVE=false
CISO_EMAIL=
SLACK_WEBHOOK_URL=
ENSEMBLE_ALPHA=0.65
MODEL_VERSION=latest
LOG_THRESHOLD=                 # Set after Phase 4
REVIEW_THRESHOLD=              # Set after Phase 4
HIGH_RISK_THRESHOLD=           # Set after Phase 4
PSI_ALERT_THRESHOLD=0.2
FTRL_CAP_PER_INVESTIGATOR=15
LEIDEN_DEPLOYED=false
```

---

## NEW DATABASE MIGRATIONS

| Migration | Change | Status |
|-----------|--------|--------|
| 003_shap_audit.py | `shap_access_log` table | TODO |
| 004_evidence_seal.py | `evidence_seal` table (INSERT-only) | TODO |
| 005_gate0_pilot.py | `gate0_pilot_log` table | TODO |
| 006_account_profile_stats.py | `account_profile_stats` table | TODO |
| 007_novelty_source.py | `source` column on `novelty_queue` | TODO |
| 008_dpdp_compliance.py | `data_categories` table | TODO |
| 009_retention_encryption.py | INSERT-only triggers + pgcrypto | TODO |
| 010_sanctions_list.py | `sanctions_list` table | TODO |

---

## NEW FILES TO CREATE

```
app/utils/model_integrity.py
app/utils/sanitize.py
app/utils/auth.py
app/utils/evidence_seal.py
app/utils/stats.py
app/detection/tier2/rapid_relay_gate.py
app/integrations/finnet_client.py
app/integrations/npci_client.py
app/integrations/sanctions_client.py
app/compliance/ciso_notifier.py
app/compliance/psi_monitor.py
app/api/v1/data_principal.py
app/graph/tasks.py
app/graph/queries/device_vpa_queries.cypher
app/graph/queries/layering_queries.cypher
celeryconfig.py
ml/feature_registry.py
ml/ieee_cis_bridge.py
ml/adbench_bridge.py
ml/archetype_blender.py
ml/train_hgt.py
ml/train_xgbod.py
scripts/sync_sanctions_lists.py
scripts/seed_canary_accounts.py
scripts/rotate_pseudonymization_key.py
tests/load/locustfile.py
docs/LOAD_TEST_RESULTS.md
```

---

## TESTING REQUIREMENTS BY PHASE

After each phase: `pytest tests/ -v` — all N tests must pass (count grows with each phase).

### New tests to add (by phase)

**Phase 3:**
- Gate 0: 3 scenarios (fires / exempted merchant / exempted salary processor)
- Gate 2 Path B: 4 scenarios (fraud / NRI / wedding / seasonal merchant)
- Festival multiplier: 4 scenarios (daytime known / night new VPA / daytime new VPA / senior night)
- Micro test payment: 2 scenarios (precursor present / absent)
- Shell company: 2 scenarios (fake merchant fires / real merchant exempt)

**Phase 4:**
- Ensemble HGT score: 0.0-1.0 for all 8 existing fraud scenarios
- Calibration: fraud mean score 0.55-0.85
- Staleness penalty: staleness >20h + burst >0.5 → score inflated
- Feature registry consistency: `FEATURE_NAMES == get_feature_names()` at all times
- Archetype regression: no archetype drops >0.05 from pre-retraining scores

---

## DEFINITION OF DONE PER PHASE

Before marking any phase complete:
- [ ] `pytest tests/ -v` all pass
- [ ] CLAUDE.md updated with phase completion
- [ ] `.env.example` updated with new vars
- [ ] Alembic migrations run on clean DB
- [ ] No hardcoded credentials (grep check: `grep -r "API_KEY\s*=" app/ ml/`)
- [ ] `/metrics` endpoint returns new counters for this phase
- [ ] This file updated with completion timestamp
- [ ] Agent docs updated with schema/API changes

---

*Generated: 2026-05-28 | Owner: BLING Blue Team*
*Do not distribute outside the BLING Blue Team*
