# BLING Blue Team — Gotchas
# Maintained by Claude. Read first when something unexpected happens.
# Pre-populated from spec review. Add entries as project is built.

## Entry Format
**[Short title]**
- Symptom: [what you observe]
- Cause: [why it happens]
- Fix: [exact solution]
- File: `[affected file or module]`

---

## Detection Architecture

**Tier 1 must produce exactly THREE outputs — never binary**
- Symptom: First-time payees at normal hours pass straight through without graph analysis
- Cause: Tier 1 implemented as pass/fail — UNCERTAIN bucket was missing
- Fix: `tier1_classify()` returns `FAST_CLEAN`, `UNCERTAIN`, or `SUSPICIOUS`. UNCERTAIN (first-time payees with no hard flags but not clearly clean) MUST proceed to Tier 2.
- File: `app/detection/tier1/heuristics.py`

**FAST_CLEAN does NOT mean ignored**
- Symptom: Slow mules (45-day warmup accounts) never get caught
- Cause: FAST_CLEAN was treated as "done, skip forever"
- Fix: FAST_CLEAN exits the real-time pipeline only. Nightly APScheduler batch still scores ALL FAST_CLEAN accounts for slow behavioral drift. Slow mule detection is acceptable overnight — this is forensic, not real-time.
- File: `app/graph/precompute/nightly_batch.py`

**Cycle gate must NEVER suppress silently**
- Symptom: A round-trip cycle is detected but no alert is raised and no log explains why
- Cause: Legitimacy filter code returned early without logging the named reason
- Fix: After cycle gate fires, run ALL 5 legitimacy filters in exact order. Each filter either returns `{'explained': True, 'reason': 'named_reason'}` and stops, OR passes to next filter. If none explain it: ESCALATE with score=1.0. The named reason must be logged. No silent passes.
- File: `app/detection/tier2/legitimacy_filter.py`

**Legitimacy filter order is non-negotiable**
- Symptom: Salary advance returns being flagged as fraud
- Cause: Filters applied in wrong order — amount-reduction filter (Filter 5) checked before salary advance filter (Filter 3), incorrectly explaining a cycle that should have escalated
- Fix: Always run in this exact order: (1) internal/treasury, (2) KYC relationship, (3) salary advance, (4) all-merchant, (5) amount <70%. Never reorder.
- File: `app/detection/tier2/legitimacy_filter.py`

---

## Graph / Neo4j

**Full Neo4j traversal times out at 3+ hops**
- Symptom: Tier 2 cycle gate queries time out under load; p99 spikes to seconds
- Cause: Cypher `MATCH path = (a)-[:SENT*2..8]->(a)` on live Neo4j with 10M+ transactions traverses the full graph
- Fix: Nightly batch pre-computes cycle-prone subgraphs and account attributes. Real-time Cypher only checks delta (new edges since last nightly run). Never run full traversal at query time.
- File: `app/graph/queries/cycle_queries.py`, `app/graph/precompute/nightly_batch.py`

**Neo4j connection pool exhaustion under load**
- Symptom: `ServiceUnavailable: Failed to establish connection` errors under concurrent load
- Cause: Neo4j Community Edition max connections 400; API + Celery workers share pool without limit
- Fix: Set `max_connection_pool_size` explicitly in `neo4j_client.py` driver config. Default leaves it unbounded on client side.
- File: `app/graph/neo4j_client.py`

**All Neo4j Cypher must use parameterized queries**
- Symptom: Cypher injection via account_id parameter
- Cause: f-string used to build Cypher query
- Fix: Always use `session.run("MATCH (a:Account {id: $account_id}) RETURN a", account_id=account_id)`. Never f-strings in Cypher.
- File: All files in `app/graph/queries/`

---

## Cash Mule Gate (ATM / Ghost Node)

**ATM transactions have no UPI device fingerprint**
- Symptom: Ghost node gate tries to match device IDs across ATM transactions and always misses
- Cause: ATM transactions use card+PIN authentication — no UPI device ID is attached
- Fix: Use Cash Mule Sink Detector (receive→ATM_withdraw→digital_silence) instead. All data is in PostgreSQL. No device ID matching across ATMs ever.
- File: `app/detection/tier2/cash_mule_sink_gate.py`

---

## ML / XGBoost

**scale_pos_weight must be 99 — never omit**
- Symptom: Model predicts "not fraud" for everything, achieves 99% accuracy on validation set, completely useless
- Cause: Training data is ~1% fraud. Without class weight correction, model learns to always predict majority class.
- Fix: `XGBClassifier(scale_pos_weight=99, ...)` always. Never train without this.
- File: `ml/train.py`, `app/detection/tier3/ensemble.py`

**Use eval_metric='aucpr' not 'auc'**
- Symptom: Model appears to perform well (ROC-AUC 0.95) but misses most fraud at operational thresholds
- Cause: ROC-AUC is misleading for heavily imbalanced data — it inflates performance
- Fix: Always use PR-AUC (precision-recall). `XGBClassifier(eval_metric='aucpr', ...)`.
- File: `ml/train.py`, `ml/evaluate.py`

**Online learning must use warm start — never full retrain**
- Symptom: Model freezes for 30-60 seconds after each investigator feedback event
- Cause: Full retraining of XGBoost on all historical data triggered on every feedback
- Fix: `model.fit(new_features, [new_label], xgb_model=model)` — warm start only. River FTRL handles weight adaptation incrementally.
- File: `app/detection/tier3/online_learning.py`

**feature_builder must handle missing account history gracefully**
- Symptom: New accounts (<30 days) cause feature builder to crash or return NaN
- Cause: Features like `txn_count_90d`, `avg_txn_amount_30d` have no data for new accounts
- Fix: XGBoost handles NaN natively. Return `float('nan')` for unavailable features instead of imputing or crashing.
- File: `app/detection/tier3/feature_builder.py`

---

## Audit / Compliance

**model_audit trigger silently swallows UPDATEs**
- Symptom: UPDATE on model_audit returns success but changes nothing
- Cause: `DO INSTEAD NOTHING` rule — designed to prevent tampering, side effect is silent no-op
- Fix: model_audit is INSERT ONLY. Never attempt UPDATE or DELETE. Design audit events as new INSERT rows, not modifications.
- File: `app/utils/audit_logger.py`

**Audit INSERT must be atomic with score response**
- Symptom: Scoring decision returned to caller but not recorded in audit trail
- Cause: Audit write happened after response was returned — network failure left a gap
- Fix: Audit INSERT must complete before returning the API response. If audit write fails, the entire /score request fails with 500. No silent audit skips.
- File: `app/api/v1/score.py`, `app/utils/audit_logger.py`

---

## PII / Logging

**Never log real account IDs**
- Symptom: Account numbers appear in structlog output — PII leak
- Cause: `account_id` logged directly
- Fix: Always pseudonymize: `sha256(settings.SALT + account_id)[:12]`. Use this in every log statement.
- File: All files that call `structlog.get_logger()`

---

## Evidence / Fund Trail

**Celery trail reconstruction — never synchronous**
- Symptom: POST /score takes 8 minutes to respond under load
- Cause: Trail reconstruction called synchronously inside the API handler
- Fix: Always enqueue via Celery: `trail_builder.reconstruct_fund_trail.delay(txn_id)`. Return `alert_id` immediately. Investigator polls GET /alerts/{id} for trail_status=COMPLETE.
- File: `app/evidence/trail_builder.py`, `app/api/v1/score.py`

---

## graph_features_cache Staleness

**Stale feature cache after 24+ hour gap**
- Symptom: Tier 3 features reflect 2-day-old graph state; new fraud accounts have zero contamination score
- Cause: APScheduler nightly batch didn't run (container restart, etc.) — `computed_at` is stale
- Fix: `feature_builder.py` checks `computed_at`. If stale >26h, fall back to real-time computation for highest-stakes features (pagerank_fraud_seeded, community_fraud_ratio). Log a warning.
- File: `app/detection/tier3/feature_builder.py`, `app/graph/precompute/nightly_batch.py`

---

## Isolation Forest Novelty Detection

**NEVER put IF score into XGBoost features or fraud_score**
- Symptom: False positives flood investigator queue — Jan Dhan first-time payments, pension arrears, post-vacation transactions all flagged REVIEW
- Cause: Isolation Forest flags "unusual" transactions. Unusual ≠ fraud. Plugging IF score into XGBoost means every legitimate edge case raises the fraud score.
- Fix: Separation is absolute. `novelty_router.py` writes to `novelty_queue` only. `fraud_score` in API response is always XGBoost output. Never IF output.
- File: `app/detection/novelty/novelty_router.py`, `app/api/v1/score.py`

**Novelty threshold limits — do not cross**
- Symptom (too permissive at -0.15): Developer queue floods 50+ items/hour — signal lost in noise
- Symptom (too strict at -0.30): Novel evasion patterns undetected for weeks — fingerprint count never reaches 10
- Fix: `NOVELTY_THRESHOLD = -0.20` is calibrated. If `novelty_flags_total / scoring_requests > 1%`, tighten to -0.25. Never go below -0.15.
- File: `app/detection/novelty/isolation_forest.py`

**Model not found → degraded mode, NOT a crash**
- Symptom: Startup log shows `"novelty_detection_disabled"` — all fraud detection works normally
- Cause: `ml/models/isolation_forest_v1.joblib` missing — model not trained yet
- Fix: Run `python ml/train_isolation_forest.py`. Expected on first setup. API does not fail — novelty block silently skips.
- File: `ml/train_isolation_forest.py`, `app/detection/novelty/isolation_forest.py`

**novelty_router must NEVER raise — all exceptions caught and logged**
- Symptom: POST /score returns 500 for a Redis/DB outage that should only affect novelty logging
- Cause: Exception from `route_novelty()` propagated to scoring response
- Fix: Outer `try/except Exception` in `route_novelty()` catches everything. The novelty block in `score.py` also has its own guard. Redis or DB failure here never fails the scoring response.
- File: `app/detection/novelty/novelty_router.py`, `app/api/v1/score.py`

**novelty_queue is NOT immutable — this is intentional**
- Symptom: Developer hesitates to UPDATE status, expecting silent no-op like model_audit
- Cause: Two tables look similar but serve different purposes
- Fix: `model_audit` = RBI PMLA immutable audit trail (DB triggers block UPDATE/DELETE). `novelty_queue` = developer working queue — updating status and adding notes is correct and expected.
- File: `app/api/v1/novelty.py`

**17 features only for IF — never use all 59**
- Symptom: IF flags too many legit transactions (night workers, large one-off payments)
- Cause: Using time/amount features (hour_of_day, txn_amount) that vary legitimately
- Fix: IF uses only the 17 structural graph features in `ISOLATION_FOREST_FEATURES`. Time and amount features explicitly excluded from both training and inference.
- File: `app/detection/novelty/isolation_forest.py`, `ml/train_isolation_forest.py`
