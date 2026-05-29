# BLING Blue Team — Architecture
# Maintained by Claude. Updated when system design changes.
# Last updated: 2026-05-30

## System Overview

Blue Team is a **post-transaction forensic fraud detection system** for Union Bank of India. Money has already moved. The system scores every settled transaction for fraud likelihood, reconstructs the full fund trail when suspicious, and packages a complete SHAP-explained evidence bundle for a human investigator. Humans decide at every point — no automated blocking.

---

## Component Diagram

```
Core Banking (Finacle)
        ↓ CDC
     Kafka
        ↓
  Graph Engine (TEAMMATE) → builds Neo4j live graph
        ↓ POST /api/v1/score (each settled txn)
┌───────────────────────────────────────────────────────────────────┐
│                     BLUE TEAM (this codebase)                      │
│                                                                    │
│  Tier 1: Fast Heuristic Rules              5ms                    │
│  Redis velocity counters + 6 rule signals                          │
│  FAST_CLEAN (78%) | UNCERTAIN (14%) | SUSPICIOUS (8%)             │
│           ↓ UNCERTAIN + SUSPICIOUS (22%)                           │
│  Tier 2: Hard Graph Gates                  +15ms                  │
│  cycle | sink D-01 | bipartite | cash_mule | merchant_terminal    │
│  + Gate 0 rapid relay (LOG-ONLY until GATE0_LIVE=true)            │
│  gate fires → legitimacy filters → score=1.0, action=REVIEW       │
│           ↓ SUSPICIOUS that cleared gates (~5%)                   │
│  Tier 3: Committee Engine                  +30ms                  │
│  ┌─ Committee (shadow mode) ──────────────────────────────────┐   │
│  │  Scorer A: XGBoost GBM (113 features + UPI session)        │   │
│  │  Scorer B: PC-GNN + Hypergraph embedding (32-dim)          │   │
│  │  Scorer C: FAISS prototype vault (k-NN archetype match)    │   │
│  │  Scorer D: Behavioral set RF (7 features, 90-day history)  │   │
│  │  Scorer F: MiniLM remark screener (Hindi + English)        │   │
│  │  → shadow_score_committee (silent, zero effect on score)   │   │
│  └────────────────────────────────────────────────────────────┘   │
│  Live XGBoost score → Indian context adjustment → threshold map   │
│  PASS | LOG | REVIEW | HIGH_RISK                                   │
│           ↓ REVIEW or HIGH_RISK                                    │
│  Alert created + Celery async trail reconstruction                  │
│           ↓ PASS stream only (background, silent)                  │
│  ┌── Discovery Ensemble (PASS stream only) ─────────────────────┐ │
│  │  IsoForest + ECOD · NEVER affects fraud_score or alerts      │ │
│  │  Novel → novelty_queue (developer review)                    │ │
│  │         + Red Team escalation (10+ same pattern in 7 days)   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│           ↓                                                        │
│  GET /api/v1/alerts/{id}        ← Investigator Dashboard           │
│  POST /api/v1/feedback          ← Investigator decision            │
│           ↓ confirmed fraud                ↓ false positive        │
│  blockchain_client.seal()      curated_dataset_queue (label=0)    │
│  red_team_client.send_dna()    + reviewed_novelty_registry         │
│  prototype_injection_candidates (PENDING_REVIEW, developer queue)  │
└───────────────────────────────────────────────────────────────────┘
        ↓                          ↓                    ↓
Investigator Dashboard      Private Blockchain     Red Team Sandbox
    (TEAMMATE)                 (TEAMMATE)            (TEAMMATE)
```

---

## GNN Embedding Pipeline

```
Neo4j graph (read-only)
    ↓ nightly batch (3am, Celery Beat)
app/graph/gnn_embedder.py
    FraudPCGNN (2× PCGNNLayer + HypergraphConv)
    PCGNNLayer: camouflage-resistant attention
      message() = sigmoid(att_src(x_j) + att_dst(x_i)) * lin(x_j)
      suppresses fraudsters hiding among legit neighbors
    HypergraphConv: Leiden community = hyperedge
      10-account mule farm in same community = 1 group signal
    ↓
Redis gnn_emb:{account}    32-dim float list, TTL 25h
    + refresh every 5min for recently-active accounts
    ↓
Scorer B: [32-dim GNN emb || 8 structural features] → MLPClassifier

Fallback chain: gnn_emb:{account} → emb:{account} (Node2Vec) → missing_flag=True

P2-2 stub: build_hetero_data_stub() ready for Device+VPA nodes
           when Graph Engine teammate delivers heterogeneous schema
```

---

## Request Flow

1. Graph Engine POSTs settled transaction → `POST /api/v1/score`
2. `score.py` validates Pydantic schema, checks X-API-Key or RS256 JWT
3. `audit_logger` inserts pending score event to `model_audit` (immutable)
4. `detection/pipeline.py` runs Tier 1 → Tier 2 → Tier 3 sequentially
5. **Tier 1** (`tier1/heuristics.py`): Redis ZSET velocity counters + 6 rules → `FAST_CLEAN`, `UNCERTAIN`, or `SUSPICIOUS`
6. `FAST_CLEAN` → return `{action: PASS, score: 0.05}`
7. **Tier 2** (`tier2/gates.py`): All gates in order. Fire → legitimacy filters. If unexplained → `{action: REVIEW, score: 1.0}`
8. `UNCERTAIN` and all gates clear → `{action: LOG, score: 0.10}`
9. **Tier 3** (`tier3/committee_scorer.py`): Build ~107 features, run 5 scorers in shadow, live XGBoost inference, Indian context, threshold + jitter
10. `REVIEW` or `HIGH_RISK` → create Alert, enqueue Celery trail reconstruction
11. `PASS` action → discovery ensemble runs in background thread (non-blocking)
12. Audit INSERT — atomic with response (failure = 500)
13. Response padded to 55ms constant time (timing oracle prevention)
14. Celery: fund trail → ghost node reconnection → STR draft (156 fields) → Alert updated

---

## Feedback Flow (post-FTRL)

FTRL (River online learning) was removed in Phase 3. Feedback routes to structured queues:

| Investigator action | Route | Effect |
|---------------------|-------|--------|
| False positive | `curated_dataset_queue` (label=0) + `reviewed_novelty_registry` | Batch retraining data |
| Confirmed fraud | `prototype_injection_candidates` (PENDING_REVIEW) | Developer reviews → inject into Scorer C FAISS vault |
| Both paths | Blockchain seal + Red Team DNA | Always |

Developer reviews `GET /api/v1/developer-queue/prototype-candidates` (INTERNAL_KEY only).

---

## Module Map

| Path | Responsibility | Must NOT |
|------|---------------|---------|
| `app/api/v1/score.py` | HTTP routing, Pydantic validation, audit trigger | Business logic |
| `app/detection/pipeline.py` | Orchestrate T1→T2→T3, assemble context_features | Contain rule or ML logic. Max 2 total changes (done). |
| `app/detection/tier1/heuristics.py` | Redis ZSET velocity + 6 rule flags | Call Neo4j or PostgreSQL |
| `app/detection/tier2/gates.py` | Route to gates, return first-fired result | Make scoring decisions |
| `app/detection/tier2/legitimacy_filter.py` | Explain legitimate cycle patterns | Skip any of 5 filters or reorder |
| `app/detection/tier2/rapid_relay_gate.py` | Gate 0 — LOG-ONLY until GATE0_LIVE=true | Use amounts[-1]; always total_outflow/total_inflow |
| `app/detection/tier3/committee_scorer.py` | Shadow/live mode, 5-scorer orchestration | Raise on scorer failure; shadow failures must be silent |
| `app/detection/tier3/scorer_a.py` | XGBoost GBM (Scorer A) | Pass CalibratedClassifierCV to SHAP |
| `app/detection/tier3/scorer_b.py` | PC-GNN embedding + structural (Scorer B) | Fetch extra Redis keys (use cached feat: hash) |
| `app/detection/tier3/scorer_c.py` | FAISS prototype vault wrapper (Scorer C) | Return raw feature vectors to any caller |
| `app/detection/tier3/scorer_d.py` | Behavioral set RF / Mamba stub (Scorer D) | Raise on <5 txn history — return unavailable |
| `app/detection/tier3/scorer_f.py` | MiniLM remark screener (Scorer F) | Block >5ms — MiniLM only, no large LLM |
| `app/detection/tier3/shadow_writer.py` | Write shadow_score_committee | Ever raise to caller — all failures must be caught |
| `app/detection/tier3/committee_auditor.py` | INSERT model_audit for committee events | Silently pass on DB failure — must raise AuditWriteError |
| `app/detection/tier3/feature_builder.py` | Assemble ~107 features from Redis + PostgreSQL | Hardcode feature list (import from feature_registry.py) |
| `app/detection/tier3/ensemble.py` | Legacy live XGBoost (SHAP source) | Add functionality — marked for removal post go-live |
| `app/detection/tier3/prototype_vault.py` | FAISS k-NN archetype matching | Return raw vectors; expose inject endpoint without INTERNAL_KEY |
| `app/detection/context/indian_adjuster.py` | Multiply raw score by context factors | Gate logic; always cap result to [0.0, 1.0] |
| `app/detection/novelty/discovery_ensemble.py` | IsoForest + ECOD scoring (PASS stream) | Affect fraud_score or create alerts |
| `app/detection/novelty/discovery_router.py` | Write novelty_queue; escalate at 10+ occurrences | Create investigator alerts; use non-atomic incr+expire |
| `app/detection/feedback/feedback_router.py` | Route feedback to curated_dataset_queue or prototype_injection_candidates | Import River or FTRL anything |
| `app/graph/gnn_embedder.py` | PC-GNN + Hypergraph training and inference | Import torch at module level (lazy-load only); write to Neo4j |
| `app/graph/neo4j_client.py` | Connection pool + parameterized Cypher runner | Write to Neo4j; f-strings in Cypher |
| `app/graph/precompute/nightly_batch.py` | Compute 35+ graph features → Redis nightly + GNN embeddings | Run at scoring time |
| `app/evidence/trail_builder.py` | Async Celery fund trail reconstruction | Block the API thread — always .delay() |
| `app/utils/audit_logger.py` | INSERT to model_audit only | UPDATE or DELETE audit records |
| `app/integrations/` | POST to teammate APIs (stubs when URL not set) | Own business logic |
| `ml/feature_registry.py` | Single source of truth for feature names/order | Never duplicate feature lists elsewhere |

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Hard gates before ML | Dimensional veto | Weighted ensemble gives cycle gate 0.40 max; REVIEW threshold is 0.62. Cycle would be missed without hard veto. |
| Pre-computed graph features | Nightly batch → Redis | Full Cypher traversal times out at 3+ hops on 10M+ transactions. Pre-compute nightly, delta-check only in real time. |
| Async trail reconstruction | Celery + Redis queue | Trail traverses 10 hops — takes 5-15 minutes. Blocking /score causes retry floods. |
| Committee in shadow mode | 5-scorer parallel, zero live impact | Allows safe accumulation of calibration data (50K shadow rows) before switching live. |
| PC-GNN over GraphSAGE | Pick-and-Choose attention | Fraud accounts hide among legit neighbors. PC-GNN's attention = sigmoid(att_src + att_dst) suppresses camouflage neighbors explicitly. GraphSAGE averages them equally. |
| Hypergraph over pairwise edges | Leiden community = hyperedge | A 10-account mule farm in same community = 1 group signal via hyperedge, not 10 noisy pairwise edges. HypergraphConv is more expressive for mule network structure. |
| FTRL removed | Structured feedback queues | FTRL created untracked model drift and had no per-investigator audit trail beyond a Redis counter. Feedback queues give full observability + batch retraining control. |
| XGBoost over DNN | XGBoost 2.x + Platt calibration | SHAP explainability mandatory for RBI STR filings. DNN is a black box. XGBoost handles NaN natively, fast inference (2-5ms). |
| SHAP on base (uncalibrated) | Keep two model references | CalibratedClassifierCV breaks TreeExplainer. scorer_a_base.joblib = SHAP source; scorer_a_v1.joblib = scoring. |
| Audit table immutable | DB trigger blocks UPDATE/DELETE | RBI PMLA Section 12 requires tamper-proof audit trails. Cannot be bypassed by application code. |
| Discovery PASS-only | Gate in score.py before executor launch | Prevents anomaly detector from creating noise on REVIEW/HIGH_RISK transactions that already have a fraud explanation. |
| Constant 55ms response | asyncio.sleep padding | Timing oracle: response time reveals which tier exited. Padding makes Tier 2 (20ms) and Tier 3 (47ms) indistinguishable. |
| Score jitter ±0.01 | Applied before threshold decision | Anti-model-extraction: makes repeated probing return different values. Action and score use same jitter draw for consistency. |

---

## Boundaries (Import Rules)

- Routes call pipeline. Pipeline calls tier modules. Tier modules call DB clients.
- Neo4j is read-only. No writes from Blue Team ever.
- Integrations are thin HTTP clients — no business logic in `app/integrations/`.
- `audit_logger` is insert-only. No updates anywhere.
- `feature_registry.py` is the only place feature names are defined. Both `train.py` and `feature_builder.py` import from it.
- `torch` imported lazily inside functions in `gnn_embedder.py` — API starts cleanly without it.
- Shadow failures never propagate. Audit failures always propagate.

---

## What NOT to Build

- **Graph Engine**: teammate owns Neo4j population, CDC pipeline, Kafka consumer
- **Investigator Dashboard**: teammate owns frontend UI
- **Blockchain layer**: teammate owns sealing; Blue Team POSTs to their API
- **Red Team sandbox**: teammate owns attack simulation; Blue Team sends fraud DNA only
- **Real-time blocking**: money has already moved — forensic post-settlement only
- **HGT model**: blocked on teammate P2-2 (Device+VPA nodes). Stub ready in `build_hetero_data_stub()`.

---

## Planned / In Progress

| Item | Status | Blocker |
|------|--------|---------|
| Committee meta-learner | Waiting for 10K shadow rows | Real traffic |
| Committee go-live | Waiting for 50K shadow rows + threshold derivation | Real traffic |
| HGT (Heterogeneous GNN) | Stub ready | Teammate P2-2 |
| Mamba sequence scorer | Stub in `ml/train_mamba.py` | Sequence dataset + `pip install mamba-ssm` |
| Gate 0 live promotion | LOG-ONLY pilot | 2-week pilot review |
| FINnet 2.0 submission | Stub wired | `FINNET_LIVE=true` + `FINNET_API_KEY` |
| NPCI pre-settlement | Stub wired | `NPCI_LIVE=true` + `NPCI_API_KEY` |
| OFAC/UN/MHA sanctions | URL fetch wired | Live URLs + XML/CSV parser |
