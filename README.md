<div align="center">

# BLING Blue Team
### Forensic Fraud Detection Engine — Union Bank of India

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Redis](https://img.shields.io/badge/Redis-7.x-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Tests](https://img.shields.io/badge/Tests-102%2B%20passing-brightgreen?style=flat-square)](tests/)
[![Migrations](https://img.shields.io/badge/Migrations-007%20%28head%29-blue?style=flat-square)](alembic/versions/)

</div>

---

## What This Is

Post-transaction forensic fraud detection engine for Indian UPI payments. Money has already moved. This system scores every settled transaction, reconstructs fund trails when suspicious, and delivers SHAP-explained evidence bundles with draft STR reports (156 FINnet fields) to human investigators.

> **Core principle:** Investigators stay in control at every decision point. No automated blocking. Full explainability on every alert.

---

## Full System Architecture

```mermaid
flowchart TB
    subgraph INPUTS["Input Layer"]
        GE[Graph Engine\ntransaction POST]
        INV[Investigator\nDashboard]
    end

    subgraph TIER1["Tier 1 — Fast Rules ~5ms"]
        T1{Classify}
        FC[FAST_CLEAN\n78% of traffic]
        UC[UNCERTAIN]
        SU[SUSPICIOUS]
    end

    subgraph TIER2["Tier 2 — Hard Graph Gates ~15ms"]
        G1[Cycle Gate\nA→B→C→A]
        G2[Sink Gate D-01\ndormant reactivation]
        G3[Bipartite Gate\n7+ senders → collector]
        G4[Cash Mule Sink\nATM withdrawal pattern]
        G5[Merchant Terminal\nPOS round-trip]
        G0[Gate 0 Rapid Relay\nLOG-ONLY pilot]
        LF[5 Legitimacy\nFilters]
    end

    subgraph TIER3["Tier 3 — Committee Engine ~30ms"]
        XGB[Calibrated XGBoost\nlive decisions]
        COMM[5-Scorer Committee\nshadow mode]
        ICA[Indian Context\nAdjuster]
        THR[Threshold + Jitter\n0.38 / 0.62 / 0.83]
    end

    subgraph OUTPUTS["Output Layer"]
        PASS[PASS\nscore lt 0.38]
        LOG[LOG\n0.38–0.61]
        REV[REVIEW\n0.62–0.82]
        HR[HIGH_RISK\nge 0.83]
        ALERT[Alert Created]
        TRAIL[Celery: Fund Trail\nasync 5–15 min]
        SHAP[SHAP Explainer\nasync Celery]
    end

    subgraph FEEDBACK["Investigator Feedback"]
        FP[False Positive\ncurated_dataset_queue]
        CF[Confirmed Fraud\nprototype_injection_candidates]
        BC[Blockchain Seal]
        RT[Red Team DNA]
    end

    GE --> T1
    T1 -->|78%| FC --> PASS
    T1 --> UC & SU
    UC & SU --> G1 & G2 & G3 & G4 & G5 & G0
    G1 & G2 & G3 & G4 & G5 -->|gate fires| LF
    LF -->|not explained| REV --> ALERT
    UC -->|all gates clear| LOG
    SU -->|all gates clear| XGB
    XGB --> COMM
    XGB --> ICA --> THR
    THR --> PASS & LOG & REV & HR
    HR --> ALERT
    ALERT --> TRAIL & SHAP
    INV -->|GET /alerts| TRAIL
    INV -->|POST /feedback| FP & CF
    CF --> BC & RT

    style HR fill:#d93025,color:#fff
    style REV fill:#e8710a,color:#fff
    style PASS fill:#1e8e3e,color:#fff
    style GE fill:#1a73e8,color:#fff
    style COMM fill:#9c27b0,color:#fff
```

---

## 3-Tier Detection Pipeline

```mermaid
flowchart LR
    A[POST /api/v1/score\nGraph Engine] --> B

    subgraph T1["Tier 1 — 5ms"]
        B{6 hard rules\nvelocity · VPA age\namount · time · KYC\nchannel}
    end

    B -->|FAST_CLEAN\n78%| P[PASS 0.05]
    B -->|UNCERTAIN or\nSUSPICIOUS| T2

    subgraph T2["Tier 2 — 15ms"]
        C[Cycle\nGate]
        D[Sink\nD-01]
        E[Bipartite\nGate]
        F[Cash Mule\nSink]
        G[Merchant\nTerminal]
    end

    T2 -->|gate fires| LF[Legitimacy\nFilters × 5]
    LF -->|explained| LOG1[LOG or lower]
    LF -->|not explained| R1[REVIEW 1.0]

    T2 -->|UNCERTAIN\nall clear| LOG2[LOG 0.10]
    T2 -->|SUSPICIOUS\nall clear| T3

    subgraph T3["Tier 3 — 30ms"]
        H[Build ~107 features\nRedis + PostgreSQL]
        I[XGBoost calibrated\n+ committee shadow]
        J[Indian context\nadjustment]
        K{Threshold\n+ jitter}
    end

    H --> I --> J --> K
    K -->|lt 0.38| PASS2[PASS]
    K -->|0.38| LOG3[LOG]
    K -->|0.62| REV2[REVIEW]
    K -->|0.83| HR[HIGH_RISK]

    style HR fill:#d93025,color:#fff
    style R1 fill:#e8710a,color:#fff
    style P fill:#1e8e3e,color:#fff
    style PASS2 fill:#1e8e3e,color:#fff
```

---

## Tier 3 — 5-Scorer Committee Engine

Committee runs in **shadow mode** alongside the existing XGBoost. Every SUSPICIOUS-path transaction writes to `shadow_score_committee`. Live decisions unchanged until ≥50K shadow rows and meta-learner trained.

```mermaid
flowchart TB
    F[Feature vector\n~107 features] --> A & B & C & D & E

    subgraph SCORERS["5 Scorers — parallel via run_in_executor"]
        A["Scorer A — GBM\nCalibrated XGBoost\n+ UPI session features\n(existing model fallback)"]
        B["Scorer B — Graph\nNode2Vec 32-dim\n+ 8 structural features\nfrom feat:{account}"]
        C["Scorer C — Prototype\nFAISS k-NN vault\nknown archetype proximity\nweighted fraud ratio"]
        D["Scorer D — Sequence\nSet-based RF\n(Mamba stub until\nsequences available)"]
        E["Scorer F — Remark\nMultilingual phrase screener\n7 Hindi/English clusters\n< 5ms per remark"]
    end

    A & B & C & D & E --> SO[ScorerOutput\nscore · confidence · missing_flag]

    SO --> OV{Track B\nSpecialist Override?}
    OV -->|any score ≥ threshold\n0.92/0.90/0.88/–/–| OVERRIDE[return 1.0\nhigh-confidence veto]
    OV -->|no override| ML

    subgraph META["Meta-Learner — Phase 2+"]
        ML[Stacking meta-learner\n15 features: 5×score\n5×confidence · 5×missing_flag\nLogReg or XGBoost]
        FALL[Fallback aggregate\nA×0.40 B×0.20 C×0.20\nD×0.10 F×0.10]
    end

    ML --> SHADOW[Write shadow_score_committee\nsilent on failure]
    FALL --> SHADOW

    SHADOW --> LIVE[Returns EXISTING\nXGBoost score\nunchanged in shadow mode]

    style OVERRIDE fill:#d93025,color:#fff
    style LIVE fill:#1e8e3e,color:#fff
    style SHADOW fill:#9c27b0,color:#fff
```

---

## Discovery Pipeline — PASS Stream Only

```mermaid
flowchart LR
    SCORE[Transaction scored\naction == PASS only] --> ENS

    subgraph DISC["Discovery Ensemble — run_in_executor"]
        ENS[ECOD + IsoForest\nstructural features only\nno amount or time]
    end

    ENS -->|not novel| SKIP[No action]
    ENS -->|novel| DD{Dedup check\nreviewed_novelty_registry\nSHA-256 fingerprint}

    DD -->|label=0 seen| SKIP
    DD -->|new pattern| NQ[novelty_queue\nDeveloper review]

    NQ --> COUNT{Same fingerprint\n10+ times in 7d?}
    COUNT -->|No| TRIAGE[Batch triage]
    COUNT -->|Yes| ESC[Red Team\nEscalation]

    style ESC fill:#d93025,color:#fff
    style NQ fill:#e8710a,color:#fff
    style SKIP fill:#1e8e3e,color:#fff
```

> **Invariant:** Anomaly score **never** enters `fraud_score`. Investigators never see it. Only fires when `action == "PASS"`.

---

## Investigator Feedback Routing

```mermaid
flowchart TD
    INV[Investigator submits\nPOST /api/v1/feedback] --> AUTH[Auth check\nINVESTIGATOR_API_KEY\nor RS256 JWT]

    AUTH --> ALWAYS[Always:\nBlockchain seal\nRed Team DNA delivery\nAlert status update\nfeedback_log INSERT\nmodel_audit INSERT]

    AUTH --> ROUTE{Decision type}

    ROUTE -->|confirmed_fraud=false\nFalse positive| FP

    subgraph FP_PATH["False Positive Path"]
        FP[curated_dataset_queue\nlabel=0\nbatch_exported=false]
        REG[reviewed_novelty_registry\nfingerprint dedup\nlabel=0]
    end

    ROUTE -->|confirmed_fraud=true\nTrue positive| CF

    subgraph CF_PATH["Confirmed Fraud Path"]
        CF[prototype_injection_candidates\nstatus=PENDING_REVIEW]
        DEV[Developer reviews\nGET /developer-queue]
        INJ[Inject into\nScorer C FAISS vault\nPOST /inject]
    end

    FP --> BATCH[Batch export every N rows\n→ python ml/train.py retrain]
    CF --> DEV --> INJ

    ALWAYS -.->|immutable| AUDIT[(model_audit\nDB trigger blocks\nUPDATE/DELETE)]

    style AUDIT fill:#d93025,color:#fff
    style CF fill:#e8710a,color:#fff
    style FP fill:#1a73e8,color:#fff
```

---

## Nightly Graph Feature Pipeline

```mermaid
sequenceDiagram
    participant BEAT as Celery Beat<br/>3am UTC
    participant NB as nightly_batch.py
    participant NEO as Neo4j<br/>(read-only)
    participant NX as NetworkX
    participant LEIDEN as leidenalg
    participant REDIS as Redis<br/>feat:{account}
    participant NODE2VEC as node2vec_runner.py
    participant MODEL as train.py

    BEAT->>NB: run_nightly_feature_computation()
    NB->>NEO: MATCH (a:Account) 30-day edge query
    NEO-->>NB: account rows + edge rows (≤2M)
    NB->>NX: build DiGraph
    NX-->>NB: G
    NB->>NX: fraud-seeded PageRank
    NB->>LEIDEN: find_partition(ModularityVertexPartition)
    LEIDEN-->>NB: community_map (non-empty = Leiden OK)
    NB->>NX: betweenness · clustering · sink/bipartite scores
    NB->>REDIS: SET feat:{account} ~107 fields per account
    alt community_map non-empty
        NB->>REDIS: SET leiden:deployed = true
    else empty — Leiden failed
        NB->>NB: WARNING — flag NOT set
    end

    BEAT->>NODE2VEC: run_node2vec_embeddings() (nightly)
    NODE2VEC->>REDIS: SET emb:{account} = JSON array 32-dim

    MODEL->>REDIS: GET leiden:deployed
    MODEL-->>MODEL: retrain only when flag = true
```

---

## The 5 Hard Graph Gates (Tier 2)

Gates fire `score=1.0` or pass through. No partial scores. Based on RBI FATF layering detection guidance.

| Gate | Detects | Key signal |
|------|---------|-----------|
| **Cycle** | Circular fund trails A→B→C→A (2-8 hops) | `cycle_membership` — Leiden nightly batch |
| **Sink D-01** | Dormant account suddenly receives large inflow | `days_since_last_send` from Redis — NOT account age |
| **Bipartite** | 7+ senders → 1 collector (density >0.7) | `bipartite_score` |
| **Cash Mule Sink** | Receive → ATM withdrawal → digital silence | PostgreSQL only — no device ID needed |
| **Merchant Terminal** | Round-trip through POS terminal | `merchant_terminal_id` correlation |
| **Gate 0 Rapid Relay** | Relay ≥80% of inflow within 1h | **LOG-ONLY** until `GATE0_LIVE=true` |

After a gate fires, **5 legitimacy filters run in order** — internal/treasury → KYC relationship → salary advance → all-merchant → amount <70%. Never skipped. Never reordered.

---

## Feature Engineering

~107 total features across 4 groups. `ml/feature_registry.py` is the **single source of truth** — `train.py` and `feature_builder.py` both import from it. No file may hardcode a feature list.

```mermaid
graph LR
    subgraph REDIS["Redis feat:{account} — Leiden nightly batch"]
        R1[pagerank_fraud_seeded]
        R2[community_id · community_fraud_ratio]
        R3[sink_score · bipartite_score]
        R4[betweenness · clustering · degree]
        R5[burst_score · velocity_ratio]
        R6[multi-hop windows 1h/6h/24h/7d]
        R7[days_since_last_send/receive]
        R8[graph_staleness_hours]
    end

    subgraph EMB["Redis emb:{account} — Node2Vec nightly"]
        E1[32-dim embedding\nJSON array]
    end

    subgraph PG["PostgreSQL — real-time at scoring"]
        P1[txn_amount · log · rounded]
        P2[velocity counts 1h/24h/7d]
        P3[payee_vpa_age_days]
        P4[payee_in_alert_log]
        P5[benford_deviation]
        P6[fan_in_sender_zscore]
        P7[micro_test_payment]
    end

    subgraph TXN["Transaction — inline"]
        T1[channel · hour · day]
        T2[is_night · is_weekend]
        T3[is_festival_period]
        T4[amount_vs_threshold_50K/1L/10L]
    end

    REDIS & EMB & PG & TXN --> FB[feature_builder.py\nassembles ~107 features]
    FB --> XGB[XGBoost scorer]
    FB --> COMM[Committee scorers]
```

---

## Indian Context Adjustment

```mermaid
graph TD
    RAW[Raw XGBoost score\n0.0 – 1.0] --> MULT

    subgraph MULT["Context Multipliers — apply_indian_context()"]
        M1["Festival season\nOct 1 – Nov 15\n× 0.70"]
        M2["Gig worker\nkyc_occupation\n× 0.85"]
        M3["Senior 60+ + night\n× 1.50"]
        M4["Senior + VPA < 7d\n× 1.30"]
        M5["Jan Dhan first digital\n× 0.65"]
        M6["Rural + geo switch\n× 0.75"]
        M7["Graph staleness > 24h\npenalty proportional\nto staleness_hours"]
    end

    MULT --> ADJ[Adjusted score\ncap 0.0 – 1.0]
    ADJ --> JITTER["Score jitter ±0.01\nanti-model-extraction\nP6-1"]
    JITTER --> ACTION{Threshold decision\nfrom SAME jitter draw}

    ACTION -->|lt 0.38| PASS[PASS]
    ACTION -->|0.38| LOG[LOG]
    ACTION -->|0.62| REV[REVIEW]
    ACTION -->|0.83| HR[HIGH_RISK]

    style HR fill:#d93025,color:#fff
    style REV fill:#e8710a,color:#fff
    style PASS fill:#1e8e3e,color:#fff
```

> Score and action are always derived from the **same jitter draw** — prevents score/action inconsistency at boundaries.

---

## API Reference

### `POST /api/v1/score`
Auth: `GRAPH_ENGINE_API_KEY` or RS256 JWT · Rate limit: 1000/min

```json
// Request
{
  "transaction_id": "TXN_001",
  "account_id": "ACC123456789",
  "amount": "500000.00",
  "channel": "UPI",
  "timestamp": "2026-05-17T02:14:00Z",
  "payee_vpa": "recipient@upi",
  "payee_vpa_created_at": "2026-05-15T10:00:00Z"
}

// Response — always padded to 55ms (timing oracle prevention)
{
  "transaction_id": "TXN_001",
  "score": 0.9654,
  "action": "HIGH_RISK",
  "gate_fired": null,
  "alert_id": "a1b2c3d4-...",
  "processing_ms": 47
}
```

### `GET /api/v1/alerts/{alert_id}`
Auth: `INVESTIGATOR_API_KEY` or JWT — evidence package: fund trail + SHAP values + committee breakdown + STR draft (156 FINnet fields).

### `POST /api/v1/feedback`
Auth: `INVESTIGATOR_API_KEY` — false positive → `curated_dataset_queue`; confirmed fraud → `prototype_injection_candidates` + blockchain + Red Team.

### `GET/POST /api/v1/developer-queue/prototype-candidates`
Auth: `INTERNAL_API_KEY` only — review and inject confirmed novel fraud prototypes into Scorer C FAISS vault.

### `GET/POST /api/v1/internal/model/versions` · `/activate`
Auth: `INTERNAL_API_KEY` only — model versioning and rollback with SHA-256 integrity check (P6-2).

### `POST /api/v1/analyze-graph`
Auth: any valid key — pure NetworkX topology analysis on a graph snapshot. No DB required.

---

## 16+ Fraud Archetypes

| Archetype | Description | Test Score |
|-----------|-------------|-----------|
| `structuring` | Multiple txns just below ₹50K/₹1L/₹10L | 0.867 |
| `romance_scam` | Escalating transfers to new VPA | 0.845 |
| `pig_butchering` | Small trust-building then large exit | 0.833 |
| `merchant_terminal` | Round-trip through POS | 0.813 |
| `cash_in_mule` | Cash deposit → digital → ATM | 0.813 |
| `otp_fraud` | Failed attempts → success post-OTP | 0.803 |
| `digital_arrest` | Senior + night + large + new VPA | 0.802 |
| `investment_fraud` | High return promise + crypto gateway | 0.807 |
| `account_takeover` | Device change + velocity + new payees | 0.799 |
| `low_slow_mule` | 45-day warmup then 1.8L spike at 2am | 0.798 |
| `cycle_round_trip` | Circular flow — Tier 2 gate catches | 0.794 |
| `salary_mule` | Legit salary in, immediately forwarded | 0.768 |
| `rapid_layering` | 4+ hops, declining amounts, <20min | 0.759 |
| `sim_swap` | Device change + immediate high-value UPI | 0.745 |
| `ghost_node_cash` | ATM withdrawal + deposit different city 18h later | 0.706 |
| `bipartite_mule` | 7+ senders → 1 collector, density 0.85 | 0.698 |
| `hawala` | Informal value transfer network pattern | Phase 3 |
| `crypto_on_ramp` | Cash → crypto gateway layering | Phase 3 |
| `benami` | Nominee account concealment pattern | Phase 3 |

---

## Database Schema

```mermaid
erDiagram
    accounts {
        varchar id PK
        int account_age_days
        varchar kyc_occupation
        int kyc_age
        varchar account_type
        float kyc_completeness_score
        timestamptz created_at
        timestamptz updated_at
    }
    transactions {
        varchar id PK
        varchar account_id FK
        varchar payee_account_id FK
        numeric amount
        varchar channel
        timestamptz timestamp
        timestamptz payee_vpa_created_at
    }
    fraud_scores {
        bigserial id PK
        varchar transaction_id FK
        float score
        varchar action
        varchar gate_fired
        jsonb tier1_flags
        jsonb feature_vector
        jsonb shap_values
        varchar model_version
        int processing_ms
    }
    alerts {
        varchar id PK
        varchar transaction_id FK
        float score
        varchar action
        varchar status
        varchar trail_status
    }
    model_audit {
        bigserial id PK
        varchar transaction_id
        varchar event_type
        jsonb event_data
        timestamptz created_at
    }
    shadow_score_committee {
        bigserial id PK
        varchar transaction_id FK
        float scorer_a_score
        float scorer_b_score
        float scorer_c_score
        float scorer_d_score
        float scorer_f_score
        float meta_score
        boolean specialist_override
        float final_committee_score
        float live_score
        varchar live_action
    }
    curated_dataset_queue {
        bigserial id PK
        varchar transaction_id
        int label
        varchar label_source
        jsonb feature_vector
        boolean batch_exported
    }
    prototype_injection_candidates {
        bigserial id PK
        varchar transaction_id
        varchar fraud_type
        jsonb feature_vector
        varchar status
    }

    accounts ||--o{ transactions : "sends"
    transactions ||--o{ fraud_scores : "scored_as"
    fraud_scores ||--o{ alerts : "triggers"
    transactions ||--o{ shadow_score_committee : "shadow_scored"
```

Migrations: `001 → 002 → 003 → 004 → 005 → 006 → 007` (current head)

---

## How to Run

```bash
# Infrastructure
docker-compose up -d

# Env setup
cp .env.example .env   # fill credentials — see .env.example for all vars

# Database
alembic upgrade head   # runs migrations 001 → 007

# Build committee engine assets
python ml/scripts/build_phrase_dict.py        # Scorer F phrase embeddings
python ml/scripts/build_initial_prototypes.py # Scorer C FAISS vault seed

# Train models
python ml/train.py                    # XGBoost + Platt calibration (~2 min)
python ml/train_isolation_forest.py   # IsoForest base (~30 sec)

# Seed Redis + demo data
python scripts/seed_redis.py
python scripts/generate_test_data.py && python scripts/load_sample_data.py

# Start Celery (separate terminal)
celery -A app.celery_app worker -l info -Q default,evidence,graph
celery -A app.celery_app beat -l info

# Start API
uvicorn app.main:app --reload --port 8000

# Verify
curl http://localhost:8000/health
pytest tests/ -v   # 102+ passing
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.111 + Uvicorn |
| ML scoring | XGBoost 2.x + Platt calibration + SHAP 0.44 |
| Committee | 5-scorer committee engine (shadow mode → live Phase 5) |
| Discovery | IsolationForest + ECOD (PASS stream only) |
| Prototype vault | FAISS-cpu k-NN |
| Phrase screener | sentence-transformers (multilingual MiniLM, CPU-fast) |
| Graph features | Leiden community + Node2Vec (leidenalg + igraph + networkx) |
| Primary DB | PostgreSQL 15 (JSONB, pgcrypto, INET, BIGSERIAL) |
| Graph DB | Neo4j Community 5.x (read-only) |
| Cache | Redis 7.x (AOF persistence, connection pool, ZSET velocity windows) |
| Async tasks | Celery 5.x (fund trail, SHAP, graph refresh, betweenness) |
| Scheduler | Celery Beat (nightly 3am · betweenness 2h · micro-batch 5min) |
| Auth | X-API-Key per caller + RS256 JWT (dual mode, P1-7) |
| Compliance | FINnet 2.0 stub · NPCI pre-settlement stub · DPDP Act 2023 |
| Sanctions | OFAC + UN + MHA India (Redis SET, atomic rename, daily sync) |
| Observability | structlog + Prometheus |
| Deployment | Docker + Docker Compose |

---

## Security

| Concern | Implementation |
|---------|---------------|
| Auth | X-API-Key per caller + RS256 JWT; router-level `Depends()` |
| Rate limiting | 1000/min POST /score · per-endpoint via slowapi |
| SQL injection | Parameterized queries only — zero f-strings in any SQL |
| PII in logs | `HMAC-SHA256(PSEUDONYMIZATION_KEY, account_id)[:12]` |
| HTTP headers | HSTS · X-Frame-Options DENY · X-Content-Type-Options · CSP |
| Secrets | `.env` only — never in source, never logged |
| Audit integrity | DB trigger blocks UPDATE/DELETE on `model_audit` (RBI PMLA §12) |
| Anti-model-extraction | ±0.01 jitter before threshold decision; constant 55ms response time |
| Model integrity | SHA-256 sidecar hashes verified at startup (P0-3) |
| Sanctions | OFAC + UN + MHA lists; atomic Redis rename (no race window) |
| SMTP | STARTTLS enforced before send on port 587 |

---

## Core Invariants

1. No automated blocking. Investigators decide at every decision point.
2. XGBOD / IsolationForest / ECOD scores **never** enter `fraud_score`. Investigators never see them.
3. Gate 0 (rapid relay) is LOG-ONLY until `GATE0_LIVE=true` after 2-week pilot review.
4. SHAP always runs on uncalibrated base XGBoost — **never** on `CalibratedClassifierCV`.
5. Leiden + XGBoost retrain deploy atomically. `leiden:deployed` flag set only when `community_map` is non-empty.
6. `feature_registry.py` is the only place feature names/order are defined.
7. `model_audit` INSERT is atomic with every scoring response — if it fails, the request fails.
8. Blue Team never writes to Neo4j.
9. Score and action always derive from the **same jitter draw**.
10. Shadow failures (write to `shadow_score_committee`) never raise to caller or affect live score.

---

## Teammate Integration

| Teammate | Direction | What |
|---------|-----------|------|
| Graph Engine | → Blue Team | POST /api/v1/score per settled transaction |
| Graph Engine | → Neo4j | Builds live graph — Blue Team reads only |
| Investigator Dashboard | → Blue Team | GET /alerts/{id} · POST /feedback |
| Blockchain | ← Blue Team | Seals evidence on confirmed fraud |
| Red Team | ← Blue Team | Fraud DNA on feedback + novelty escalations |

---

<div align="center">
<b>BLING Hackathon · Blue Team · Union Bank of India</b><br>
Post-transaction forensic fraud detection with graph intelligence<br><br>
<i>102+ tests · 7 migrations · ~107 features · 5-scorer committee · 16+ archetypes</i>
</div>
