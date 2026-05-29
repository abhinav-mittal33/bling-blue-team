"""
ml/train_gnn.py — Training script for FraudPCGNN (PC-GNN + HypergraphConv).

Steps:
  1. Connect Neo4j, load 30-day edge list.
  2. Build NetworkX DiGraph -> convert to PyG Data (edge_index).
  3. Load node features from Redis feat:{account} for all accounts.
  4. Load labels from PostgreSQL: HIGH_RISK / REVIEW alerts = fraud=1.
  5. Build hyperedge_index from Leiden community_id groupings.
  6. Train FraudPCGNN with BCEWithLogitsLoss on labeled nodes.
  7. After training: run inference on ALL nodes -> 32-dim embeddings.
  8. Write gnn_emb:{account} to Redis (JSON list, TTL=90000s).
  9. Save model (without fraud_head) to ml/models/pcgnn_v1.pt.
 10. Print PR-AUC on validation set.

Fallback: If Neo4j is unavailable, falls back to a synthetic random graph for
testing the training loop. A WARNING is printed and final embeddings are meaningless.

Run: python ml/train_gnn.py

INVARIANTS:
  - eval_metric: PR-AUC on val set (never ROC-AUC — misleading on imbalanced data)
  - scale_pos_weight: computed from actual label distribution, printed for CLAUDE.md update
  - Parameterized SQL only — never f-strings in queries
  - All torch imports inside function bodies or guarded by try/except ImportError
"""
import os
import sys
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

# Allow imports from project root regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

log = structlog.get_logger()

# ---- Constants ---------------------------------------------------------------
_MODEL_SAVE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models", "pcgnn_v1.pt")
)
_GNN_NODE_FEATURES = [
    "pagerank_fraud_seeded",
    "community_fraud_ratio",
    "sink_score",
    "degree_centrality",
    "betweenness_centrality",
    "burst_score",
    "clustering_coefficient",
    "account_age_days",
    "kyc_completeness_score",
    "txn_count_30d",
]
_IN_CHANNELS = len(_GNN_NODE_FEATURES)  # 10
_HIDDEN_CHANNELS = 64
_OUT_CHANNELS = 32
_EPOCHS = 100
_LEARNING_RATE = 0.001
_EARLY_STOP_PATIENCE = 10
_VAL_FRACTION = 0.2
_GNN_EMB_TTL = 90_000  # 25h

# 30-day edge query — same as nightly_batch.py _EDGE_QUERY (parameterized, no f-strings)
_EDGE_QUERY = (
    "MATCH (a:Account)-[r:SENT]->(b:Account) "
    "WHERE r.timestamp > datetime() - duration({days: 30}) "
    "RETURN a.id AS src, b.id AS dst "
    "LIMIT 2000000"
)

# Labeled fraud accounts: HIGH_RISK or REVIEW alert actions
# Uses parameterized query — $actions is a list literal passed as parameter
_FRAUD_LABEL_QUERY = """
SELECT DISTINCT t.account_id
FROM alerts a
JOIN transactions t ON t.id = a.transaction_id
WHERE a.action IN ('HIGH_RISK', 'REVIEW')
LIMIT 50000
"""

# Sample equal-count clean (non-alert) accounts for fraud=0 labels
_CLEAN_LABEL_QUERY = """
SELECT DISTINCT account_id
FROM accounts
WHERE account_id NOT IN (
    SELECT DISTINCT t.account_id
    FROM alerts al
    JOIN transactions t ON t.id = al.transaction_id
    WHERE al.action IN ('HIGH_RISK', 'REVIEW')
)
ORDER BY RANDOM()
LIMIT %s
"""


# ---- Neo4j edge loader -------------------------------------------------------

def _load_edges_from_neo4j():
    """
    Load 30-day transaction edges from Neo4j.

    Returns:
        list of (src, dst) tuples.
        Falls back to empty list if Neo4j unavailable (caller detects and uses synthetic graph).
    """
    try:
        from app.graph.neo4j_client import run_query
        rows = run_query(_EDGE_QUERY, {})
        edges = [(r["src"], r["dst"]) for r in rows if r.get("src") and r.get("dst")]
        log.info("neo4j_edges_loaded", count=len(edges))
        return edges
    except Exception as exc:
        log.warning("neo4j_edge_load_failed", error=str(exc), fallback="synthetic_graph")
        return None


# ---- Feature loader ----------------------------------------------------------

def _load_node_features(account_list):
    """
    Load feat:{account} hashes from Redis for each account in account_list.

    Returns:
        {account_id: {feature_name: float_value}}
        Missing features default to 0.0 in the tensor builder.
    """
    try:
        from app.utils.redis_client import get_redis
        r = get_redis()
        feat_dict = {}
        pipe = r.pipeline()
        for acc_id in account_list:
            pipe.hgetall(f"feat:{acc_id}")
        results = pipe.execute()
        for acc_id, raw in zip(account_list, results):
            if raw:
                parsed = {}
                for k, v in raw.items():
                    try:
                        parsed[k] = float(v)
                    except (TypeError, ValueError):
                        parsed[k] = v
                feat_dict[acc_id] = parsed
            else:
                feat_dict[acc_id] = {}
        log.info("redis_features_loaded", accounts=len(feat_dict), with_data=sum(1 for v in feat_dict.values() if v))
        return feat_dict
    except Exception as exc:
        log.warning("redis_feature_load_failed", error=str(exc))
        return {acc: {} for acc in account_list}


# ---- Label loader ------------------------------------------------------------

def _load_labels_from_postgres(account_list):
    """
    Load binary fraud labels from PostgreSQL.

    Fraud=1: accounts that appear in HIGH_RISK or REVIEW alerts.
    Fraud=0: equal-count randomly sampled clean (non-alert) accounts.

    Uses parameterized queries — never f-strings in SQL.

    Returns:
        {account_id: 0_or_1} for the subset of account_list that has labels.
    """
    try:
        import sqlalchemy
        from app.core.config import settings

        engine = sqlalchemy.create_engine(settings.postgres_url)
        with engine.connect() as conn:
            # Fraud accounts
            result = conn.execute(sqlalchemy.text(_FRAUD_LABEL_QUERY))
            fraud_accounts = {row[0] for row in result}

            # Filter to accounts we actually have in the graph
            fraud_in_graph = {acc for acc in fraud_accounts if acc in set(account_list)}
            fraud_count = len(fraud_in_graph)

            if fraud_count == 0:
                log.warning("no_fraud_labels_found", note="All labels will be 0 — check alerts table")
                return {}

            # Sample equal-count clean accounts from the graph
            # Parameterized: %s placeholder for SQLAlchemy text queries
            clean_query = sqlalchemy.text(_CLEAN_LABEL_QUERY)
            result = conn.execute(clean_query, (fraud_count,))
            clean_accounts = {row[0] for row in result if row[0] in set(account_list)}

        labels = {}
        for acc in fraud_in_graph:
            labels[acc] = 1
        for acc in clean_accounts:
            if acc not in labels:  # never overwrite a fraud label
                labels[acc] = 0

        fraud_final = sum(1 for v in labels.values() if v == 1)
        clean_final = sum(1 for v in labels.values() if v == 0)
        spw = clean_final / max(fraud_final, 1)
        print(f"\n[train_gnn] Label distribution: fraud={fraud_final}, clean={clean_final}")
        print(f"[train_gnn] scale_pos_weight (BCEWithLogitsLoss pos_weight): {spw:.1f}")
        print(f"[train_gnn] UPDATE .claude/CLAUDE.md with: scale_pos_weight = {spw:.1f}\n")
        log.info("labels_loaded", fraud=fraud_final, clean=clean_final, pos_weight=round(spw, 1))
        return labels

    except Exception as exc:
        log.warning("postgres_label_load_failed", error=str(exc))
        return {}


# ---- Hyperedge construction (same logic as gnn_embedder.py) ------------------

def _build_hyperedge_index_train(account_list, feat_dict, torch):
    """
    Build hyperedge_index tensor from Leiden community_id groupings.

    Returns torch.LongTensor [2, total_memberships] or empty [2, 0] tensor.
    """
    community_groups = defaultdict(list)
    for node_idx, acc_id in enumerate(account_list):
        raw_comm = feat_dict.get(acc_id, {}).get("community_id", -1)
        try:
            comm_id = int(float(raw_comm))
        except (TypeError, ValueError):
            comm_id = -1
        if comm_id >= 0:
            community_groups[comm_id].append(node_idx)

    node_idx_list = []
    hedge_idx_list = []
    for hedge_idx, members in enumerate(community_groups.values()):
        if len(members) >= 2:
            for m in members:
                node_idx_list.append(m)
                hedge_idx_list.append(hedge_idx)

    if not node_idx_list:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([node_idx_list, hedge_idx_list], dtype=torch.long)


# ---- Node feature tensor builder ---------------------------------------------

def _build_feature_tensor(account_list, feat_dict, torch):
    """Build [N, 10] FloatTensor from feat_dict. Missing values -> 0.0."""
    rows = []
    for acc_id in account_list:
        feats = feat_dict.get(acc_id, {})
        row = []
        for fname in _GNN_NODE_FEATURES:
            raw = feats.get(fname, 0.0)
            try:
                val = float(raw)
                if val != val:  # NaN guard
                    val = 0.0
            except (TypeError, ValueError):
                val = 0.0
            row.append(val)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float)


# ---- Graph to edge_index converter -------------------------------------------

def _graph_to_edge_index(G, account_index, torch):
    """Convert nx.DiGraph to PyG edge_index [2, E]."""
    src_list, dst_list = [], []
    for src, dst in G.edges():
        s = account_index.get(src)
        d = account_index.get(dst)
        if s is not None and d is not None:
            src_list.append(s)
            dst_list.append(d)
    if not src_list:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src_list, dst_list], dtype=torch.long)


# ---- PR-AUC evaluation -------------------------------------------------------

def _compute_pr_auc(y_true, y_scores):
    """Compute PR-AUC using sklearn. Returns float."""
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, y_scores))
    except Exception as exc:
        log.warning("pr_auc_computation_failed", error=str(exc))
        return 0.0


# ---- Training loop -----------------------------------------------------------

def _train(model, fraud_head, optimizer, criterion, X, edge_index, hyperedge_index,
           train_mask, train_labels, val_mask, val_labels):
    """
    Semi-supervised node classification training loop.

    Args:
        model: FraudPCGNN instance (embedding layers only).
        fraud_head: Linear(32, 1) — classification head used during training only.
        optimizer: Adam optimizer.
        criterion: BCEWithLogitsLoss with pos_weight.
        X: Node feature matrix [N, 10].
        edge_index: [2, E].
        hyperedge_index: [2, M] or [2, 0].
        train_mask: Boolean tensor [N] marking training nodes.
        train_labels: Float tensor [train_count] with 0/1 labels.
        val_mask: Boolean tensor [N].
        val_labels: Float tensor [val_count].

    Returns:
        (best_val_pr_auc, best_epoch)
    """
    import torch
    import torch.nn.functional as F

    best_val_loss = float("inf")
    best_val_pr_auc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, _EPOCHS + 1):
        # Training pass
        model.train()
        fraud_head.train()
        optimizer.zero_grad()

        embeddings = model(X, edge_index, hyperedge_index)
        train_logits = fraud_head(embeddings[train_mask]).squeeze(-1)
        loss = criterion(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        # Validation pass
        model.eval()
        fraud_head.eval()
        with torch.no_grad():
            embeddings_val = model(X, edge_index, hyperedge_index)
            val_logits = fraud_head(embeddings_val[val_mask]).squeeze(-1)
            val_loss = criterion(val_logits, val_labels).item()
            val_scores = torch.sigmoid(val_logits).cpu().numpy()

        val_pr_auc = _compute_pr_auc(val_labels.cpu().numpy(), val_scores)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{_EPOCHS} | "
                f"train_loss={loss.item():.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_pr_auc={val_pr_auc:.4f}"
            )

        # Early stopping on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= _EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={_EARLY_STOP_PATIENCE})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_pr_auc, best_epoch


# ---- Embedding write to Redis ------------------------------------------------

def _write_all_embeddings(account_list, emb_np):
    """Write 32-dim embedding for every account to Redis as gnn_emb:{account}."""
    try:
        from app.utils.redis_client import get_redis
        r = get_redis()
        pipe = r.pipeline()
        for idx, acc_id in enumerate(account_list):
            key = f"gnn_emb:{acc_id}"
            emb_list = emb_np[idx].tolist()
            pipe.setex(key, _GNN_EMB_TTL, json.dumps(emb_list))
        pipe.execute()
        log.info("embeddings_written_to_redis", count=len(account_list))
        print(f"[train_gnn] Wrote {len(account_list)} embeddings to Redis (gnn_emb:*)")
    except Exception as exc:
        log.error("redis_embedding_write_failed", error=str(exc))
        print(f"[train_gnn] WARNING: Failed to write embeddings to Redis: {exc}")


# ---- Synthetic graph fallback ------------------------------------------------

def _build_synthetic_graph(n_nodes=500, n_edges=2000):
    """
    Generate a small random graph for testing the training loop when Neo4j is unavailable.
    Embeddings produced from this graph are meaningless — for code verification only.
    """
    import networkx as nx
    import random as rng

    print("[train_gnn] WARNING: Using synthetic random graph — Neo4j unavailable.")
    print("[train_gnn] Embeddings from this run are NOT suitable for production deployment.")
    G = nx.DiGraph()
    nodes = [f"synthetic_{i}" for i in range(n_nodes)]
    G.add_nodes_from(nodes)
    for _ in range(n_edges):
        s = rng.choice(nodes)
        d = rng.choice(nodes)
        if s != d:
            G.add_edge(s, d)
    return G


# ---- Main entry point --------------------------------------------------------

def main():
    """
    Full training pipeline for FraudPCGNN.

    Imports torch inside main() to ensure ImportError is handled cleanly.
    """
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.data import Data
    except ImportError as exc:
        print(f"[train_gnn] FATAL: torch or torch_geometric not installed. {exc}")
        print("[train_gnn] Install with: pip install torch torch_geometric")
        sys.exit(1)

    from app.graph.gnn_embedder import build_pcgnn_model
    import networkx as nx

    print("[train_gnn] Starting FraudPCGNN training...")
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load edges from Neo4j
    # ------------------------------------------------------------------
    edges = _load_edges_from_neo4j()
    if edges is None:
        # Neo4j unavailable — use synthetic graph for testing
        G = _build_synthetic_graph()
    else:
        G = nx.DiGraph()
        G.add_edges_from(edges)
        if len(G) == 0:
            print("[train_gnn] WARNING: Graph is empty after loading edges. Using synthetic graph.")
            G = _build_synthetic_graph()

    account_list = list(G.nodes())
    account_index = {acc: idx for idx, acc in enumerate(account_list)}
    N = len(account_list)
    print(f"[train_gnn] Graph: {N} accounts, {G.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Step 2: Build PyG Data (edge_index)
    # ------------------------------------------------------------------
    edge_index = _graph_to_edge_index(G, account_index, torch)

    # ------------------------------------------------------------------
    # Step 3: Load node features from Redis feat:{account}
    # ------------------------------------------------------------------
    print("[train_gnn] Loading node features from Redis...")
    feat_dict = _load_node_features(account_list)

    # ------------------------------------------------------------------
    # Step 4: Build node feature tensor X
    # ------------------------------------------------------------------
    X = _build_feature_tensor(account_list, feat_dict, torch)
    print(f"[train_gnn] Feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # Step 5: Load labels from PostgreSQL
    # ------------------------------------------------------------------
    print("[train_gnn] Loading labels from PostgreSQL...")
    label_dict = _load_labels_from_postgres(account_list)

    if len(label_dict) == 0:
        print("[train_gnn] WARNING: No labels found. Skipping supervised training.")
        print("[train_gnn] Running unsupervised embedding generation only...")
        supervised = False
    else:
        supervised = True

    # ------------------------------------------------------------------
    # Step 6: Build hyperedge_index from community_id groupings
    # ------------------------------------------------------------------
    hyperedge_index = _build_hyperedge_index_train(account_list, feat_dict, torch)
    hedge_count = hyperedge_index.size(1) if hyperedge_index.size(1) > 0 else 0
    print(f"[train_gnn] Hyperedge memberships: {hedge_count}")

    # ------------------------------------------------------------------
    # Step 7: Build model + training head
    # ------------------------------------------------------------------
    model = build_pcgnn_model(
        in_channels=_IN_CHANNELS,
        hidden_channels=_HIDDEN_CHANNELS,
        out_channels=_OUT_CHANNELS,
    )
    if model is None:
        print("[train_gnn] FATAL: build_pcgnn_model returned None")
        sys.exit(1)

    # fraud_head is used during training only — stripped before saving
    fraud_head = nn.Linear(_OUT_CHANNELS, 1)

    if supervised and len(label_dict) > 0:
        # ------------------------------------------------------------------
        # Step 8: Set up labels, masks, pos_weight, and loss
        # ------------------------------------------------------------------
        labeled_accounts = [acc for acc in account_list if acc in label_dict]
        labeled_indices = [account_index[acc] for acc in labeled_accounts]
        label_values = [float(label_dict[acc]) for acc in labeled_accounts]

        fraud_count = sum(1 for v in label_values if v == 1.0)
        clean_count = sum(1 for v in label_values if v == 0.0)
        pos_weight_val = clean_count / max(fraud_count, 1)
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        all_params = list(model.parameters()) + list(fraud_head.parameters())
        optimizer = torch.optim.Adam(all_params, lr=_LEARNING_RATE)

        # 80/20 train/val split on labeled nodes
        n_labeled = len(labeled_accounts)
        n_val = max(1, int(n_labeled * _VAL_FRACTION))
        n_train = n_labeled - n_val

        # Shuffle labeled indices deterministically
        import random as rng
        rng.seed(42)
        shuffled = list(range(n_labeled))
        rng.shuffle(shuffled)
        train_idx_local = shuffled[:n_train]
        val_idx_local = shuffled[n_train:]

        # Build full-graph masks (boolean [N])
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        label_tensor = torch.tensor(label_values, dtype=torch.float)

        for li in train_idx_local:
            train_mask[labeled_indices[li]] = True
        for li in val_idx_local:
            val_mask[labeled_indices[li]] = True

        train_labels = label_tensor[train_idx_local]
        val_labels = label_tensor[val_idx_local]

        print(f"[train_gnn] Train labeled: {n_train}, Val labeled: {n_val}")
        print(f"[train_gnn] pos_weight (BCEWithLogitsLoss): {pos_weight_val:.2f}")

        # ------------------------------------------------------------------
        # Step 9: Training loop
        # ------------------------------------------------------------------
        print(f"[train_gnn] Training {_EPOCHS} epochs (early stop patience={_EARLY_STOP_PATIENCE})...")
        best_pr_auc, best_epoch = _train(
            model, fraud_head, optimizer, criterion,
            X, edge_index, hyperedge_index,
            train_mask, train_labels, val_mask, val_labels,
        )
        print(f"\n[train_gnn] Best epoch={best_epoch}, val_pr_auc={best_pr_auc:.4f}")
    else:
        print("[train_gnn] Skipping supervised training (no labels).")
        best_pr_auc = 0.0

    # ------------------------------------------------------------------
    # Step 10: Run inference on ALL nodes -> 32-dim embeddings
    # ------------------------------------------------------------------
    print("[train_gnn] Running inference on all nodes to generate embeddings...")
    model.eval()
    with torch.no_grad():
        all_embeddings = model(X, edge_index, hyperedge_index)

    emb_np = all_embeddings.cpu().numpy()
    print(f"[train_gnn] Embeddings shape: {emb_np.shape}")

    # ------------------------------------------------------------------
    # Step 11: Write gnn_emb:{account} to Redis for each account
    # ------------------------------------------------------------------
    print("[train_gnn] Writing embeddings to Redis...")
    _write_all_embeddings(account_list, emb_np)

    # ------------------------------------------------------------------
    # Step 12: Save model (WITHOUT fraud_head) to ml/models/pcgnn_v1.pt
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(_MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), _MODEL_SAVE_PATH)
    print(f"[train_gnn] Model saved (embedding layers only): {_MODEL_SAVE_PATH}")
    log.info("pcgnn_model_saved", path=_MODEL_SAVE_PATH)

    # ------------------------------------------------------------------
    # Print PR-AUC summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n[train_gnn] ============= TRAINING COMPLETE =============")
    print(f"[train_gnn] Val PR-AUC:     {best_pr_auc:.4f}")
    print(f"[train_gnn] Accounts:       {N}")
    print(f"[train_gnn] Model saved:    {_MODEL_SAVE_PATH}")
    print(f"[train_gnn] Elapsed:        {elapsed:.1f}s")
    print(f"[train_gnn] =================================================\n")

    if best_pr_auc < 0.3 and supervised:
        print("[train_gnn] WARNING: val_pr_auc < 0.30. Consider:")
        print("  - More labeled data (run scripts to generate feedback_log entries)")
        print("  - Feature quality (verify feat:{account} hashes are populated in Redis)")
        print("  - Graph connectivity (sparse graphs degrade GNN performance)")


if __name__ == "__main__":
    main()
