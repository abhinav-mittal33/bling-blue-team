"""
app/graph/gnn_embedder.py — PC-GNN + Hypergraph embedding component (Scorer B).

Architecture:
  Transaction graph (Neo4j Account->Account)
      |
  PC-GNN — 2 layers, camouflage-resistant attention aggregation
      | 32-dim per-node embeddings
  Hypergraph layer — Leiden community hyperedges (group-level mule farm signal)
      | enriched 32-dim embeddings
  Redis gnn_emb:{account} — 25h TTL (90000s)

PC-GNN "pick" mechanism: attention = sigmoid(att_src(x_src) + att_dst(x_dst)).
Camouflage neighbors (fraudsters hiding among legit accounts) have dissimilar embeddings
-> low attention -> suppressed. This is the camouflage-resistance property.

Hypergraph: uses torch_geometric.nn.HypergraphConv. Hyperedges = Leiden communities.
A 10-account mule farm in the same community appears as one coordinated hyperedge,
not 10 pairwise relationships. This captures group-level mule farm signal.

CRITICAL: torch is optional. All torch imports are lazy (inside function bodies).
If torch is not installed, every public function logs a warning and returns None/0.
The API must start and operate normally without torch installed.

Model file: ml/models/pcgnn_v1.pt
Redis key:  gnn_emb:{account_id}  (JSON list of 32 floats, TTL=90000s)
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from typing import Optional

import structlog

log = structlog.get_logger()

# ---- Constants ---------------------------------------------------------------
_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml", "models", "pcgnn_v1.pt")
)
_GNN_EMB_TTL = 90_000  # 25h in seconds — slightly longer than nightly batch cycle
_GNN_EMB_KEY_PREFIX = "gnn_emb:"

# The 10 node features consumed from feat:{account} Redis hash.
# Graph-structural features only — time/amount features excluded because they vary
# legitimately (night workers, large one-off payments) and would pollute the
# structural embedding space with non-structural variance.
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

# Model singleton + thread-safe double-check lock
_model_lock = threading.Lock()
_model_instance = None
_model_loaded = False  # True once we have attempted load (avoids repeated file I/O)

# Class singleton lock
_FraudPCGNN_class = None
_FraudPCGNN_lock = threading.Lock()


# ---- Module-level class builder ----------------------------------------------
# FraudPCGNN is built lazily so the module can be imported without torch.
# train_gnn.py imports build_pcgnn_model() which triggers the class build.

def _build_fraud_pcgnn_class():
    """
    Build and return the FraudPCGNN nn.Module class.
    Raises ImportError if torch_geometric is unavailable.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import MessagePassing, HypergraphConv, Linear

    class PCGNNLayer(MessagePassing):
        """
        Single PC-GNN layer with camouflage-resistant attention.

        attention = sigmoid(att_src(x_src) + att_dst(x_dst)) for each (src, dst) edge.
        Fraudsters hiding among legit neighbors have dissimilar embeddings ->
        low attention weight -> their influence on the target node is suppressed.
        """

        def __init__(self, in_channels, out_channels):
            super().__init__(aggr="add")
            self.lin = Linear(in_channels, out_channels, bias=False)
            # Separate source/destination attention projections
            self.att_src = Linear(in_channels, 1, bias=False)
            self.att_dst = Linear(in_channels, 1, bias=False)
            self.bias = nn.Parameter(torch.zeros(out_channels))

        def forward(self, x, edge_index):
            out = self.propagate(edge_index, x=x)
            out = out + self.bias
            return F.elu(out)

        def message(self, x_i, x_j):
            # x_i = destination features, x_j = source (neighbor) features
            att = torch.sigmoid(self.att_src(x_j) + self.att_dst(x_i))
            return att * self.lin(x_j)

    class FraudPCGNN(nn.Module):
        """
        PC-GNN (Pick-and-Choose GNN) + HypergraphConv for fraud detection.

        Layer structure:
          1. PCGNNLayer(in_channels=10, hidden=64) — camouflage-resistant aggregation
          2. PCGNNLayer(64, out_channels=32) — 32-dim graph representation
          3. HypergraphConv(32, 32) — Leiden community hyperedge enrichment

        fraud_head Linear(32, 1) is added during training (train_gnn.py) and stripped
        before saving to pcgnn_v1.pt so inference uses only the embedding layers.
        """

        def __init__(self, in_channels=10, hidden_channels=64, out_channels=32):
            super().__init__()
            self.pcgnn1 = PCGNNLayer(in_channels, hidden_channels)
            self.pcgnn2 = PCGNNLayer(hidden_channels, out_channels)
            # HypergraphConv: Leiden community hyperedges provide group-level signal.
            # A 10-account mule farm in the same community forms one hyperedge
            # instead of 10 disconnected pairwise edges.
            self.hyper_conv = HypergraphConv(out_channels, out_channels)
            self.dropout = nn.Dropout(p=0.3)

        def forward(self, x, edge_index, hyperedge_index=None):
            """
            Args:
                x: Node feature matrix [N, in_channels]
                edge_index: COO edge index [2, E]
                hyperedge_index: Hyperedge incidence [2, total_memberships] or None.
                    Row 0 = node indices, Row 1 = hyperedge indices.
                    If None, HypergraphConv is skipped (no community features).
            Returns:
                embeddings: [N, out_channels=32]
            """
            x = self.pcgnn1(x, edge_index)
            x = self.dropout(x)
            x = self.pcgnn2(x, edge_index)

            if hyperedge_index is not None and hyperedge_index.size(1) > 0:
                x = self.hyper_conv(x, hyperedge_index)

            return x

    return FraudPCGNN


def _get_fraud_pcgnn_class():
    """Thread-safe lazy loader for the FraudPCGNN class."""
    global _FraudPCGNN_class
    if _FraudPCGNN_class is not None:
        return _FraudPCGNN_class
    with _FraudPCGNN_lock:
        if _FraudPCGNN_class is None:
            _FraudPCGNN_class = _build_fraud_pcgnn_class()
    return _FraudPCGNN_class


# ---- Public API: model factory -----------------------------------------------

def build_pcgnn_model(in_channels=10, hidden_channels=64, out_channels=32):
    """
    Factory — creates and returns a new FraudPCGNN instance.

    Imported by both gnn_embedder.py (inference) and ml/train_gnn.py (training).
    Returns None if torch or torch_geometric is not installed.

    Args:
        in_channels: Number of input node features (default 10 — matches _GNN_NODE_FEATURES).
        hidden_channels: Hidden layer width (default 64).
        out_channels: Embedding dimensionality (default 32).
    """
    try:
        FraudPCGNN = _get_fraud_pcgnn_class()
        return FraudPCGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
        )
    except ImportError:
        log.warning("pcgnn_torch_unavailable", reason="torch or torch_geometric not installed")
        return None
    except Exception as exc:
        log.error("pcgnn_build_failed", error=str(exc))
        return None


# ---- Model loader (lazy, thread-safe) ----------------------------------------

def load_gnn_model():
    """
    Lazy-load the trained PC-GNN model from ml/models/pcgnn_v1.pt.

    Double-check locking for thread safety. Returns None if:
    - torch not installed
    - model file does not exist (degraded mode — embeddings default to zeros)
    - model file is corrupt

    Model is cached as module-level singleton after first successful load.
    """
    global _model_instance, _model_loaded

    # Fast path — already loaded or confirmed absent
    if _model_loaded:
        return _model_instance

    with _model_lock:
        if _model_loaded:
            return _model_instance

        try:
            import torch

            if not os.path.exists(_MODEL_PATH):
                log.warning(
                    "pcgnn_model_file_absent",
                    path=_MODEL_PATH,
                    note="Run ml/train_gnn.py to generate pcgnn_v1.pt",
                )
                _model_instance = None
                _model_loaded = True
                return None

            FraudPCGNN = _get_fraud_pcgnn_class()
            model = FraudPCGNN()
            state = torch.load(_MODEL_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            model.eval()
            _model_instance = model
            _model_loaded = True
            log.info("pcgnn_model_loaded", path=_MODEL_PATH)
            return model

        except ImportError:
            log.warning("pcgnn_torch_unavailable", note="API continues without GNN embeddings")
            _model_instance = None
            _model_loaded = True
            return None
        except Exception as exc:
            log.error("pcgnn_model_load_failed", path=_MODEL_PATH, error=str(exc))
            _model_instance = None
            _model_loaded = True
            return None


# ---- Hyperedge construction ---------------------------------------------------

def _build_hyperedge_index(account_list, feat_dict):
    """
    Build a hyperedge_index tensor from Leiden community_id groupings.

    Each Leiden community with >=2 members becomes one hyperedge. This is how a
    10-account mule farm registers as a single coordinated group signal rather than
    10 isolated edges.

    Returns:
        torch.LongTensor [2, total_memberships] or empty [2, 0] tensor.
        Returns None if torch unavailable.
    """
    try:
        import torch

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
            if len(members) >= 2:  # skip singletons — no group signal to capture
                for m in members:
                    node_idx_list.append(m)
                    hedge_idx_list.append(hedge_idx)

        if not node_idx_list:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor([node_idx_list, hedge_idx_list], dtype=torch.long)

    except ImportError:
        return None


# ---- Node feature extraction -------------------------------------------------

def _extract_node_features(account_list, feat_dict):
    """
    Build node feature matrix X from feat_dict.

    For each account, reads 10 features in _GNN_NODE_FEATURES order.
    Missing or non-numeric values -> 0.0 (GNN requires real values;
    unlike XGBoost it has no built-in NaN missing-value path).

    Returns:
        torch.FloatTensor [N, 10] or None if torch unavailable.
    """
    try:
        import torch

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

    except ImportError:
        return None


# ---- Graph to edge_index -----------------------------------------------------

def _graph_to_edge_index(G, account_index):
    """
    Convert a NetworkX DiGraph to a PyG edge_index tensor.

    Args:
        G: nx.DiGraph with account IDs as nodes.
        account_index: {account_id: integer_node_index}

    Returns:
        torch.LongTensor [2, E] or None if torch unavailable.
    """
    try:
        import torch

        src_list, dst_list = [], []
        for src, dst in G.edges():
            s_idx = account_index.get(src)
            d_idx = account_index.get(dst)
            if s_idx is not None and d_idx is not None:
                src_list.append(s_idx)
                dst_list.append(d_idx)

        if not src_list:
            # Isolated-node graph — empty edge tensor is valid input for MessagePassing
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor([src_list, dst_list], dtype=torch.long)

    except ImportError:
        return None


# ---- Main embedding generation -----------------------------------------------

def _load_feat_dict_from_redis(account_list: list) -> dict:
    """Load feat:{account} hashes from Redis for all accounts in list."""
    from app.utils.redis_client import get_redis
    r = get_redis()
    feat_dict: dict = {}
    for acc_id in account_list:
        raw = r.hgetall(f"feat:{acc_id}")
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
    return feat_dict


def generate_embeddings(G, feat_dict=None):
    """
    Run PC-GNN + HypergraphConv inference on the full account graph and write
    32-dim embeddings to Redis as gnn_emb:{account_id}.

    Args:
        G: nx.DiGraph — accounts as nodes, transaction edges.
        feat_dict: {account_id: {feature_name: value}} from Redis feat:{account}.
            If None, features are loaded from Redis automatically (used when called
            from nightly_batch.py after features have been written to Redis).

    Returns:
        Number of embeddings written to Redis.
        0 if torch unavailable, model absent, or graph is empty.

    Side effects:
        Writes JSON list (32 floats) to gnn_emb:{account_id} with TTL=90000s.
    """
    try:
        import torch
    except ImportError:
        log.warning("generate_embeddings_skipped", reason="torch not installed")
        return 0

    if len(G) == 0:
        log.warning("generate_embeddings_skipped", reason="empty graph")
        return 0

    model = load_gnn_model()
    if model is None:
        log.warning("generate_embeddings_skipped", reason="model not loaded — run ml/train_gnn.py")
        return 0

    try:
        account_list = list(G.nodes())
        account_index = {acc: idx for idx, acc in enumerate(account_list)}

        # Load from Redis when caller has no in-memory feat_dict
        if feat_dict is None:
            feat_dict = _load_feat_dict_from_redis(account_list)

        X = _extract_node_features(account_list, feat_dict)
        edge_index = _graph_to_edge_index(G, account_index)
        hyperedge_index = _build_hyperedge_index(account_list, feat_dict)

        if X is None or edge_index is None:
            log.error("generate_embeddings_failed", reason="tensor construction returned None")
            return 0

        with torch.no_grad():
            embeddings = model(X, edge_index, hyperedge_index)

        emb_np = embeddings.cpu().numpy()
        count = _write_embeddings_to_redis(account_list, emb_np)
        log.info("generate_embeddings_complete", accounts=len(account_list), written=count)
        return count

    except Exception as exc:
        log.error("generate_embeddings_failed", error=str(exc))
        return 0


def _write_embeddings_to_redis(account_list, emb_np):
    """Pipeline-write embedding rows to Redis. Returns count of successful writes."""
    try:
        from app.utils.redis_client import get_redis
        r = get_redis()
        pipe = r.pipeline()
        for idx, acc_id in enumerate(account_list):
            key = f"{_GNN_EMB_KEY_PREFIX}{acc_id}"
            emb_list = emb_np[idx].tolist()
            pipe.setex(key, _GNN_EMB_TTL, json.dumps(emb_list))
        pipe.execute()
        return len(account_list)
    except Exception as exc:
        log.error("redis_embedding_write_failed", error=str(exc))
        return 0


# ---- Partial refresh (called by tasks.py every 5 minutes) --------------------

def refresh_recent_embeddings(lookback_minutes=60):
    """
    Re-embed only accounts active in the last `lookback_minutes`.

    Called by app/graph/tasks.py:refresh_gnn_embeddings_task every 5 minutes.
    Queries Neo4j for recently-active accounts, fetches feat:{account} hashes,
    runs partial inference, updates only those gnn_emb:{account} Redis keys.

    Args:
        lookback_minutes: Window to check for recent activity.

    Returns:
        Number of embeddings updated. 0 if torch unavailable, model absent, or
        no recently-active accounts found.
    """
    try:
        import torch
    except ImportError:
        log.warning("refresh_recent_embeddings_skipped", reason="torch not installed")
        return 0

    model = load_gnn_model()
    if model is None:
        log.warning("refresh_recent_embeddings_skipped", reason="model not loaded")
        return 0

    try:
        from app.graph.neo4j_client import run_query
        from app.utils.redis_client import get_redis
        import networkx as nx

        # Parameterized Cypher — never f-strings in queries
        cypher = (
            "MATCH (a:Account)-[r:SENT]->(b:Account) "
            "WHERE r.timestamp > datetime() - duration({minutes: $lookback_minutes}) "
            "RETURN DISTINCT a.id AS account_id "
            "UNION "
            "MATCH (a:Account)<-[r:SENT]-(b:Account) "
            "WHERE r.timestamp > datetime() - duration({minutes: $lookback_minutes}) "
            "RETURN DISTINCT a.id AS account_id "
            "LIMIT 10000"
        )
        rows = run_query(cypher, {"lookback_minutes": lookback_minutes})
        recent_accounts = [row["account_id"] for row in rows if row.get("account_id")]

        if not recent_accounts:
            log.debug("refresh_recent_embeddings_no_accounts", lookback_minutes=lookback_minutes)
            return 0

        # Fetch feat:{account} from Redis for each recently-active account
        r = get_redis()
        feat_dict = {}
        for acc_id in recent_accounts:
            raw = r.hgetall(f"feat:{acc_id}")
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

        # Build partial graph for these accounts.
        # Using isolated-node graph as base — edges between these accounts are
        # added below. Full-graph embeddings regenerate nightly via run_node2vec_task.
        G_partial = nx.DiGraph()
        G_partial.add_nodes_from(recent_accounts)

        # Pull edges between recently-active accounts for better embedding context
        edge_cypher = (
            "MATCH (a:Account)-[r:SENT]->(b:Account) "
            "WHERE a.id IN $accounts AND b.id IN $accounts "
            "AND r.timestamp > datetime() - duration({minutes: $lookback_minutes}) "
            "RETURN a.id AS src, b.id AS dst "
            "LIMIT 50000"
        )
        edge_rows = run_query(
            edge_cypher,
            {"accounts": recent_accounts, "lookback_minutes": lookback_minutes},
        )
        for row in edge_rows:
            src = row.get("src")
            dst = row.get("dst")
            if src and dst:
                G_partial.add_edge(src, dst)

        count = generate_embeddings(G_partial, feat_dict)
        log.info(
            "refresh_recent_embeddings_complete",
            accounts_checked=len(recent_accounts),
            embeddings_updated=count,
            lookback_minutes=lookback_minutes,
        )
        return count

    except Exception as exc:
        log.error("refresh_recent_embeddings_failed", error=str(exc))
        return 0


# ---- Heterogeneous data stub (P2-2 integration point) ------------------------

def build_hetero_data_stub(account_x, edge_index, device_edges=None, vpa_edges=None):
    """
    Build PyG HeteroData when Device + VPA node types are available (P2-2).

    CURRENT BEHAVIOR (P2-2 not yet deployed):
        Builds homogeneous Data(x=account_x, edge_index=edge_index).
        device_edges and vpa_edges are accepted but ignored.

    P2-2 INTEGRATION POINT:
        When teammate adds Device and VPA nodes to the Neo4j schema, update this
        function to build HeteroData with three node types:
          'account': account_x features [N_acc, 10]
          'device':  device features [N_dev, 2] (device_shared_count, device_age_days)
          'vpa':     VPA features [N_vpa, 2] (vpa_age_days, vpa_fraud_count)
        And three edge types:
          ('account', 'sent', 'account'): edge_index
          ('account', 'uses', 'device'):  device_edges
          ('account', 'linked_to', 'vpa'): vpa_edges
        The HGT model (ml/train_hgt.py) will then replace FraudPCGNN for full hetero training.

    Args:
        account_x: Account node feature tensor [N_acc, in_channels].
        edge_index: Account->Account edge index [2, E].
        device_edges: Account->Device edge index [2, E_dev] — None until P2-2.
        vpa_edges: Account->VPA edge index [2, E_vpa] — None until P2-2.

    Returns:
        torch_geometric.data.Data (homogeneous) currently.
        torch_geometric.data.HeteroData once P2-2 is deployed.
    """
    try:
        from torch_geometric.data import Data, HeteroData

        if device_edges is None and vpa_edges is None:
            # Homogeneous path — current production behavior
            return Data(x=account_x, edge_index=edge_index)

        # P2-2 hetero path — activates once teammate deploys Device + VPA nodes
        data = HeteroData()
        data["account"].x = account_x
        data["account", "sent", "account"].edge_index = edge_index

        if device_edges is not None:
            data["account", "uses", "device"].edge_index = device_edges

        if vpa_edges is not None:
            data["account", "linked_to", "vpa"].edge_index = vpa_edges

        return data

    except ImportError:
        log.warning("build_hetero_data_stub_failed", reason="torch_geometric not installed")
        return None
    except Exception as exc:
        log.error("build_hetero_data_stub_failed", error=str(exc))
        return None
