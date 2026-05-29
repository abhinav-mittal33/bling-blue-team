"""
ml/train_mamba.py — Train Scorer D in full Mamba (S4 state-space) mode.

STATUS: STUB — cannot train until sequence data exists.

=============================================================================
WHAT YOU NEED BEFORE RUNNING THIS
=============================================================================

1. INSTALL DEPENDENCIES (not in requirements.txt yet):
   pip install mamba-ssm causal-conv1d
   # Or for CPU-only (slower):
   pip install mamba-ssm --no-build-isolation

   Note: mamba-ssm requires CUDA for fast training. CPU training is 20-50x slower.
   For CPU-only environments, consider using a hosted GPU (Colab/Vast.ai) for training,
   then copy the .pt file to the server.

   After verifying it works, add to requirements.txt:
     mamba-ssm>=1.2.0
     causal-conv1d>=1.1.0

2. BUILD THE SEQUENCE DATASET:
   The dataset must be a table (or export) of ordered transaction sequences per account.
   Required schema:
     account_id   VARCHAR
     seq_position INTEGER      -- order within account's history (1 = oldest)
     features     JSONB/FLOAT[] -- same 7 set features as scorer_d.py OR raw txn features
     label        INTEGER       -- 1 = fraud account, 0 = legit (account-level label)

   Minimum: 5,000 fraud accounts with ≥10 transactions each.
   Script to build from existing transactions table:
     python scripts/build_sequence_dataset.py  ← WRITE THIS when ready

3. SET ENV VAR:
   MAMBA_LIMITED_MODE=false
   SCORER_D_SEQUENCE_DATA_PATH=/path/to/sequences.parquet

=============================================================================
ARCHITECTURE (fill in when building)
=============================================================================

  Input:  [batch_size, seq_len, feature_dim] tensor
          seq_len = last N transactions per account (pad/truncate to fixed length)
          feature_dim = 7 (same set features) OR full txn feature vector (70+)

  Model:  Mamba S4 block → Global average pool → Sigmoid head
          Reference: https://github.com/state-spaces/mamba

  Output: Per-account fraud probability (scalar, 0-1)
          Saved to: ml/models/scorer_d_mamba_v1.pt (embedding layers only, no head)

=============================================================================
INTEGRATION STEPS (after training)
=============================================================================

  1. Add mamba-ssm to requirements.txt
  2. Set MAMBA_LIMITED_MODE=false in .env
  3. Update app/detection/tier3/scorer_d.py:
       - Replace the `not settings.mamba_limited_mode → return unavailable` stub
       - Load scorer_d_mamba_v1.pt
       - Build input tensor from last-N transactions queried from DB
       - Forward pass → sigmoid → ScorerOutput
  4. Test: python -c "from app.detection.tier3 import scorer_d; print(scorer_d.score('test', db))"

=============================================================================
STUB IMPLEMENTATION (fill in when sequence data is available)
=============================================================================
"""
import sys

# --- STUB: remove this block and implement below when ready ---
print("train_mamba.py is a stub — sequence data not yet available.")
print()
print("To enable full Mamba mode:")
print("  1. pip install mamba-ssm causal-conv1d")
print("  2. Build sequence dataset: python scripts/build_sequence_dataset.py")
print("  3. Implement training loop below")
print("  4. Set MAMBA_LIMITED_MODE=false in .env")
print("  5. Update app/detection/tier3/scorer_d.py to load mamba model")
sys.exit(0)
# --- END STUB ---


# ============================================================
# IMPLEMENT BELOW (when sequence data is available)
# ============================================================

def build_sequence_tensors(data_path: str):
    """
    Load sequence dataset and return (X, y) tensors.

    X: shape [n_accounts, seq_len, feature_dim] — float32
    y: shape [n_accounts] — int (0 or 1, account-level label)

    TODO: implement when build_sequence_dataset.py output is ready.
    """
    raise NotImplementedError("Build sequence dataset first.")


def train(data_path: str, output_path: str) -> None:
    """
    Train FraudMamba model on account transaction sequences.

    TODO: implement. Rough structure:
      1. X, y = build_sequence_tensors(data_path)
      2. model = MambaFraudDetector(d_model=64, n_layers=2, seq_len=seq_len)
      3. optimizer = Adam(lr=1e-3)
      4. loss = BCEWithLogitsLoss(pos_weight=torch.tensor([n_legit/n_fraud]))
      5. Train 50 epochs, early stop patience=10 on val PR-AUC
      6. Strip classifier head → save embedding encoder to scorer_d_mamba_v1.pt
    """
    raise NotImplementedError("Implement when sequence data is ready.")


if __name__ == "__main__":
    import os
    data_path = os.getenv("SCORER_D_SEQUENCE_DATA_PATH", "")
    if not data_path:
        print("ERROR: set SCORER_D_SEQUENCE_DATA_PATH env var")
        sys.exit(1)
    train(data_path, "ml/models/scorer_d_mamba_v1.pt")
