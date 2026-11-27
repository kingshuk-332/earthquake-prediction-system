"""
Leak-Free Dilated 1D-CNN Baseline for Sequential Seismic Activity Forecasting
=============================================================================
Remediated per codebase_audit_and_fusion_proposal.md (Priority 0 & Priority 1):
  1. Removed flattened 3D SMOTE synthetic sequence fabrication and Mixup label blending.
  2. Removed silent hardcoded MIN_MAG/MAX_MAG fallback; requires verified unscaled
     `target_mw_raw` and `dl_temporal_scaler.pkl` provenance.
  3. Enforced an explicit embargo gap (>= seq_len) between chronological train, validation,
     and test sequence partitions so zero events overlap across split boundaries.
  4. Uses inverse-prevalence `pos_weight` in BCEWithLogitsLoss (proper scoring rule)
     with fold-scoped standardization fitted strictly on training sequences.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATE = SCRIPT_DIR.parent
if str(REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_CANDIDATE))

from seismic_fusion.config import PipelineConfig, resolve_repo_root


class DilatedConvBlock(nn.Module):
    """Causal-padded / fixed-window 1D dilated residual block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.dropout(F.gelu(self.bn(self.conv(x))))
        if out.shape[-1] != res.shape[-1]:
            res = res[..., : out.shape[-1]]
        return out + res


class EfficientCNN1D(nn.Module):
    """Parameter-efficient dilated 1D-CNN for causal look-back event sequences."""

    def __init__(self, n_features: int, seq_len: int = 15):
        super().__init__()
        self.seq_len = seq_len
        self.channel_reduce = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.block1 = DilatedConvBlock(32, 32, kernel_size=3, dilation=1, dropout=0.20)
        self.block2 = DilatedConvBlock(32, 48, kernel_size=3, dilation=2, dropout=0.25)
        self.block3 = DilatedConvBlock(48, 64, kernel_size=3, dilation=4, dropout=0.30)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, seq_len, n_features) -> transpose to (B, n_features, seq_len)
        h = x.transpose(1, 2)
        h = self.channel_reduce(h)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        pooled = h.mean(dim=-1)
        return self.classifier(pooled).squeeze(-1)


def build_embargoed_causal_sequences(
    features: np.ndarray,
    targets_mw: np.ndarray,
    timestamps_sec: np.ndarray,
    seq_len: int = 15,
    mw_threshold: float = 4.5,
    embargo_sec: float = 30.0 * 86400.0,
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Constructs strictly causal (past-only) sequences X[k] = features[k - seq_len : k]
    predicting y[k] = 1(targets_mw[k] >= mw_threshold), partitioned chronologically
    with a hard temporal embargo >= embargo_sec AND index gap >= seq_len so zero
    input events are shared across split boundaries.
    """
    n = len(features)
    if n <= 3 * seq_len:
        raise ValueError(f"Insufficient catalog length ({n}) for seq_len={seq_len}.")

    seq_X, seq_y, seq_t_start, seq_t_target, seq_end_idx = [], [], [], [], []
    for target_idx in range(seq_len, n):
        seq_X.append(features[target_idx - seq_len : target_idx])
        seq_y.append(1.0 if targets_mw[target_idx] >= mw_threshold else 0.0)
        seq_t_start.append(timestamps_sec[target_idx - seq_len])
        seq_t_target.append(timestamps_sec[target_idx])
        seq_end_idx.append(target_idx)

    X_all = np.asarray(seq_X, dtype=np.float32)
    y_all = np.asarray(seq_y, dtype=np.float32)
    t_start_arr = np.asarray(seq_t_start, dtype=np.float64)
    t_target_arr = np.asarray(seq_t_target, dtype=np.float64)
    end_idx_arr = np.asarray(seq_end_idx, dtype=np.int64)

    n_seq = len(X_all)
    tr_cut = int(n_seq * 0.70)
    va_cut = int(n_seq * 0.85)

    train_mask = np.arange(n_seq) < tr_cut
    max_train_time = float(t_target_arr[train_mask][-1])
    max_train_event_idx = int(end_idx_arr[train_mask][-1])

    # Validation starts strictly after embargo_sec AND after all training target indices leave the lookback window
    val_mask = (
        (np.arange(n_seq) >= tr_cut)
        & (np.arange(n_seq) < va_cut)
        & (t_start_arr > max_train_time + embargo_sec)
        & ((end_idx_arr - seq_len) > max_train_event_idx)
    )
    if not np.any(val_mask):
        raise ValueError("Validation split empty after applying temporal embargo.")

    max_val_time = float(t_target_arr[val_mask][-1])
    max_val_event_idx = int(end_idx_arr[val_mask][-1])

    test_mask = (
        (np.arange(n_seq) >= va_cut)
        & (t_start_arr > max_val_time + embargo_sec)
        & ((end_idx_arr - seq_len) > max_val_event_idx)
    )
    if not np.any(test_mask):
        raise ValueError("Test split empty after applying temporal embargo.")

    return {
        "train": (X_all[train_mask], y_all[train_mask]),
        "validation": (X_all[val_mask], y_all[val_mask]),
        "test": (X_all[test_mask], y_all[test_mask]),
    }


def run_cnn_forecast_benchmark(
    repo_root: Path | None = None,
    epochs: int = 12,
    seq_len: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    root = repo_root or resolve_repo_root(SCRIPT_DIR)
    cfg = PipelineConfig()
    data_path = root / "preprocessed_dl_temporal.csv"
    scaler_path = root / "dl_temporal_scaler.pkl"

    if not data_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError(
            "Missing 'preprocessed_dl_temporal.csv' or 'dl_temporal_scaler.pkl'. "
            "Run 'preprocess_dl_temporal.py' first; synthetic fallback is disabled."
        )

    df = pd.read_csv(data_path)
    with scaler_path.open("rb") as fh:
        scaler_info = pickle.load(fh)

    predictor_cols: list[str] = scaler_info["all_predictor_columns"]
    missing = [c for c in [*predictor_cols, "target_mw_raw", "origin_time_utc"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in '{data_path.name}': {missing}")

    timestamps = pd.to_datetime(df["origin_time_utc"]).astype("int64").values / 1e9
    splits = build_embargoed_causal_sequences(
        features=df[predictor_cols].values,
        targets_mw=df["target_mw_raw"].values,
        timestamps_sec=timestamps,
        seq_len=seq_len,
        mw_threshold=cfg.validation.target_mw_threshold,
        embargo_sec=cfg.validation.embargo_days * 86400.0,
    )

    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["validation"]
    X_te, y_te = splits["test"]

    # Fold-scoped sequence standardization strictly on X_tr
    mu = X_tr.mean(axis=(0, 1), keepdims=True)
    sd = np.where(X_tr.std(axis=(0, 1), keepdims=True) > 1e-8, X_tr.std(axis=(0, 1), keepdims=True), 1.0)
    X_tr_n = (X_tr - mu) / sd
    X_va_n = (X_va - mu) / sd
    X_te_n = (X_te - mu) / sd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientCNN1D(n_features=X_tr_n.shape[-1], seq_len=seq_len).to(device)
    n_pos = float(np.sum(y_tr == 1.0))
    n_neg = float(np.sum(y_tr == 0.0))
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(y_tr)),
        batch_size=64,
        shuffle=True,
    )

    best_val_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(torch.from_numpy(X_va_n).to(device))
            va_loss = float(criterion(va_logits, torch.from_numpy(y_va).to(device)).item())
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(model(torch.from_numpy(X_te_n).to(device))).cpu().numpy()

    results = {
        "model": "EfficientCNN1D_LeakFree",
        "smote_applied": False,
        "embargo_days": cfg.validation.embargo_days,
        "split_sizes": {"train": len(X_tr), "validation": len(X_va), "test": len(X_te)},
        "test_roc_auc": round(float(roc_auc_score(y_te, te_probs)), 6) if len(np.unique(y_te)) > 1 else None,
        "test_pr_auc": round(float(average_precision_score(y_te, te_probs)), 6) if len(np.unique(y_te)) > 1 else None,
        "test_brier": round(float(brier_score_loss(y_te, te_probs)), 6),
    }
    (SCRIPT_DIR / "cnn_1d_leakfree_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(run_cnn_forecast_benchmark(), indent=2))
