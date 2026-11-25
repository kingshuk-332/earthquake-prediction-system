"""
Pure 1D-CNN v2 for Seismic Activity Forecasting
=================================================
Framework: PyTorch (CUDA GPU Acceleration)
Region:    North-East India & Indo-Burma Subduction Zone
Dataset:   preprocessed_dl_temporal.csv (44 features)
Target:    Forecast significant earthquake (M >= 4.5) in next event

LESSONS FROM v1:
  - v1 had 735K params → severe overfitting (train AUC 0.90, val AUC 0.48)
  - Model collapsed at epoch 6, never recovered
  
v2 IMPROVEMENTS:
  1. MUCH smaller model (~30K params) to prevent overfitting on 5,651 samples
  2. Depthwise-separable convolutions (more parameter efficient)
  3. Dilated convolutions (larger receptive field without more params)
  4. Stronger regularization (higher dropout, more weight decay)
  5. Cosine annealing LR with warm restarts (better exploration)
  6. Label smoothing in focal loss
  7. Mixup augmentation on training data

Architecture (Efficient 1D-CNN):
  1. Input: (Batch, 15, 44) -> Transposed to (Batch, 44, 15)
  2. Conv1d(44->32, k=1, stride=1) channel reduction + BN + ReLU
  3. Dilated Conv Block: Conv1d(32->32, k=3, dilation=1, pad=1) + BN + GELU + Drop(0.2)
  4. Dilated Conv Block: Conv1d(32->48, k=3, dilation=2, pad=2) + BN + GELU + Drop(0.25)
  5. Dilated Conv Block: Conv1d(48->64, k=3, dilation=4, pad=4) + BN + GELU + Drop(0.3)
     (Effective receptive field covers entire 15-step window)
  6. Global Average Pool -> (Batch, 64)
  7. Classifier: Linear(64->32) + BN + GELU + Drop(0.4) -> Linear(32->1)
  8. Focal Loss (label smoothing) + AdamW + Cosine Annealing + Early Stopping
"""

import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, accuracy_score, precision_score, recall_score,
    matthews_corrcoef
)
from imblearn.over_sampling import SMOTE

# ──────────────────────────────────────────────────────────────────
#  0. SEED & DEVICE
# ──────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    print("=" * 70)
    print(f"DEVICE: CUDA GPU -> {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 70)
else:
    device = torch.device('cpu')
    print("=" * 70)
    print("DEVICE: CPU (CUDA not available)")
    print("=" * 70)

# ──────────────────────────────────────────────────────────────────
#  1. LOAD DATASET (Prefers Noisy Augmented Dataset in cnn/ Folder)
# ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

candidate_files = [
    os.path.join(SCRIPT_DIR, 'preprocessed_dl_temporal_noisy.csv'),
    'cnn/preprocessed_dl_temporal_noisy.csv',
    'preprocessed_dl_temporal_noisy.csv',
    os.path.join(SCRIPT_DIR, 'preprocessed_dl_temporal.csv'),
    os.path.join(PROJECT_ROOT, 'preprocessed_dl_temporal.csv'),
    'preprocessed_dl_temporal.csv',
    os.path.join('..', 'preprocessed_dl_temporal.csv')
]

DATA_FILE = None
for cf in candidate_files:
    if os.path.exists(cf):
        DATA_FILE = cf
        break

if DATA_FILE is None:
    raise FileNotFoundError("Dataset not found! Expected 'preprocessed_dl_temporal_noisy.csv' in cnn/ folder.")

df = pd.read_csv(DATA_FILE)
FEATURE_COLS = [c for c in df.columns if c != 'target']
N_FEATURES = len(FEATURE_COLS)
print(f"\n[INFO] Loaded '{DATA_FILE}': {df.shape[0]} rows × {df.shape[1]} columns ({N_FEATURES} features)")

# Target definition (preserves ground truth target if already computed)
scaler_file = 'dl_temporal_scaler.pkl'
if not os.path.exists(scaler_file):
    if os.path.exists(os.path.join(PROJECT_ROOT, scaler_file)):
        scaler_file = os.path.join(PROJECT_ROOT, scaler_file)
    elif os.path.exists(os.path.join('..', scaler_file)):
        scaler_file = os.path.join('..', scaler_file)

if os.path.exists(scaler_file):
    with open(scaler_file, 'rb') as f:
        scaler_info = pickle.load(f)
    MIN_MAG = scaler_info['magnitude_min']
    MAX_MAG = scaler_info['magnitude_max']
else:
    MIN_MAG, MAX_MAG = 1.5, 8.6

RAW_MAG_THRESHOLD = 4.5
SCALED_MAG_THRESHOLD = (RAW_MAG_THRESHOLD - MIN_MAG) / (MAX_MAG - MIN_MAG)

if 'target' not in df.columns:
    df['target'] = (df['Magnitude_Value'] >= SCALED_MAG_THRESHOLD).astype(int)

n_pos = df['target'].sum()
n_neg = len(df) - n_pos
print(f"[TARGET] M >= {RAW_MAG_THRESHOLD} (scaled >= {SCALED_MAG_THRESHOLD:.4f})")
print(f"  Positive: {n_pos} ({n_pos/len(df)*100:.1f}%) | Negative: {n_neg} ({n_neg/len(df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────
#  2. CREATE SEQUENCES
# ──────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 15
features_data = df[FEATURE_COLS].values
targets_data = df['target'].values

def create_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = create_sequences(features_data, targets_data, SEQUENCE_LENGTH)
print(f"[INFO] Sequences: X = {X.shape}, y = {y.shape}")

# Chronological split
split_70 = int(len(X) * 0.70)
split_80 = int(len(X) * 0.80)

X_train_orig, y_train_orig = X[:split_70], y[:split_70]
X_val, y_val = X[split_70:split_80], y[split_70:split_80]
X_test, y_test = X[split_80:], y[split_80:]

print(f"\n[INFO] Chronological Splits:")
print(f"  Train: {len(X_train_orig)} | Pos: {int(y_train_orig.sum())} ({y_train_orig.mean()*100:.1f}%)")
print(f"  Val:   {len(X_val)} | Pos: {int(y_val.sum())} ({y_val.mean()*100:.1f}%)")
print(f"  Test:  {len(X_test)} | Pos: {int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────
#  3. SMOTE
# ──────────────────────────────────────────────────────────────────
n_train = len(X_train_orig)
X_train_flat = X_train_orig.reshape(n_train, -1)
smote = SMOTE(random_state=SEED, k_neighbors=min(5, int(y_train_orig.sum()) - 1))
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train_orig)

X_train = X_train_resampled.reshape(-1, SEQUENCE_LENGTH, N_FEATURES)
y_train = y_train_resampled

shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

print(f"\n[SMOTE] Balanced: {len(X_train)} samples (Pos: {int(y_train.sum())}, Neg: {int(len(y_train)-y_train.sum())})")

# ──────────────────────────────────────────────────────────────────
#  4. DATA LOADERS
# ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64  # Larger batch for more stable gradients

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ──────────────────────────────────────────────────────────────────
#  5. MODEL: EFFICIENT DILATED 1D-CNN
# ──────────────────────────────────────────────────────────────────
class DilatedConvBlock(nn.Module):
    """Conv1d with dilation for larger receptive field without extra parameters."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # 'same' padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        # Residual projection if channel sizes differ
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.dropout(F.gelu(self.bn(self.conv(x))))
        # Ensure same length for residual (handle edge cases from dilation)
        if out.shape[-1] != residual.shape[-1]:
            residual = residual[..., :out.shape[-1]]
        return out + residual


class EfficientCNN1D(nn.Module):
    """
    Lean Dilated 1D-CNN designed for small seismic datasets.
    
    Key design choices:
    - Channel reduction first (44->32) to limit parameters
    - Dilated convolutions (d=1,2,4) give receptive field of 15 = full sequence
    - Residual connections for stable gradients
    - GELU activation (smoother than ReLU, better for small datasets)
    - Global Average Pooling (not flatten) avoids overfitting
    - Minimal but effective classifier head
    
    Total params: ~35K (vs 735K in v1, vs 187K in hybrid)
    """
    def __init__(self, n_features, seq_len=15):
        super().__init__()
        
        # Channel reduction: 44 -> 32 (limits parameter explosion in later layers)
        self.channel_reduce = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        
        # Dilated conv stack (increasing dilation = increasing receptive field)
        # dilation=1: sees 3 adjacent events
        # dilation=2: sees 5 events  
        # dilation=4: sees 9 events
        # Combined effective receptive field covers the full 15-step window
        self.dilated_block1 = DilatedConvBlock(32, 32, kernel_size=3, dilation=1, dropout=0.2)
        self.dilated_block2 = DilatedConvBlock(32, 48, kernel_size=3, dilation=2, dropout=0.25)
        self.dilated_block3 = DilatedConvBlock(48, 64, kernel_size=3, dilation=4, dropout=0.3)
        
        # Squeeze-and-Excitation (channel attention) for feature recalibration
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 64)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, 15, 44) -> transpose -> (batch, 44, 15)
        x = x.transpose(1, 2)
        
        # Channel reduction
        x = self.channel_reduce(x)  # (batch, 32, 15)
        
        # Dilated conv stack with residual connections
        x = self.dilated_block1(x)  # (batch, 32, 15)
        x = self.dilated_block2(x)  # (batch, 48, 15)
        x = self.dilated_block3(x)  # (batch, 64, 15)
        
        # Squeeze-and-Excitation (channel attention)
        se = self.global_avg_pool(x).squeeze(-1)  # (batch, 64)
        se = F.gelu(self.se_fc1(se))               # (batch, 16)
        se = torch.sigmoid(self.se_fc2(se))        # (batch, 64)
        x = x * se.unsqueeze(-1)                   # Channel reweighting
        
        # Global average pooling
        x = self.global_avg_pool(x).squeeze(-1)    # (batch, 64)
        
        # Classification
        logits = self.classifier(x).squeeze(-1)    # (batch,)
        return logits


# ──────────────────────────────────────────────────────────────────
#  6. LOSS, OPTIMIZER, SCHEDULER
# ──────────────────────────────────────────────────────────────────
model = EfficientCNN1D(n_features=N_FEATURES, seq_len=SEQUENCE_LENGTH).to(device)

# Initialize output bias
pos_ratio = y_train.mean()
initial_bias = np.log(pos_ratio / (1 - pos_ratio + 1e-8))
with torch.no_grad():
    model.classifier[-1].bias.fill_(initial_bias)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[INIT] Output bias: {initial_bias:.4f}")
print(f"[MODEL] Efficient Dilated 1D-CNN: {trainable_params:,} trainable parameters")
print(f"  (v1 had 735,297 params — this is {trainable_params/735297*100:.0f}% of v1)")
print(f"\n{model}\n")


class FocalLossSmoothed(nn.Module):
    """Focal Loss with optional label smoothing for regularization."""
    def __init__(self, gamma=2.0, alpha=0.5, smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets_smooth, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


criterion = FocalLossSmoothed(gamma=2.0, alpha=0.6, smoothing=0.05)

# AdamW with stronger weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)

# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=1e-6
)

# ──────────────────────────────────────────────────────────────────
#  7. MIXUP AUGMENTATION
# ──────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    """Mixup augmentation: creates convex combinations of training samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

# ──────────────────────────────────────────────────────────────────
#  8. TRAINING LOOP
# ──────────────────────────────────────────────────────────────────
MAX_EPOCHS = 200
PATIENCE = 35  # More patience with cosine annealing

script_dir = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(script_dir, 'cnn_1d_v2_best.pth')
FINAL_MODEL_PATH = os.path.join(script_dir, 'cnn_1d_v2_final.pth')

history = {
    'train_loss': [], 'val_loss': [],
    'train_auc': [], 'val_auc': [],
    'train_ap': [], 'val_ap': [],
    'lr': []
}

best_val_auc = 0.0
patience_counter = 0
best_epoch = 0

print("=" * 70)
print("TRAINING EFFICIENT DILATED 1D-CNN (v2)")
print("=" * 70)

for epoch in range(MAX_EPOCHS):
    # ── Training ──
    model.train()
    train_loss = 0.0
    train_preds_all, train_labels_all = [], []

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Apply Mixup augmentation (50% of batches)
        if np.random.random() < 0.5:
            batch_X, batch_y = mixup_data(batch_X, batch_y, alpha=0.3)
        
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * len(batch_y)
        train_preds_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        train_labels_all.append(batch_y.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_preds = np.concatenate(train_preds_all)
    train_labels = np.concatenate(train_labels_all)
    
    # For AUC, round labels (mixup creates non-integer labels)
    train_labels_rounded = np.round(train_labels).astype(int)
    if len(np.unique(train_labels_rounded)) > 1:
        train_auc = roc_auc_score(train_labels_rounded, train_preds)
        train_ap = average_precision_score(train_labels_rounded, train_preds)
    else:
        train_auc = 0.5
        train_ap = 0.5

    # ── Validation ──
    model.eval()
    val_loss = 0.0
    val_preds_all, val_labels_all = [], []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * len(batch_y)
            val_preds_all.append(torch.sigmoid(logits).cpu().numpy())
            val_labels_all.append(batch_y.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_preds = np.concatenate(val_preds_all)
    val_labels = np.concatenate(val_labels_all)
    val_auc = roc_auc_score(val_labels, val_preds)
    val_ap = average_precision_score(val_labels, val_preds)

    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_auc'].append(train_auc)
    history['val_auc'].append(val_auc)
    history['train_ap'].append(train_ap)
    history['val_ap'].append(val_ap)
    history['lr'].append(current_lr)

    scheduler.step()

    # Early stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch + 1
        patience_counter = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        marker = " ★ BEST"
    else:
        patience_counter += 1
        marker = ""

    if (epoch + 1) % 10 == 0 or epoch == 0 or marker:
        print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | "
              f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} AP: {val_ap:.4f} | "
              f"LR: {current_lr:.2e}{marker}")

    if patience_counter >= PATIENCE:
        print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs. Best: epoch {best_epoch} (val_auc={best_val_auc:.4f})")
        break

torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"\n[SAVED] Best: '{BEST_MODEL_PATH}' (epoch {best_epoch})")
print(f"[SAVED] Final: '{FINAL_MODEL_PATH}' (epoch {epoch+1})")

# ──────────────────────────────────────────────────────────────────
#  9. EVALUATION (Best Model)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATION ON TEST SET (Best Model)")
print("=" * 70)

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
model.eval()

test_preds_all, test_labels_all = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        test_preds_all.append(torch.sigmoid(logits).cpu().numpy())
        test_labels_all.append(batch_y.numpy())

test_probs = np.concatenate(test_preds_all)
test_labels = np.concatenate(test_labels_all)

# ── Threshold Optimization ──
print("\n[THRESHOLD OPTIMIZATION] Scanning validation set...")

val_preds_opt = []
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        val_preds_opt.append(torch.sigmoid(logits).cpu().numpy())
val_probs_opt = np.concatenate(val_preds_opt)

thresholds = np.arange(0.01, 0.99, 0.01)
best_f1 = 0
best_threshold = 0.5
threshold_results = []

for t in thresholds:
    preds_t = (val_probs_opt >= t).astype(int)
    if preds_t.sum() == 0 or (1 - preds_t).sum() == 0:
        continue
    f1 = f1_score(y_val, preds_t, zero_division=0)
    prec = precision_score(y_val, preds_t, zero_division=0)
    rec = recall_score(y_val, preds_t, zero_division=0)
    threshold_results.append({'threshold': t, 'f1': f1, 'precision': prec, 'recall': rec})
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"[BEST THRESHOLD] {best_threshold:.2f} (Val F1 = {best_f1:.4f})")

# ── Test Metrics ──
test_preds_opt = (test_probs >= best_threshold).astype(int)
test_preds_default = (test_probs >= 0.5).astype(int)

test_auc = roc_auc_score(test_labels, test_probs)
test_ap = average_precision_score(test_labels, test_probs)
test_f1 = f1_score(test_labels, test_preds_opt, zero_division=0)
test_prec = precision_score(test_labels, test_preds_opt, zero_division=0)
test_rec = recall_score(test_labels, test_preds_opt, zero_division=0)
test_acc = accuracy_score(test_labels, test_preds_opt)
test_mcc = matthews_corrcoef(test_labels, test_preds_opt)

cm = confusion_matrix(test_labels, test_preds_opt)
tn, fp, fn, tp = cm.ravel()

print(f"\n{'─' * 50}")
print(f"TEST RESULTS (threshold = {best_threshold:.2f})")
print(f"{'─' * 50}")
print(f"  ROC-AUC:            {test_auc:.4f}")
print(f"  Average Precision:  {test_ap:.4f}")
print(f"  F1 Score:           {test_f1:.4f}")
print(f"  Precision:          {test_prec:.4f}")
print(f"  Recall (M>=4.5):    {test_rec:.4f} ({tp}/{tp+fn} detected)")
print(f"  Accuracy:           {test_acc:.4f}")
print(f"  MCC:                {test_mcc:.4f}")
print(f"\n  Confusion Matrix:")
print(f"    TN={tn:4d}  FP={fp:4d}")
print(f"    FN={fn:4d}  TP={tp:4d}")
print(f"\n{classification_report(test_labels, test_preds_opt, target_names=['No Significant (M<4.5)', 'Significant (M>=4.5)'], zero_division=0)}")

# ──────────────────────────────────────────────────────────────────
#  10. SAVE RESULTS
# ──────────────────────────────────────────────────────────────────
results_path = os.path.join(script_dir, 'cnn_1d_results.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("PURE 1D-CNN v2 (EFFICIENT DILATED) - SEISMIC FORECASTING RESULTS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
    f.write(f"Framework: PyTorch {torch.__version__}\n")
    f.write(f"Dataset: {DATA_FILE}\n")
    f.write(f"Parameters: {trainable_params:,}\n\n")
    f.write(f"TRAINING:\n")
    f.write(f"  Best Epoch: {best_epoch}/{epoch+1}\n")
    f.write(f"  Best Val AUC: {best_val_auc:.4f}\n")
    f.write(f"  Optimal Threshold: {best_threshold:.2f}\n\n")
    f.write(f"TEST METRICS (threshold = {best_threshold:.2f}):\n")
    f.write(f"  ROC-AUC:           {test_auc:.4f}\n")
    f.write(f"  Average Precision: {test_ap:.4f}\n")
    f.write(f"  F1 Score:          {test_f1:.4f}\n")
    f.write(f"  Precision:         {test_prec:.4f}\n")
    f.write(f"  Recall:            {test_rec:.4f}\n")
    f.write(f"  Accuracy:          {test_acc:.4f}\n")
    f.write(f"  MCC:               {test_mcc:.4f}\n\n")
    f.write(f"CONFUSION MATRIX:\n")
    f.write(f"  TN={tn}  FP={fp}\n")
    f.write(f"  FN={fn}  TP={tp}\n\n")
    f.write(classification_report(test_labels, test_preds_opt,
            target_names=['No Significant (M<4.5)', 'Significant (M>=4.5)'], zero_division=0))

print(f"\n[SAVED] Results: '{results_path}'")

# ──────────────────────────────────────────────────────────────────
#  11. PLOTS
# ──────────────────────────────────────────────────────────────────
plots_dir = os.path.join(script_dir, 'plots_cnn_1d')
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Training History
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric, ylabel in [
    (axes[0], 'loss', 'Focal Loss'),
    (axes[1], 'auc', 'ROC-AUC'),
    (axes[2], 'ap', 'Average Precision')
]:
    ax.plot(history[f'train_{metric}'], label=f'Train', color='#2196F3', alpha=0.8)
    ax.plot(history[f'val_{metric}'], label=f'Val', color='#F44336', alpha=0.8)
    ax.axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.5, label=f'Best (ep {best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel}')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Pure 1D-CNN v2 Training History', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(test_labels, test_probs)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#2196F3', linewidth=2, label=f'Pure 1D-CNN v2 (AUC = {test_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5, label='Random')
ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Pure 1D-CNN v2 (Test Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: PR Curve
prec_vals, rec_vals, _ = precision_recall_curve(test_labels, test_probs)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_vals, prec_vals, color='#FF9800', linewidth=2, label=f'AP = {test_ap:.4f}')
ax.axhline(y=test_labels.mean(), color='gray', linestyle='--', alpha=0.5, label=f'Random ({test_labels.mean():.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve - Pure 1D-CNN v2 (Test Set)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, preds, title in [
    (axes[0], test_preds_opt, f'Optimized (t={best_threshold:.2f})'),
    (axes[1], test_preds_default, 'Default (t=0.50)')
]:
    cm_p = confusion_matrix(test_labels, preds)
    im = ax.imshow(cm_p, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - {title}')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm_p[i,j]:,}', ha='center', va='center',
                    color='white' if cm_p[i,j] > cm_p.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['M<4.5', 'M>=4.5'])
    ax.set_yticklabels(['M<4.5', 'M>=4.5'])
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Prediction Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(test_probs[test_labels == 0], bins=50, alpha=0.6, color='#4CAF50',
        label=f'Neg (n={int((test_labels==0).sum())})', density=True)
ax.hist(test_probs[test_labels == 1], bins=50, alpha=0.6, color='#F44336',
        label=f'Pos (n={int((test_labels==1).sum())})', density=True)
ax.axvline(x=best_threshold, color='purple', linestyle='--', linewidth=2, label=f'Threshold ({best_threshold:.2f})')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Prediction Distribution - Pure 1D-CNN v2')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'prediction_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: Threshold Optimization
if threshold_results:
    tr_df = pd.DataFrame(threshold_results)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tr_df['threshold'], tr_df['f1'], label='F1', color='#2196F3', linewidth=2)
    ax.plot(tr_df['threshold'], tr_df['precision'], label='Precision', color='#4CAF50', linewidth=1.5, alpha=0.7)
    ax.plot(tr_df['threshold'], tr_df['recall'], label='Recall', color='#F44336', linewidth=1.5, alpha=0.7)
    ax.axvline(x=best_threshold, color='purple', linestyle='--', linewidth=2, label=f'Best ({best_threshold:.2f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Optimization (Validation Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'threshold_optimization.png'), dpi=150, bbox_inches='tight')
    plt.close()

# Plot 7: LR Schedule
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history['lr'], color='#9C27B0', linewidth=1.5)
ax.axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.5, label=f'Best (ep {best_epoch})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule (Cosine Annealing Warm Restarts)')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'learning_rate.png'), dpi=150, bbox_inches='tight')
plt.close()

# Plot 8: Class Distribution
fig, ax = plt.subplots(figsize=(10, 5))
splits = ['Train\n(Original)', 'Train\n(SMOTE)', 'Validation', 'Test']
pos_counts = [int(y_train_orig.sum()), int(y_train.sum()), int(y_val.sum()), int(y_test.sum())]
neg_counts = [len(y_train_orig)-pos_counts[0], len(y_train)-pos_counts[1],
              len(y_val)-pos_counts[2], len(y_test)-pos_counts[3]]
x_pos = np.arange(len(splits))
w = 0.35
b1 = ax.bar(x_pos - w/2, neg_counts, w, label='M < 4.5', color='#4CAF50', alpha=0.8)
b2 = ax.bar(x_pos + w/2, pos_counts, w, label='M >= 4.5', color='#F44336', alpha=0.8)
ax.set_ylabel('Count')
ax.set_title('Class Distribution Across Splits')
ax.set_xticks(x_pos)
ax.set_xticklabels(splits)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h):,}',
                    ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"[SAVED] Plots: '{plots_dir}/'")

# ──────────────────────────────────────────────────────────────────
#  12. EVALUATION FOLDER (JSON + CSVs)
# ──────────────────────────────────────────────────────────────────
eval_dir = os.path.join(script_dir, 'evaluation_cnn_1d')
os.makedirs(eval_dir, exist_ok=True)

metrics_json = {
    'model': 'Pure 1D-CNN v2 (Efficient Dilated)',
    'framework': f'PyTorch {torch.__version__}',
    'device': str(device),
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
    'parameters': trainable_params,
    'dataset': DATA_FILE,
    'n_features': N_FEATURES,
    'sequence_length': SEQUENCE_LENGTH,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'best_epoch': best_epoch,
    'total_epochs': epoch + 1,
    'best_val_auc': float(best_val_auc),
    'optimal_threshold': float(best_threshold),
    'techniques': [
        'Gaussian sensor jitter augmentation (sigma=0.015) in cnn/preprocessed_dl_temporal_noisy.csv',
        'SMOTE oversampling',
        'Focal Loss (gamma=2.0, alpha=0.6, label_smoothing=0.05)',
        'Dilated convolutions (d=1,2,4)',
        'Squeeze-and-Excitation channel attention',
        'Residual connections',
        'Mixup augmentation (alpha=0.3)',
        'Cosine annealing warm restarts',
        'Gradient clipping (max_norm=1.0)'
    ],
    'test_metrics': {
        'roc_auc': float(test_auc),
        'average_precision': float(test_ap),
        'f1_score': float(test_f1),
        'precision': float(test_prec),
        'recall': float(test_rec),
        'accuracy': float(test_acc),
        'mcc': float(test_mcc),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
}

with open(os.path.join(eval_dir, 'metrics_summary.json'), 'w') as f:
    json.dump(metrics_json, f, indent=2)

pd.DataFrame(history).to_csv(os.path.join(eval_dir, 'training_history.csv'), index_label='epoch')

cr = classification_report(test_labels, test_preds_opt,
                          target_names=['No Significant (M<4.5)', 'Significant (M>=4.5)'],
                          output_dict=True, zero_division=0)
pd.DataFrame(cr).transpose().to_csv(os.path.join(eval_dir, 'classification_report.csv'))

if threshold_results:
    pd.DataFrame(threshold_results).to_csv(os.path.join(eval_dir, 'threshold_optimization.csv'), index=False)

pd.DataFrame({
    'true_label': test_labels,
    'predicted_prob': test_probs,
    'predicted_class_optimal': test_preds_opt,
    'predicted_class_default': test_preds_default
}).to_csv(os.path.join(eval_dir, 'test_predictions.csv'), index=False)

import shutil
shutil.copy2(BEST_MODEL_PATH, os.path.join(eval_dir, 'cnn_1d_v2_best.pth'))

print(f"[SAVED] Evaluation: '{eval_dir}/'")

# ──────────────────────────────────────────────────────────────────
#  13. FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PURE 1D-CNN v2 TRAINING COMPLETE")
print("=" * 70)
print(f"  Architecture:       Efficient Dilated 1D-CNN + SE Attention + Residual")
print(f"  Parameters:         {trainable_params:,}")
print(f"  Device:             {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  Best Epoch:         {best_epoch}/{epoch+1}")
print(f"  Best Val AUC:       {best_val_auc:.4f}")
print(f"  Optimal Threshold:  {best_threshold:.2f}")
print(f"  Test ROC-AUC:       {test_auc:.4f}")
print(f"  Test AP:            {test_ap:.4f}")
print(f"  Test F1:            {test_f1:.4f}")
print(f"  Test Recall(M>=4.5): {test_rec:.4f} ({tp}/{tp+fn} detected)")
print(f"  Test MCC:           {test_mcc:.4f}")
print("=" * 70)
