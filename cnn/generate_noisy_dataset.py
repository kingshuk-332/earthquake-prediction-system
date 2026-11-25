"""
Seismic Dataset Noise Injection Pipeline for Pure 1D-CNN
=========================================================
Purpose:
  Synthesize realistic seismic instrument noise, hypocenter location error,
  and phase picking uncertainty into the preprocessed dataset.
  
Geophysical & Sensor Rationale:
  - Epicenter coordinates (Lat, Long): Catalog location solutions in the Indo-Burma
    subduction zone have standard error ellipses of ±5 to 15 km (~0.05° - 0.15°).
  - Focal Depth (Depth_km): Subduction zone depths exhibit ±5 to 20 km uncertainty.
  - Magnitude (Magnitude_Value, log_energy): Catalog magnitudes carry standard
    estimation errors of ±0.1 to 0.2 magnitude units.
  - Phase Arrivals (p_arrival, s_arrival, sp_travel_time): Picker jitter of ±5-10 samples.
  - SNR (snr_db): Ambient cultural and environmental noise fluctuations.
  - Continuous Features: Injected with zero-mean Gaussian jitter N(0, sigma^2) where
    sigma = 0.015 (1.5% of the normalized scale), with hard boundary clipping to [0, 1].
  - Discrete One-Hot Encodings: Kept untouched to preserve categorical station/type validity.
  - Ground Truth Target: Computed from original true magnitude BEFORE noise injection,
    guaranteeing zero label noise or target corruption.

Output File:
  cnn/preprocessed_dl_temporal_noisy.csv
"""

import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────
#  0. PATH RESOLUTION
# ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

INPUT_CSV = 'preprocessed_dl_temporal.csv'
if not os.path.exists(INPUT_CSV):
    if os.path.exists(os.path.join(PROJECT_ROOT, INPUT_CSV)):
        INPUT_CSV = os.path.join(PROJECT_ROOT, INPUT_CSV)
    elif os.path.exists(os.path.join('..', INPUT_CSV)):
        INPUT_CSV = os.path.join('..', INPUT_CSV)
    else:
        raise FileNotFoundError(f"Input file '{INPUT_CSV}' not found!")

SCALER_FILE = 'dl_temporal_scaler.pkl'
if not os.path.exists(SCALER_FILE):
    if os.path.exists(os.path.join(PROJECT_ROOT, SCALER_FILE)):
        SCALER_FILE = os.path.join(PROJECT_ROOT, SCALER_FILE)
    elif os.path.exists(os.path.join('..', SCALER_FILE)):
        SCALER_FILE = os.path.join('..', SCALER_FILE)

OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'preprocessed_dl_temporal_noisy.csv')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots_cnn_1d')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
#  1. LOAD DATA & EXTRACT TRUE GROUND-TRUTH TARGET
# ──────────────────────────────────────────────────────────────────
df_orig = pd.read_csv(INPUT_CSV)
print("=" * 70)
print("SEISMIC NOISE INJECTION PIPELINE FOR 1D-CNN")
print("=" * 70)
print(f"[LOAD] Source: '{INPUT_CSV}' ({df_orig.shape[0]} rows x {df_orig.shape[1]} columns)")

if os.path.exists(SCALER_FILE):
    with open(SCALER_FILE, 'rb') as f:
        scaler_info = pickle.load(f)
    MIN_MAG = scaler_info['magnitude_min']
    MAX_MAG = scaler_info['magnitude_max']
else:
    MIN_MAG, MAX_MAG = 1.5, 8.6

RAW_MAG_THRESHOLD = 4.5
SCALED_MAG_THRESHOLD = (RAW_MAG_THRESHOLD - MIN_MAG) / (MAX_MAG - MIN_MAG)

# Calculate ground-truth binary target from TRUE unperturbed magnitude
true_target = (df_orig['Magnitude_Value'] >= SCALED_MAG_THRESHOLD).astype(int)
n_pos = true_target.sum()
n_neg = len(true_target) - n_pos
print(f"[TARGET] Ground truth M >= {RAW_MAG_THRESHOLD} (scaled >= {SCALED_MAG_THRESHOLD:.4f})")
print(f"  Positive (M >= 4.5): {n_pos} ({n_pos/len(true_target)*100:.1f}%) | Negative: {n_neg} ({n_neg/len(true_target)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────
#  2. IDENTIFY FEATURE SUBSETS
# ──────────────────────────────────────────────────────────────────
continuous_cols = [
    'Lat', 'Long', 'Depth_km', 'Magnitude_Value',
    'receiver_latitude', 'receiver_longitude', 'receiver_elevation_m',
    'p_arrival_sample', 's_arrival_sample', 'snr_db',
    'epicentral_dist_km', 'sp_travel_time', 'year_raw',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'time_diff_hours', 'log_time_diff', 'depth_mag_ratio', 'log_energy'
]

categorical_cols = [c for c in df_orig.columns if c not in continuous_cols and c != 'target']

print(f"\n[FEATURES] Total features: {len(df_orig.columns)}")
print(f"  Continuous physical/temporal features to receive noise: {len(continuous_cols)}")
print(f"  One-hot categorical columns kept exact (unperturbed):   {len(categorical_cols)}")

# ──────────────────────────────────────────────────────────────────
#  3. INJECT CONTROLLED GAUSSIAN NOISE
# ──────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
NOISE_SIGMA = 0.015  # 1.5% standard deviation on [0, 1] scale

df_noisy = df_orig.copy()

# Generate Gaussian noise matrix
noise_matrix = np.random.normal(0.0, NOISE_SIGMA, size=df_orig[continuous_cols].shape)

# Apply noise and clip strictly to valid physical bounds [0.0, 1.0]
df_noisy[continuous_cols] = np.clip(df_orig[continuous_cols].values + noise_matrix, 0.0, 1.0)

# Attach true target (ensures zero label distortion / zero leakage)
df_noisy['target'] = true_target

# Statistical audit
diff_matrix = np.abs(df_noisy[continuous_cols].values - df_orig[continuous_cols].values)
mean_err = diff_matrix.mean()
max_err = diff_matrix.max()

print(f"\n[NOISE INJECTION STATS]")
print(f"  Distribution:         Gaussian N(0, sigma^2) with sigma = {NOISE_SIGMA}")
print(f"  Mean Absolute Jitter: {mean_err:.4f} (~{mean_err*100:.2f}% of normalized scale)")
print(f"  Max Absolute Jitter:  {max_err:.4f} (~{max_err*100:.2f}% of normalized scale)")
print(f"  Value Range:          Min = {df_noisy[continuous_cols].values.min():.4f}, Max = {df_noisy[continuous_cols].values.max():.4f}")
print(f"  Missing Values:       {df_noisy.isnull().sum().sum()} NaNs")

# ──────────────────────────────────────────────────────────────────
#  4. SAVE NOISY DATASET TO CNN FOLDER
# ──────────────────────────────────────────────────────────────────
df_noisy.to_csv(OUTPUT_CSV, index=False)
print(f"\n[SAVED] Noisy dataset created successfully:")
print(f"  Path:  '{OUTPUT_CSV}'")
print(f"  Shape: {df_noisy.shape[0]} rows x {df_noisy.shape[1]} columns (44 features + 1 target)")

# ──────────────────────────────────────────────────────────────────
#  5. GENERATE COMPARISON VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────
key_features = ['Lat', 'Long', 'Depth_km', 'Magnitude_Value', 'snr_db', 'sp_travel_time']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(key_features):
    ax = axes[i]
    ax.hist(df_orig[col], bins=40, alpha=0.55, color='#2196F3', label='Original Clean', density=True)
    ax.hist(df_noisy[col], bins=40, alpha=0.55, color='#FF5722', label='Noisy (Augmented)', density=True)
    ax.set_title(f'{col} (Original vs. Noisy)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Normalized Value [0, 1]', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

plt.suptitle(f'Seismic Dataset Feature Distribution: Original vs. Gaussian Jitter (sigma={NOISE_SIGMA})',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
comp_plot_path = os.path.join(PLOTS_DIR, 'noise_distribution_comparison.png')
plt.savefig(comp_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[SAVED] Noise comparison plot: '{comp_plot_path}'")

# Residual error distribution plot
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(noise_matrix.flatten(), bins=80, color='#9C27B0', alpha=0.7, density=True, label=f'Injected Noise (sigma={NOISE_SIGMA})')
# Overlay theoretical Gaussian curve
x_grid = np.linspace(-4*NOISE_SIGMA, 4*NOISE_SIGMA, 200)
gaussian_pdf = (1.0 / (NOISE_SIGMA * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_grid / NOISE_SIGMA)**2)
ax.plot(x_grid, gaussian_pdf, 'r--', linewidth=2, label=f'Theoretical N(0, {NOISE_SIGMA}^2)')
ax.set_title('Injected Gaussian Sensor Jitter Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Delta (Perturbation on Normalized Scale)', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
error_plot_path = os.path.join(PLOTS_DIR, 'noise_residual_histogram.png')
plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[SAVED] Residual histogram plot: '{error_plot_path}'")

print("=" * 70)
print("NOISE GENERATION COMPLETE")
print("=" * 70)

