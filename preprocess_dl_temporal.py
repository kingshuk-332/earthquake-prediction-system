"""
Temporal-Aware DL Preprocessing for Seismic Prediction
=======================================================
Reads:  dataset_2.csv  (NEVER modified)
Writes: preprocessed_dl_temporal.csv  — DL-ready with temporal features

Key improvements over preprocessed_dl.csv:
  1. Rich temporal features (year, month, hour, day_of_week, day_of_year)
  2. Cyclical encoding for periodic features (sin/cos)
  3. Inter-event time gap (critical for seismic sequences)
  4. S-P travel time as engineered feature
  5. All features MinMax normalized [0, 1]
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────
#  0. LOAD
# ──────────────────────────────────────────────────────────────────
INPUT_FILE = 'dataset_2.csv'
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Loaded '{INPUT_FILE}': {df.shape[0]} rows × {df.shape[1]} columns")

# ──────────────────────────────────────────────────────────────────
#  1. PARSE DATETIMES & SORT CHRONOLOGICALLY
# ──────────────────────────────────────────────────────────────────
df['origin_time_utc'] = df['origin_time_utc'].str.replace(' UTC', '', regex=False)
df['origin_time_utc'] = pd.to_datetime(df['origin_time_utc'], errors='coerce')

# Drop rows with unparseable dates (critical for temporal model)
n_before = len(df)
df.dropna(subset=['origin_time_utc'], inplace=True)
print(f"[INFO] Dropped {n_before - len(df)} rows with invalid dates")

# Sort chronologically
df.sort_values('origin_time_utc', inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"[INFO] Sorted chronologically: {df['origin_time_utc'].min()} → {df['origin_time_utc'].max()}")

# ──────────────────────────────────────────────────────────────────
#  2. EXTRACT TEMPORAL FEATURES
# ──────────────────────────────────────────────────────────────────
print("\n── Extracting Temporal Features ──")

# Raw temporal components
df['year_raw'] = df['year']
df['month'] = df['origin_time_utc'].dt.month
df['day_of_week'] = df['origin_time_utc'].dt.dayofweek   # 0=Monday
df['hour'] = df['origin_time_utc'].dt.hour
df['day_of_year'] = df['origin_time_utc'].dt.dayofyear

# Cyclical encoding (captures periodicity — e.g., Dec↔Jan are close)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Inter-event time gap (hours since previous earthquake — crucial for seismicity)
df['time_diff_hours'] = df['origin_time_utc'].diff().dt.total_seconds() / 3600.0
df['time_diff_hours'].fillna(0, inplace=True)  # First event has no predecessor
# Cap extreme gaps at 99th percentile to avoid outlier distortion
cap = df['time_diff_hours'].quantile(0.99)
df['time_diff_hours'] = df['time_diff_hours'].clip(upper=cap)

# Log-transformed time gap (compress the long tail)
df['log_time_diff'] = np.log1p(df['time_diff_hours'])

print(f"  Temporal features added: month_sin/cos, dow_sin/cos, hour_sin/cos, doy_sin/cos")
print(f"  Inter-event gap: mean={df['time_diff_hours'].mean():.1f}h, max={df['time_diff_hours'].max():.1f}h")

# ──────────────────────────────────────────────────────────────────
#  3. ENGINEERED SEISMIC FEATURES
# ──────────────────────────────────────────────────────────────────
print("\n── Engineering Seismic Features ──")

# S-P travel time (proxy for epicentral distance, very important for seismology)
df['sp_travel_time'] = df['s_arrival_sample'] - df['p_arrival_sample']

# Depth-magnitude interaction
df['depth_mag_ratio'] = df['Depth_km'] / (df['Magnitude_Value'] + 0.01)

# Energy proxy (log10 of seismic energy from magnitude)
# log10(E) ≈ 1.5*M + 4.8 (Gutenberg-Richter)
df['log_energy'] = 1.5 * df['Magnitude_Value'] + 4.8

print(f"  sp_travel_time, depth_mag_ratio, log_energy added")

# ──────────────────────────────────────────────────────────────────
#  4. SELECT FINAL FEATURE SET
# ──────────────────────────────────────────────────────────────────
# Numeric features for the model
numeric_features = [
    # Core seismic parameters
    'Lat', 'Long', 'Depth_km', 'Magnitude_Value',
    # Receiver/station parameters
    'receiver_latitude', 'receiver_longitude', 'receiver_elevation_m',
    # Waveform features
    'p_arrival_sample', 's_arrival_sample', 'snr_db',
    'epicentral_dist_km', 'sp_travel_time',
    # Temporal features (cyclical)
    'year_raw', 'month_sin', 'month_cos',
    'dow_sin', 'dow_cos', 'hour_sin', 'hour_cos',
    'doy_sin', 'doy_cos',
    # Inter-event dynamics
    'time_diff_hours', 'log_time_diff',
    # Engineered
    'depth_mag_ratio', 'log_energy',
]

# Categorical features (will be one-hot encoded)
categorical_features = ['Magnitude_Type', 'source_magnitude_type', 'receiver_code']

# Build DL dataframe
df_dl = df[numeric_features].copy()

# ──────────────────────────────────────────────────────────────────
#  5. HANDLE MISSING VALUES
# ──────────────────────────────────────────────────────────────────
for col in df_dl.columns:
    n_null = df_dl[col].isnull().sum()
    if n_null > 0:
        median_val = df_dl[col].median()
        df_dl[col].fillna(median_val, inplace=True)
        print(f"  Filled {n_null} NaN in '{col}' with median={median_val:.4f}")

# ──────────────────────────────────────────────────────────────────
#  6. ONE-HOT ENCODE CATEGORICALS
# ──────────────────────────────────────────────────────────────────
for cat_col in categorical_features:
    if cat_col in df.columns:
        col_data = df[cat_col].str.upper().str.strip()
        dummies = pd.get_dummies(col_data, prefix=cat_col, drop_first=False, dtype=int)
        df_dl = pd.concat([df_dl, dummies], axis=1)
        print(f"  One-hot encoded '{cat_col}': {dummies.shape[1]} categories")

# ──────────────────────────────────────────────────────────────────
#  7. MIN-MAX NORMALIZE ALL FEATURES [0, 1]
# ──────────────────────────────────────────────────────────────────
all_numeric_cols = df_dl.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
df_dl[all_numeric_cols] = scaler.fit_transform(df_dl[all_numeric_cols])

# Ensure no NaN remain after scaling (edge cases from one-hot or gap calc)
df_dl.fillna(0, inplace=True)

# Save scaler info for reference
import pickle
scaler_info = {
    'scaler': scaler,
    'feature_names': all_numeric_cols,
    'magnitude_min': df['Magnitude_Value'].min(),
    'magnitude_max': df['Magnitude_Value'].max(),
}
with open('dl_temporal_scaler.pkl', 'wb') as f:
    pickle.dump(scaler_info, f)

print(f"\n[INFO] MinMaxScaler applied to {len(all_numeric_cols)} columns")
print(f"[INFO] Magnitude raw range: [{scaler_info['magnitude_min']}, {scaler_info['magnitude_max']}]")

# ──────────────────────────────────────────────────────────────────
#  8. SAVE
# ──────────────────────────────────────────────────────────────────
OUTPUT_FILE = 'preprocessed_dl_temporal.csv'
df_dl.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*70}")
print(f"OUTPUT: {OUTPUT_FILE}")
print(f"Shape: {df_dl.shape[0]} rows × {df_dl.shape[1]} columns")
print(f"Features ({df_dl.shape[1]}):")
for i, col in enumerate(df_dl.columns):
    print(f"  {i+1:>2}. {col}")
print(f"Value range: [{df_dl.min().min():.4f}, {df_dl.max().max():.4f}]")
print(f"Any NaN: {df_dl.isnull().any().any()}")
print(f"{'='*70}")

# Class distribution preview
MAG_COL_IDX = list(df_dl.columns).index('Magnitude_Value')
raw_mag_min = scaler_info['magnitude_min']
raw_mag_max = scaler_info['magnitude_max']
threshold_scaled = (4.5 - raw_mag_min) / (raw_mag_max - raw_mag_min)
n_pos = (df_dl['Magnitude_Value'] >= threshold_scaled).sum()
n_neg = (df_dl['Magnitude_Value'] < threshold_scaled).sum()
print(f"\nClass distribution (M>=4.5):")
print(f"  Positive: {n_pos} ({n_pos/len(df_dl)*100:.1f}%)")
print(f"  Negative: {n_neg} ({n_neg/len(df_dl)*100:.1f}%)")
print(f"  Imbalance ratio: 1:{n_neg/max(n_pos,1):.1f}")
