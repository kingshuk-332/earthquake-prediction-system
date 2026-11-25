"""
Earthquake Dataset Preprocessing Pipeline
==========================================
Reads: dataset_2.csv  (NEVER modified)
Writes:
  - preprocessed_cleaned.csv          -- general cleaning output
  - preprocessed_eda.csv              -- EDA-ready (enriched features)
  - preprocessed_ml.csv               -- ML-ready (encoded + scaled)
  - preprocessed_dl.csv               -- DL-ready (normalized for waveform models)
  - ml_scaler.pkl                     -- fitted StandardScaler (for inverse transforms)
  - ml_label_encoders.pkl             -- fitted LabelEncoders for categorical cols
  - preprocessing_report.txt          -- summary report

Author: Auto-generated preprocessing pipeline
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────
#  0. LOAD (original file is NEVER touched)
# ──────────────────────────────────────────────────────────────────
INPUT_FILE = 'dataset_2.csv'
df_raw = pd.read_csv(INPUT_FILE)
print(f"[INFO] Loaded '{INPUT_FILE}': {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

report_lines = []
def log(msg):
    print(msg)
    report_lines.append(msg)

log("=" * 70)
log("EARTHQUAKE DATASET PREPROCESSING REPORT")
log(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Source file: {INPUT_FILE} (NOT modified)")
log(f"Original shape: {df_raw.shape}")
log("=" * 70)

# ──────────────────────────────────────────────────────────────────
#  1. GENERAL CLEANING  →  preprocessed_cleaned.csv
# ──────────────────────────────────────────────────────────────────
log("\n── STEP 1: GENERAL CLEANING ──")
df = df_raw.copy()

# 1a. Parse datetime columns
df['origin_time_ist'] = df['origin_time_ist'].str.replace(' IST', '', regex=False)
df['origin_time_utc'] = df['origin_time_utc'].str.replace(' UTC', '', regex=False)
df['origin_time_ist'] = pd.to_datetime(df['origin_time_ist'], errors='coerce')
df['origin_time_utc'] = pd.to_datetime(df['origin_time_utc'], errors='coerce')
log(f"  Parsed datetime columns. Null IST: {df['origin_time_ist'].isnull().sum()}, Null UTC: {df['origin_time_utc'].isnull().sum()}")

# 1b. Drop exact duplicate rows (if any)
n_before = len(df)
df.drop_duplicates(inplace=True)
n_dropped_dup = n_before - len(df)
log(f"  Dropped {n_dropped_dup} exact duplicate rows")

# 1c. Strip whitespace from string columns
str_cols = df.select_dtypes(include='object').columns
for col in str_cols:
    df[col] = df[col].str.strip()
log(f"  Stripped whitespace from {len(str_cols)} string columns")

# 1d. Validate numeric ranges
invalid_lat = (df['Lat'] < -90) | (df['Lat'] > 90)
invalid_lon = (df['Long'] < -180) | (df['Long'] > 180)
invalid_depth = df['Depth_km'] < 0
invalid_mag = (df['Magnitude_Value'] < 0) | (df['Magnitude_Value'] > 12)
invalid_snr = df['snr_db'] < 0

n_invalid = invalid_lat.sum() + invalid_lon.sum() + invalid_depth.sum() + invalid_mag.sum() + invalid_snr.sum()
log(f"  Invalid values found — Lat:{invalid_lat.sum()}, Lon:{invalid_lon.sum()}, "
    f"Depth:{invalid_depth.sum()}, Mag:{invalid_mag.sum()}, SNR:{invalid_snr.sum()}")

# Replace invalid values with NaN instead of dropping rows
df.loc[invalid_lat, 'Lat'] = np.nan
df.loc[invalid_lon, 'Long'] = np.nan
df.loc[invalid_depth, 'Depth_km'] = np.nan
df.loc[invalid_mag, 'Magnitude_Value'] = np.nan
df.loc[invalid_snr, 'snr_db'] = np.nan

# 1e. Ensure source_* and Lat/Long/Depth/Magnitude consistency
lat_mismatch = (df['Lat'] != df['source_latitude']).sum()
lon_mismatch = (df['Long'] != df['source_longitude']).sum()
depth_mismatch = (df['Depth_km'] != df['source_depth_km']).sum()
mag_mismatch = (df['Magnitude_Value'] != df['source_magnitude']).sum()
log(f"  Source column mismatches — Lat:{lat_mismatch}, Lon:{lon_mismatch}, "
    f"Depth:{depth_mismatch}, Mag:{mag_mismatch}")

# 1f. Drop redundant columns (source_* duplicates + composite 'Magnitude' string + 'Region' = State+Country)
redundant_cols = ['source_latitude', 'source_longitude', 'source_depth_km',
                  'source_magnitude', 'Magnitude']
df.drop(columns=redundant_cols, inplace=True)
log(f"  Dropped {len(redundant_cols)} redundant columns: {redundant_cols}")

# 1g. Standardize Magnitude_Type and source_magnitude_type
df['Magnitude_Type'] = df['Magnitude_Type'].str.upper().str.strip()
df['source_magnitude_type'] = df['source_magnitude_type'].str.upper().str.strip()
log(f"  Standardized magnitude types: {df['Magnitude_Type'].unique().tolist()}")

# 1h. Sort by time
df.sort_values('origin_time_utc', inplace=True)
df.reset_index(drop=True, inplace=True)
log(f"  Sorted by origin_time_utc")

log(f"  Cleaned shape: {df.shape}")
df.to_csv('preprocessed_cleaned.csv', index=False)
log(f"  Saved → preprocessed_cleaned.csv")

# ──────────────────────────────────────────────────────────────────
#  2. EDA-READY  →  preprocessed_eda.csv
# ──────────────────────────────────────────────────────────────────
log("\n── STEP 2: EDA-READY ENRICHMENT ──")
df_eda = df.copy()

# 2a. Extract temporal features
df_eda['month'] = df_eda['origin_time_utc'].dt.month
df_eda['day_of_week'] = df_eda['origin_time_utc'].dt.dayofweek  # 0=Mon
df_eda['hour'] = df_eda['origin_time_utc'].dt.hour
df_eda['day_of_year'] = df_eda['origin_time_utc'].dt.dayofyear
df_eda['decade'] = (df_eda['year'] // 10) * 10
log(f"  Added temporal features: month, day_of_week, hour, day_of_year, decade")

# 2b. Magnitude bins
def magnitude_category(mag):
    if pd.isna(mag):
        return 'Unknown'
    elif mag < 3.0:
        return 'Minor (<3)'
    elif mag < 4.0:
        return 'Light (3-4)'
    elif mag < 5.0:
        return 'Moderate (4-5)'
    elif mag < 6.0:
        return 'Strong (5-6)'
    elif mag < 7.0:
        return 'Major (6-7)'
    else:
        return 'Great (7+)'

df_eda['magnitude_category'] = df_eda['Magnitude_Value'].apply(magnitude_category)
log(f"  Added magnitude_category: {df_eda['magnitude_category'].value_counts().to_dict()}")

# 2c. Depth classification
def depth_class(d):
    if pd.isna(d):
        return 'Unknown'
    elif d <= 70:
        return 'Shallow (0-70km)'
    elif d <= 300:
        return 'Intermediate (70-300km)'
    else:
        return 'Deep (300+km)'

df_eda['depth_class'] = df_eda['Depth_km'].apply(depth_class)
log(f"  Added depth_class: {df_eda['depth_class'].value_counts().to_dict()}")

# 2d. S-P travel time (proxy for distance)
df_eda['sp_travel_time_samples'] = df_eda['s_arrival_sample'] - df_eda['p_arrival_sample']
log(f"  Added sp_travel_time_samples")

# 2e. Log-transformed magnitude (for visualization)
df_eda['log_magnitude'] = np.log1p(df_eda['Magnitude_Value'])
df_eda['log_depth'] = np.log1p(df_eda['Depth_km'])
log(f"  Added log_magnitude, log_depth")

# 2f. Distance between source and receiver (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

df_eda['src_recv_dist_km'] = haversine(
    df_eda['Lat'].values, df_eda['Long'].values,
    df_eda['receiver_latitude'].values, df_eda['receiver_longitude'].values
)
log(f"  Added src_recv_dist_km (Haversine)")

df_eda.to_csv('preprocessed_eda.csv', index=False)
log(f"  EDA shape: {df_eda.shape}")
log(f"  Saved → preprocessed_eda.csv")

# ──────────────────────────────────────────────────────────────────
#  3. ML-READY  →  preprocessed_ml.csv
# ──────────────────────────────────────────────────────────────────
log("\n── STEP 3: ML-READY (ENCODED + SCALED) ──")
df_ml = df_eda.copy()

# 3a. Drop identifier / free-text columns not useful for ML
drop_for_ml = ['origin_time_ist', 'origin_time_utc', 'source_id', 'trace_name',
               'Region', 'Location', 'Magnitude']
# Filter to only columns that exist
drop_for_ml = [c for c in drop_for_ml if c in df_ml.columns]
df_ml.drop(columns=drop_for_ml, inplace=True)
log(f"  Dropped non-ML columns: {drop_for_ml}")

# 3b. Handle remaining NaN (fill numeric with median, categorical with mode)
numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_ml.select_dtypes(include='object').columns.tolist()

for col in numeric_cols:
    n_null = df_ml[col].isnull().sum()
    if n_null > 0:
        median_val = df_ml[col].median()
        df_ml[col].fillna(median_val, inplace=True)
        log(f"  Filled {n_null} NaN in '{col}' with median={median_val:.4f}")

for col in cat_cols:
    n_null = df_ml[col].isnull().sum()
    if n_null > 0:
        mode_val = df_ml[col].mode()[0]
        df_ml[col].fillna(mode_val, inplace=True)
        log(f"  Filled {n_null} NaN in '{col}' with mode='{mode_val}'")

# 3c. Label-encode categorical columns
from sklearn.preprocessing import LabelEncoder, StandardScaler

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le
    log(f"  Label-encoded '{col}': {len(le.classes_)} classes")

with open('ml_label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
log(f"  Saved label encoders → ml_label_encoders.pkl")

# 3d. Detect and handle outliers using IQR method (cap instead of remove)
outlier_report = {}
numeric_cols_updated = df_ml.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols_updated:
    Q1 = df_ml[col].quantile(0.25)
    Q3 = df_ml[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df_ml[col] < lower) | (df_ml[col] > upper)).sum()
    if n_outliers > 0:
        outlier_report[col] = n_outliers
        df_ml[col] = df_ml[col].clip(lower=lower, upper=upper)

if outlier_report:
    log(f"  Outliers capped (IQR method): {outlier_report}")
else:
    log(f"  No outliers found via IQR method")

# 3e. Feature scaling (StandardScaler)
scaler = StandardScaler()
scale_cols = [c for c in numeric_cols_updated if c not in ['year']]
df_ml[scale_cols] = scaler.fit_transform(df_ml[scale_cols])

with open('ml_scaler.pkl', 'wb') as f:
    pickle.dump({'scaler': scaler, 'columns': scale_cols}, f)
log(f"  StandardScaler applied to {len(scale_cols)} columns")
log(f"  Saved scaler → ml_scaler.pkl")

df_ml.to_csv('preprocessed_ml.csv', index=False)
log(f"  ML shape: {df_ml.shape}")
log(f"  Saved → preprocessed_ml.csv")

# ──────────────────────────────────────────────────────────────────
#  4. DL-READY  →  preprocessed_dl.csv
# ──────────────────────────────────────────────────────────────────
log("\n── STEP 4: DL-READY (NORMALIZED FOR WAVEFORM MODELS) ──")
df_dl = df.copy()

# 4a. Keep only columns relevant for deep learning seismic models
dl_keep_cols = [
    'Lat', 'Long', 'Depth_km', 'Magnitude_Value', 'Magnitude_Type',
    'source_magnitude_type', 'receiver_code',
    'receiver_latitude', 'receiver_longitude', 'receiver_elevation_m',
    'p_arrival_sample', 's_arrival_sample', 'snr_db',
    'epicentral_dist_km'
]
dl_keep_cols = [c for c in dl_keep_cols if c in df_dl.columns]
df_dl = df_dl[dl_keep_cols].copy()
log(f"  Kept {len(dl_keep_cols)} DL-relevant columns")

# 4b. Fill any NaN with median for numeric
for col in df_dl.select_dtypes(include=[np.number]).columns:
    n_null = df_dl[col].isnull().sum()
    if n_null > 0:
        df_dl[col].fillna(df_dl[col].median(), inplace=True)
        log(f"  Filled {n_null} NaN in '{col}' with median")

# 4c. One-hot encode categorical columns (better for DL than label encoding)
cat_dl = df_dl.select_dtypes(include='object').columns.tolist()
df_dl = pd.get_dummies(df_dl, columns=cat_dl, drop_first=False, dtype=int)
log(f"  One-hot encoded: {cat_dl} → {df_dl.shape[1]} total columns")

# 4d. Min-Max normalization (0-1 range, better for neural networks)
from sklearn.preprocessing import MinMaxScaler

dl_numeric = df_dl.select_dtypes(include=[np.number]).columns.tolist()
mm_scaler = MinMaxScaler()
df_dl[dl_numeric] = mm_scaler.fit_transform(df_dl[dl_numeric])
log(f"  MinMaxScaler (0-1) applied to {len(dl_numeric)} columns")

df_dl.to_csv('preprocessed_dl.csv', index=False)
log(f"  DL shape: {df_dl.shape}")
log(f"  Saved → preprocessed_dl.csv")

# ──────────────────────────────────────────────────────────────────
#  5. SAVE REPORT
# ──────────────────────────────────────────────────────────────────
log("\n" + "=" * 70)
log("SUMMARY OF OUTPUT FILES")
log("=" * 70)
log(f"  {'File':<35} {'Description'}")
log(f"  {'─'*35} {'─'*35}")
log(f"  {'dataset_2.csv':<35} Original file (UNTOUCHED)")
log(f"  {'preprocessed_cleaned.csv':<35} General cleaning output")
log(f"  {'preprocessed_eda.csv':<35} EDA-ready (enriched features)")
log(f"  {'preprocessed_ml.csv':<35} ML-ready (encoded + scaled)")
log(f"  {'preprocessed_dl.csv':<35} DL-ready (normalized, one-hot)")
log(f"  {'ml_scaler.pkl':<35} Fitted StandardScaler")
log(f"  {'ml_label_encoders.pkl':<35} Fitted LabelEncoders")
log(f"  {'preprocessing_report.txt':<35} This report")

with open('preprocessing_report.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print("\n✅ All preprocessing complete! Original dataset_2.csv is untouched.")
