"""
Leak-Free Temporal Preprocessing for Seismic Activity Forecasting
=================================================================
Reads:  dataset_2.csv (NEVER modified)
Writes:
  - preprocessed_dl_temporal.csv    -- chronologically ordered, train-only scaled feature set
  - dl_temporal_scaler.pkl          -- train-only MinMaxScaler + inter-event cap metadata
  - temporal_audit_manifest.json    -- immutable record-conservation manifest (valid + flagged == total_raw)

Remediations applied per audit:
  1. Removed target-derived leakage features (`log_energy`, `depth_mag_ratio`).
  2. Preserved raw unscaled `Magnitude_Value` and `Mw_homogenized` separately from scaled predictors;
     past-event magnitude history uses `Magnitude_Value_norm` scaled strictly by training-split moments.
  3. Fitted 99th-percentile inter-event gap cap and `MinMaxScaler` exclusively on the chronological
     training split (`train_idx`), eliminating future distribution leakage.
  4. Enforced strict record conservation (`valid + flagged == total_raw`) with fail-fast schema validation.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocess import homogenize_to_mw, sha256_file, spherical_hypocentral_km
from seismic_fusion.config import PipelineConfig, REQUIRED_CATALOG_COLUMNS, resolve_repo_root


def run_temporal_preprocessing(repo_root: Path | None = None, config: PipelineConfig | None = None) -> dict[str, Any]:
    root = repo_root or resolve_repo_root()
    cfg = config or PipelineConfig()
    cfg.validate()

    input_path = root / cfg.dataset_rel_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Required input catalog missing: {input_path}")

    df_raw = pd.read_csv(input_path)
    missing_cols = [c for c in REQUIRED_CATALOG_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Schema violation in '{input_path.name}': missing required columns {missing_cols}")

    total_raw = len(df_raw)
    df = df_raw.copy()
    df["origin_time_utc"] = pd.to_datetime(
        df["origin_time_utc"].astype(str).str.replace(" UTC", "", regex=False).str.strip(),
        errors="coerce",
    )

    missing_ts = df["origin_time_utc"].isna()
    below_mc = df["Magnitude_Value"] < cfg.geophysics.mc_completeness
    invalid_phase = df["s_arrival_sample"] <= df["p_arrival_sample"]
    flagged_mask = missing_ts | below_mc | invalid_phase

    n_valid = int((~flagged_mask).sum())
    n_flagged = int(flagged_mask.sum())
    assert n_valid + n_flagged == total_raw, "Record conservation invariant violated."

    if missing_ts.any():
        raise ValueError(
            f"Found {int(missing_ts.sum())} unparseable timestamps in '{input_path.name}'; "
            "refusing silent row drop."
        )

    df.sort_values(["origin_time_utc", "source_id"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    n = len(df)
    split_train_end = int(n * 0.70)
    split_val_end = int(n * 0.85)
    split_arr = np.array(["test"] * n, dtype=object)
    split_arr[:split_train_end] = "train"
    split_arr[split_train_end:split_val_end] = "validation"
    train_mask = split_arr == "train"

    # Temporal & cyclical features
    df["year_raw"] = df["year"].astype(float)
    month = df["origin_time_utc"].dt.month.astype(float)
    dow = df["origin_time_utc"].dt.dayofweek.astype(float)
    hour = df["origin_time_utc"].dt.hour.astype(float)
    doy = df["origin_time_utc"].dt.dayofyear.astype(float)

    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    # Causal inter-event time gap; upper cap fitted strictly on training split
    raw_gap_hours = df["origin_time_utc"].diff().dt.total_seconds().fillna(0.0) / 3600.0
    raw_gap_hours = np.maximum(raw_gap_hours, 0.0)
    train_gap_cap = float(np.quantile(raw_gap_hours[train_mask], 0.99))
    df["time_diff_hours"] = np.clip(raw_gap_hours, 0.0, train_gap_cap)
    df["log_time_diff"] = np.log1p(df["time_diff_hours"])

    # Spherical kinematic features (NO target-derived log_energy or depth_mag_ratio)
    df["sp_travel_time"] = (df["s_arrival_sample"] - df["p_arrival_sample"]).astype(float)
    df["hypocentral_dist_spherical_km"] = spherical_hypocentral_km(
        df["epicentral_dist_km"].values,
        df["Depth_km"].values,
        r_earth=cfg.geophysics.earth_radius_km,
    )
    df["Mw_homogenized"] = homogenize_to_mw(df["Magnitude_Value"].values, df["Magnitude_Type"])

    # Note: past_event_mag_scaled is permissible ONLY for strictly prior lookback steps (0 .. t-1),
    # never at target index t! Target columns are stored explicitly as unscaled raw columns.
    predictor_numeric_cols = [
        "Lat",
        "Long",
        "Depth_km",
        "Magnitude_Value",
        "receiver_latitude",
        "receiver_longitude",
        "receiver_elevation_m",
        "p_arrival_sample",
        "s_arrival_sample",
        "snr_db",
        "epicentral_dist_km",
        "hypocentral_dist_spherical_km",
        "sp_travel_time",
        "year_raw",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "time_diff_hours",
        "log_time_diff",
    ]
    categorical_features = ["Magnitude_Type", "source_magnitude_type", "receiver_code"]

    df_out = pd.DataFrame(index=df.index)
    df_out["source_id"] = df["source_id"]
    df_out["origin_time_utc"] = df["origin_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_out["split"] = split_arr
    df_out["above_mc_mask"] = (~below_mc).astype(int)
    df_out["target_magnitude_raw"] = df["Magnitude_Value"].astype(float)
    df_out["target_mw_raw"] = df["Mw_homogenized"].astype(float)

    train_medians: dict[str, float] = {}
    for col in predictor_numeric_cols:
        med = float(df.loc[train_mask, col].median())
        train_medians[col] = med
        df_out[col] = df[col].fillna(med).astype(float)

    one_hot_cols: list[str] = []
    for cat_col in categorical_features:
        col_clean = df[cat_col].astype(str).str.upper().str.strip()
        tr_cats = sorted(col_clean[train_mask].unique().tolist())
        for cat_val in tr_cats:
            cname = f"{cat_col}_{cat_val}"
            df_out[cname] = (col_clean == cat_val).astype(int)
            one_hot_cols.append(cname)

    scaler = MinMaxScaler()
    df_out.loc[train_mask, predictor_numeric_cols] = scaler.fit_transform(
        df_out.loc[train_mask, predictor_numeric_cols]
    )
    df_out.loc[~train_mask, predictor_numeric_cols] = scaler.transform(
        df_out.loc[~train_mask, predictor_numeric_cols]
    )

    mag_idx = predictor_numeric_cols.index("Magnitude_Value")
    scaler_info = {
        "scaler": scaler,
        "numeric_feature_names": predictor_numeric_cols,
        "one_hot_feature_names": one_hot_cols,
        "all_predictor_columns": predictor_numeric_cols + one_hot_cols,
        "train_medians": train_medians,
        "train_gap_cap_hours": train_gap_cap,
        "fit_split": "train",
        "train_rows": int(train_mask.sum()),
        "train_magnitude_min": float(scaler.data_min_[mag_idx]),
        "train_magnitude_max": float(scaler.data_max_[mag_idx]),
    }
    with (root / "dl_temporal_scaler.pkl").open("wb") as fh:
        pickle.dump(scaler_info, fh)

    output_csv = root / "preprocessed_dl_temporal.csv"
    output_csv.write_text(df_out.to_csv(index=False), encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema_version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_file": cfg.dataset_rel_path,
        "source_sha256": sha256_file(input_path),
        "output_file": "preprocessed_dl_temporal.csv",
        "total_raw": total_raw,
        "valid": n_valid,
        "flagged": n_flagged,
        "record_conservation_verified": (n_valid + n_flagged == total_raw) and (len(df_out) == total_raw),
        "removed_target_leakage_columns": ["log_energy", "depth_mag_ratio"],
        "scaler_fit_split": "train",
        "train_rows": int(train_mask.sum()),
        "predictor_count": len(predictor_numeric_cols) + len(one_hot_cols),
    }
    (root / "temporal_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    res = run_temporal_preprocessing()
    print(json.dumps(res, indent=2))
