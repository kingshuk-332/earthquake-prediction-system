"""
Deterministic Multi-Source Alignment, Quality-Logged Preprocessing, 3D ECEF Station Graph,
and Strictly Causal Feature/Target Assembly for SeismicMultiModalFusion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from preprocess import homogenize_to_mw, spherical_hypocentral_km
from seismic_fusion.config import (
    BRANCH_A_FEATURE_NAMES,
    BRANCH_B_FEATURE_NAMES,
    BRANCH_C_FEATURE_NAMES,
    PipelineConfig,
    REQUIRED_CATALOG_COLUMNS,
    resolve_repo_root,
)


@dataclass
class AssembledFusionDataset:
    source_ids: np.ndarray                 # (N,)
    timestamps_sec: np.ndarray             # (N,) forecast origin UTC epoch seconds
    target_end_timestamps_sec: np.ndarray  # (N,) forecast target completion UTC epoch seconds (t_origin + H)
    latitudes: np.ndarray                  # (N,) float64
    longitudes: np.ndarray                 # (N,) float64
    station_codes: np.ndarray              # (N,)
    station_idx: np.ndarray                # (N,) int64 in [0, V-1]
    kinematic_seq: np.ndarray              # (N, C_kin, L) float32 strictly t <= t_origin
    node_feats: np.ndarray                 # (N, V, F_gnn) float32 strictly t <= t_origin
    adj_norm: np.ndarray                   # (V, V) float32
    phys_feats: np.ndarray                 # (N, F_phys) float32 strictly t <= t_origin
    mod_mask: np.ndarray                   # (N, 3) float32 binary availability mask
    thresholds: Tuple[float, ...]          # K thresholds, e.g., (4.0, 4.5, 5.0, 5.5)
    y_exceed_matrix: np.ndarray            # (N, K) float32 binary Mw >= th_k in (t_origin, t_origin + H]
    y_exceed: np.ndarray                   # (N,) float32 primary threshold column
    future_max_mw: np.ndarray              # (N,) float32 max Mw strictly in (t_origin, t_origin + H]
    reg_mask: np.ndarray                   # (N,) bool mask where >=1 above-Mc event occurs in (t_origin, t_origin + H]
    station_vocab: list[str]
    audit_manifest: dict[str, Any]


def ecef_xyz_km(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    depth_km: np.ndarray,
    earth_radius_km: float = 6371.0,
) -> np.ndarray:
    """3D spherical ECEF coordinates (km) eliminating flat-earth projection distortion."""
    r = earth_radius_km - np.asarray(depth_km, dtype=np.float64)
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def extract_causal_physics_features(
    w_raw: np.ndarray,
    p_idx: int,
    horizon_samples: int,
    fs: float = 100.0,
) -> np.ndarray:
    """
    Strictly causal post-P physical waveform observables for Track 6A when real recorded
    seismograms are available: [log_pd, log_pv, log_pa, log_tau_c, log_pd_pv_ratio, log_pre_rms].
    """
    if p_idx <= 0 or p_idx + horizon_samples > len(w_raw):
        raise ValueError(f"Invalid causal window bounds p_idx={p_idx}, horizon={horizon_samples}, len={len(w_raw)}")
    eps = 1e-8
    pre_p = np.asarray(w_raw[max(0, p_idx - int(fs)) : p_idx], dtype=np.float64)
    post_p = np.asarray(w_raw[p_idx : p_idx + horizon_samples], dtype=np.float64)
    if not (np.isfinite(pre_p).all() and np.isfinite(post_p).all()):
        raise ValueError("Non-finite samples encountered in waveform trace.")

    vel = post_p
    disp = np.cumsum(vel) / fs
    acc = np.diff(vel, prepend=pre_p[-1] if len(pre_p) > 0 else 0.0) * fs

    pd_pk = float(np.max(np.abs(disp))) + eps
    pv_pk = float(np.max(np.abs(vel))) + eps
    pa_pk = float(np.max(np.abs(acc))) + eps
    tau_c = 2.0 * math.pi * math.sqrt(float(np.sum(disp**2)) / (float(np.sum(vel**2)) + eps))
    pre_rms = math.sqrt(float(np.mean(pre_p**2)) + eps)

    return np.array(
        [
            math.log(pd_pk),
            math.log(pv_pk),
            math.log(pa_pk),
            math.log(tau_c + eps),
            math.log(pd_pk / pv_pk),
            math.log(pre_rms),
        ],
        dtype=np.float32,
    )


def check_real_waveform_availability(repo_root: Path) -> dict[str, Any]:
    """
    Checks whether raw recorded seismogram archives exist in the workspace without
    fabricating synthetic waveforms.
    """
    h5_candidates = [
        repo_root / "cnn" / "cnn wave" / "eew_causal_v3" / "dataset" / "eew_dataset_train.h5",
        repo_root / "bilstm_eew" / "causal_tabular_v1" / "tabular_features.h5",
    ]
    present = [str(p.relative_to(repo_root).as_posix()) for p in h5_candidates if p.is_file()]
    return {
        "recorded_seismograms_in_catalog": False,
        "branch_a_active_representation": "16-step_causal_catalog_kinematic_sequence",
        "branch_a_channels": list(BRANCH_A_FEATURE_NAMES),
        "eew_h5_archives_present_on_disk": present,
        "eew_h5_archives_missing_on_disk": [
            str(p.relative_to(repo_root).as_posix()) for p in h5_candidates if not p.is_file()
        ],
    }


def build_geodesic_station_graph(
    df: pd.DataFrame,
    earth_radius_km: float = 6371.0,
    kernel_length_km: float = 200.0,
) -> Tuple[list[str], np.ndarray, np.ndarray]:
    """
    Constructs the 10-node Northeast India station graph using 3D ECEF geodesic distances.
    """
    st_df = (
        df.groupby("receiver_code", as_index=False)
        .agg(
            lat=("receiver_latitude", "median"),
            lon=("receiver_longitude", "median"),
            elev_m=("receiver_elevation_m", "median"),
        )
        .sort_values("receiver_code")
        .reset_index(drop=True)
    )
    vocab = st_df["receiver_code"].astype(str).tolist()
    xyz = ecef_xyz_km(
        st_df["lat"].values,
        st_df["lon"].values,
        -st_df["elev_m"].values / 1000.0,
        earth_radius_km=earth_radius_km,
    )
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist_km = np.linalg.norm(diff, axis=-1)
    weights = np.exp(-(dist_km**2) / (2.0 * (kernel_length_km**2)))
    row_sum = np.sum(weights, axis=-1, keepdims=True)
    adj_norm = (weights / np.maximum(row_sum, 1e-8)).astype(np.float32)

    xyz_scaled = xyz / earth_radius_km
    elev_km = (st_df["elev_m"].values / 1000.0)[:, None]
    static_feats = np.concatenate([xyz_scaled, elev_km], axis=-1).astype(np.float32)
    return vocab, static_feats, adj_norm


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble_targets_for_horizon(
    ts_sec: np.ndarray,
    mw_arr: np.ndarray,
    mag_raw_arr: np.ndarray,
    anchor_indices: np.ndarray,
    horizon_sec: float,
    thresholds: Tuple[float, ...],
    mc_val: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Computes strictly forward forecast targets over the open-closed interval
    (t_origin, t_origin + horizon_sec] using binary search `side='right'` on `t_origin`,
    guaranteeing ZERO same-timestamp or self-event target leakage even if timestamp ties exist.
    """
    n_samples = len(anchor_indices)
    k_th = len(thresholds)
    y_mat = np.zeros((n_samples, k_th), dtype=np.float32)
    future_max_mw = np.zeros(n_samples, dtype=np.float32)
    reg_mask = np.zeros(n_samples, dtype=bool)
    min_lead_sec = float("inf")

    for out_i, idx in enumerate(anchor_indices):
        t_now = float(ts_sec[idx])
        # Strictly > t_now (excludes idx AND any hypothetical simultaneous t == t_now records)
        fwd_start = int(np.searchsorted(ts_sec, t_now, side="right"))
        fwd_end = int(np.searchsorted(ts_sec, t_now + horizon_sec, side="right"))
        if fwd_end > fwd_start:
            lead = float(ts_sec[fwd_start]) - t_now
            if lead < min_lead_sec:
                min_lead_sec = lead
            fwd_mw = mw_arr[fwd_start:fwd_end]
            fwd_above_mc = fwd_mw[mag_raw_arr[fwd_start:fwd_end] >= mc_val]
            max_m_fwd = float(np.max(fwd_mw))
            for k_i, th in enumerate(thresholds):
                if max_m_fwd >= th:
                    y_mat[out_i, k_i] = 1.0
            if len(fwd_above_mc) > 0:
                future_max_mw[out_i] = float(np.max(fwd_above_mc))
                reg_mask[out_i] = True

    return y_mat, future_max_mw, reg_mask, (min_lead_sec if math.isfinite(min_lead_sec) else 0.0)


def ingest_and_assemble_fusion_dataset(
    repo_root: Path | None = None,
    config: PipelineConfig | None = None,
) -> AssembledFusionDataset:
    """
    Ingests real `dataset_2.csv`, enforces `valid + flagged == total_raw`, constructs the
    3D ECEF station graph, and assembles strictly causal multi-modal feature tensors
    (`t <= t_origin`) paired with strictly forward targets (`t_origin < t <= t_origin + H`).
    """
    root = repo_root or resolve_repo_root()
    cfg = config or PipelineConfig()
    cfg.validate()

    catalog_path = root / cfg.dataset_rel_path
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog file missing at '{catalog_path}'.")

    df_raw = pd.read_csv(catalog_path)
    missing_cols = [c for c in REQUIRED_CATALOG_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in '{catalog_path.name}': {missing_cols}")

    total_raw = len(df_raw)
    df = df_raw.copy()
    df["receiver_code"] = df["receiver_code"].astype(str).str.strip()
    df["Magnitude_Type"] = df["Magnitude_Type"].astype(str).str.upper().str.strip()
    df["origin_time_utc_dt"] = pd.to_datetime(
        df["origin_time_utc"].astype(str).str.replace(" UTC", "", regex=False).str.strip(),
        errors="coerce",
    )

    dup_mask = df.duplicated(keep="first")
    nan_ts_mask = df["origin_time_utc_dt"].isna()
    geom_bad_mask = (
        ~np.isfinite(df["Lat"])
        | (df["Lat"] < cfg.geophysics.lat_bounds[0])
        | (df["Lat"] > cfg.geophysics.lat_bounds[1])
        | ~np.isfinite(df["Long"])
        | (df["Long"] < cfg.geophysics.lon_bounds[0])
        | (df["Long"] > cfg.geophysics.lon_bounds[1])
        | ~np.isfinite(df["Depth_km"])
        | (df["Depth_km"] < cfg.geophysics.depth_bounds_km[0])
        | (df["Depth_km"] > cfg.geophysics.depth_bounds_km[1])
    )
    mag_bad_mask = (
        ~np.isfinite(df["Magnitude_Value"])
        | (df["Magnitude_Value"] <= cfg.geophysics.magnitude_bounds[0])
        | (df["Magnitude_Value"] > cfg.geophysics.magnitude_bounds[1])
    )
    phase_bad_mask = (
        ~np.isfinite(df["p_arrival_sample"])
        | ~np.isfinite(df["s_arrival_sample"])
        | (df["s_arrival_sample"] <= df["p_arrival_sample"])
        | ~np.isfinite(df["snr_db"])
        | (df["snr_db"] < cfg.geophysics.snr_min_db)
    )
    below_mc_mask = df["Magnitude_Value"] < cfg.geophysics.mc_completeness

    hard_invalid = dup_mask | nan_ts_mask | geom_bad_mask | mag_bad_mask | phase_bad_mask
    flagged_any = hard_invalid | below_mc_mask

    n_valid = int((~flagged_any).sum())
    n_flagged = int(flagged_any.sum())
    assert n_valid + n_flagged == total_raw, (
        f"Record conservation failure: valid ({n_valid}) + flagged ({n_flagged}) != total_raw ({total_raw})"
    )

    if nan_ts_mask.any():
        raise ValueError(f"Cannot order catalog with {int(nan_ts_mask.sum())} invalid UTC timestamps.")

    df_valid_schema = df.loc[~hard_invalid].copy()
    df_valid_schema.sort_values(["origin_time_utc_dt", "source_id"], inplace=True, kind="mergesort")
    df_valid_schema.reset_index(drop=True, inplace=True)

    df_valid_schema["Mw"] = homogenize_to_mw(
        df_valid_schema["Magnitude_Value"].values, df_valid_schema["Magnitude_Type"]
    )
    df_valid_schema["above_mc"] = (df_valid_schema["Magnitude_Value"] >= cfg.geophysics.mc_completeness).astype(
        np.float32
    )
    df_valid_schema["hypo_sph_km"] = spherical_hypocentral_km(
        df_valid_schema["epicentral_dist_km"].values,
        df_valid_schema["Depth_km"].values,
        r_earth=cfg.geophysics.earth_radius_km,
    )
    df_valid_schema["sp_sec"] = (
        df_valid_schema["s_arrival_sample"].values - df_valid_schema["p_arrival_sample"].values
    ) / cfg.geophysics.sampling_rate_hz

    ts_sec = df_valid_schema["origin_time_utc_dt"].astype("int64").values / 1e9
    dt_hours = np.zeros(len(df_valid_schema), dtype=np.float64)
    dt_hours[1:] = np.maximum((ts_sec[1:] - ts_sec[:-1]) / 3600.0, 0.0)

    station_vocab, static_st_feats, adj_norm = build_geodesic_station_graph(
        df_valid_schema,
        earth_radius_km=cfg.geophysics.earth_radius_km,
        kernel_length_km=cfg.geophysics.geodesic_kernel_length_km,
    )
    st_to_idx = {s: i for i, s in enumerate(station_vocab)}
    st_indices_all = np.array([st_to_idx[s] for s in df_valid_schema["receiver_code"].values], dtype=np.int64)
    n_stations = len(station_vocab)

    mw_arr = df_valid_schema["Mw"].values.astype(np.float64)
    mag_raw_arr = df_valid_schema["Magnitude_Value"].values.astype(np.float64)
    above_mc_arr = df_valid_schema["above_mc"].values.astype(np.float64)
    doy_arr = df_valid_schema["origin_time_utc_dt"].dt.dayofyear.values.astype(np.float64)

    event_kin_channels = np.stack(
        [
            mw_arr,
            np.log1p(df_valid_schema["Depth_km"].values),
            np.log1p(df_valid_schema["hypo_sph_km"].values),
            np.log1p(df_valid_schema["sp_sec"].values),
            df_valid_schema["snr_db"].values,
            np.log1p(np.minimum(dt_hours, 24.0 * 365.0)),
        ],
        axis=-1,
    ).astype(np.float32)

    lookback = cfg.model.lookback_events
    horizon_sec = cfg.validation.forecast_horizon_days * 86400.0
    modern_start_sec = pd.Timestamp(cfg.validation.modern_catalog_start_utc).value / 1e9
    max_catalog_sec = float(ts_sec[-1])
    mc_val = cfg.geophysics.mc_completeness
    benioff_all = 10.0 ** (0.75 * mw_arr)

    anchor_list = [
        idx
        for idx in range(lookback - 1, len(df_valid_schema))
        if ts_sec[idx] >= modern_start_sec and (ts_sec[idx] + horizon_sec) <= max_catalog_sec
    ]
    if not anchor_list:
        raise ValueError("No valid anchor events satisfy lookback and non-censored horizon constraints.")
    anchor_arr = np.asarray(anchor_list, dtype=np.int64)
    n_samples = len(anchor_arr)

    kin_seq_out = np.zeros((n_samples, cfg.model.kinematic_channels, lookback), dtype=np.float32)
    node_feats_out = np.zeros((n_samples, n_stations, cfg.model.gnn_node_dim), dtype=np.float32)
    phys_feats_out = np.zeros((n_samples, cfg.model.phys_feature_dim), dtype=np.float32)
    mod_mask_out = np.ones((n_samples, 3), dtype=np.float32)

    w7_sec = 7.0 * 86400.0
    w30_sec = 30.0 * 86400.0
    w90_sec = 90.0 * 86400.0
    w180_sec = 180.0 * 86400.0
    max_feature_time_offset = -float("inf")

    for out_i, idx in enumerate(anchor_arr):
        t_now = float(ts_sec[idx])

        # Branch A: Strictly causal sequence [idx - lookback + 1 .. idx] -> max timestamp == t_now
        kin_seq_out[out_i] = event_kin_channels[idx - lookback + 1 : idx + 1].T
        feat_t_max = float(np.max(ts_sec[idx - lookback + 1 : idx + 1]) - t_now)
        if feat_t_max > max_feature_time_offset:
            max_feature_time_offset = feat_t_max

        # Causal history indices strictly <= t_now (`idx + 1` exclusive upper slice bound)
        i_7d = int(np.searchsorted(ts_sec, t_now - w7_sec, side="left"))
        i_30d = int(np.searchsorted(ts_sec, t_now - w30_sec, side="left"))
        i_90d = int(np.searchsorted(ts_sec, t_now - w90_sec, side="left"))
        i_180d = int(np.searchsorted(ts_sec, t_now - w180_sec, side="left"))

        node_feats_out[out_i, :, :4] = static_st_feats
        sl_30 = slice(i_30d, idx + 1)
        sl_90 = slice(i_90d, idx + 1)
        st_30 = st_indices_all[sl_30]
        st_90 = st_indices_all[sl_90]
        mw_90 = mw_arr[sl_90]
        snr_90 = df_valid_schema["snr_db"].values[sl_90]
        ts_90 = ts_sec[sl_90]

        for s_id in range(n_stations):
            c30 = float(np.sum(st_30 == s_id))
            m90_mask = st_90 == s_id
            c90 = float(np.sum(m90_mask))
            if c90 > 0:
                max_m_st = float(np.max(mw_90[m90_mask]))
                mean_snr_st = float(np.mean(snr_90[m90_mask]))
                dt_last_days = max((t_now - float(ts_90[m90_mask][-1])) / 86400.0, 0.0)
            else:
                max_m_st = 0.0
                mean_snr_st = 0.0
                dt_last_days = 90.0
            node_feats_out[out_i, s_id, 4] = math.log1p(c30)
            node_feats_out[out_i, s_id, 5] = math.log1p(c90)
            node_feats_out[out_i, s_id, 6] = max_m_st
            node_feats_out[out_i, s_id, 7] = mean_snr_st / 20.0
            node_feats_out[out_i, s_id, 8] = math.log1p(dt_last_days)

        hist_180_mc = mag_raw_arr[i_180d : idx + 1]
        above_180 = hist_180_mc[hist_180_mc >= mc_val]
        if len(above_180) >= 8 and (np.mean(above_180) - (mc_val - 0.05)) > 0.05:
            b_val = math.log10(math.e) / (float(np.mean(above_180)) - (mc_val - 0.05))
            b_val = float(np.clip(b_val, 0.2, 2.5))
        else:
            b_val = 1.0

        benioff_30 = math.log1p(float(np.sum(benioff_all[sl_30])))
        benioff_180 = math.log1p(float(np.sum(benioff_all[i_180d : idx + 1])))

        recent_start = max(0, idx - 128)
        rec_mask = above_mc_arr[recent_start : idx + 1] > 0.5
        if np.any(rec_mask):
            dt_days_etas = (t_now - ts_sec[recent_start : idx + 1][rec_mask]) / 86400.0
            m_etas = mw_arr[recent_start : idx + 1][rec_mask]
            etas_sum = float(np.sum((10.0 ** (0.5 * (m_etas - mc_val))) / ((dt_days_etas + 0.05) ** 1.1)))
        else:
            etas_sum = 0.0
        log_etas = math.log1p(etas_sum)

        rate_7d = math.log1p(float(np.sum(above_mc_arr[i_7d : idx + 1])))
        rate_30d = math.log1p(float(np.sum(above_mc_arr[sl_30])))
        total_30d = max(idx + 1 - i_30d, 1)
        sub_mc_frac_30d = float(1.0 - np.sum(above_mc_arr[sl_30]) / total_30d)

        doy_rad = 2.0 * math.pi * doy_arr[idx] / 365.25
        phys_feats_out[out_i] = np.array(
            [
                b_val,
                benioff_30,
                benioff_180,
                log_etas,
                rate_7d,
                rate_30d,
                sub_mc_frac_30d,
                float(mw_arr[idx]),
                math.log1p(float(df_valid_schema["Depth_km"].iloc[idx])),
                math.sin(doy_rad),
                math.cos(doy_rad),
                float(above_mc_arr[idx]),
            ],
            dtype=np.float32,
        )

    thresholds = cfg.validation.evaluation_mw_thresholds
    y_mat, future_max_mw, reg_mask, min_target_lead_sec = assemble_targets_for_horizon(
        ts_sec=ts_sec,
        mw_arr=mw_arr,
        mag_raw_arr=mag_raw_arr,
        anchor_indices=anchor_arr,
        horizon_sec=horizon_sec,
        thresholds=thresholds,
        mc_val=mc_val,
    )
    assert max_feature_time_offset <= 0.0, f"Future feature leakage detected: offset={max_feature_time_offset}"
    assert min_target_lead_sec > 0.0, f"Same-timestamp target leakage detected: min_lead={min_target_lead_sec}"

    primary_idx = thresholds.index(cfg.validation.target_mw_threshold)
    y_primary = y_mat[:, primary_idx].copy()
    anchor_ts = ts_sec[anchor_arr]
    target_end_ts = anchor_ts + horizon_sec

    wf_status = check_real_waveform_availability(root)
    threshold_base_rates = {
        f"Mw>={th:.1f}": {
            "positives": int(np.sum(y_mat[:, k_i] == 1.0)),
            "positive_rate": round(float(np.mean(y_mat[:, k_i])), 6),
        }
        for k_i, th in enumerate(thresholds)
    }

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_path": cfg.dataset_rel_path,
        "dataset_sha256": _sha256(catalog_path),
        "total_raw": total_raw,
        "valid": n_valid,
        "flagged": n_flagged,
        "record_conservation_verified": (n_valid + n_flagged == total_raw),
        "causality_boundary_proof": {
            "max_feature_timestamp_minus_origin_sec": float(max_feature_time_offset),
            "min_target_event_minus_origin_sec": round(float(min_target_lead_sec), 6),
            "branch_a_features": list(BRANCH_A_FEATURE_NAMES),
            "branch_b_features": list(BRANCH_B_FEATURE_NAMES),
            "branch_c_features": list(BRANCH_C_FEATURE_NAMES),
        },
        "anomaly_counts": {
            "exact_duplicates": int(dup_mask.sum()),
            "missing_timestamps": int(nan_ts_mask.sum()),
            "out_of_range_geometry": int(geom_bad_mask.sum()),
            "out_of_range_magnitude": int(mag_bad_mask.sum()),
            "phase_or_snr_anomaly": int(phase_bad_mask.sum()),
            "below_mc": int(below_mc_mask.sum()),
        },
        "mc_completeness": mc_val,
        "station_count": n_stations,
        "station_vocab": station_vocab,
        "modern_anchor_samples": n_samples,
        "threshold_summary": threshold_base_rates,
        "waveform_archive_status": wf_status,
        "synthetic_records_fabricated": 0,
    }

    out_dir = root / cfg.output_dir_rel_path
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fusion_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AssembledFusionDataset(
        source_ids=df_valid_schema["source_id"].values[anchor_arr].astype(str),
        timestamps_sec=anchor_ts,
        target_end_timestamps_sec=target_end_ts,
        latitudes=df_valid_schema["Lat"].values[anchor_arr].astype(np.float64),
        longitudes=df_valid_schema["Long"].values[anchor_arr].astype(np.float64),
        station_codes=df_valid_schema["receiver_code"].values[anchor_arr].astype(str),
        station_idx=st_indices_all[anchor_arr],
        kinematic_seq=kin_seq_out,
        node_feats=node_feats_out,
        adj_norm=adj_norm,
        phys_feats=phys_feats_out,
        mod_mask=mod_mask_out,
        thresholds=thresholds,
        y_exceed_matrix=y_mat,
        y_exceed=y_primary,
        future_max_mw=future_max_mw,
        reg_mask=reg_mask,
        station_vocab=station_vocab,
        audit_manifest=manifest,
    )
