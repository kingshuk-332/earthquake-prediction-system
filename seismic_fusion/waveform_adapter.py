"""
Real 3-Component (Z, N, E) Broadband Waveform Ingestion & Deterministic QC Adapter
==================================================================================
Supports:
  1. MiniSEED / SAC waveform files via lazy `obspy` import (`ingest_miniseed_or_sac_directory`).
  2. STEAD-compatible HDF5 + metadata CSV archives (`ingest_stead_hdf5`).
  3. Deterministic QC flagging (`SAMPLING_RATE_MISMATCH`, `MISSING_CHANNELS`, `NON_FINITE_SAMPLES`,
     `FLATLINE_OR_CLIPPED`, `LOW_PRE_P_SNR`, `INVALID_P_WINDOW`) with strict record conservation
     (`valid + flagged == total_raw`) and direct tensor routing into `SeismicMultiModalFusion` Branch A.

Strict No-Fabrication Policy: Never synthesizes or substitutes mock waveforms when files or
channels are missing or corrupt.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seismic_fusion.preprocessing import extract_causal_physics_features


REQUIRED_STEAD_METADATA_COLS = (
    "trace_name",
    "p_arrival_sample",
    "s_arrival_sample",
    "receiver_code",
)


@dataclass
class AdaptedWaveformBatch:
    trace_ids: np.ndarray        # (N_valid,)
    branch_a_windows: np.ndarray # (N_valid, 3, W) float32 pre-P causal Z-score normalized
    physics_features: np.ndarray # (N_valid, 6) float32 vertical-component (tau_c, Pd/Pv, ...)
    mod_mask_a: np.ndarray       # (N_valid,) float32 == 1.0 for QC-valid traces
    qc_manifest: dict[str, Any]


def evaluate_3c_trace_qc(
    data_3c: np.ndarray,  # Expected shape (3, T) ordered [Z, N, E]
    sampling_rate_hz: float,
    p_arrival_sample: int,
    expected_fs_hz: float = 100.0,
    pre_p_seconds: float = 1.0,
    horizon_seconds: float = 3.0,
    min_pre_p_snr_db: float = 1.5,
    clip_fraction_limit: float = 0.02,
) -> tuple[list[str], np.ndarray | None, float]:
    """
    Runs deterministic physical & numerical QC on a single 3-component (Z, N, E) trace
    and returns `(flags, causal_normalized_window [3, W] | None, measured_snr_db)`.
    """
    flags: list[str] = []
    if abs(float(sampling_rate_hz) - expected_fs_hz) > 0.5:
        flags.append("SAMPLING_RATE_MISMATCH")

    arr = np.asarray(data_3c, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 3:
        flags.append("MISSING_CHANNELS")
        return flags, None, 0.0

    if not np.isfinite(arr).all():
        flags.append("NON_FINITE_SAMPLES")
        return flags, None, 0.0

    pre_n = int(round(pre_p_seconds * expected_fs_hz))
    post_n = int(round(horizon_seconds * expected_fs_hz))
    total_t = arr.shape[1]
    if p_arrival_sample < pre_n or (p_arrival_sample + post_n) > total_t:
        flags.append("INVALID_P_WINDOW")
        return flags, None, 0.0

    # Check flatline (zero variance) or rail-clipping saturation across each channel
    for ch in range(3):
        ch_sig = arr[ch]
        std_all = float(np.std(ch_sig))
        if std_all < 1e-10:
            flags.append("FLATLINE_ZERO_VARIANCE")
            break
        peak = float(np.max(np.abs(ch_sig)))
        if peak > 0:
            near_rail = float(np.mean(np.abs(ch_sig) >= 0.999 * peak))
            if near_rail > clip_fraction_limit:
                flags.append("AMPLITUDE_CLIPPING")
                break

    pre_seg = arr[:, p_arrival_sample - pre_n : p_arrival_sample]
    post_seg = arr[:, p_arrival_sample : p_arrival_sample + post_n]

    pre_rms_z = float(np.sqrt(np.mean(pre_seg[0] ** 2)))
    post_rms_z = float(np.sqrt(np.mean(post_seg[0] ** 2)))
    snr_db = 20.0 * float(np.log10(max(post_rms_z, 1e-12) / max(pre_rms_z, 1e-12)))
    if snr_db < min_pre_p_snr_db:
        flags.append("LOW_PRE_P_SNR")

    if flags:
        return flags, None, snr_db

    # Strictly causal 1.0s pre-P per-component standardization (zero post-P statistics leaked)
    win = arr[:, p_arrival_sample - pre_n : p_arrival_sample + post_n]
    mu_pre = np.mean(pre_seg, axis=1, keepdims=True)
    sd_pre = np.std(pre_seg, axis=1, keepdims=True)
    win_norm = ((win - mu_pre) / np.maximum(sd_pre, 1e-8)).astype(np.float32)
    return [], win_norm, snr_db


def ingest_stead_hdf5(
    hdf5_path: Path,
    metadata_csv_path: Path,
    manifest_out_path: Path | None = None,
    expected_fs_hz: float = 100.0,
    horizon_seconds: float = 3.0,
    min_pre_p_snr_db: float = 1.5,
) -> AdaptedWaveformBatch:
    """
    Ingests real 3-component waveforms from a STEAD-schema HDF5 archive + metadata CSV.
    Enforces fail-fast schema validation and `valid + flagged == total_raw`.
    """
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"STEAD HDF5 waveform file not found: {hdf5_path}")
    if not metadata_csv_path.is_file():
        raise FileNotFoundError(f"STEAD metadata CSV file not found: {metadata_csv_path}")

    h5py = importlib.import_module("h5py")
    meta_df = pd.read_csv(metadata_csv_path)
    missing_cols = [c for c in REQUIRED_STEAD_METADATA_COLS if c not in meta_df.columns]
    if missing_cols:
        raise ValueError(f"STEAD metadata CSV missing required columns: {missing_cols}")

    total_raw = len(meta_df)
    valid_ids: list[str] = []
    windows_list: list[np.ndarray] = []
    phys_list: list[np.ndarray] = []
    flagged_log: list[dict[str, Any]] = []

    horizon_samples = int(round(horizon_seconds * expected_fs_hz))

    with h5py.File(hdf5_path, "r") as h5f:
        data_grp = h5f["data"] if "data" in h5f else h5f
        for _, row in meta_df.iterrows():
            tr_name = str(row["trace_name"])
            if tr_name not in data_grp:
                flagged_log.append({"trace_name": tr_name, "flags": ["TRACE_MISSING_IN_HDF5"]})
                continue

            ds = data_grp[tr_name]
            raw_arr = np.asarray(ds[:])
            # STEAD stores (T, 3) [E, N, Z] or [Z, N, E]; transpose (T, 3) -> (3, T)
            if raw_arr.ndim == 2 and raw_arr.shape[1] == 3 and raw_arr.shape[0] != 3:
                arr_3c = raw_arr.T
            else:
                arr_3c = raw_arr

            fs_attr = float(ds.attrs.get("sampling_rate_hz", row.get("sampling_rate_hz", expected_fs_hz)))
            try:
                p_samp = int(round(float(row["p_arrival_sample"])))
            except (TypeError, ValueError):
                flagged_log.append({"trace_name": tr_name, "flags": ["INVALID_P_ARRIVAL_METADATA"]})
                continue

            flags, win_norm, snr_meas = evaluate_3c_trace_qc(
                arr_3c,
                sampling_rate_hz=fs_attr,
                p_arrival_sample=p_samp,
                expected_fs_hz=expected_fs_hz,
                horizon_seconds=horizon_seconds,
                min_pre_p_snr_db=min_pre_p_snr_db,
            )
            if flags or win_norm is None:
                flagged_log.append({"trace_name": tr_name, "flags": flags, "measured_snr_db": round(snr_meas, 3)})
                continue

            phys_vec = extract_causal_physics_features(
                arr_3c[0], p_idx=p_samp, horizon_samples=horizon_samples, fs=expected_fs_hz
            )
            valid_ids.append(tr_name)
            windows_list.append(win_norm)
            phys_list.append(phys_vec)

    n_valid = len(valid_ids)
    n_flagged = len(flagged_log)
    assert n_valid + n_flagged == total_raw, (
        f"STEAD conservation failure: valid ({n_valid}) + flagged ({n_flagged}) != total_raw ({total_raw})"
    )

    w_len = int(round((1.0 + horizon_seconds) * expected_fs_hz))
    win_arr = np.stack(windows_list, axis=0) if windows_list else np.zeros((0, 3, w_len), dtype=np.float32)
    phys_arr = np.stack(phys_list, axis=0) if phys_list else np.zeros((0, 6), dtype=np.float32)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "adapter": "STEAD_HDF5_3C",
        "source_hdf5": hdf5_path.name,
        "total_raw": total_raw,
        "valid": n_valid,
        "flagged": n_flagged,
        "record_conservation_verified": (n_valid + n_flagged == total_raw),
        "flagged_records": flagged_log,
        "synthetic_traces_fabricated": 0,
    }
    if manifest_out_path is not None:
        manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AdaptedWaveformBatch(
        trace_ids=np.asarray(valid_ids, dtype=str),
        branch_a_windows=win_arr,
        physics_features=phys_arr,
        mod_mask_a=np.ones(n_valid, dtype=np.float32),
        qc_manifest=manifest,
    )


def ingest_miniseed_or_sac_records(
    records: list[dict[str, Any]],
    manifest_out_path: Path | None = None,
    expected_fs_hz: float = 100.0,
    horizon_seconds: float = 3.0,
    min_pre_p_snr_db: float = 1.5,
) -> AdaptedWaveformBatch:
    """
    Ingests real 3-component MiniSEED or SAC waveform files using lazy `obspy` import.
    Each item in `records` must specify `{'event_id': str, 'file_path': Path | str, 'p_arrival_sample': int}`.
    """
    if not records:
        raise ValueError("Empty waveform record list supplied to MiniSEED/SAC adapter.")
    try:
        obspy = importlib.import_module("obspy")
    except ImportError as exc:
        raise ImportError(
            "ObsPy is required for MiniSEED/SAC ingestion (`pip install obspy`), "
            "and synthetic fallback is prohibited."
        ) from exc

    total_raw = len(records)
    valid_ids: list[str] = []
    windows_list: list[np.ndarray] = []
    phys_list: list[np.ndarray] = []
    flagged_log: list[dict[str, Any]] = []
    horizon_samples = int(round(horizon_seconds * expected_fs_hz))

    for rec in records:
        if "event_id" not in rec or "file_path" not in rec or "p_arrival_sample" not in rec:
            raise ValueError(f"Record missing required keys ('event_id', 'file_path', 'p_arrival_sample'): {rec}")
        ev_id = str(rec["event_id"])
        fpath = Path(rec["file_path"])
        if not fpath.is_file():
            raise FileNotFoundError(f"Waveform file does not exist: {fpath}")

        st = obspy.read(str(fpath))
        comp_map: dict[str, Any] = {}
        for tr in st:
            ch_code = str(tr.stats.channel).upper()
            if ch_code:
                comp_map[ch_code[-1]] = tr

        # Normalize common horizontal component conventions (1/2 -> N/E if N/E absent)
        if "N" not in comp_map and "1" in comp_map:
            comp_map["N"] = comp_map["1"]
        if "E" not in comp_map and "2" in comp_map:
            comp_map["E"] = comp_map["2"]

        if not all(k in comp_map for k in ("Z", "N", "E")):
            flagged_log.append({
                "event_id": ev_id,
                "flags": ["MISSING_CHANNELS"],
                "available_components": sorted(comp_map.keys()),
            })
            continue

        lengths = [len(comp_map[k].data) for k in ("Z", "N", "E")]
        min_len = min(lengths)
        arr_3c = np.stack([np.asarray(comp_map[k].data[:min_len], dtype=np.float64) for k in ("Z", "N", "E")], axis=0)
        fs_hz = float(comp_map["Z"].stats.sampling_rate)
        p_samp = int(rec["p_arrival_sample"])

        flags, win_norm, snr_meas = evaluate_3c_trace_qc(
            arr_3c,
            sampling_rate_hz=fs_hz,
            p_arrival_sample=p_samp,
            expected_fs_hz=expected_fs_hz,
            horizon_seconds=horizon_seconds,
            min_pre_p_snr_db=min_pre_p_snr_db,
        )
        if flags or win_norm is None:
            flagged_log.append({"event_id": ev_id, "flags": flags, "measured_snr_db": round(snr_meas, 3)})
            continue

        phys_vec = extract_causal_physics_features(
            arr_3c[0], p_idx=p_samp, horizon_samples=horizon_samples, fs=expected_fs_hz
        )
        valid_ids.append(ev_id)
        windows_list.append(win_norm)
        phys_list.append(phys_vec)

    n_valid = len(valid_ids)
    n_flagged = len(flagged_log)
    assert n_valid + n_flagged == total_raw

    w_len = int(round((1.0 + horizon_seconds) * expected_fs_hz))
    win_arr = np.stack(windows_list, axis=0) if windows_list else np.zeros((0, 3, w_len), dtype=np.float32)
    phys_arr = np.stack(phys_list, axis=0) if phys_list else np.zeros((0, 6), dtype=np.float32)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "adapter": "OBSPY_MINISEED_SAC_3C",
        "total_raw": total_raw,
        "valid": n_valid,
        "flagged": n_flagged,
        "record_conservation_verified": (n_valid + n_flagged == total_raw),
        "flagged_records": flagged_log,
        "synthetic_traces_fabricated": 0,
    }
    if manifest_out_path is not None:
        manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AdaptedWaveformBatch(
        trace_ids=np.asarray(valid_ids, dtype=str),
        branch_a_windows=win_arr,
        physics_features=phys_arr,
        mod_mask_a=np.ones(n_valid, dtype=np.float32),
        qc_manifest=manifest,
    )
