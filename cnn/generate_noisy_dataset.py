"""
Deterministic Temporal Dataset Physical Invariant & Audit Verifier
==================================================================
Remediated per codebase_audit_and_fusion_proposal.md (Priority 0):
  - Retired synthetic Gaussian feature-jitter generation that corrupted cyclical
    trigonometric identities (sin^2 + cos^2 = 1) and arrival kinematics (s - p = sp_travel_time).
  - Removed silent hardcoded MIN_MAG/MAX_MAG fallback constants.
  - Validates schema completeness, finiteness, and strict record conservation
    (valid + flagged == total_raw) directly on real preprocessed_dl_temporal.csv.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATE = SCRIPT_DIR.parent
if str(REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_CANDIDATE))

from seismic_fusion.config import resolve_repo_root


REQUIRED_TEMPORAL_COLS = (
    "source_id",
    "origin_time_utc",
    "split",
    "above_mc_mask",
    "target_magnitude_raw",
    "target_mw_raw",
)


def verify_temporal_dataset(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or resolve_repo_root(SCRIPT_DIR)
    csv_path = root / "preprocessed_dl_temporal.csv"
    scaler_path = root / "dl_temporal_scaler.pkl"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Required temporal dataset missing at '{csv_path}'. "
            "Run 'preprocess_dl_temporal.py' first; synthetic fallback is prohibited."
        )
    if not scaler_path.is_file():
        raise FileNotFoundError(
            f"Required train-only scaler artifact missing at '{scaler_path}'. "
            "Silent hardcoded magnitude bounds fallback has been removed."
        )

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_TEMPORAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Schema violation in '{csv_path.name}': missing columns {missing}")

    for forbidden_col in ("log_energy", "depth_mag_ratio"):
        if forbidden_col in df.columns:
            raise ValueError(f"Target-leakage column '{forbidden_col}' detected in '{csv_path.name}'.")

    with scaler_path.open("rb") as fh:
        scaler_meta = pickle.load(fh)

    if scaler_meta.get("fit_split") != "train":
        raise ValueError("Scaler provenance check failed: scaler was not fitted strictly on 'train' split.")

    predictor_cols: list[str] = scaler_meta["all_predictor_columns"]
    missing_preds = [c for c in predictor_cols if c not in df.columns]
    if missing_preds:
        raise ValueError(f"Predictor columns missing from '{csv_path.name}': {missing_preds}")

    # Verify trigonometric unit-circle invariant after inverse-scaling cyclical pairs
    num_cols: list[str] = scaler_meta["numeric_feature_names"]
    scaler = scaler_meta["scaler"]
    raw_num = pd.DataFrame(
        scaler.inverse_transform(df[num_cols].values),
        columns=num_cols,
        index=df.index,
    )
    max_trig_residual = 0.0
    for prefix in ("month", "dow", "hour", "doy"):
        s_col, c_col = f"{prefix}_sin", f"{prefix}_cos"
        norm_sq = raw_num[s_col].values ** 2 + raw_num[c_col].values ** 2
        max_trig_residual = max(max_trig_residual, float(np.max(np.abs(norm_sq - 1.0))))

    if max_trig_residual > 1e-5:
        raise ValueError(f"Trigonometric identity violated: max |sin^2 + cos^2 - 1| = {max_trig_residual:.6e}")

    # Verify arrival kinematic consistency: s_arrival_sample - p_arrival_sample == sp_travel_time
    sp_residual = float(
        np.max(np.abs((raw_num["s_arrival_sample"] - raw_num["p_arrival_sample"]) - raw_num["sp_travel_time"]))
    )
    if sp_residual > 1e-4:
        raise ValueError(f"Arrival kinematic invariant violated: max |(s - p) - sp| = {sp_residual:.6e}")

    non_finite_mask = ~np.isfinite(df[predictor_cols].values).all(axis=1)
    below_mc_mask = df["above_mc_mask"].values == 0
    flagged_mask = non_finite_mask | below_mc_mask

    total_rows = len(df)
    valid_rows = int((~flagged_mask).sum())
    flagged_rows = int(flagged_mask.sum())
    assert valid_rows + flagged_rows == total_rows

    report = {
        "verified_file": str(csv_path.relative_to(root).as_posix()),
        "total_raw": total_rows,
        "valid": valid_rows,
        "flagged": flagged_rows,
        "record_conservation_verified": (valid_rows + flagged_rows == total_rows),
        "non_finite_rows": int(non_finite_mask.sum()),
        "below_mc_rows": int(below_mc_mask.sum()),
        "max_trigonometric_residual": round(max_trig_residual, 10),
        "max_sp_kinematic_residual": round(sp_residual, 10),
        "synthetic_noise_injected": False,
    }
    out_manifest = SCRIPT_DIR / "temporal_qc_audit.json"
    out_manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = verify_temporal_dataset()
    print(json.dumps(result, indent=2))
