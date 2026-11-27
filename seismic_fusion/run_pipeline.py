"""
CLI Entrypoint for the Leak-Free Multi-Modal Seismic Fusion, Multi-Threshold & Spatial
Validation, Diagnostic Figure Export, and Post-Hoc XAI Attribution Pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATE = SCRIPT_DIR.parent
if str(REPO_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_CANDIDATE))

from seismic_fusion.config import PipelineConfig, resolve_repo_root
from seismic_fusion.explainability import export_xai_report_and_figures
from seismic_fusion.preprocessing import ingest_and_assemble_fusion_dataset
from seismic_fusion.validation import run_walk_forward_and_spatial_validation


def write_remediated_sha256_manifest(repo_root: Path) -> Path:
    """
    Writes SHA-256 digests of active remediated Python modules and notebooks into a
    dedicated new manifest (`configs/remediated_sources_sha256.json`) without mutating
    historical audit snapshots.
    """
    hashes: dict[str, str] = {}
    for ext in ("*.py", "*.ipynb"):
        for fp in sorted(repo_root.rglob(ext)):
            if any(part in fp.parts for part in (".git", "archive", "graphify-out", "__pycache__")):
                continue
            h = hashlib.sha256()
            with fp.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            hashes[fp.relative_to(repo_root).as_posix()] = h.hexdigest()

    out_path = repo_root / "configs" / "remediated_sources_sha256.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "file_count": len(hashes),
        "sha256": hashes,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run end-to-end SeismicMultiModalFusion validation, spatial holdout, plots, and XAI."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to JSON configuration file (defaults to configs/seismic_fusion_config.json).",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root(SCRIPT_DIR)
    cfg_path = Path(args.config) if args.config else (repo_root / "configs" / "seismic_fusion_config.json")
    cfg = PipelineConfig.from_json(cfg_path) if cfg_path.is_file() else PipelineConfig()

    out_dir = repo_root / cfg.output_dir_rel_path
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = ingest_and_assemble_fusion_dataset(repo_root=repo_root, config=cfg)
    validation_report, last_ensemble, heldout_tensors = run_walk_forward_and_spatial_validation(
        dataset, cfg, figures_dir=figures_dir
    )

    primary_col = dataset.thresholds.index(cfg.validation.target_mw_threshold)
    xai_report = export_xai_report_and_figures(
        ensemble=last_ensemble,
        heldout_tensors=heldout_tensors,
        adj_norm=dataset.adj_norm,
        output_dir=out_dir,
        figures_dir=figures_dir,
        target_col=primary_col,
        target_label=f"Mw>={cfg.validation.target_mw_threshold:.1f}",
    )
    sha_manifest_path = write_remediated_sha256_manifest(repo_root)

    full_output = {
        "audit_manifest_summary": {
            "total_raw": dataset.audit_manifest["total_raw"],
            "valid": dataset.audit_manifest["valid"],
            "flagged": dataset.audit_manifest["flagged"],
            "record_conservation_verified": dataset.audit_manifest["record_conservation_verified"],
            "modern_anchor_samples": dataset.audit_manifest["modern_anchor_samples"],
            "causality_boundary_proof": dataset.audit_manifest["causality_boundary_proof"],
            "threshold_summary": dataset.audit_manifest["threshold_summary"],
            "waveform_archive_status": dataset.audit_manifest["waveform_archive_status"],
            "remediated_sha256_manifest": str(sha_manifest_path.relative_to(repo_root).as_posix()),
        },
        "validation": validation_report,
        "xai_summary": {
            "evaluated_split": xai_report["evaluated_split"],
            "explained_target": xai_report["explained_target"],
            "mean_completeness_mae_logit": xai_report["integrated_gradients"]["mean_completeness_mae_logit"],
            "stratified_modality_gate_weights": xai_report["stratified_modality_gate_weights"],
            "top_branch_a": dict(list(xai_report["integrated_gradients"]["branch_a_channel_importance"].items())[:3]),
            "top_branch_b": dict(
                list(xai_report["integrated_gradients"]["branch_b_station_feature_importance"].items())[:3]
            ),
            "top_branch_c": dict(
                list(xai_report["integrated_gradients"]["branch_c_physics_feature_importance"].items())[:3]
            ),
        },
    }

    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(full_output, indent=2), encoding="utf-8")
    print(json.dumps(full_output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
