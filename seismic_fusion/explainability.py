"""
Native Post-Hoc Explainability (XAI) Module for `SeismicMultiModalFusion`
=========================================================================
Computes strictly on held-out out-of-sample test folds:
  1. Modality gate-weight attributions stratified across peak horizon magnitude regimes
     (`Mw < 4.5`, `4.5 <= Mw < 5.0`, `Mw >= 5.0`).
  2. Deterministic Integrated Gradients (IG) feature attributions across all 3 branches
     (Branch A kinematic channels, Branch B station GNN features, Branch C catalog/ETAS features)
     with exact Riemann completeness residual checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from seismic_fusion.config import (
    BRANCH_A_FEATURE_NAMES,
    BRANCH_B_FEATURE_NAMES,
    BRANCH_C_FEATURE_NAMES,
)
from seismic_fusion.model import DeepSeismicEnsemble


def compute_stratified_gate_attributions(
    ensemble: DeepSeismicEnsemble,
    heldout_tensors: dict[str, np.ndarray],
    adj_norm: np.ndarray,
    primary_col: int = 2,
) -> dict[str, Any]:
    """Stratifies learned modality gate weights by observed forward-horizon peak Mw regime."""
    pred = ensemble.predict(heldout_tensors, adj_norm, primary_col=primary_col)
    gates = pred["gate_weights"]  # (N, 3)
    mw = heldout_tensors["future_max_mw"]
    reg = heldout_tensors["reg_mask"]

    strata = {
        "all_heldout": np.ones(len(mw), dtype=bool),
        "Mw_lt_4.5": reg & (mw < 4.5),
        "4.5_le_Mw_lt_5.0": reg & (mw >= 4.5) & (mw < 5.0),
        "Mw_ge_5.0": reg & (mw >= 5.0),
    }
    branch_labels = ("branch_a_kinematic_seq", "branch_b_geodesic_gnn", "branch_c_catalog_etas")
    out: dict[str, Any] = {}
    for s_name, mask in strata.items():
        cnt = int(np.sum(mask))
        if cnt == 0:
            out[s_name] = {"count": 0, "status": "empty_stratum"}
            continue
        sub = gates[mask]
        out[s_name] = {
            "count": cnt,
            "mean": {b: round(float(np.mean(sub[:, i])), 5) for i, b in enumerate(branch_labels)},
            "std": {b: round(float(np.std(sub[:, i])), 5) for i, b in enumerate(branch_labels)},
            "median": {b: round(float(np.median(sub[:, i])), 5) for i, b in enumerate(branch_labels)},
        }
    return out


def compute_integrated_gradients_heldout(
    ensemble: DeepSeismicEnsemble,
    heldout_tensors: dict[str, np.ndarray],
    adj_norm: np.ndarray,
    target_col: int = 2,
    n_steps: int = 32,
    max_eval_records: int = 256,
) -> dict[str, Any]:
    """
    Computes multi-branch Integrated Gradients (Sundararajan et al., 2017) on held-out
    test records relative to the zero-standardized training-mean baseline.
    """
    if not ensemble.members:
        raise RuntimeError("Ensemble must be fitted before computing Integrated Gradients.")

    n_total = len(heldout_tensors["y_exceed"])
    # Deterministic evenly spaced subsample across the held-out fold if large
    if n_total > max_eval_records:
        sel = np.linspace(0, n_total - 1, max_eval_records, dtype=np.int64)
    else:
        sel = np.arange(n_total, dtype=np.int64)

    device = ensemble.device
    adj_t = torch.from_numpy(adj_norm).to(device)
    wf_x = torch.from_numpy(heldout_tensors["kinematic_seq"][sel]).to(device)
    nf_x = torch.from_numpy(heldout_tensors["node_feats"][sel]).to(device)
    si_x = torch.from_numpy(heldout_tensors["station_idx"][sel]).to(device)
    pf_x = torch.from_numpy(heldout_tensors["phys_feats"][sel]).to(device)
    mm_x = torch.from_numpy(heldout_tensors["mod_mask"][sel]).to(device)

    wf_0 = torch.zeros_like(wf_x)
    nf_0 = torch.zeros_like(nf_x)
    pf_0 = torch.zeros_like(pf_x)

    alphas = torch.linspace(0.5 / n_steps, 1.0 - 0.5 / n_steps, n_steps, device=device)

    ig_wf_members = []
    ig_nf_members = []
    ig_pf_members = []
    completeness_errors = []

    for model in ensemble.members:
        model.eval()
        # Disable cuDNN RNN backward restriction by running deterministic eval forward with grad enabled on CPU/GPU
        grad_wf_sum = torch.zeros_like(wf_x)
        grad_nf_sum = torch.zeros_like(nf_x)
        grad_pf_sum = torch.zeros_like(pf_x)

        for alpha in alphas:
            wf_in = (wf_0 + alpha * (wf_x - wf_0)).detach().requires_grad_(True)
            nf_in = (nf_0 + alpha * (nf_x - nf_0)).detach().requires_grad_(True)
            pf_in = (pf_0 + alpha * (pf_x - pf_0)).detach().requires_grad_(True)

            out = model(wf_in, nf_in, adj_t, si_x, pf_in, mm_x)
            score = out["det_logits"][:, target_col].sum()
            g_wf, g_nf, g_pf = torch.autograd.grad(score, (wf_in, nf_in, pf_in))
            grad_wf_sum += g_wf.detach()
            grad_nf_sum += g_nf.detach()
            grad_pf_sum += g_pf.detach()

        ig_wf = (wf_x - wf_0) * (grad_wf_sum / float(n_steps))
        ig_nf = (nf_x - nf_0) * (grad_nf_sum / float(n_steps))
        ig_pf = (pf_x - pf_0) * (grad_pf_sum / float(n_steps))

        with torch.no_grad():
            f_x = model(wf_x, nf_x, adj_t, si_x, pf_x, mm_x)["det_logits"][:, target_col]
            f_0 = model(wf_0, nf_0, adj_t, si_x, pf_0, mm_x)["det_logits"][:, target_col]
            total_ig = ig_wf.sum(dim=(1, 2)) + ig_nf.sum(dim=(1, 2)) + ig_pf.sum(dim=1)
            comp_err = float((total_ig - (f_x - f_0)).abs().mean().item())
            completeness_errors.append(comp_err)

        ig_wf_members.append(ig_wf.cpu().numpy())
        ig_nf_members.append(ig_nf.cpu().numpy())
        ig_pf_members.append(ig_pf.cpu().numpy())

    mean_ig_wf = np.mean(np.stack(ig_wf_members, axis=0), axis=0)  # (B, C, L)
    mean_ig_nf = np.mean(np.stack(ig_nf_members, axis=0), axis=0)  # (B, V, F_gnn)
    mean_ig_pf = np.mean(np.stack(ig_pf_members, axis=0), axis=0)  # (B, F_phys)

    # Aggregate mean |IG| per named feature
    wf_imp = np.mean(np.sum(np.abs(mean_ig_wf), axis=2), axis=0)   # (C,)
    nf_imp = np.mean(np.sum(np.abs(mean_ig_nf), axis=1), axis=0)   # (F_gnn,)
    pf_imp = np.mean(np.abs(mean_ig_pf), axis=0)                   # (F_phys,)

    def ranked_dict(names: tuple[str, ...], vals: np.ndarray) -> dict[str, float]:
        pairs = sorted(zip(names, vals.tolist()), key=lambda kv: kv[1], reverse=True)
        return {k: round(float(v), 6) for k, v in pairs}

    return {
        "evaluated_heldout_records": int(len(sel)),
        "riemann_steps": n_steps,
        "mean_completeness_mae_logit": round(float(np.mean(completeness_errors)), 6),
        "branch_a_channel_importance": ranked_dict(BRANCH_A_FEATURE_NAMES, wf_imp),
        "branch_b_station_feature_importance": ranked_dict(BRANCH_B_FEATURE_NAMES, nf_imp),
        "branch_c_physics_feature_importance": ranked_dict(BRANCH_C_FEATURE_NAMES, pf_imp),
    }


def export_xai_report_and_figures(
    ensemble: DeepSeismicEnsemble,
    heldout_tensors: dict[str, np.ndarray],
    adj_norm: np.ndarray,
    output_dir: Path,
    figures_dir: Path,
    target_col: int = 2,
    target_label: str = "Mw>=5.0",
) -> dict[str, Any]:
    """Generates `xai_report.json` and `xai_feature_and_gate_attributions.png` strictly on held-out data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    strat_gates = compute_stratified_gate_attributions(
        ensemble, heldout_tensors, adj_norm, primary_col=target_col
    )
    ig_summary = compute_integrated_gradients_heldout(
        ensemble, heldout_tensors, adj_norm, target_col=target_col
    )

    # Render 2-panel XAI summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    regimes = ["Mw_lt_4.5", "4.5_le_Mw_lt_5.0", "Mw_ge_5.0"]
    regime_titles = ["Mw < 4.5", "4.5 <= Mw < 5.0", "Mw >= 5.0"]
    valid_regimes = [r for r in regimes if strat_gates.get(r, {}).get("count", 0) > 0]
    valid_titles = [t for r, t in zip(regimes, regime_titles) if r in valid_regimes]

    x_pos = np.arange(len(valid_regimes))
    width = 0.25
    b_keys = [
        ("branch_a_kinematic_seq", "Branch A (Kinematic Seq)", "#4c72b0"),
        ("branch_b_geodesic_gnn", "Branch B (3D Station GNN)", "#55a868"),
        ("branch_c_catalog_etas", "Branch C (Catalog / ETAS)", "#c44e52"),
    ]
    for offset_idx, (bk, blabel, bcol) in enumerate(b_keys):
        vals = [strat_gates[r]["mean"][bk] for r in valid_regimes]
        ax1.bar(x_pos + (offset_idx - 1) * width, vals, width=width, label=blabel, color=bcol, alpha=0.85)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(valid_titles)
    ax1.set_ylabel("Mean Softmax Gate Weight")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Held-Out Modality Gate Weights by Magnitude Regime")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8.5)

    # Top-10 features across all branches by mean |IG|
    combined_feats: list[tuple[str, float]] = []
    for k, v in ig_summary["branch_a_channel_importance"].items():
        combined_feats.append((f"[A] {k}", v))
    for k, v in ig_summary["branch_b_station_feature_importance"].items():
        combined_feats.append((f"[B] {k}", v))
    for k, v in ig_summary["branch_c_physics_feature_importance"].items():
        combined_feats.append((f"[C] {k}", v))
    combined_feats.sort(key=lambda kv: kv[1], reverse=True)
    top10 = combined_feats[:10][::-1]

    ax2.barh([x[0] for x in top10], [x[1] for x in top10], color="#2b5c8f", alpha=0.85)
    ax2.set_xlabel(f"Mean |Integrated Gradients| ({target_label} Logit)")
    ax2.set_title(f"Top-10 Held-Out Feature Attributions ({target_label})")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    fig_path = figures_dir / "xai_feature_and_gate_attributions.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    report = {
        "evaluated_split": "held_out_final_walk_forward_test_fold",
        "explained_target": target_label,
        "stratified_modality_gate_weights": strat_gates,
        "integrated_gradients": ig_summary,
        "figure_exported": fig_path.name,
    }
    (output_dir / "xai_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
