"""
Strict Target-Completed Walk-Forward Validation, Multi-Threshold Probabilistic Scoring
(`Mw >= 4.0, 4.5, 5.0, 5.5`), Tectonic Sub-Zone Spatial Holdout, Modality Ablation,
and Reproducible Diagnostic Plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from seismic_fusion.config import PipelineConfig
from seismic_fusion.model import DeepSeismicEnsemble
from seismic_fusion.preprocessing import AssembledFusionDataset


@dataclass
class FoldFeatureScaler:
    """
    Fold-scoped tensor & target normalizer fitted EXCLUSIVELY on training indices (`fit_idx`),
    guaranteeing zero future distribution leakage into calibration, validation, or test folds.
    """

    kin_mean: np.ndarray | None = None
    kin_std: np.ndarray | None = None
    node_mean: np.ndarray | None = None
    node_std: np.ndarray | None = None
    phys_mean: np.ndarray | None = None
    phys_std: np.ndarray | None = None
    mag_center: float = 4.5
    mag_scale: float = 1.0
    fit_max_timestamp_sec: float | None = None

    def fit(self, ds: AssembledFusionDataset, fit_idx: np.ndarray) -> "FoldFeatureScaler":
        if len(fit_idx) == 0:
            raise ValueError("Cannot fit FoldFeatureScaler on empty index array.")
        kin = ds.kinematic_seq[fit_idx]
        self.kin_mean = kin.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        k_sd = kin.std(axis=(0, 2), keepdims=True)
        self.kin_std = np.where(k_sd > 1e-6, k_sd, 1.0).astype(np.float32)

        nf = ds.node_feats[fit_idx]
        self.node_mean = nf.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        n_sd = nf.std(axis=(0, 1), keepdims=True)
        self.node_std = np.where(n_sd > 1e-6, n_sd, 1.0).astype(np.float32)

        pf = ds.phys_feats[fit_idx]
        self.phys_mean = pf.mean(axis=0, keepdims=True).astype(np.float32)
        p_sd = pf.std(axis=0, keepdims=True)
        self.phys_std = np.where(p_sd > 1e-6, p_sd, 1.0).astype(np.float32)

        reg_fit = ds.reg_mask[fit_idx]
        if np.any(reg_fit):
            m_fit = ds.future_max_mw[fit_idx][reg_fit]
            self.mag_center = float(np.mean(m_fit))
            m_sd = float(np.std(m_fit))
            self.mag_scale = m_sd if m_sd > 1e-4 else 1.0
        else:
            self.mag_center = 4.5
            self.mag_scale = 1.0

        self.fit_max_timestamp_sec = float(np.max(ds.target_end_timestamps_sec[fit_idx]))
        return self

    def transform(self, ds: AssembledFusionDataset, idx: np.ndarray) -> dict[str, np.ndarray]:
        if self.kin_mean is None or self.node_mean is None or self.phys_mean is None:
            raise RuntimeError("FoldFeatureScaler must be fitted before transform().")
        kin_n = ((ds.kinematic_seq[idx] - self.kin_mean) / self.kin_std).astype(np.float32)
        node_n = ((ds.node_feats[idx] - self.node_mean) / self.node_std).astype(np.float32)
        phys_n = ((ds.phys_feats[idx] - self.phys_mean) / self.phys_std).astype(np.float32)
        mw_raw = ds.future_max_mw[idx].astype(np.float32)
        mw_norm = ((mw_raw - self.mag_center) / self.mag_scale).astype(np.float32)
        return {
            "kinematic_seq": kin_n,
            "node_feats": node_n,
            "station_idx": ds.station_idx[idx],
            "phys_feats": phys_n,
            "mod_mask": ds.mod_mask[idx].copy(),
            "y_exceed_matrix": ds.y_exceed_matrix[idx],
            "y_exceed": ds.y_exceed[idx],
            "future_max_mw": mw_raw,
            "future_max_mw_norm": mw_norm,
            "reg_mask": ds.reg_mask[idx],
        }


def walk_forward_embargo_splits(
    timestamps_sec: np.ndarray,
    n_folds: int = 5,
    embargo_sec: float = 30.0 * 86400.0,
    horizon_sec: float | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window chronological splits enforcing BOTH:
      1. Full target completion before decision cutoff: `t_train + horizon_sec <= cutoff_t`
      2. Post-cutoff embargo gap before validation origins: `t_val >= cutoff_t + embargo_sec`
    Consequently, `min(t_val) - max(t_train_target_end) >= embargo_sec` and
    `min(t_val) - max(t_train_origin) >= horizon_sec + embargo_sec`.
    """
    if embargo_sec < 0:
        raise ValueError("embargo_sec must be non-negative.")
    eff_horizon = embargo_sec if horizon_sec is None else float(horizon_sec)
    order = np.argsort(timestamps_sec, kind="mergesort")
    ts = timestamps_sec[order]
    t_min, t_max = float(ts[0]), float(ts[-1])
    span = (t_max - t_min) / float(n_folds + 1)

    for fold_k in range(1, n_folds + 1):
        cutoff_t = t_min + fold_k * span
        val_start = cutoff_t + embargo_sec
        val_end = min(val_start + span, t_max)
        tr_idx = order[(ts + eff_horizon) <= cutoff_t]
        va_idx = order[(ts >= val_start) & (ts <= val_end)]
        if len(tr_idx) >= 64 and len(va_idx) >= 16:
            assert float(np.min(timestamps_sec[va_idx]) - np.max(timestamps_sec[tr_idx] + eff_horizon)) >= embargo_sec - 1e-6
            yield tr_idx, va_idx


def split_inner_train_calibration(
    timestamps_sec: np.ndarray,
    train_idx: np.ndarray,
    embargo_sec: float,
    horizon_sec: float | None = None,
    cal_fraction: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a fold's training set chronologically into `inner_tr` and `inner_cal` such that
    every `inner_tr` target window completes BEFORE `min(timestamps_sec[inner_cal])`.
    """
    eff_horizon = embargo_sec if horizon_sec is None else float(horizon_sec)
    ordered_tr = train_idx[np.argsort(timestamps_sec[train_idx], kind="mergesort")]
    cal_count = max(int(len(ordered_tr) * cal_fraction), 20)
    if len(ordered_tr) <= cal_count + 20:
        raise ValueError("Insufficient training events inside fold for target-isolated calibration split.")

    inner_cal = ordered_tr[-cal_count:]
    cal_start_t = float(timestamps_sec[inner_cal[0]])
    inner_tr = ordered_tr[(timestamps_sec[ordered_tr] + eff_horizon) <= cal_start_t]
    if len(inner_tr) < 24:
        # Relax calibration size slightly while preserving target completion invariance
        inner_cal = ordered_tr[-16:]
        cal_start_t = float(timestamps_sec[inner_cal[0]])
        inner_tr = ordered_tr[(timestamps_sec[ordered_tr] + eff_horizon) <= cal_start_t]
    if len(inner_tr) < 16 or len(inner_cal) < 8:
        raise ValueError("Insufficient temporal span inside training fold to isolate inner target horizon.")
    return inner_tr, inner_cal


def compute_probabilistic_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    p_null: np.ndarray,
    p_etas: np.ndarray | None = None,
    mu_mag: np.ndarray | None = None,
    std_mag: np.ndarray | None = None,
    m_true: np.ndarray | None = None,
    reg_mask: np.ndarray | None = None,
    ece_bins: int = 10,
    min_positives: int = 5,
) -> dict[str, Any]:
    """
    Computes proper probabilistic scoring metrics for a binary exceedance threshold.
    Explicitly logs `'insufficient_positives'` when positives or negatives < `min_positives`
    instead of forcing degenerate ranking/skill ratios.
    """
    eps = 1e-7
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    p0 = np.clip(np.asarray(p_null, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=np.float64)

    n_total = int(len(y))
    n_pos = int(np.sum(y == 1.0))
    n_neg = n_total - n_pos
    brier = float(np.mean((p - y) ** 2))
    brier_null = float(np.mean((p0 - y) ** 2))

    bin_edges = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    for b_lo, b_hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (p >= b_lo) & (p < b_hi if b_hi < 1.0 else p <= b_hi)
        if np.any(in_bin):
            ece += (float(np.sum(in_bin)) / max(n_total, 1)) * abs(float(np.mean(p[in_bin]) - np.mean(y[in_bin])))

    sufficient = (n_pos >= min_positives) and (n_neg >= min_positives)
    out: dict[str, Any] = {
        "n_samples": n_total,
        "positives": n_pos,
        "negatives": n_neg,
        "positive_rate": round(float(np.mean(y)), 6) if n_total > 0 else 0.0,
        "brier_score": round(brier, 6),
        "brier_poisson_null": round(brier_null, 6),
        "ece": round(float(ece), 6),
    }

    if not sufficient:
        out["status"] = "insufficient_positives"
        out["brier_skill_score"] = "insufficient_positives"
        out["bss_vs_etas"] = "insufficient_positives"
        out["igpe_bits_per_eq"] = "insufficient_positives"
        out["log_likelihood_ratio_nats"] = "insufficient_positives"
        out["pr_auc"] = "insufficient_positives"
        out["roc_auc"] = "insufficient_positives"
    else:
        out["status"] = "valid"
        bss = 1.0 - (brier / max(brier_null, 1e-12))
        ll_model_nats = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
        ll_null_nats = y * np.log(p0) + (1.0 - y) * np.log(1.0 - p0)
        llr_nats = float(np.sum(ll_model_nats - ll_null_nats))
        igpe_bits = float(np.sum((ll_model_nats - ll_null_nats) / np.log(2.0)) / float(n_pos))
        out["brier_skill_score"] = round(bss, 6)
        if p_etas is not None:
            pe = np.clip(np.asarray(p_etas, dtype=np.float64), eps, 1.0 - eps)
            brier_etas = float(np.mean((pe - y) ** 2))
            out["brier_etas"] = round(brier_etas, 6)
            out["bss_vs_etas"] = round(1.0 - (brier / max(brier_etas, 1e-12)), 6)
        out["igpe_bits_per_eq"] = round(igpe_bits, 6)
        out["log_likelihood_ratio_nats"] = round(llr_nats, 4)
        out["pr_auc"] = round(float(average_precision_score(y, p)), 6)
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 6)

    if mu_mag is not None and m_true is not None and reg_mask is not None and np.any(reg_mask):
        m_obs = m_true[reg_mask]
        m_hat = mu_mag[reg_mask]
        out["mag_mae"] = round(float(np.mean(np.abs(m_hat - m_obs))), 6)
        if np.std(m_obs) > 1e-6:
            out["mag_sd_ratio"] = round(float(np.std(m_hat) / np.std(m_obs)), 6)
        strong = m_obs >= 5.0
        if np.any(strong):
            out["strong_mw_ge_5_count"] = int(np.sum(strong))
            out["strong_mw_ge_5_mae"] = round(float(np.mean(np.abs(m_hat[strong] - m_obs[strong]))), 6)
        if std_mag is not None:
            s_hat = std_mag[reg_mask]
            z90 = 1.6448536
            covered = (m_obs >= (m_hat - z90 * s_hat)) & (m_obs <= (m_hat + z90 * s_hat))
            out["picp_90"] = round(float(np.mean(covered)), 6)

    return out


def compute_multithreshold_table(
    probs_mat: np.ndarray,
    y_mat: np.ndarray,
    p_null_mat: np.ndarray,
    p_etas_mat: np.ndarray | None,
    thresholds: Tuple[float, ...],
    ece_bins: int = 10,
    min_positives: int = 5,
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for k_i, th in enumerate(thresholds):
        key = f"Mw>={th:.1f}"
        table[key] = compute_probabilistic_metrics(
            probs=probs_mat[:, k_i],
            y_true=y_mat[:, k_i],
            p_null=p_null_mat[:, k_i],
            p_etas=(p_etas_mat[:, k_i] if p_etas_mat is not None else None),
            ece_bins=ece_bins,
            min_positives=min_positives,
        )
    return table


def fit_etas_multithreshold_baseline(
    tr_phys: np.ndarray,
    tr_y_mat: np.ndarray,
    va_phys: np.ndarray,
    p_null_mat: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    etas_cols = [1, 2, 3, 4, 5]
    k_th = tr_y_mat.shape[1]
    out = np.zeros((len(va_phys), k_th), dtype=np.float64)
    for k_i in range(k_th):
        yk = tr_y_mat[:, k_i]
        if int(np.sum(yk == 1.0)) >= 3 and int(np.sum(yk == 0.0)) >= 3:
            clf = LogisticRegression(random_state=seed, max_iter=500)
            clf.fit(tr_phys[:, etas_cols], yk)
            out[:, k_i] = clf.predict_proba(va_phys[:, etas_cols])[:, 1]
        else:
            out[:, k_i] = p_null_mat[:, k_i]
    return out


def evaluate_modality_ablation(
    ensemble: DeepSeismicEnsemble,
    eval_tensors: dict[str, np.ndarray],
    adj_norm: np.ndarray,
    p_null_primary: np.ndarray,
    primary_col: int = 2,
    ece_bins: int = 10,
    min_positives: int = 5,
) -> dict[str, dict[str, Any]]:
    n = len(eval_tensors["y_exceed"])
    configs = {
        "full_multimodal": np.tile(np.array([[1.0, 1.0, 1.0]], dtype=np.float32), (n, 1)),
        "ablate_branch_a_kinematic_wf": np.tile(np.array([[0.0, 1.0, 1.0]], dtype=np.float32), (n, 1)),
        "ablate_branch_b_spatial_gnn": np.tile(np.array([[1.0, 0.0, 1.0]], dtype=np.float32), (n, 1)),
        "ablate_branch_c_catalog_etas": np.tile(np.array([[1.0, 1.0, 0.0]], dtype=np.float32), (n, 1)),
    }
    ablation_out: dict[str, dict[str, Any]] = {}
    y_col = eval_tensors["y_exceed_matrix"][:, primary_col]
    for name, mask_mat in configs.items():
        pred = ensemble.predict(eval_tensors, adj_norm, override_mod_mask=mask_mat, primary_col=primary_col)
        metrics = compute_probabilistic_metrics(
            probs=pred["prob"],
            y_true=y_col,
            p_null=p_null_primary,
            mu_mag=pred["mu_mag"],
            std_mag=pred["total_mag_std"],
            m_true=eval_tensors["future_max_mw"],
            reg_mask=eval_tensors["reg_mask"],
            ece_bins=ece_bins,
            min_positives=min_positives,
        )
        metrics["mean_gate_weights"] = [round(float(x), 4) for x in np.mean(pred["gate_weights"], axis=0)]
        ablation_out[name] = metrics
    return ablation_out


def generate_validation_figures(
    probs_mat: np.ndarray,
    y_mat: np.ndarray,
    gate_weights: np.ndarray,
    thresholds: Tuple[float, ...],
    figures_dir: Path,
) -> list[str]:
    """Saves publication-clean reliability curves, multi-threshold PR curves, and gate distributions."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # 1. Reliability / Calibration Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration (1:1)")
    bins = np.linspace(0.0, 1.0, 11)
    for k_i, th in enumerate(thresholds):
        p = probs_mat[:, k_i]
        y = y_mat[:, k_i]
        xs, ys = [], []
        for b_lo, b_hi in zip(bins[:-1], bins[1:]):
            m = (p >= b_lo) & (p < b_hi if b_hi < 1.0 else p <= b_hi)
            if np.sum(m) >= 5:
                xs.append(float(np.mean(p[m])))
                ys.append(float(np.mean(y[m])))
        if xs:
            ax.plot(xs, ys, marker="o", lw=2.0, color=colors[k_i % len(colors)], label=f"Mw >= {th:.1f}")
    ax.set_xlabel("Predicted Exceedance Probability")
    ax.set_ylabel("Observed Empirical Frequency")
    ax.set_title("Walk-Forward Out-of-Sample Reliability Diagrams (30-Day Horizon)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    rel_path = figures_dir / "reliability_calibration_curves.png"
    fig.tight_layout()
    fig.savefig(rel_path, dpi=220)
    plt.close(fig)
    saved.append(rel_path.name)

    # 2. Multi-Threshold Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for k_i, th in enumerate(thresholds):
        p = probs_mat[:, k_i]
        y = y_mat[:, k_i]
        if len(np.unique(y)) > 1:
            prec, rec, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            base = float(np.mean(y))
            c = colors[k_i % len(colors)]
            ax.plot(rec, prec, lw=2.0, color=c, label=f"Mw >= {th:.1f} (PR-AUC={ap:.3f}, base={base:.2f})")
            ax.axhline(base, linestyle=":", color=c, alpha=0.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves Across Magnitude Thresholds")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=9)
    pr_path = figures_dir / "pr_curves_multi_threshold.png"
    fig.tight_layout()
    fig.savefig(pr_path, dpi=220)
    plt.close(fig)
    saved.append(pr_path.name)

    # 3. Modality Gate-Weight Distributions
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Branch A:\nKinematic Seq", "Branch B:\n3D Geodesic GNN", "Branch C:\nCatalog / ETAS"]
    bp = ax.boxplot(
        [gate_weights[:, 0], gate_weights[:, 1], gate_weights[:, 2]],
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
    )
    palette = ["#4c72b0", "#55a868", "#c44e52"]
    for patch, col in zip(bp["boxes"], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax.set_ylabel("Learned Softmax Gate Weight")
    ax.set_title("Out-of-Sample Modality Gate-Weight Distribution")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    gate_path = figures_dir / "gate_weight_distributions.png"
    fig.tight_layout()
    fig.savefig(gate_path, dpi=220)
    plt.close(fig)
    saved.append(gate_path.name)

    return saved


def run_walk_forward_and_spatial_validation(
    ds: AssembledFusionDataset,
    cfg: PipelineConfig,
    figures_dir: Path | None = None,
) -> Tuple[dict[str, Any], DeepSeismicEnsemble, dict[str, np.ndarray]]:
    """
    Executes 5-fold target-completed walk-forward validation across `Mw >= 4.0, 4.5, 5.0, 5.5`,
    geographically disjoint Indo-Burma tectonic sub-zone spatial holdout, station holdout,
    modality ablations, and figure export. Also returns the final-fold trained ensemble
    and held-out test tensors for deterministic post-hoc XAI inspection.
    """
    embargo_sec = cfg.validation.embargo_days * 86400.0
    horizon_sec = cfg.validation.forecast_horizon_days * 86400.0
    if embargo_sec < horizon_sec:
        raise ValueError("Validation embargo must be >= forecast horizon.")

    thresholds = ds.thresholds
    k_th = len(thresholds)
    primary_col = thresholds.index(cfg.validation.target_mw_threshold)
    min_pos = cfg.validation.min_positives_for_metric

    fold_reports: list[dict[str, Any]] = []
    pooled_probs_mat: list[np.ndarray] = []
    pooled_null_mat: list[np.ndarray] = []
    pooled_etas_mat: list[np.ndarray] = []
    pooled_y_mat: list[np.ndarray] = []
    pooled_mu: list[np.ndarray] = []
    pooled_std: list[np.ndarray] = []
    pooled_m_true: list[np.ndarray] = []
    pooled_reg_mask: list[np.ndarray] = []
    pooled_gates: list[np.ndarray] = []

    last_ensemble: DeepSeismicEnsemble | None = None
    last_va_tensors: dict[str, np.ndarray] | None = None
    last_ablation: dict[str, dict[str, Any]] = {}

    for fold_idx, (tr_idx, va_idx) in enumerate(
        walk_forward_embargo_splits(
            ds.timestamps_sec,
            n_folds=cfg.validation.n_walk_forward_folds,
            embargo_sec=embargo_sec,
            horizon_sec=horizon_sec,
        ),
        start=1,
    ):
        inner_tr, inner_cal = split_inner_train_calibration(
            ds.timestamps_sec, tr_idx, embargo_sec=embargo_sec, horizon_sec=horizon_sec
        )

        scaler = FoldFeatureScaler().fit(ds, inner_tr)
        tr_tensors = scaler.transform(ds, inner_tr)
        cal_tensors = scaler.transform(ds, inner_cal)
        va_tensors = scaler.transform(ds, va_idx)

        base_rates_tr = np.clip(np.mean(ds.y_exceed_matrix[tr_idx], axis=0), 1e-4, 1.0 - 1e-4)
        p_null_mat_va = np.tile(base_rates_tr[None, :], (len(va_idx), 1))

        p_etas_mat_va = fit_etas_multithreshold_baseline(
            tr_tensors["phys_feats"],
            tr_tensors["y_exceed_matrix"],
            va_tensors["phys_feats"],
            p_null_mat_va,
            seed=cfg.validation.random_seed,
        )

        ensemble = DeepSeismicEnsemble(cfg.model, n_thresholds=k_th)
        ensemble.fit(
            tr_tensors,
            cal_tensors,
            ds.adj_norm,
            mag_center=scaler.mag_center,
            mag_scale=scaler.mag_scale,
        )
        pred_va = ensemble.predict(va_tensors, ds.adj_norm, primary_col=primary_col)

        fold_multi = compute_multithreshold_table(
            probs_mat=pred_va["probs_matrix"],
            y_mat=va_tensors["y_exceed_matrix"],
            p_null_mat=p_null_mat_va,
            p_etas_mat=p_etas_mat_va,
            thresholds=thresholds,
            ece_bins=cfg.validation.ece_bins,
            min_positives=min_pos,
        )
        primary_metrics = compute_probabilistic_metrics(
            probs=pred_va["prob"],
            y_true=va_tensors["y_exceed_matrix"][:, primary_col],
            p_null=p_null_mat_va[:, primary_col],
            p_etas=p_etas_mat_va[:, primary_col],
            mu_mag=pred_va["mu_mag"],
            std_mag=pred_va["total_mag_std"],
            m_true=va_tensors["future_max_mw"],
            reg_mask=va_tensors["reg_mask"],
            ece_bins=cfg.validation.ece_bins,
            min_positives=min_pos,
        )

        origin_gap_days = float((np.min(ds.timestamps_sec[va_idx]) - np.max(ds.timestamps_sec[tr_idx])) / 86400.0)
        target_end_gap_days = float(
            (np.min(ds.timestamps_sec[va_idx]) - np.max(ds.target_end_timestamps_sec[tr_idx])) / 86400.0
        )

        fold_reports.append(
            {
                "fold": fold_idx,
                "train_size": int(len(tr_idx)),
                "inner_train_size": int(len(inner_tr)),
                "inner_cal_size": int(len(inner_cal)),
                "val_size": int(len(va_idx)),
                "origin_gap_days": round(origin_gap_days, 3),
                "post_target_completion_embargo_days": round(target_end_gap_days, 3),
                "calibrated_temperatures": [
                    [round(t, 4) for t in m_temps] for m_temps in ensemble.temperatures
                ],
                "primary_threshold_metrics": primary_metrics,
                "multi_threshold_metrics": fold_multi,
            }
        )

        pooled_probs_mat.append(pred_va["probs_matrix"])
        pooled_null_mat.append(p_null_mat_va)
        pooled_etas_mat.append(p_etas_mat_va)
        pooled_y_mat.append(va_tensors["y_exceed_matrix"])
        pooled_mu.append(pred_va["mu_mag"])
        pooled_std.append(pred_va["total_mag_std"])
        pooled_m_true.append(va_tensors["future_max_mw"])
        pooled_reg_mask.append(va_tensors["reg_mask"])
        pooled_gates.append(pred_va["gate_weights"])

        last_ensemble = ensemble
        last_va_tensors = va_tensors
        last_p_null_primary = p_null_mat_va[:, primary_col]

    assert last_ensemble is not None and last_va_tensors is not None
    last_ablation = evaluate_modality_ablation(
        last_ensemble,
        last_va_tensors,
        ds.adj_norm,
        last_p_null_primary,
        primary_col=primary_col,
        ece_bins=cfg.validation.ece_bins,
        min_positives=min_pos,
    )

    all_probs_mat = np.concatenate(pooled_probs_mat, axis=0)
    all_null_mat = np.concatenate(pooled_null_mat, axis=0)
    all_etas_mat = np.concatenate(pooled_etas_mat, axis=0)
    all_y_mat = np.concatenate(pooled_y_mat, axis=0)
    all_mu = np.concatenate(pooled_mu, axis=0)
    all_std = np.concatenate(pooled_std, axis=0)
    all_m_true = np.concatenate(pooled_m_true, axis=0)
    all_reg = np.concatenate(pooled_reg_mask, axis=0)
    all_gates = np.concatenate(pooled_gates, axis=0)

    pooled_multi_fusion = compute_multithreshold_table(
        all_probs_mat,
        all_y_mat,
        all_null_mat,
        all_etas_mat,
        thresholds=thresholds,
        ece_bins=cfg.validation.ece_bins,
        min_positives=min_pos,
    )
    pooled_multi_etas = compute_multithreshold_table(
        all_etas_mat,
        all_y_mat,
        all_null_mat,
        None,
        thresholds=thresholds,
        ece_bins=cfg.validation.ece_bins,
        min_positives=min_pos,
    )
    pooled_primary = compute_probabilistic_metrics(
        all_probs_mat[:, primary_col],
        all_y_mat[:, primary_col],
        all_null_mat[:, primary_col],
        p_etas=all_etas_mat[:, primary_col],
        mu_mag=all_mu,
        std_mag=all_std,
        m_true=all_m_true,
        reg_mask=all_reg,
        ece_bins=cfg.validation.ece_bins,
        min_positives=min_pos,
    )

    # 1. Geographically Disjoint Tectonic Sub-Zone Holdout:
    # Train on Western Shillong-Himalayan Foredeep (Long <= lon_min - buffer),
    # exclude buffer strip, evaluate on Eastern Indo-Burma Subduction Arc (Long >= lon_min) after temporal cutoff.
    lon_min = cfg.validation.spatial_holdout_lon_min
    buf_deg = cfg.validation.spatial_buffer_deg
    west_idx = np.flatnonzero(ds.longitudes <= (lon_min - buf_deg))
    east_idx = np.flatnonzero(ds.longitudes >= lon_min)

    tectonic_holdout_report: dict[str, Any] = {
        "train_region": f"West_Shillong_Himalayan (Long <= {lon_min - buf_deg:.2f}E)",
        "buffer_excluded_region": f"Buffer ({lon_min - buf_deg:.2f}E < Long < {lon_min:.2f}E)",
        "eval_region": f"East_Indo_Burma_Subduction (Long >= {lon_min:.2f}E)",
    }
    if len(west_idx) >= 96 and len(east_idx) >= 32:
        w_ord = west_idx[np.argsort(ds.timestamps_sec[west_idx], kind="mergesort")]
        t_cut_sp = float(np.quantile(ds.timestamps_sec[w_ord], 0.72))
        sp_tr = w_ord[ds.target_end_timestamps_sec[w_ord] <= t_cut_sp]
        sp_va = east_idx[ds.timestamps_sec[east_idx] >= (t_cut_sp + embargo_sec)]
        if len(sp_tr) >= 64 and len(sp_va) >= 24:
            sp_in_tr, sp_in_cal = split_inner_train_calibration(
                ds.timestamps_sec, sp_tr, embargo_sec=embargo_sec, horizon_sec=horizon_sec
            )
            sp_sc = FoldFeatureScaler().fit(ds, sp_in_tr)
            sp_tr_t = sp_sc.transform(ds, sp_in_tr)
            sp_cal_t = sp_sc.transform(ds, sp_in_cal)
            sp_va_t = sp_sc.transform(ds, sp_va)

            sp_base = np.clip(np.mean(ds.y_exceed_matrix[sp_tr], axis=0), 1e-4, 1.0 - 1e-4)
            sp_null_mat = np.tile(sp_base[None, :], (len(sp_va), 1))
            sp_etas_mat = fit_etas_multithreshold_baseline(
                sp_tr_t["phys_feats"],
                sp_tr_t["y_exceed_matrix"],
                sp_va_t["phys_feats"],
                sp_null_mat,
                seed=cfg.validation.random_seed,
            )

            sp_ens = DeepSeismicEnsemble(cfg.model, n_thresholds=k_th)
            sp_ens.fit(sp_tr_t, sp_cal_t, ds.adj_norm, mag_center=sp_sc.mag_center, mag_scale=sp_sc.mag_scale)
            sp_pred = sp_ens.predict(sp_va_t, ds.adj_norm, primary_col=primary_col)

            tectonic_holdout_report["train_size"] = int(len(sp_tr))
            tectonic_holdout_report["eval_size"] = int(len(sp_va))
            tectonic_holdout_report["multi_threshold_metrics"] = compute_multithreshold_table(
                sp_pred["probs_matrix"],
                sp_va_t["y_exceed_matrix"],
                sp_null_mat,
                sp_etas_mat,
                thresholds=thresholds,
                ece_bins=cfg.validation.ece_bins,
                min_positives=min_pos,
            )

    # 2. Disjoint Station Holdout (KOH, SHL)
    holdout_set = set(cfg.validation.holdout_stations)
    is_holdout_st = np.isin(ds.station_codes, list(holdout_set))
    non_holdout_idx = np.flatnonzero(~is_holdout_st)
    holdout_idx = np.flatnonzero(is_holdout_st)
    station_holdout_report: dict[str, Any] = {"holdout_stations": sorted(holdout_set)}
    if len(non_holdout_idx) >= 96 and len(holdout_idx) >= 16:
        nh_ord = non_holdout_idx[np.argsort(ds.timestamps_sec[non_holdout_idx], kind="mergesort")]
        t_cut_st = float(np.quantile(ds.timestamps_sec[nh_ord], 0.72))
        st_tr = nh_ord[ds.target_end_timestamps_sec[nh_ord] <= t_cut_st]
        st_va = holdout_idx[ds.timestamps_sec[holdout_idx] >= (t_cut_st + embargo_sec)]
        if len(st_tr) >= 64 and len(st_va) >= 16:
            st_in_tr, st_in_cal = split_inner_train_calibration(
                ds.timestamps_sec, st_tr, embargo_sec=embargo_sec, horizon_sec=horizon_sec
            )
            st_sc = FoldFeatureScaler().fit(ds, st_in_tr)
            st_tr_t = st_sc.transform(ds, st_in_tr)
            st_cal_t = st_sc.transform(ds, st_in_cal)
            st_va_t = st_sc.transform(ds, st_va)

            st_base = np.clip(np.mean(ds.y_exceed_matrix[st_tr], axis=0), 1e-4, 1.0 - 1e-4)
            st_null_mat = np.tile(st_base[None, :], (len(st_va), 1))
            st_etas_mat = fit_etas_multithreshold_baseline(
                st_tr_t["phys_feats"],
                st_tr_t["y_exceed_matrix"],
                st_va_t["phys_feats"],
                st_null_mat,
                seed=cfg.validation.random_seed,
            )

            st_ens = DeepSeismicEnsemble(cfg.model, n_thresholds=k_th)
            st_ens.fit(st_tr_t, st_cal_t, ds.adj_norm, mag_center=st_sc.mag_center, mag_scale=st_sc.mag_scale)
            st_pred = st_ens.predict(st_va_t, ds.adj_norm, primary_col=primary_col)

            station_holdout_report["train_non_holdout_size"] = int(len(st_tr))
            station_holdout_report["eval_holdout_future_size"] = int(len(st_va))
            station_holdout_report["multi_threshold_metrics"] = compute_multithreshold_table(
                st_pred["probs_matrix"],
                st_va_t["y_exceed_matrix"],
                st_null_mat,
                st_etas_mat,
                thresholds=thresholds,
                ece_bins=cfg.validation.ece_bins,
                min_positives=min_pos,
            )

    saved_figs: list[str] = []
    if figures_dir is not None:
        saved_figs = generate_validation_figures(
            all_probs_mat, all_y_mat, all_gates, thresholds=thresholds, figures_dir=figures_dir
        )

    report = {
        "primary_target": (
            f"Probabilistic exceedance Mw >= {cfg.validation.target_mw_threshold:.1f} "
            f"within forward {cfg.validation.forecast_horizon_days:.0f}-day horizon"
        ),
        "evaluated_thresholds": [f"Mw>={th:.1f}" for th in thresholds],
        "target_completion_and_embargo_enforced": True,
        "embargo_days_post_target_completion": cfg.validation.embargo_days,
        "ensemble_members": len(cfg.model.ensemble_seeds),
        "walk_forward_folds": fold_reports,
        "pooled_out_of_sample_primary": pooled_primary,
        "pooled_multi_threshold_fusion": pooled_multi_fusion,
        "pooled_multi_threshold_etas_baseline": pooled_multi_etas,
        "modality_ablation_final_fold": last_ablation,
        "tectonic_subzone_spatial_holdout": tectonic_holdout_report,
        "station_spatial_holdout": station_holdout_report,
        "generated_figures": saved_figs,
    }
    return report, last_ensemble, last_va_tensors
