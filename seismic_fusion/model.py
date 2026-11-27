"""
Multi-Modal Gated Seismic Fusion Network & Calibrated Multi-Threshold Deep Ensemble
===================================================================================
Upgraded after Forensic Audit:
  1. Per-branch `LayerNorm(hidden)` prior to convex softmax gating eliminates uncalibrated
     magnitude baseline shifts when individual modalities are ablated (`mod_mask[:, k] = 0`).
  2. Multi-threshold `det_head` (`Mw >= 4.0, 4.5, 5.0, 5.5`) with independent per-threshold
     post-hoc validation temperature scaling (`log_temperatures`).
  3. Standardized magnitude regression in `FoldFeatureScaler` space (`mu_mag_norm`, `log_var_norm`)
     inverted to physical `Mw` units at prediction time.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from seismic_fusion.config import FusionModelConfig


class SeismicMultiModalFusion(nn.Module):
    """
    Quality-Gated Multi-Modal Fusion Network supporting multi-threshold probabilistic
    exceedance forecasting and heteroscedastic peak-magnitude estimation.
    """

    def __init__(
        self,
        wf_channels: int = 6,
        gnn_dim: int = 9,
        phys_dim: int = 12,
        hidden: int = 64,
        n_thresholds: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_thresholds = n_thresholds

        # Branch A: Dilated Conv1d + BiGRU + Temporal Attention
        self.wf_conv = nn.Sequential(
            nn.Conv1d(wf_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 48, kernel_size=3, dilation=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(48, hidden, kernel_size=3, dilation=4, padding=4),
            nn.SiLU(),
        )
        self.wf_gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.wf_att = nn.Linear(hidden, 1, bias=False)
        self.norm_a = nn.LayerNorm(hidden)

        # Branch B: 2-hop 3D ECEF Geodesic Station Graph Neural Network
        self.gnn_proj1 = nn.Linear(gnn_dim, hidden)
        self.gnn_norm1 = nn.LayerNorm(hidden)
        self.gnn_proj2 = nn.Linear(hidden, hidden)
        self.gnn_norm2 = nn.LayerNorm(hidden)
        self.norm_b = nn.LayerNorm(hidden)

        # Branch C: Gutenberg-Richter / Benioff / ETAS Physics Encoder
        self.phys_enc = nn.Sequential(
            nn.Linear(phys_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.norm_c = nn.LayerNorm(hidden)

        # Quality-aware cross-modal softmax gate
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden * 3 + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

        self.fusion_drop = nn.Dropout(dropout)
        self.det_head = nn.Linear(hidden, n_thresholds)
        self.mag_head = nn.Linear(hidden, 2)  # [mu_mag_norm, log_var_mag_norm]
        self.log_temperatures = nn.Parameter(torch.zeros(n_thresholds), requires_grad=False)
        self.cal_biases = nn.Parameter(torch.zeros(n_thresholds), requires_grad=False)

    def forward(
        self,
        wf: torch.Tensor,           # (B, C, L)
        node_feats: torch.Tensor,   # (B, V, F_gnn)
        adj_norm: torch.Tensor,     # (V, V)
        station_idx: torch.Tensor,  # (B,)
        phys_feats: torch.Tensor,   # (B, F_phys)
        mod_mask: torch.Tensor,     # (B, 3) binary mask in {0, 1}
    ) -> Dict[str, torch.Tensor]:
        x = torch.sign(wf) * torch.log1p(wf.abs())
        seq = self.wf_conv(x).transpose(1, 2)
        gru_out, _ = self.wf_gru(seq)
        alpha = F.softmax(self.wf_att(gru_out).squeeze(-1), dim=-1)
        h_wf = self.norm_a(torch.sum(gru_out * alpha.unsqueeze(-1), dim=1))

        h_v = F.silu(self.gnn_norm1(torch.matmul(adj_norm, self.gnn_proj1(node_feats))))
        h_v = h_v + F.silu(self.gnn_norm2(torch.matmul(adj_norm, self.gnn_proj2(h_v))))
        batch_arange = torch.arange(wf.size(0), device=wf.device)
        h_gnn = self.norm_b(0.5 * h_v[batch_arange, station_idx] + 0.5 * h_v.mean(dim=1))

        h_phys = self.norm_c(self.phys_enc(phys_feats))

        active_any = mod_mask.sum(dim=-1, keepdim=True) > 0.5
        safe_mask = torch.where(active_any, mod_mask, torch.ones_like(mod_mask))
        gate_logits = self.modality_gate(torch.cat([h_wf, h_gnn, h_phys, safe_mask], dim=-1))
        gate_logits = gate_logits.masked_fill(safe_mask < 0.5, -1e9)
        w = F.softmax(gate_logits, dim=-1)

        fused = self.fusion_drop(w[:, 0:1] * h_wf + w[:, 1:2] * h_gnn + w[:, 2:3] * h_phys)
        raw_logits = self.det_head(fused)  # (B, K)
        temps = self.log_temperatures.exp().clamp(min=0.25, max=5.0).unsqueeze(0)
        det_logits = (raw_logits / temps) + self.cal_biases.unsqueeze(0)

        mag_out = self.mag_head(fused)
        mu_mag = mag_out[:, 0]
        log_var_mag = mag_out[:, 1].clamp(min=-6.0, max=4.0)

        return {
            "raw_logits": raw_logits,
            "det_logits": det_logits,
            "det_logit": det_logits[:, 0] if det_logits.size(1) == 1 else det_logits[:, -2],
            "mu_mag": mu_mag,
            "log_var_mag": log_var_mag,
            "gate_weights": w,
        }


def heteroscedastic_multithreshold_loss(
    raw_logits: torch.Tensor,        # (B, K)
    mu_m_norm: torch.Tensor,         # (B,)
    log_var_m_norm: torch.Tensor,    # (B,)
    y_det_mat: torch.Tensor,         # (B, K)
    m_true_norm: torch.Tensor,       # (B,)
    m_true_raw: torch.Tensor,        # (B,)
    reg_mask: torch.Tensor,          # (B,)
    pos_weights: torch.Tensor | None = None,  # (K,)
    tail_alpha: float = 0.75,
    reg_lambda: float = 0.35,
) -> torch.Tensor:
    """Proper multi-threshold BCE + tail-weighted standardized heteroscedastic Gaussian NLL."""
    loss_det = F.binary_cross_entropy_with_logits(raw_logits, y_det_mat, pos_weight=pos_weights)
    if not bool(reg_mask.any()):
        return loss_det
    err2 = (m_true_norm[reg_mask] - mu_m_norm[reg_mask]) ** 2
    lv = log_var_m_norm[reg_mask]
    nll = 0.5 * (torch.exp(-lv) * err2 + lv)
    tail_w = 1.0 + tail_alpha * torch.exp(torch.clamp(m_true_raw[reg_mask] - 4.5, min=0.0, max=2.5))
    loss_mag = torch.sum(tail_w * nll) / torch.sum(tail_w)
    return loss_det + reg_lambda * loss_mag


def fit_multithreshold_temperatures(
    model: SeismicMultiModalFusion,
    raw_val_logits: torch.Tensor,  # (N_cal, K)
    y_val_mat: torch.Tensor,       # (N_cal, K)
) -> list[float]:
    """
    Fits post-hoc affine temperature scaling (`raw_logit / T_k + b_k`, `T_k in [0.25, 5.0]`)
    with L2 regularization toward identity (`T_k=1, b_k=0`) on inner calibration data.
    """
    k_th = raw_val_logits.size(1)
    fitted_log_t = torch.zeros(k_th, device=raw_val_logits.device)
    fitted_bias = torch.zeros(k_th, device=raw_val_logits.device)

    for k_i in range(k_th):
        yk = y_val_mat[:, k_i].detach()
        rk = raw_val_logits[:, k_i].detach()
        if float(yk.sum().item()) < 5.0 or float((1.0 - yk).sum().item()) < 5.0:
            fitted_log_t[k_i] = 0.0
            fitted_bias[k_i] = 0.0
            continue
        log_tk = nn.Parameter(torch.zeros(1, device=raw_val_logits.device))
        bias_k = nn.Parameter(torch.zeros(1, device=raw_val_logits.device))
        opt = torch.optim.LBFGS([log_tk, bias_k], lr=0.1, max_iter=40)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            t = log_tk.exp().clamp(min=0.25, max=5.0)
            reg = 0.02 * (log_tk.pow(2).sum() + bias_k.pow(2).sum())
            loss = F.binary_cross_entropy_with_logits((rk / t) + bias_k, yk) + reg
            loss.backward()
            return loss

        opt.step(closure)
        fitted_log_t[k_i] = log_tk.detach().clamp(min=-1.386, max=1.609)[0]
        fitted_bias[k_i] = bias_k.detach().clamp(min=-2.0, max=2.0)[0]

    with torch.no_grad():
        model.log_temperatures.copy_(fitted_log_t)
        model.cal_biases.copy_(fitted_bias)
    return [float(x) for x in model.log_temperatures.exp().cpu().tolist()]


class DeepSeismicEnsemble:
    """
    K-member independently initialized multi-threshold ensemble providing calibrated
    exceedance probabilities and decomposed aleatoric + epistemic magnitude uncertainty.
    """

    def __init__(self, cfg: FusionModelConfig, n_thresholds: int = 4, device: torch.device | None = None):
        self.cfg = cfg
        self.n_thresholds = n_thresholds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members: list[SeismicMultiModalFusion] = []
        self.temperatures: list[list[float]] = []
        self.mag_center: float = 0.0
        self.mag_scale: float = 1.0

    def fit(
        self,
        train_tensors: dict[str, np.ndarray],
        val_tensors: dict[str, np.ndarray],
        adj_norm: np.ndarray,
        mag_center: float = 0.0,
        mag_scale: float = 1.0,
    ) -> None:
        self.members.clear()
        self.temperatures.clear()
        self.mag_center = float(mag_center)
        self.mag_scale = float(max(mag_scale, 1e-6))

        adj_t = torch.from_numpy(adj_norm).to(self.device)
        # Strictly proper unweighted Bernoulli NLL preserves natural base-rate calibration across all 4 thresholds
        pos_weights = None

        tr_ds = TensorDataset(
            torch.from_numpy(train_tensors["kinematic_seq"]),
            torch.from_numpy(train_tensors["node_feats"]),
            torch.from_numpy(train_tensors["station_idx"]),
            torch.from_numpy(train_tensors["phys_feats"]),
            torch.from_numpy(train_tensors["mod_mask"]),
            torch.from_numpy(train_tensors["y_exceed_matrix"]),
            torch.from_numpy(train_tensors["future_max_mw_norm"]),
            torch.from_numpy(train_tensors["future_max_mw"]),
            torch.from_numpy(train_tensors["reg_mask"]),
        )

        va_wf = torch.from_numpy(val_tensors["kinematic_seq"]).to(self.device)
        va_nf = torch.from_numpy(val_tensors["node_feats"]).to(self.device)
        va_si = torch.from_numpy(val_tensors["station_idx"]).to(self.device)
        va_pf = torch.from_numpy(val_tensors["phys_feats"]).to(self.device)
        va_mm = torch.from_numpy(val_tensors["mod_mask"]).to(self.device)
        va_y = torch.from_numpy(val_tensors["y_exceed_matrix"]).to(self.device)
        va_mn = torch.from_numpy(val_tensors["future_max_mw_norm"]).to(self.device)
        va_mr = torch.from_numpy(val_tensors["future_max_mw"]).to(self.device)
        va_rm = torch.from_numpy(val_tensors["reg_mask"]).to(self.device)

        for seed in self.cfg.ensemble_seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = SeismicMultiModalFusion(
                wf_channels=self.cfg.kinematic_channels,
                gnn_dim=self.cfg.gnn_node_dim,
                phys_dim=self.cfg.phys_feature_dim,
                hidden=self.cfg.hidden_dim,
                n_thresholds=self.n_thresholds,
                dropout=self.cfg.dropout,
            ).to(self.device)

            gen = torch.Generator().manual_seed(seed)
            loader = DataLoader(tr_ds, batch_size=self.cfg.batch_size, shuffle=True, generator=gen)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )

            best_val = float("inf")
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0

            for _ in range(self.cfg.max_epochs):
                model.train()
                for b_wf, b_nf, b_si, b_pf, b_mm, b_y, b_mn, b_mr, b_rm in loader:
                    b_wf = b_wf.to(self.device)
                    b_nf = b_nf.to(self.device)
                    b_si = b_si.to(self.device)
                    b_pf = b_pf.to(self.device)
                    b_mm = b_mm.to(self.device)
                    b_y = b_y.to(self.device)
                    b_mn = b_mn.to(self.device)
                    b_mr = b_mr.to(self.device)
                    b_rm = b_rm.to(self.device)

                    # Balanced per-sample modality dropout so ablated evaluations remain well-calibrated
                    if self.cfg.modality_dropout_prob > 0.0:
                        u = torch.rand(b_mm.size(0), generator=gen, device=b_mm.device)
                        drop_active = u < self.cfg.modality_dropout_prob
                        if bool(drop_active.any()):
                            b_mm = b_mm.clone()
                            drop_cols = torch.randint(
                                0, 3, (int(drop_active.sum().item()),), generator=gen, device=b_mm.device
                            )
                            b_mm[drop_active, drop_cols] = 0.0

                    optimizer.zero_grad(set_to_none=True)
                    out = model(b_wf, b_nf, adj_t, b_si, b_pf, b_mm)
                    loss = heteroscedastic_multithreshold_loss(
                        out["raw_logits"],
                        out["mu_mag"],
                        out["log_var_mag"],
                        b_y,
                        b_mn,
                        b_mr,
                        b_rm,
                        pos_weights=pos_weights,
                        tail_alpha=self.cfg.tail_alpha,
                    )
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    v_out = model(va_wf, va_nf, adj_t, va_si, va_pf, va_mm)
                    v_loss = float(
                        heteroscedastic_multithreshold_loss(
                            v_out["raw_logits"],
                            v_out["mu_mag"],
                            v_out["log_var_mag"],
                            va_y,
                            va_mn,
                            va_mr,
                            va_rm,
                            pos_weights=pos_weights,
                            tail_alpha=self.cfg.tail_alpha,
                        ).item()
                    )
                if v_loss < best_val - 1e-4:
                    best_val = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.cfg.patience:
                        break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                v_raw = model(va_wf, va_nf, adj_t, va_si, va_pf, va_mm)["raw_logits"]
            temps = fit_multithreshold_temperatures(model, v_raw, va_y)
            self.members.append(model)
            self.temperatures.append(temps)

    @torch.no_grad()
    def predict(
        self,
        tensors: dict[str, np.ndarray],
        adj_norm: np.ndarray,
        override_mod_mask: np.ndarray | None = None,
        primary_col: int = 2,
    ) -> dict[str, np.ndarray]:
        if not self.members:
            raise RuntimeError("DeepSeismicEnsemble must be fitted before calling predict().")

        adj_t = torch.from_numpy(adj_norm).to(self.device)
        wf = torch.from_numpy(tensors["kinematic_seq"]).to(self.device)
        nf = torch.from_numpy(tensors["node_feats"]).to(self.device)
        si = torch.from_numpy(tensors["station_idx"]).to(self.device)
        pf = torch.from_numpy(tensors["phys_feats"]).to(self.device)
        mm_arr = override_mod_mask if override_mod_mask is not None else tensors["mod_mask"]
        mm = torch.from_numpy(mm_arr).to(self.device)

        member_probs = []
        member_mu_phys = []
        member_alea_var_phys = []
        member_gates = []

        scale2 = self.mag_scale**2
        for model in self.members:
            model.eval()
            out = model(wf, nf, adj_t, si, pf, mm)
            member_probs.append(torch.sigmoid(out["det_logits"]).cpu().numpy())
            mu_phys = out["mu_mag"].cpu().numpy() * self.mag_scale + self.mag_center
            alea_phys = torch.exp(out["log_var_mag"]).cpu().numpy() * scale2
            member_mu_phys.append(mu_phys)
            member_alea_var_phys.append(alea_phys)
            member_gates.append(out["gate_weights"].cpu().numpy())

        probs_stack = np.stack(member_probs, axis=0)      # (M, N, K)
        mu_stack = np.stack(member_mu_phys, axis=0)       # (M, N)
        alea_stack = np.stack(member_alea_var_phys, axis=0)
        gates_stack = np.stack(member_gates, axis=0)

        mean_probs_mat = np.mean(probs_stack, axis=0)     # (N, K)
        mean_mu = np.mean(mu_stack, axis=0)
        aleatoric_var = np.mean(alea_stack, axis=0)
        epistemic_var = np.var(mu_stack, axis=0)
        total_std = np.sqrt(np.maximum(aleatoric_var + epistemic_var, 1e-8))

        p_col = min(primary_col, mean_probs_mat.shape[1] - 1)
        return {
            "probs_matrix": mean_probs_mat,
            "prob": mean_probs_mat[:, p_col],
            "prob_epistemic_std": np.std(probs_stack[:, :, p_col], axis=0),
            "mu_mag": mean_mu,
            "aleatoric_var": aleatoric_var,
            "epistemic_var": epistemic_var,
            "total_mag_std": total_std,
            "gate_weights": np.mean(gates_stack, axis=0),
        }
