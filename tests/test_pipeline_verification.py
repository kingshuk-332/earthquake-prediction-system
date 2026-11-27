"""
Comprehensive Automated Verification Suite (Tests 01 through 08)
================================================================
Verifies:
  1. Record conservation (`valid + flagged == total_raw == 8089`) & fail-fast schema errors.
  2. Zero temporal leakage in `FoldFeatureScaler` and target-completed `walk_forward_embargo_splits`.
  3. 3D spherical ECEF geodesy, Brune corner-frequency scaling, and causal post-P physics.
  4. Input sensitivity and modality gate masking across all 3 branches.
  5. Multi-threshold (`Mw >= 4.0, 4.5, 5.0, 5.5`) ensemble calibration, `"insufficient_positives"`
     handling, and modality ablation.
  6. Forensic causal boundary proof (`test_06_strict_causal_feature_target_boundary`): mutating
     future events or shifting forecast horizons produces exact 0.0 change in Branches A, B, C,
     and simultaneous-timestamp ties never leak into forward targets.
  7. Held-out Integrated Gradients & stratified gate XAI attributions (`explainability.py`).
  8. Real 3-component (Z, N, E) STEAD HDF5 & ObsPy MiniSEED waveform adapter QC & fail-fast
     behavior without synthetic trace fabrication (`waveform_adapter.py`).
"""

from __future__ import annotations

import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SIM_PARENT = REPO_ROOT / "cnn" / "cnn wave" / "data wave"
if str(SIM_PARENT) not in sys.path:
    sys.path.insert(0, str(SIM_PARENT))

from cnn.cnn_1d_model import build_embargoed_causal_sequences
from cnn.generate_noisy_dataset import verify_temporal_dataset
from preprocess import run_preprocessing, spherical_hypocentral_km
from preprocess_dl_temporal import run_temporal_preprocessing
from seismic_fusion.config import FusionModelConfig, PipelineConfig, ValidationConfig, resolve_repo_root
from seismic_fusion.explainability import (
    compute_integrated_gradients_heldout,
    compute_stratified_gate_attributions,
)
from seismic_fusion.model import DeepSeismicEnsemble, SeismicMultiModalFusion
from seismic_fusion.preprocessing import (
    assemble_targets_for_horizon,
    ecef_xyz_km,
    extract_causal_physics_features,
    ingest_and_assemble_fusion_dataset,
)
from seismic_fusion.validation import (
    FoldFeatureScaler,
    compute_multithreshold_table,
    compute_probabilistic_metrics,
    evaluate_modality_ablation,
    split_inner_train_calibration,
    walk_forward_embargo_splits,
)
from seismic_fusion.waveform_adapter import (
    ingest_miniseed_or_sac_records,
    ingest_stead_hdf5,
)

_sim_iwg = importlib.import_module("synthetic_waveform_simulator.improved_waveform_generator")
ImprovedSimulationParameters = getattr(_sim_iwg, "ImprovedSimulationParameters")
ImprovedWaveformGenerator = getattr(_sim_iwg, "ImprovedWaveformGenerator")


class TestPipelineVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = resolve_repo_root(REPO_ROOT)
        cls.cfg = PipelineConfig()
        cls.fusion_ds = ingest_and_assemble_fusion_dataset(cls.repo_root, cls.cfg)

    def test_01_record_conservation_and_fail_fast_schema(self) -> None:
        m_fus = self.fusion_ds.audit_manifest
        self.assertEqual(m_fus["total_raw"], 8089)
        self.assertEqual(m_fus["valid"] + m_fus["flagged"], m_fus["total_raw"])
        self.assertTrue(m_fus["record_conservation_verified"])
        self.assertEqual(m_fus["synthetic_records_fabricated"], 0)

        m_pre = run_preprocessing(self.repo_root, self.cfg)
        self.assertEqual(m_pre["valid"] + m_pre["flagged"], 8089)
        self.assertFalse(m_pre["iqr_target_clipping_applied"])
        df_ml = pd.read_csv(self.repo_root / "preprocessed_ml.csv")
        self.assertAlmostEqual(float(df_ml["Magnitude_Value"].max()), 8.6, places=3)
        self.assertEqual(int((df_ml["Magnitude_Value"] > 6.45).sum()), 60)

        m_tmp = run_temporal_preprocessing(self.repo_root, self.cfg)
        self.assertEqual(m_tmp["valid"] + m_tmp["flagged"], 8089)
        qc_rep = verify_temporal_dataset(self.repo_root)
        self.assertTrue(qc_rep["record_conservation_verified"])
        self.assertLessEqual(qc_rep["max_trigonometric_residual"], 1e-6)
        self.assertLessEqual(qc_rep["max_sp_kinematic_residual"], 1e-5)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            df_corrupt = pd.read_csv(self.repo_root / "dataset_2.csv", nrows=25).drop(columns=["Magnitude_Value"])
            (tmp_path / "dataset_2.csv").write_text(df_corrupt.to_csv(index=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_preprocessing(tmp_path, self.cfg)
            with self.assertRaises(ValueError):
                ingest_and_assemble_fusion_dataset(tmp_path, self.cfg)

    def test_02_zero_temporal_leakage_in_splits_and_scalers(self) -> None:
        embargo_sec = self.cfg.validation.embargo_days * 86400.0
        horizon_sec = self.cfg.validation.forecast_horizon_days * 86400.0
        splits = list(
            walk_forward_embargo_splits(
                self.fusion_ds.timestamps_sec,
                n_folds=self.cfg.validation.n_walk_forward_folds,
                embargo_sec=embargo_sec,
                horizon_sec=horizon_sec,
            )
        )
        self.assertGreaterEqual(len(splits), 3)

        for tr_idx, va_idx in splits:
            self.assertEqual(len(set(tr_idx).intersection(set(va_idx))), 0)
            # Post-target-completion embargo check: min(t_val) >= max(t_train + H) + embargo
            post_target_gap = float(
                np.min(self.fusion_ds.timestamps_sec[va_idx])
                - np.max(self.fusion_ds.target_end_timestamps_sec[tr_idx])
            )
            self.assertGreaterEqual(post_target_gap, embargo_sec - 1e-6)

            ds_mutated = copy.deepcopy(self.fusion_ds)
            ds_mutated.kinematic_seq[va_idx] += 1.0e6
            ds_mutated.node_feats[va_idx] += 1.0e6
            ds_mutated.phys_feats[va_idx] += 1.0e6
            ds_mutated.future_max_mw[va_idx] += 50.0

            sc_clean = FoldFeatureScaler().fit(self.fusion_ds, tr_idx)
            sc_mut = FoldFeatureScaler().fit(ds_mutated, tr_idx)

            np.testing.assert_allclose(sc_clean.kin_mean, sc_mut.kin_mean, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(sc_clean.node_mean, sc_mut.node_mean, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(sc_clean.phys_mean, sc_mut.phys_mean, rtol=0.0, atol=0.0)
            self.assertEqual(sc_clean.mag_center, sc_mut.mag_center)
            self.assertEqual(sc_clean.mag_scale, sc_mut.mag_scale)

        n_check = 600
        dummy_feats = np.arange(n_check * 4, dtype=np.float32).reshape(n_check, 4)
        dummy_mags = np.linspace(3.0, 5.5, n_check)
        dummy_ts = np.arange(n_check, dtype=np.float64) * 5.0 * 86400.0
        seq_splits = build_embargoed_causal_sequences(
            dummy_feats, dummy_mags, dummy_ts, seq_len=10, mw_threshold=4.5, embargo_sec=30.0 * 86400.0
        )
        tr_vals = set(seq_splits["train"][0][:, :, 0].ravel().tolist())
        va_vals = set(seq_splits["validation"][0][:, :, 0].ravel().tolist())
        te_vals = set(seq_splits["test"][0][:, :, 0].ravel().tolist())
        self.assertEqual(len(tr_vals.intersection(va_vals)), 0)
        self.assertEqual(len(va_vals.intersection(te_vals)), 0)

    def test_03_geodesic_3d_ecef_and_brune_conformance(self) -> None:
        r_sph = spherical_hypocentral_km(np.array([683.9]), np.array([100.0]), r_earth=6371.0)[0]
        r_flat = float(np.sqrt(683.9**2 + 100.0**2))
        self.assertLess(r_sph, r_flat)
        self.assertGreater(abs(r_flat - r_sph), 5.0)

        xyz_a = ecef_xyz_km(np.array([25.0]), np.array([92.0]), np.array([10.0]))
        xyz_b = ecef_xyz_km(np.array([26.0]), np.array([93.0]), np.array([35.0]))
        self.assertTrue(100.0 < float(np.linalg.norm(xyz_a - xyz_b)) < 200.0)

        gen_brune = ImprovedWaveformGenerator(ImprovedSimulationParameters(use_brune_corner_freq=True))
        fp_small, fs_small = gen_brune.brune_corner_frequencies(3.5)
        fp_large, fs_large = gen_brune.brune_corner_frequencies(6.5)
        self.assertLess(fp_large, fp_small)
        self.assertLess(fs_large, fs_small)

        rng = np.random.default_rng(42)
        wf_base = rng.normal(0.0, 0.05, size=1000)
        wf_pert = wf_base.copy()
        wf_pert[500:] += 100.0
        f_base = extract_causal_physics_features(wf_base, p_idx=200, horizon_samples=300, fs=100.0)
        f_pert = extract_causal_physics_features(wf_pert, p_idx=200, horizon_samples=300, fs=100.0)
        np.testing.assert_allclose(f_base, f_pert, rtol=0.0, atol=0.0)

    def test_04_input_sensitivity_and_modality_gating(self) -> None:
        torch.manual_seed(42)
        model = SeismicMultiModalFusion(
            wf_channels=self.cfg.model.kinematic_channels,
            gnn_dim=self.cfg.model.gnn_node_dim,
            phys_dim=self.cfg.model.phys_feature_dim,
            hidden=32,
            n_thresholds=len(self.fusion_ds.thresholds),
        ).eval()

        batch_sl = slice(0, 8)
        wf = torch.tensor(self.fusion_ds.kinematic_seq[batch_sl], dtype=torch.float32, requires_grad=True)
        nf = torch.tensor(self.fusion_ds.node_feats[batch_sl], dtype=torch.float32, requires_grad=True)
        adj = torch.tensor(self.fusion_ds.adj_norm, dtype=torch.float32)
        si = torch.tensor(self.fusion_ds.station_idx[batch_sl], dtype=torch.int64)
        pf = torch.tensor(self.fusion_ds.phys_feats[batch_sl], dtype=torch.float32, requires_grad=True)
        mm = torch.ones((8, 3), dtype=torch.float32)

        out_base = model(wf, nf, adj, si, pf, mm)
        (out_base["det_logits"].sum() + out_base["mu_mag"].sum()).backward()

        self.assertGreater(float(wf.grad.norm().item()), 0.0)
        self.assertGreater(float(nf.grad.norm().item()), 0.0)
        self.assertGreater(float(pf.grad.norm().item()), 0.0)

        with torch.no_grad():
            out_wf = model(wf.detach() + 0.5, nf.detach(), adj, si, pf.detach(), mm)
            out_nf = model(wf.detach(), nf.detach() + 0.5, adj, si, pf.detach(), mm)
            out_pf = model(wf.detach(), nf.detach(), adj, si, pf.detach() + 0.5, mm)
            self.assertGreater(float((out_wf["det_logits"] - out_base["det_logits"]).abs().max().item()), 1e-6)
            self.assertGreater(float((out_nf["det_logits"] - out_base["det_logits"]).abs().max().item()), 1e-6)
            self.assertGreater(float((out_pf["det_logits"] - out_base["det_logits"]).abs().max().item()), 1e-6)

            mm_drop_a = torch.tensor([[0.0, 1.0, 1.0]] * 8, dtype=torch.float32)
            out_masked = model(wf.detach(), nf.detach(), adj, si, pf.detach(), mm_drop_a)
            self.assertLess(float(out_masked["gate_weights"][:, 0].max().item()), 1e-6)

    def test_05_multithreshold_ensemble_insufficient_positives_and_xai(self) -> None:
        # Verify explicit "insufficient_positives" guard when positives < min_positives
        deg_m = compute_probabilistic_metrics(
            probs=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            y_true=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),  # Only 1 positive (< 5)
            p_null=np.full(6, 0.15),
            min_positives=5,
        )
        self.assertEqual(deg_m["status"], "insufficient_positives")
        self.assertEqual(deg_m["pr_auc"], "insufficient_positives")

        fast_model_cfg = FusionModelConfig(
            lookback_events=self.cfg.model.lookback_events,
            kinematic_channels=self.cfg.model.kinematic_channels,
            gnn_node_dim=self.cfg.model.gnn_node_dim,
            phys_feature_dim=self.cfg.model.phys_feature_dim,
            hidden_dim=32,
            dropout=0.1,
            ensemble_seeds=(42, 1337),
            batch_size=128,
            max_epochs=6,
            patience=3,
        )
        embargo_sec = 30.0 * 86400.0
        tr_idx, va_idx = next(
            walk_forward_embargo_splits(
                self.fusion_ds.timestamps_sec, n_folds=3, embargo_sec=embargo_sec, horizon_sec=embargo_sec
            )
        )
        in_tr, in_cal = split_inner_train_calibration(
            self.fusion_ds.timestamps_sec, tr_idx, embargo_sec=embargo_sec, horizon_sec=embargo_sec
        )
        scaler = FoldFeatureScaler().fit(self.fusion_ds, in_tr)
        tr_t = scaler.transform(self.fusion_ds, in_tr)
        cal_t = scaler.transform(self.fusion_ds, in_cal)
        va_t = scaler.transform(self.fusion_ds, va_idx)

        ens = DeepSeismicEnsemble(fast_model_cfg, n_thresholds=len(self.fusion_ds.thresholds))
        ens.fit(tr_t, cal_t, self.fusion_ds.adj_norm, mag_center=scaler.mag_center, mag_scale=scaler.mag_scale)
        preds = ens.predict(va_t, self.fusion_ds.adj_norm, primary_col=2)

        self.assertEqual(preds["probs_matrix"].shape, (len(va_idx), 4))
        base_tr = np.clip(np.mean(self.fusion_ds.y_exceed_matrix[tr_idx], axis=0), 1e-4, 1.0 - 1e-4)
        p_null_mat = np.tile(base_tr[None, :], (len(va_idx), 1))
        mt_table = compute_multithreshold_table(
            preds["probs_matrix"], va_t["y_exceed_matrix"], p_null_mat, None, self.fusion_ds.thresholds
        )
        self.assertEqual(set(mt_table.keys()), {"Mw>=4.0", "Mw>=4.5", "Mw>=5.0", "Mw>=5.5"})

        ablation = evaluate_modality_ablation(ens, va_t, self.fusion_ds.adj_norm, p_null_mat[:, 2], primary_col=2)
        self.assertIn("full_multimodal", ablation)

        # Task 4: Verify native XAI attributions on held-out va_t
        gates_xai = compute_stratified_gate_attributions(ens, va_t, self.fusion_ds.adj_norm, primary_col=2)
        self.assertIn("all_heldout", gates_xai)
        ig_xai = compute_integrated_gradients_heldout(
            ens, va_t, self.fusion_ds.adj_norm, target_col=2, n_steps=16, max_eval_records=32
        )
        self.assertLess(ig_xai["mean_completeness_mae_logit"], 0.05)
        self.assertEqual(len(ig_xai["branch_a_channel_importance"]), 6)
        self.assertEqual(len(ig_xai["branch_b_station_feature_importance"]), 9)
        self.assertEqual(len(ig_xai["branch_c_physics_feature_importance"]), 12)

    def test_06_strict_causal_feature_target_boundary(self) -> None:
        """
        Forensic Causality Proof:
          1. Shifting forecast_horizon_days (10d vs 25d) leaves ALL Branch A, B, and C features
             for overlapping anchor timestamps 100% bit-identical while changing forward targets.
          2. Mutating future catalog rows (t > cutoff) to extreme magnitudes/depths/SNR produces
             ZERO change in Branch A, B, and C feature tensors for all origins <= cutoff.
          3. Simultaneous-timestamp ties (`t_i == t_{i+1}`) are strictly excluded from forward targets.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            df_sub = pd.read_csv(self.repo_root / "dataset_2.csv").tail(450).reset_index(drop=True)
            (tmp_root / "dataset_2.csv").write_text(df_sub.to_csv(index=False), encoding="utf-8")

            cfg_10d = PipelineConfig(
                validation=ValidationConfig(
                    modern_catalog_start_utc="1990-01-01 00:00:00",
                    forecast_horizon_days=10.0,
                    embargo_days=10.0,
                )
            )
            cfg_25d = PipelineConfig(
                validation=ValidationConfig(
                    modern_catalog_start_utc="1990-01-01 00:00:00",
                    forecast_horizon_days=25.0,
                    embargo_days=25.0,
                )
            )
            ds_10 = ingest_and_assemble_fusion_dataset(tmp_root, cfg_10d)
            ds_25 = ingest_and_assemble_fusion_dataset(tmp_root, cfg_25d)

            common_n = len(ds_25.source_ids)
            self.assertGreater(common_n, 50)
            np.testing.assert_array_equal(ds_10.source_ids[:common_n], ds_25.source_ids)
            np.testing.assert_array_equal(ds_10.kinematic_seq[:common_n], ds_25.kinematic_seq)
            np.testing.assert_array_equal(ds_10.node_feats[:common_n], ds_25.node_feats)
            np.testing.assert_array_equal(ds_10.phys_feats[:common_n], ds_25.phys_feats)
            # Horizon expansion must monotonically satisfy y_25 >= y_10 everywhere
            self.assertTrue(np.all(ds_25.y_exceed_matrix >= ds_10.y_exceed_matrix[:common_n]))
            self.assertGreater(
                float(np.sum(ds_25.y_exceed_matrix - ds_10.y_exceed_matrix[:common_n])), 0.0
            )

            # Mutate the final 100 future catalog rows to extreme values and verify past features stay bit-identical
            df_mut = df_sub.copy()
            # Keep station static medians fixed by only mutating earthquake event attributes (Mag, Depth, SNR, S-arrival)
            df_mut.loc[len(df_mut) - 100 :, "Magnitude_Value"] = 8.9
            df_mut.loc[len(df_mut) - 100 :, "Depth_km"] = 400.0
            df_mut.loc[len(df_mut) - 100 :, "snr_db"] = 75.0
            df_mut.loc[len(df_mut) - 100 :, "s_arrival_sample"] = df_mut.loc[len(df_mut) - 100 :, "p_arrival_sample"] + 3000

            tmp_mut = Path(tmpdir) / "mut"
            tmp_mut.mkdir()
            (tmp_mut / "dataset_2.csv").write_text(df_mut.to_csv(index=False), encoding="utf-8")
            ds_mut = ingest_and_assemble_fusion_dataset(tmp_mut, cfg_10d)

            # First 200 anchors lie strictly before the mutated tail
            safe_k = 200
            np.testing.assert_array_equal(ds_10.kinematic_seq[:safe_k], ds_mut.kinematic_seq[:safe_k])
            np.testing.assert_array_equal(ds_10.node_feats[:safe_k], ds_mut.node_feats[:safe_k])
            np.testing.assert_array_equal(ds_10.phys_feats[:safe_k], ds_mut.phys_feats[:safe_k])

        # Simultaneous timestamp tie check (`t_0 == t_1 == 100.0`): event 1 at t=100.0 has Mw=7.0,
        # event 2 at t=200.0 has Mw=3.8 -> anchor 0 at t=100.0 MUST NOT see event 1 (Mw=7.0) in its forward target!
        ts_tie = np.array([100.0, 100.0, 200.0], dtype=np.float64)
        mw_tie = np.array([4.0, 7.0, 3.8], dtype=np.float64)
        y_tie, max_mw_tie, _, min_lead = assemble_targets_for_horizon(
            ts_sec=ts_tie,
            mw_arr=mw_tie,
            mag_raw_arr=mw_tie,
            anchor_indices=np.array([0], dtype=np.int64),
            horizon_sec=500.0,
            thresholds=(4.0, 5.0),
            mc_val=3.7,
        )
        self.assertAlmostEqual(float(max_mw_tie[0]), 3.8, places=5)
        self.assertEqual(float(y_tie[0, 1]), 0.0)
        self.assertEqual(min_lead, 100.0)

    def test_07_real_3c_waveform_adapter_qc_and_fail_fast(self) -> None:
        h5py = importlib.import_module("h5py")
        obspy = importlib.import_module("obspy")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)

            # Deterministic analytical oscillatory test signals for schema/QC verification (not benchmark data)
            t = np.arange(1000, dtype=np.float64) / 100.0
            good_3c = np.zeros((1000, 3), dtype=np.float32)
            for ch, freq in enumerate((4.0, 5.0, 6.0)):
                good_3c[:200, ch] = 0.02 * np.sin(2.0 * np.pi * (freq + 1.3) * t[:200])
                good_3c[200:, ch] = 0.80 * np.sin(2.0 * np.pi * freq * t[200:]) * np.exp(-0.3 * (t[200:] - 2.0))

            nan_3c = good_3c.copy()
            nan_3c[250, 0] = np.nan

            flat_3c = np.zeros((1000, 3), dtype=np.float32)

            h5_path = tmp_dir / "stead_test.h5"
            with h5py.File(h5_path, "w") as hf:
                grp = hf.create_group("data")
                d1 = grp.create_dataset("TR_VALID_01", data=good_3c)
                d1.attrs["sampling_rate_hz"] = 100.0
                d2 = grp.create_dataset("TR_NAN_02", data=nan_3c)
                d2.attrs["sampling_rate_hz"] = 100.0
                d3 = grp.create_dataset("TR_FLAT_03", data=flat_3c)
                d3.attrs["sampling_rate_hz"] = 100.0
                d4 = grp.create_dataset("TR_BADFS_04", data=good_3c)
                d4.attrs["sampling_rate_hz"] = 50.0

            meta_csv = tmp_dir / "stead_meta.csv"
            pd.DataFrame(
                [
                    {"trace_name": "TR_VALID_01", "p_arrival_sample": 200, "s_arrival_sample": 600, "receiver_code": "GHY"},
                    {"trace_name": "TR_NAN_02", "p_arrival_sample": 200, "s_arrival_sample": 600, "receiver_code": "IMP"},
                    {"trace_name": "TR_FLAT_03", "p_arrival_sample": 200, "s_arrival_sample": 600, "receiver_code": "TEZ"},
                    {"trace_name": "TR_BADFS_04", "p_arrival_sample": 200, "s_arrival_sample": 600, "receiver_code": "KOH"},
                    {"trace_name": "TR_MISSING_05", "p_arrival_sample": 200, "s_arrival_sample": 600, "receiver_code": "SHL"},
                ]
            ).to_csv(meta_csv, index=False)

            batch = ingest_stead_hdf5(h5_path, meta_csv, manifest_out_path=tmp_dir / "stead_qc.json")
            self.assertEqual(batch.qc_manifest["total_raw"], 5)
            self.assertEqual(batch.qc_manifest["valid"], 1)
            self.assertEqual(batch.qc_manifest["flagged"], 4)
            self.assertTrue(batch.qc_manifest["record_conservation_verified"])
            self.assertEqual(batch.branch_a_windows.shape, (1, 3, 400))

            # Verify direct routing of 3-component (1, 3, 400) window into SeismicMultiModalFusion Branch A
            fusion_3c = SeismicMultiModalFusion(wf_channels=3, gnn_dim=9, phys_dim=6, hidden=32, n_thresholds=4).eval()
            with torch.no_grad():
                out_3c = fusion_3c(
                    torch.from_numpy(batch.branch_a_windows),
                    torch.zeros((1, 10, 9), dtype=torch.float32),
                    torch.eye(10, dtype=torch.float32),
                    torch.tensor([0], dtype=torch.int64),
                    torch.from_numpy(batch.physics_features),
                    torch.ones((1, 3), dtype=torch.float32),
                )
            self.assertEqual(out_3c["det_logits"].shape, (1, 4))
            self.assertTrue(torch.isfinite(out_3c["det_logits"]).all())

            # Verify ObsPy MiniSEED 3-component ingestion & missing-channel flagging
            mseed_ok = tmp_dir / "event_ok.mseed"
            st_ok = obspy.Stream(
                [
                    obspy.Trace(data=good_3c[:, 0].copy(), header={"channel": "HHZ", "sampling_rate": 100.0}),
                    obspy.Trace(data=good_3c[:, 1].copy(), header={"channel": "HHN", "sampling_rate": 100.0}),
                    obspy.Trace(data=good_3c[:, 2].copy(), header={"channel": "HHE", "sampling_rate": 100.0}),
                ]
            )
            st_ok.write(str(mseed_ok), format="MSEED")

            mseed_1c = tmp_dir / "event_1c.mseed"
            obspy.Stream([obspy.Trace(data=good_3c[:, 0].copy(), header={"channel": "HHZ", "sampling_rate": 100.0})]).write(
                str(mseed_1c), format="MSEED"
            )

            ms_batch = ingest_miniseed_or_sac_records(
                [
                    {"event_id": "EV_OK", "file_path": mseed_ok, "p_arrival_sample": 200},
                    {"event_id": "EV_1C", "file_path": mseed_1c, "p_arrival_sample": 200},
                ]
            )
            self.assertEqual(ms_batch.qc_manifest["total_raw"], 2)
            self.assertEqual(ms_batch.qc_manifest["valid"], 1)
            self.assertEqual(ms_batch.qc_manifest["flagged"], 1)
            self.assertEqual(ms_batch.qc_manifest["flagged_records"][0]["flags"], ["MISSING_CHANNELS"])

            # Fail-fast verification on missing files and missing CSV columns
            with self.assertRaises(FileNotFoundError):
                ingest_stead_hdf5(tmp_dir / "nonexistent.h5", meta_csv)
            with self.assertRaises(FileNotFoundError):
                ingest_miniseed_or_sac_records(
                    [{"event_id": "EV_NONE", "file_path": tmp_dir / "missing.mseed", "p_arrival_sample": 200}]
                )
            bad_csv = tmp_dir / "bad_meta.csv"
            pd.DataFrame([{"trace_name": "TR_VALID_01"}]).to_csv(bad_csv, index=False)
            with self.assertRaises(ValueError):
                ingest_stead_hdf5(h5_path, bad_csv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
