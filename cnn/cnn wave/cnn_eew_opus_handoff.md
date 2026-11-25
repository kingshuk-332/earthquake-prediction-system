# CNN EEW Audit & Handoff to Opus

**Project:** `cnn/cnn wave/`  
**Model:** `wavecnn_ccb3b06b3d37` (`phase1_waveform_cnn_results/models/wavecnn_ccb3b06b3d37/best.pt`)  
**Dataset:** Phase 11 Preprocessed HDF5 (`data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/`)  
**Audit Date:** 2026-11-10  
**Overall EEW Status:** **FAIL — NOT AN EEW SYSTEM**

---

## 1. Model & Dataset Dimensions

- **Input Shape:** `(batch, 1, 10000)` — 1-channel, 10,000 samples (100.0 s at 100 Hz).
- **Architecture:** Conv1D (1→8, k9, s2) → Pool4 → Conv1D (8→16, k7, s2) → Pool4 → Conv1D (16→32, k5) → `AdaptiveAvgPool1d(1)` → Linear (32→16) → Dropout (0.3) → Linear (16→1).
  * Trainable Parameters: 4,177.
  * Inference Mode: Offline whole-trace evaluation; consumes all 100 seconds before predicting.
- **Dataset Splits:**
  * `train.h5`: 11,324 samples (5,662 Earthquake + 5,662 Noise).
  * `validation.h5`: 2,426 samples (1,213 Earthquake + 1,213 Noise).
  * `test.h5`: 2,428 samples (1,214 Earthquake + 1,214 Noise).
  * All splits pair-disjoint and stratified by magnitude and QC flags.
- **Target:** Single binary classification label $y \in \{0, 1\}$ ($0 = \text{noise}$, $1 = \text{earthquake}$).
- **Reported Test Accuracy:** 99.84% (ROC-AUC: 0.999995).

---

## 2. P/S Label & Metadata Availability

- **HDF5 Attributes:** `p_arrival_sample`, `s_arrival_sample`, `magnitude`, `depth_km`, `hypocentral_distance_km`, `snr_db`, `receiver_code`, `source_id`, `trace_name`, `qc_pass`, `pair_id`.
- **P-Arrival Artifact:** P-arrival is rigidly fixed across all earthquakes at $t_P \in [5.93\text{ s}, 6.06\text{ s}]$ (mean $6.001\text{ s}$).
- **S-Arrival Range:** Across test earthquakes, $t_S \in [7.06\text{ s}, 85.04\text{ s}]$ (mean $31.46\text{ s}$).
- **Metadata Joins:** Origin time (UTC) and station Lat/Long are absent inside HDF5, but present in `data wave/synthetic_waveform_simulator/output/phase2_travel_time/metadata_with_theoretical_arrivals.csv` (joinable via `source_id`).

---

## 3. Causality & Leakage Findings

1. **Acausal Simulator Noise Scaling (`improved_waveform_generator.py:224-243`):**
   * Pre-P ambient noise amplitude is scaled using `signal_peak = np.max(abs(clean_signal[p_arrival_sample:]))`.
   * Because S-wave amplitude dominates peak signal, pre-P noise variance is directly proportional to future S-wave magnitude.
   * **Counterfactual Evidence:** In `bilstm new updated/reports/causal_noise_counterexample.json`, quadrupling future S-wave amplitude on trace `NCS_EV_07168` scaled early noise standard deviation by $4\times$ ($0.0542 \to 0.2168$, max change $0.5076$), proving unphysical future leakage into the pre-P window.
2. **Acausal Normalization (`phase11_split_and_preprocess.py:179, 187-197`):**
   * Full 100-second linear detrending and per-trace z-score standardization (`waveforms_standardized`) divide early P-waves by the full-trace standard deviation, which is dominated by future S-wave and coda energy.
3. **Acausal Inference:**
   * Global average pooling (`AdaptiveAvgPool1d(1)`) requires the full 100 seconds. No streaming or early-warning alert can be issued prior to trace completion.

---

## 4. Current Capability Assessment

| Capability | Status | Evidence / Reference |
|---|:---:|---|
| Offline 100-Second Waveform Detection | **PASS** | 99.84% accuracy on synthetic Phase 11 test set (`phase1_waveform_cnn.ipynb`, Cell 12). |
| Real-Time P-Wave Onset Detection ($t_P$) | **FAIL** | Not implemented. CNN outputs binary class, not time-localized picks. P is anchored at 6 s. |
| Early Warning Lead Time ($\Delta t = t_S - t_{\text{alert}}$) | **FAIL** | Not implemented. Inference requires 100 s; S-waves arrive at mean 31.5 s. |
| Rapid Magnitude Estimation ($M$) | **FAIL** | Not implemented. Model outputs binary class; standardization discards absolute amplitude. |
| Operational Real-Waveform Transfer | **UNKNOWN** | Evaluated purely on synthetic simulator data. Spectral shortcut probes show drop to 94.19%. |

---

## 5. Unresolved Scientific Risks

1. **Synthetic Shortcut Learning:** Phase-randomization probe dropped accuracy to 94.19%, indicating reliance on stationary spectral coloring rather than causal wave arrival physics.
2. **Deterministic Arrival Prior:** Because $t_P \approx 6.00\text{ s}$ across all synthetic events, any temporal feature detector can overfit to arrival timing around sample 600.
3. **Amplitude Stripping:** Standardizing traces removes ground motion velocity/acceleration amplitude, which is essential for real EEW magnitude determination ($P_d$, $\tau_c$).

---

## 6. Explicit Questions for Opus Verification

1. **Quarantined Window Validation:** Does the quarantined 1/3/5-second post-P dataset in `bilstm new updated/dataset/eew_candidate_windows.npz` completely eliminate acausal noise leakage, or does `target_noise_sigma` contamination still persist in its pre-P buffer?
2. **EEW Task Definition:** Should the next model iteration focus on (a) P-wave picking on continuous streams (PhaseNet/EQTransformer style), or (b) 3-second post-P magnitude/distance regression, or (c) dual-task detection and picking?
3. **Source Waveform Replacement:** Given the simulator defects in `improved_waveform_generator.py`, should synthetic waveforms be discarded entirely in favor of real regional recordings (e.g., NE-India NCS broadband data or STEAD)?

