# Independent Scientific Verification Report: Causal EEW Dataset v2

**Target Directory:** `cnn/cnn wave/eew_causal_v2/`  
**Review Date:** 2026-11-10  
**Evaluator:** Independent Review Agent  
**Mandate:** Rigorous causality, dataset structure, and original file immutability verification.

---

## 1. Verification of Critical Requirements

| # | Check Description | Verdict | Evidence / Code Location |
|---|---|:---:|---|
| **1** | **Future S-wave/coda isolation** | **PASS** | `01_causal_simulator_and_pilot_validation.ipynb` (Cells 5–7). $4\times$ S-wave scaling and $+50\%$ coda duration caused exactly $0.00000000$ difference in pre-S and pre-P samples, both raw and normalized. |
| **2** | **Independent ambient noise** | **PASS** | `02_generate_causal_eew_dataset.ipynb` (Cell 3, lines 60–65). Noise is generated via independent Gaussian draws through a 4th-order Butterworth bandpass (0.2–25 Hz) scaled to stationary target RMS (0.05). Zero coupling to signal peak or future S-wave. |
| **3** | **No whole-trace / future normalization** | **PASS** | `02_generate_causal_eew_dataset.ipynb` (Cell 3, lines 85–101). Normalization parameters $(\mu_{\text{pre}}, \sigma_{\text{pre}})$ are derived strictly from trailing pre-P noise buffer $[t_P - 1.0\text{ s}, t_P]$. Raw physical ground motion is preserved in parallel. |
| **4** | **Variable P-wave onset distribution** | **PASS** | `02_generate_causal_eew_dataset.ipynb` (Cell 3, line 23) and `dataset_audit_metrics.json`. $t_P \in [3.00\text{ s}, 7.00\text{ s}]$ with mean $\approx 4.98\text{ s}$ and $\sigma \approx 1.15\text{ s}$. The static $t_P \approx 6.00\text{ s}$ artifact is eliminated. |
| **5** | **Pre-S observation cutoff enforcement** | **PASS** | `dataset/eew_dataset_{split}.h5:pre_s_valid_{1,2,3}s`. Horizon $\tau = 1.0\text{ s}$ is 100% valid ($\min(S-P) = 1.06\text{ s}$). Horizons $\tau = 2.0\text{ s}$ ($99.6\%$) and $\tau = 3.0\text{ s}$ ($98.8\%$) provide explicit boolean masks to filter out any S-wave intrusion. |
| **6** | **Labels, magnitude, and arrival annotations** | **PASS** | `dataset/eew_dataset_{split}.h5`. Earthquakes have $y = 1$, $M \in [1.5, 8.6]$, valid $t_P, t_S$. Noise traces have $y = 0$, $M = -1.0$, $t_P = -1, t_S = -1$. No label corruptions. |
| **7** | **Pair and source disjointness across splits** | **PASS** | `dataset/eew_dataset_{train,validation,test}.h5`. Tested `pair_id` and `source_id` intersections across train-val, train-test, and val-test: exactly 0 overlap. |
| **8** | **Dataset counts and tensor dimensions** | **PASS** | Train: 11,324; Validation: 2,426; Test: 2,428. Window tensors: `(N, 200)`, `(N, 300)`, `(N, 400)` float32. Pure post-P tensors: `(N, 100)`, `(N, 200)`, `(N, 300)` float32. |
| **9** | **Original file protection (17 protected files)** | **PASS** | `original_files_audit_result.json` and independent re-hash. All 17 original files match initial SHA256 hashes byte-for-byte (zero files modified, deleted, or moved). |
| **10** | **Oracle P-wave timing acknowledgment** | **PASS** | Documented in `reports/eew_causal_v2_completion_report.md`. Windows are anchored around theoretical $t_P$ for rapid post-trigger analysis; this does not constitute continuous stream picking. |

---

## 2. Pre-P Context Audit (200 / 300 / 400 Sample Windows)

- **Structure:**
  - 1s Horizon: 100 pre-P samples ($1.0\text{ s}$) + 100 post-P samples ($1.0\text{ s}$) = 200 samples.
  - 2s Horizon: 100 pre-P samples ($1.0\text{ s}$) + 200 post-P samples ($2.0\text{ s}$) = 300 samples.
  - 3s Horizon: 100 pre-P samples ($1.0\text{ s}$) + 300 post-P samples ($3.0\text{ s}$) = 400 samples.
- **Unavailable Information Check:**
  - In operational instrumentation, a continuous circular buffer maintains trailing ambient noise prior to trigger arrival.
  - In `eew_causal_v2`, ambient noise is generated independently of future earthquake physics. The pre-P buffer contains strictly environmental colored noise; it uses zero future or unavailable information.
  - Pure post-P windows (`windows_pure_post_p_Xs`: 100, 200, 300 samples) without pre-P context are also provided for models requiring zero pre-event buffer.

---

## 3. Scientific Limitations & Scope Boundary

1. **Synthetic Nature:** Synthesized waveforms are physics-informed mathematical models; transfer to real continuous recorded broadband seismograms requires domain adaptation.
2. **Operational Scope:** This dataset evaluates post-trigger classification and rapid magnitude regression given an initial trigger; it does not benchmark continuous streaming onset pickers.

---

## 4. Final Review Verdict

All 10 scientific checks have passed with exact machine precision. All original repository files remain completely untouched.

**VERDICT:** **APPROVED FOR SYNTHETIC CNN RESEARCH TRAINING**

