# Independent Scientific Audit: CNN Waveform EEW Feasibility

**Project:** `C:/Users/choud/Desktop/Minor project/cnn/cnn wave/`  
**Review Target:** `cnn_eew_opus_handoff.md` and referenced source implementations  
**Auditor:** Independent Review Agent  
**Date:** 2026-11-10  

---

## 1. Verification of Critical Claims

### Claim 1: Full 100-Second CNN Input and Binary-Only Output
- **Verdict:** **PASS**
- **Evidence & Source:** `phase1_waveform_cnn.ipynb`, Code Cell 4 (lines 374–385) and Cell 1 (line 63); `phase1_waveform_cnn_results/reports/wavecnn_ccb3b06b3d37/config.json`.
- **Finding:** Input tensor shape is strictly `(batch, 1, 10000)` (100.0 s at 100 Hz). The architecture `WaveformCNN` aggregates the entire trace via `AdaptiveAvgPool1d(1)` into `Linear(16, 1)` with `BCEWithLogitsLoss`. It executes offline whole-trace binary detection ($y \in \{0, 1\}$); it does not perform continuous streaming phase picking, arrival estimation, or magnitude regression.

### Claim 2: Future S-Wave Amplitude Affecting Earlier Noise
- **Verdict:** **PASS**
- **Evidence & Source:** `data wave/synthetic_waveform_simulator/improved_waveform_generator.py`, lines 224–243; counterfactual artifact `bilstm new updated/reports/causal_noise_counterexample.json`.
- **Finding:** Line 225 computes `signal_peak = float(np.max(np.abs(clean_signal[p_arrival_sample:])))`, spanning the entire waveform post-P. Because S-wave amplitude dominates the record, line 230 calculates `target_noise_sigma = signal_peak / (10^(SNR/20))` using future S-wave energy. Line 243 scales the entire noise trace, including the pre-P window. The counterfactual test proved that multiplying future S-wave amplitude by 4 scaled pre-P noise by $4\times$ ($0.054 \to 0.217$, max shift $0.508$).

### Claim 3: Full-Trace Normalization Using Future Samples
- **Verdict:** **PASS**
- **Evidence & Source:** `data wave/synthetic_waveform_simulator/phase11_split_and_preprocess.py`, lines 179 and 187–197; `phase1_waveform_cnn.ipynb`, line 63 (`waveform_key='waveforms_standardized'`).
- **Finding:** Preprocessing applies zero-phase linear detrending across all 10,000 samples and trace-level z-score standardization (`axis=-1`). Dividing early P-wave arrivals by full-record variance governed by subsequent S-wave and coda arrivals introduces mathematical acausality.

### Claim 4: Fixed P-Wave Onset Near 6 Seconds
- **Verdict:** **PASS**
- **Evidence & Source:** `output/phase2_travel_time/metadata_with_theoretical_arrivals.csv`, column 19 (`p_arrival_sample`) vs column 39 (`theoretical_p_arrival_sample`); `test.h5` attribute `p_arrival_sample`.
- **Finding:** All synthetic earthquake traces in Phase 11 have P-wave arrivals artificially pinned in the range $[593, 606]$ samples ($5.93\text{ s}$ to $6.06\text{ s}$, mean $6.001\text{ s}$, $\sigma \approx 0.025\text{ s}$), whereas physical theoretical P arrivals span $5.45\text{ s}$ to $87.99\text{ s}$. Noise traces have arrival sentinel $-1$. The CNN can exploit this fixed temporal prior as an artificial detection shortcut.

### Claim 5: Existing Phase 11 Windows Unsuitable for Causal EEW Evaluation
- **Verdict:** **PASS**
- **Evidence & Source:** Upstream generator `improved_waveform_generator.py` and preprocessor `phase11_split_and_preprocess.py`.
- **Finding:** Slicing Phase 11 waveforms post-hoc into 1–3 second post-P windows cannot remove acausal leakage because the pre-P noise samples were corrupted at synthesis time by future S-wave amplitude. Furthermore, whole-trace standardization destroys physical amplitude parameters ($P_d$) required for rapid magnitude estimation.

---

## 2. Minimum Corrections for a Causal Synthetic EEW Dataset

1. **Decoupled Ambient Noise:** Synthesize background noise independently of earthquake signal amplitude (e.g., using empirical seismic noise models or real pre-event seismic noise).
2. **Streaming Causal Preprocessing:** Standardize waveforms strictly using trailing pre-event buffers ($[t_P - 5\text{ s}, t_P]$) or sensor calibration scale factors. Zero whole-trace detrending or post-event normalization.
3. **Dynamic Arrival Alignment:** Randomize arrival onsets across the trace buffer to prevent fixed-sample shortcut learning.
4. **Causal Time Horizons:** Restrict evaluation windows to $t \le t_P + \tau$ ($\tau \in \{1, 2, 3\}\text{ s}$) enforcing $t_P + \tau < t_S$.
5. **Physically Meaningful Targets:** Implement real-time trigger picking ($t_P$) and rapid magnitude estimation ($M$).

---

## 3. Decision

- **Causal Dataset Reconstruction:** **GO** (Authorized to construct causally sound dataset pipeline).
- **Model Training / Benchmarking:** **NO-GO** (Model training remains strictly blocked until reconstructed dataset passes causality audit).

