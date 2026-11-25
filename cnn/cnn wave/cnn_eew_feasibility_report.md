# Comprehensive Feasibility and Causality Audit: Waveform CNN for Earthquake Early Warning (EEW)

**Project:** `cnn/cnn wave/`  
**Model Checkpoint:** `phase1_waveform_cnn_results/models/wavecnn_ccb3b06b3d37/best.pt` (4,177 parameters)  
**Dataset:** Phase 11 Preprocessed HDF5 (`train.h5`, `validation.h5`, `test.h5`)  
**Audit Date:** 2026-11-10  
**Status:** **INELIGIBLE FOR EEW (CRITICAL CAUSALITY & FORMULATION DEFECTS)**

---

## 1. Executive Summary

The trained Waveform CNN achieves a reported **99.84% accuracy** on the synthetic Phase 11 test set. However, an exhaustive audit demonstrates that **this model has zero Earthquake Early Warning (EEW) capability**:

1. **Acausal Input Formulation:** Inference consumes an entire **100-second full trace** (10,000 samples) through global average pooling (`AdaptiveAvgPool1d(1)`). An EEW system must issue alerts within $1–3$ seconds of P-wave arrival, well before destructive S-waves arrive. Waiting 100 seconds means destructive waves have already arrived and dissipated.
2. **Acausal Simulator Defect:** The synthetic simulator (`improved_waveform_generator.py:224-243`) scales the entire 100-second noise trace—including pre-P ambient noise—using the peak amplitude of the future signal (`signal_peak = np.max(abs(clean_signal[p_arrival_sample:]))`). Because peak amplitude is dominated by the future S-wave, pre-P noise amplitude directly encodes future S-wave magnitude.
3. **Acausal Preprocessing:** Per-trace standardization (`waveforms_standardized`) and linear detrending are computed across all 10,000 samples, allowing future S-wave and coda variance to normalize early P-wave amplitudes.
4. **Mismatched Prediction Target:** The CNN is strictly an offline binary classifier (Noise vs. Earthquake). It does not pick P onset, does not estimate earthquake magnitude, does not predict S arrival, and provides no warning lead time.
5. **Artificial P-Arrival Prior:** In all 16,178 earthquake records across splits, P-wave arrival is artificially pinned at $t \approx 6.00\text{ s}$ ($\pm 0.07\text{ s}$), making detection trivial.

---

## 2. Technical Verification of Current Model & Data

### A. Model Input & Architecture
* **Input Tensor:** Shape `(batch, 1, 10000)`, single vertical component ($Z$), 100 Hz sampling rate, 100-second duration.
* **Architecture:** 3 Conv1D blocks with batch normalization, ReLU, and MaxPool1d, followed by `nn.AdaptiveAvgPool1d(1)` and a 2-layer MLP head (4,177 total trainable parameters).
* **Global Pooling:** Global average pooling averages features across the entire 100-second temporal dimension, inherently preventing time-localized P-wave onset detection.
* **Inference Requirement:** Cannot operate on streaming, sliding, or short P-wave windows without complete structural retraining.

### B. Prediction Target & Capabilities
* **Target:** Binary classification label $y \in \{0, 1\}$ ($0 = \text{noise}$, $1 = \text{earthquake}$).
* **P-Onset Picking:** **FAIL (Not Implemented).**
* **Magnitude Estimation ($M$):** **FAIL (Not Implemented).**
* **S-Wave Arrival Prediction ($t_S$):** **FAIL (Not Implemented).**
* **Warning Lead Time ($\Delta t = t_S - t_{\text{alert}}$):** **FAIL (Not Implemented).**

### C. Label & Metadata Availability
* **P and S Arrival Labels:** Stored in HDF5 (`p_arrival_sample` and `s_arrival_sample`).
  * Test split earthquake P-arrivals: strictly span samples 593 to 606 ($5.93\text{ s}$ to $6.06\text{ s}$, mean $6.001\text{ s}$).
  * Test split earthquake S-arrivals: span samples 706 to 8504 ($7.06\text{ s}$ to $85.04\text{ s}$, mean $31.46\text{ s}$).
* **Station & Event Metadata:** `source_id`, `trace_name`, `receiver_code`, `magnitude`, `depth_km`, `hypocentral_distance_km`, `snr_db`, and `pair_id` are preserved. Origin times and station coordinates require joins with `metadata_with_theoretical_arrivals.csv`.

---

## 3. Causality & Leakage Audit

### A. Simulator Noise Scaling Leakage
In `improved_waveform_generator.py` lines 224–243:
```python
# Signal peak evaluated across full signal region including future S-wave
signal_peak = float(np.max(np.abs(clean_signal[p_arrival_sample:])))
target_noise_sigma = signal_peak / (10.0 ** (target_snr_db / 20.0))
# Entire 100-second noise trace scaled by target_noise_sigma
calibrated_noise = filtered_noise * (target_noise_sigma / pre_p_raw_std)
```
* **Impact:** S-wave amplitude ($\text{amp}_S$) is typically $3\times$ to $10\times$ larger than P-wave amplitude ($\text{amp}_P$). Therefore, `signal_peak` is governed by the future S-wave.
* **Empirical Counterfactual Evidence:** On trace `ND.DBR..BHZ__earthquake_local__NCS_EV_07168`, multiplying future S-wave amplitude by 4 (leaving P-wave, distance, SNR, and seed identical) caused pre-P noise ($t < 5.96\text{ s}$) to scale by $4\times$, shifting pre-P samples by up to $0.5076$ (documented in `causal_noise_counterexample.json`).
* **Conclusion:** Early ambient noise contains an unphysical causal leak of future earthquake energy.

### B. Preprocessing Variance Leakage
In `phase11_split_and_preprocess.py` lines 179–197:
* Linear detrending is computed across all 10,000 samples.
* Per-waveform standardization computes:
  $$\sigma_{\text{trace}} = \sqrt{\frac{1}{10000} \sum_{i=1}^{10000} (x_i - \mu)^2}$$
* Early P-wave amplitudes are divided by $\sigma_{\text{trace}}$, which is dominated by the future S-wave and surface wave envelope.

---

## 4. Minimum Corrections Required for Causal P-Wave EEW

To transition this research into a legitimate EEW system, the following 5 pipeline modifications are mandatory:

1. **Short Post-P Causal Windowing:**
   * Extract windows strictly relative to P arrival: $[t_P - 1.0\text{ s}, t_P + \tau]$ where $\tau \in \{1.0\text{ s}, 2.0\text{ s}, 3.0\text{ s}\}$.
   * Enforce causal cutoff: $t_P + \tau < t_S$ (discard windows where S arrives before cutoff).
2. **Causal Noise Calibration:**
   * Calibrate ambient noise strictly against pre-P historical noise records or standard geophysical noise models (e.g. Peterson's NLNM/NHNM), completely independent of future signal peak.
3. **Causal Streaming Preprocessing:**
   * Standardize traces using running historical statistics (e.g. trailing 5–10 second pre-P noise buffer) or absolute physical instrument calibration ($\text{m/s}$ or $\text{m/s}^2$). Never detrend or standardize using future samples.
4. **Actionable EEW Target Formulation:**
   * Reformulate model heads for:
     a) Real-time trigger / pick probability ($P_{\text{onset}}$),
     b) Magnitude estimation from initial P-wave displacement parameter ($P_d$) and characteristic period ($\tau_c$).
5. **Real-World Recorded Waveform Benchmarking:**
   * Replace synthetic waveforms with recorded regional broadband seismograms (e.g. regional North-East India NCS network) to avoid simulator-specific spectral shortcuts.

