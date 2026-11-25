# Project Inventory: Earthquake Early Warning Using CNN, BiLSTM, and Temporal Attention

**Workspace:** `/home/meghraj-dhakal/Earthquake_early_warning_and_prediction`  
**Date of Audit:** 18 November 2026  
**Auditor:** Research-Writing Team  

---

## 1. Executive Summary & Project Context

- **Working Title:** Controlled Evaluation of 1D-CNN, BiLSTM, and Temporal Attention for Synthetic Earthquake Early Warning and Magnitude Estimation
- **Problem Domain:** Real-time Earthquake Early Warning (EEW) — classifying seismic signals from non-seismic background noise and estimating source earthquake magnitude from initial P-wave observations (1.0 s, 2.0 s, and 3.0 s post-P arrival windows) using deep neural architectures.
- **Geological Motivation:** Northeast India and Indo-Burma subduction zone (Zone V, high seismic hazard).
- **Fundamental Methodological Reality:** The primary CSV (`dataset_2.csv`) contains catalog metadata (8,089 rows), not recorded physical waveform sensor traces. Continuous vertical-component (BHZ-like) seismic waveforms and background noise traces are 100% synthetically generated using a physics-informed mathematical simulation pipeline. The research evaluates architectural hypotheses on a controlled synthetic oracle-aligned benchmark; it does not claim operational deployment or real-world earthquake prediction.

---

## 2. (a) Data Synthesis and Generation Pipeline

### 2.1 File and Module Manifest

1. **Raw Metadata Source:**
   - File: `dataset_2.csv` (8,089 rows, 28 columns; SHA256 verified in audit manifests).
   - Columns include: `Magnitude`, `Depth`, `Latitude`, `Longitude`, `Origin Time`, station metadata, etc.
   - Note: Lacks empirical continuous waveform time-series records.

2. **Early Catalog Preprocessing (Phase 3 & Phase 6 exploration):**
   - File: `preprocess.py` (Phase 3 multi-stage cleaning and EDA).
     - Produced: `preprocessed_cleaned.csv` (8,089, 23), `preprocessed_eda.csv` (8,089, 34), `preprocessed_ml.csv` (8,089, 28), `preprocessed_dl.csv` (8,089, 30).
     - Summary log: `preprocessing_report.txt`.
   - File: `preprocess_dl_temporal.py` (Phase 6 temporal sequence engineering).
     - Produced: `preprocessed_dl_temporal.csv` (8,089, 44).
   - Note: These early CSV pipelines were designed for next-event catalog sequence forecasting ($M \ge 4.5$) before the project shifted to waveform-based Earthquake Early Warning (EEW).

3. **Physics-Informed Waveform Simulation Pipeline:**
   Located in `cnn/cnn wave/data wave/synthetic_waveform_simulator/`:
   - `phase1_inspect_metadata.py`: Catalog exploration and statistical boundary checks.
   - `phase2_travel_time.py`: Homogeneous crustal kinematic travel time calculations ($V_P = 6.0\text{ km/s}$, $V_S = 3.5\text{ km/s}$).
   - `phase3_clean_synthesis.py`: Clean synthetic body wave packet formulation.
   - `phase4_attenuation_noise.py`: Geometric spreading and attenuation exploration.
   - `phase5_prototype_evaluation.py`: 10-trace prototype inspection and FFT analysis.
   - `phase6_generate_full_dataset.py` & `waveform_generator.py`: Initial full-catalog synthesis (identified systemic +6.21 dB SNR bias and over-attenuation).
   - `improved_waveform_generator.py` & `phase8_improved_simulation/PHASE8_STEPS_AND_FORMULAS.md`:
     - Function: `generate_synthetic_waveform()`
     - Eliminated +6.21 dB SNR bias via pre-P window RMS empirical standard deviation calibration.
     - Implemented causal $C^1$-continuous Gamma onset envelopes.
     - Dynamic magnitude-dependent rise time $\tau_r(M)$ and distance-dependent coda decay $\tau_d(M, R)$.
     - Frequency-dependent power-law attenuation $Q(f) = Q_0 (f/f_0)^\eta$ ($Q_{P0}=350, Q_{S0}=300, \eta=0.70$).
     - Vertical-component (BHZ) amplitude ratio calibrated to $(A_S/A_P)_{\text{BHZ}} = 1.35$.
     - Multi-harmonic carrier functions for spectral richness.
   - `phase9_generate_full_dataset.py`:
     - Synthesized all 8,089 earthquake waveforms (`synthetic_seismic_waveforms.h5`).
     - QC check: 86% pass strict structural energy jumps; 100% pass numerical integrity.
   - `phase10_generate_classification_dataset.py`:
     - Synthesized 8,089 matched stochastic noise-only waveforms (total 16,178 balanced dataset).
     - Four noise generators:
       1. `gaussian_bandlimited`: 4th-order Butterworth bandpass (0.2–25 Hz).
       2. `colored_pink`: $1/f$ pink noise via Fourier filtering ($0.2\text{--}25\text{ Hz}$).
       3. `modulated_nonstationary`: Diurnal/environmental envelope modulation ($0.01\text{--}0.04\text{ Hz}$).
       4. `transient_disturbances`: Non-seismic smooth wavelets (cultural/sensor transients).
     - Calibrated to empirical pre-P RMS distribution of earthquake records.
     - Saved dataset: `output/phase10_classification_dataset/earthquake_noise_dataset.h5`.
   - `phase11_split_and_preprocess.py`:
     - Leakage-safe, pair-aware stratified split: 70% Train (11,324), 15% Validation (2,426), 15% Test (2,428).
     - Stratified by earthquake magnitude and QC pass status; co-locates paired earthquake and noise samples.
     - Preserves exact 50/50 balance in all partitions.
     - Generates `train.h5`, `validation.h5`, `test.h5`.

4. **Causal EEW Windowing & Tabular Feature Extraction:**
   - Waveform Causal Windows (`cnn/cnn wave/eew_causal_v3/`):
     - `01_causal_simulator_v3.ipynb`, `02_generate_dataset_v3.ipynb`, `03_dataset_validation_v3.ipynb`.
     - Extracts 1.0 s pre-P context (100 samples) + post-P observation horizon:
       - 1.0 s horizon: 200 samples total (2.0 s at 100 Hz)
       - 2.0 s horizon: 300 samples total (3.0 s at 100 Hz)
       - 3.0 s horizon: 400 samples total (4.0 s at 100 Hz)
     - S-wave arrival eligibility filtering: records where $t_S - t_P < h$ are masked out to guarantee strictly pre-S observations (at 1s: 0 excluded; at 2s: 7 test EQs excluded; at 3s: 14 test EQs excluded).
     - Common cohort across all horizons: 1,200 earthquakes and 1,214 noise records (2,414 total; SHA256: `862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601`).
   - Causal Tabular Features (`bilstm_eew/causal_tabular_v1/`):
     - `01_tabular_feature_extraction.ipynb`, `02_tabular_feature_validation.ipynb`.
     - Extracts 7 summary features per 10-sample (0.1 s) non-overlapping bin from normalized waveforms:
       1. RMS amplitude: $\sqrt{\frac{1}{10}\sum x_i^2}$
       2. Mean absolute amplitude: $\frac{1}{10}\sum |x_i|$
       3. Peak absolute amplitude: $\max |x_i|$
       4. Sample variance: $\frac{1}{10}\sum (x_i - \bar{x})^2$
       5. Signal energy: $\sum x_i^2 = 10 \cdot \text{RMS}^2$
       6. Zero-crossing rate: count of sign inversions / 9
       7. First-difference energy: $\sum_{i=0}^8 (x_{i+1} - x_i)^2$
     - Sequence lengths: 20 bins (1s horizon), 30 bins (2s horizon), 40 bins (3s horizon).
     - Standardized using train-only population mean and standard deviation per feature.
     - Saved dataset: `bilstm_eew/causal_tabular_v1/tabular_features.h5`.

---

## 3. (b) Every Model Trained or Evaluated, with Hyperparameters

All models were evaluated on the unified multi-task objective:
1. Binary Detection: Earthquake ($y=1$) vs. Noise ($y=0$), optimized via Binary Cross-Entropy (BCE).
2. Magnitude Regression: Earthquake magnitude $M$, optimized via Huber Loss ($\delta=1.0$) exclusively on eligible earthquake records (sentinel noise values masked out).

### 3.1 Standalone Pure 1D-CNN (`EarlyCNN`)
- **Directory:** `cnn/cnn wave/eew_causal_v3/cnn_eew_training/`
- **Input:** Raw normalized waveform, transformed pointwise by $x \mapsto \text{sign}(x) \log(1 + |x|)$.
- **Architecture:**
  - 3 strided 1D convolution blocks:
    - Conv1d(1, 16, kernel=7, stride=2, padding=3), BatchNorm1d(16), ReLU
    - Conv1d(16, 32, kernel=5, stride=2, padding=2), BatchNorm1d(32), ReLU
    - Conv1d(32, 48, kernel=3, stride=2, padding=1), BatchNorm1d(48), ReLU
  - Global Pooling: Concatenated Global Average Pooling + Global Max Pooling ($48 + 48 = 96\text{D}$)
  - Shared Dense: Linear(96, 32), ReLU, Dropout(0.2)
  - Heads: Linear(32, 1) detection logit; Linear(32, 1) standardized magnitude
  - Parameters: 14,674 trainable parameters.
- **Hyperparameters:**
  - Optimizer: AdamW, learning rate = 0.001, weight decay = 0.0001, gradient clip = 5.0
  - Batch size: 256; Maximum epochs: 50, patience: 8, min_delta: 0.0001
  - Loss: $\mathcal{L} = \mathcal{L}_{\text{BCE}} + 1.0 \cdot \mathcal{L}_{\text{Huber}}(\delta=1.0)$
  - Selected epochs: 1s: epoch 16; 2s: epoch 15; 3s: epoch 14.

### 3.2 Standalone Tabular BiLSTM
- **Directory:** `bilstm_eew/bilstm_training_v1/`
- **Input:** 7 standardized tabular features $\times$ $T$ bins ($T \in \{20, 30, 40\}$).
- **Architecture:**
  - 2-layer Bidirectional LSTM: 32 hidden units per direction (64D per step)
  - Interlayer dropout: 0.2
  - Temporal mean pooling across time steps $\to$ 64D pooled representation
  - Dropout: 0.2
  - Shared Dense: Linear(64, 32), SiLU activation
  - Heads: Linear(32, 1) detection logit; Linear(32, 1) standardized magnitude
  - Parameters: 37,730 trainable parameters.
- **Hyperparameters:**
  - Optimizer: AdamW, learning rate = 0.001, weight decay = 0.0001, gradient clip = 5.0
  - Batch size: 256; Maximum epochs: 35, patience: 7, min_delta: 0.0001
  - Validation-selected threshold: 1s: 0.47; 2s: 0.36; 3s: 0.36 (also fixed 0.5 reported).
  - Selected epochs: 1s: epoch 18; 2s: epoch 27; 3s: epoch 26.

### 3.3 Hybrid CNN-BiLSTM Feature Fusion
- **Directory:** `fusion/cnn_bilstm_v1/`
- **Input:** Frozen 32D embedding from CNN + Frozen 32D embedding from BiLSTM.
- **Architecture:**
  - Concatenation: $32 + 32 = 64\text{D}$, standardized by train-set embedding statistics
  - Fusion MLP: Linear(64, 32), SiLU, Dropout(0.2)
  - Heads: Linear(32, 1) detection logit; Linear(32, 1) standardized magnitude
  - Parameters: 2,146 trainable parameters (backbones frozen).
- **Hyperparameters:**
  - Optimizer: AdamW, learning rate = 0.001, weight decay = 0.0001, gradient clip = 5.0
  - Batch size: 256; Maximum epochs: 35, patience: 7, min_delta: 0.0001
  - Validation-selected threshold: 1s: 0.42; 2s: 0.41; 3s: 0.36.
  - Selected epochs: 1s: epoch 2; 2s: epoch 15; 3s: epoch 27.

### 3.4 Hybrid CNN-BiLSTM with Temporal Attention
- **Directory:** `fusion/cnn_bilstm_attention_v1/`
- **Input:** Frozen 32D CNN embedding + All hidden states $H \in \mathbb{R}^{T \times 64}$ from frozen BiLSTM.
- **Architecture:**
  - Additive Attention:
    $$u_t = \tanh(W_a h_t + b_a), \quad W_a \in \mathbb{R}^{32 \times 64}, b_a \in \mathbb{R}^{32}$$
    $$\alpha_t = \frac{\exp(v_a^T u_t)}{\sum_{j=1}^T \exp(v_a^T u_j)}, \quad v_a \in \mathbb{R}^{32}$$
    $$c = \sum_{t=1}^T \alpha_t h_t \in \mathbb{R}^{64}$$
  - Concatenation: CNN embedding (32D) + Attention context $c$ (64D) $\to$ 96D
  - Shared Dense: Linear(96, 32), SiLU, Dropout(0.2)
  - Heads: Linear(32, 1) detection logit; Linear(32, 1) standardized magnitude
  - Parameters: 5,282 trainable parameters (backbones frozen).
- **Hyperparameters:**
  - Optimizer: AdamW, learning rate = 0.001, weight decay = 0.0001, gradient clip = 5.0
  - Batch size: 256; Maximum epochs: 35, patience: 7, min_delta: 0.0001
  - Validation-selected threshold: 1s: 0.28; 2s: 0.41; 3s: 0.41.
  - Selected epochs: 1s: epoch 3; 2s: epoch 17; 3s: epoch 32.

### 3.5 3s Attention Hyperparameter Tuning Study
- **Directory:** `model_improvement/attention_tuning_v1/`
- **Scope:** 12 controlled trials on 3s Attention model:
  - Explored: learning rates ($10^{-4}, 3\times 10^{-4}, 10^{-3}$), dropout (0.1–0.4), hidden units (64, 128), regression weights (0.5, 1.0, 2.0), loss variants (`strong_bin`, `asymmetric`).
  - Outcome: 0 of 12 trials met the pre-declared replacement criteria over the baseline Attention model; original frozen model retained.

---

## 4. (c) Existing Metrics, Result Tables, and Figures

### 4.1 Definitive Result Tables
1. **Master Comparison Table:**
   - File: `final_comparison/four_model_eew_v1/final_metrics_table.csv`
   - Contains all 4 models $\times$ 3 horizons (1s, 2s, 3s) across both fixed 0.5 and validation-selected thresholds.
   - Evaluated on the identical common cohort: 1,200 earthquakes and 1,214 noise traces (2,414 total).
   - Metrics: Accuracy, Precision, Recall, F1, False Positive Rate (FPR), ROC-AUC, PR-AUC (Average Precision), MAE, RMSE, $R^2$, Mean Bias, Standard Deviation Ratio, Strong ($M \ge 5$) underprediction percentage, Strong Bias.
2. **Independent Audit Results:**
   - Verified in:
     - `cnn/cnn wave/eew_causal_v3/cnn_eew_training/final_cnn_eew_report.md`
     - `bilstm_eew/bilstm_training_v1/final_bilstm_eew_report.md`
     - `fusion/cnn_bilstm_v1/final_fusion_report.md`
     - `fusion/cnn_bilstm_attention_v1/final_attention_report.md`
     - `final_comparison/four_model_eew_v2/final_results_summary.md`
     - All `codex_independent_audit/audit_results.json` files.

### 4.2 Key Metrics Summary on Common Cohort (3s Horizon, Fixed Threshold 0.5)

| Metric | Standalone CNN | Standalone BiLSTM | Feature Fusion | Attention Fusion | Best Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Detection Accuracy** | 0.766777 | 0.714167 | 0.772162 | **0.776305** | Attention |
| **Detection Precision** | 0.783615 | 0.740566 | 0.832311 | **0.851812** | Attention |
| **Detection Recall** | **0.733333** | 0.654167 | 0.678333 | 0.665833 | CNN |
| **Detection F1-Score** | **0.757641** | 0.694690 | 0.747475 | 0.747428 | CNN |
| **False Positive Rate** | 0.200165 | 0.226524 | 0.135091 | **0.114498** | Attention |
| **ROC-AUC** | 0.858474 | 0.813242 | 0.863415 | **0.865393** | Attention |
| **PR-AUC (Avg. Prec.)** | 0.882498 | 0.834604 | 0.886098 | **0.887532** | Attention |
| **Magnitude MAE** | 0.613005 | 0.615082 | 0.608215 | **0.606601** | Attention |
| **Magnitude RMSE** | 0.757702 | 0.758421 | 0.753055 | **0.748621** | Attention |
| **Magnitude $R^2$** | 0.289803 | 0.288455 | 0.298488 | **0.306725** | Attention |
| **Magnitude Bias** | -0.090770 | -0.082112 | -0.061756 | **-0.004881** | Attention |
| **SD Ratio ($\sigma_{\hat{y}} / \sigma_y$)** | 0.513718 | 0.516038 | 0.569917 | **0.600623** | Attention |
| **$M \ge 5$ Underpred. %** | 96.95% | 96.18% | 92.37% | **87.02%** | Attention |
| **$M \ge 5$ Bias** | -1.164218 | -1.134616 | -1.047271 | **-0.930820** | Attention |

### 4.3 Existing Figures Manifest
- `final_comparison/four_model_eew_v1/plots/four_model_f1.png`: F1 comparison across 1s, 2s, 3s.
- `final_comparison/four_model_eew_v1/plots/four_model_mae.png`: MAE comparison across 1s, 2s, 3s.
- `final_comparison/four_model_eew_v1/plots/four_model_fpr.png`: False positive rate comparison across 1s, 2s, 3s.
- `final_comparison/four_model_eew_v1/plots/four_model_rmse.png`: RMSE comparison across 1s, 2s, 3s.
- `fusion/cnn_bilstm_attention_v1/plots/confusion_matrices.png`: Confusion matrices for Attention model.
- `fusion/cnn_bilstm_attention_v1/plots/temporal_attention_weights.png`: Temporal attention weight distribution over 40 bins.
- `fusion/cnn_bilstm_attention_v1/plots/actual_predicted_magnitude.png`: Actual vs. predicted magnitude scatter plot.
- `fusion/cnn_bilstm_attention_v1/plots/magnitude_residuals.png`: Residual error distribution.
- `explainability/final_attention_shap_v1/plots/`: 12 high-resolution SHAP plots (waveform SHAP, feature importances, temporal importances, local waterfall explanations, Attention vs. SHAP comparisons).

---

## 5. (d) Inconsistencies, Caveats, and Verifiable Boundaries

1. **Synthetic Data Reality:**
   The dataset consists entirely of synthetic mathematical signals. It does not contain field-recorded waveforms from seismic networks. Generalization to real instrument response, soil non-linearities, complex 3D geological basins, and operational telecommunication latencies cannot be established from this benchmark.
2. **Oracle P-Arrival Alignment:**
   All waveforms are strictly time-aligned using theoretical homogeneous kinematic P arrivals ($t_P = R / V_P$). In an operational EEW system, autonomous phase picking (e.g. STA/LTA or pickers) introduces arrival time uncertainty and trigger jitter.
3. **Paired Statistical Significance between Fusion and Attention:**
   While Attention achieves nominally the lowest MAE (0.606601 vs. 0.608215 for Fusion) and highest precision, paired stratified bootstrap 95% confidence intervals cross zero:
   - F1 difference (Attention minus Fusion): $[-0.0060, +0.0065]$ (crosses zero).
   - MAE difference (Attention minus Fusion): $[-0.0064, +0.0032]$ (crosses zero).
   Therefore, the empirical evidence does not establish statistically significant superiority of Attention over standard Fusion.
4. **Prediction Compression and Severe Strong-Event Underprediction:**
   All four models suffer from regression variance compression ($\sigma_{\hat{y}} / \sigma_y \approx 0.51\text{--}0.60$). For strong earthquakes ($M \ge 5.0$), models exhibit severe underprediction ($87.0\%\text{--}96.9\%$ of events underpredicted), with negative biases approaching $-0.93$ to $-1.16$ magnitude units.
5. **SHAP Explanation Stability & Numerical Reconstruction:**
   In the explainability study of the 3s Attention model:
   - Detection logit reconstruction failed strict tolerance on 4 of 12 validation cases (maximum residual 0.6226 logits).
   - Training background reference shift changed the base expected logit by 1.974 logits.
   - Tabular features are deterministic nonlinear functions of the identical waveform input, violating independent feature variation assumptions of additive attribution methods.
   - The official SHAP audit verdict is therefore **CONDITIONAL**, serving as a qualitative descriptive diagnostic rather than a causal proof.

---
*Inventory complete. All items verified against source files.*
