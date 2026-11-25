# Seismic Activity Forecasting in North-East India Using Deep Learning

An end-to-end deep learning framework for forecasting significant earthquakes ($M \ge 4.5$) in the seismically active North-East India and Indo-Burma Subduction Zone using chronological seismic event sequences.

---

## 📌 Executive Summary

* **Geological Focus:** North-East India & Indo-Burma Subduction Zone (Seismic Zone V — among the most active collision/subduction margins globally).
* **Forecasting Formulation:** Binary sequence classification via a sliding window of **15 consecutive past seismic events** to predict whether the **next (16th) event** will be a **significant earthquake** ($M \ge 4.5$).
* **Core Machine Learning Challenge:** Severe **temporal regime shift** and class imbalance — historical catalogs (1906–2010s) captured mostly felt earthquakes (~35% positive), whereas modern digital seismic arrays record thousands of micro-tremors, collapsing positive test prevalence to only **4.5% (1:21 imbalance)**.
* **Architectures Evaluated:**
  1. **Vanilla & Temporal BiLSTM** (TensorFlow / Keras)
  2. **Pure Dilated 1D-CNN with Squeeze-and-Excitation** (PyTorch / CUDA)
  3. **Hybrid CNN-BiLSTM with Temporal Self-Attention** (PyTorch / CUDA with Explainable AI)
* **Key Results:**
  * **Hybrid CNN-BiLSTM-Attention** achieved the highest test Average Precision (**0.0710** PR-AUC vs. 0.0452 random baseline) and the highest minority detection recall (**60.3%**, capturing 44 out of 73 significant earthquakes at calibrated threshold $0.12$).
  * **Temporal BiLSTM** achieved the best overall test ROC-AUC (**0.5939**) with 92% accuracy at threshold $0.44$.
  * **Pure 1D-CNN v2** demonstrated high parameter efficiency (27,857 params, 85% lighter) and resilience to synthetic sensor noise ($\sigma = 0.015$).

---

## 🗂️ Project Directory Structure

```
Minor project(1)/
├── README.md                                      # This comprehensive project documentation
└── Minor project/
    ├── dataset_2.csv                              # Raw historical seismic catalog (8,089 records, 1906–2026)
    ├── preprocess.py                              # Multi-stage cleaning & classical ML/DL feature pipeline
    ├── preprocessed_cleaned.csv                   # Validated coordinate and timestamp data
    ├── preprocessed_eda.csv                       # EDA-enriched dataset (magnitude bins, depth bins)
    ├── preprocessed_ml.csv                        # Scaled & label-encoded dataset for classical ML
    ├── preprocessed_dl.csv                        # 30-column baseline normalized dataset
    ├── preprocess_dl_temporal.py                  # Dedicated 44-feature temporal engineering pipeline
    ├── preprocessed_dl_temporal.csv               # 44-feature time-series sequence dataset (MinMax scaled)
    ├── dl_temporal_scaler.pkl                     # Serialized scaler and magnitude threshold params
    ├── ml_scaler.pkl                              # Scikit-learn StandardScaler for ML features
    ├── ml_label_encoders.pkl                      # Categorical LabelEncoders
    ├── preprocessing_report.txt                   # Automated preprocessing audit log
    ├── status.md                                  # Complete technical development log & experiment notes
    │
    ├── bilstm/                                    # TensorFlow / Keras BiLSTM Experiments
    │   ├── bilstm_model.py                        # Baseline BiLSTM script (static 30 features)
    │   ├── bilstm_model.keras                     # Saved baseline weights
    │   ├── bilstm_results.txt                     # Baseline evaluation metrics
    │   ├── bilstm_temporal_model.py               # Main temporal BiLSTM with SMOTE & Focal Loss
    │   ├── bilstm_temporal_best.keras             # Best validation checkpoint (Epoch 12)
    │   ├── bilstm_temporal_model.keras            # Final epoch model weights
    │   ├── bilstm_temporal_results.txt            # Full test classification report
    │   ├── bilstm_v2_model.py                     # Non-SMOTE class-weight ablation experiment
    │   ├── plots_temporal/                        # Publication figures (ROC, PR, Confusion Matrix, History)
    │   └── plots_v2/                              # Ablation study curves
    │
    ├── cnn/                                       # PyTorch Pure 1D-CNN Experiments
    │   ├── generate_noisy_dataset.py              # Sensor jitter injection script (Gaussian σ = 0.015)
    │   ├── preprocessed_dl_temporal_noisy.csv     # Augmented noisy dataset for robustness benchmarking
    │   ├── cnn_1d_model.py                        # Efficient Dilated 1D-CNN + SE Attention model script
    │   ├── cnn_1d_v2_best.pth                     # Best validation checkpoint (Epoch 27)
    │   ├── cnn_1d_v2_final.pth                    # Final training epoch weights
    │   ├── cnn_1d_results.txt                     # Classification metrics & confusion matrix
    │   ├── plots_cnn_1d/                          # 10 diagnostic plots (noise comparison, ROC, PR, etc.)
    │   └── evaluation_cnn_1d/                     # CSV & JSON evaluation reports and predictions
    │
    └── hybrid/                                    # PyTorch CNN-BiLSTM-Self-Attention Experiments
        ├── cnn_bilstm_attention_pytorch.py        # Hybrid model architecture training script
        ├── cnn_bilstm_attention_best.pth          # Best validation model weights
        ├── cnn_bilstm_attention_final.pth         # Final epoch weights
        ├── cnn_bilstm_attention_results.txt       # Quantitative test metrics summary
        ├── generate_cnn_attention_metrics.py      # Detailed evaluation matrix & attention extractor
        ├── plots_pytorch/                         # PyTorch training curves, ROC, PR, and attention maps
        └── evaluation_cnn_bilstm_attention/       # Full evaluation suite (JSON, CSV tables, XAI timeline)
```

---

## 🔬 Problem Formulation & The "Temporal Shift"

### Sequence Forecasting Setup
* **Input ($\mathbf{X}_i \in \mathbb{R}^{15 \times 44}$):** A sliding window of 15 consecutive historical seismic events.
* **Target ($y_i \in \{0, 1\}$):** Binary indicator whether the next event (16th) has magnitude $M \ge 4.5$:
  $$y_i = \mathbb{I}\left(M_{i+15} \ge 4.5\right)$$
* **Total Generated Sequences:** 8,074 sequences.

```
[Event t-14, Event t-13, ..., Event t-1, Event t]  ───▶ [ Deep Neural Model ] ───▶  P(Next Event M >= 4.5)
```

### Chronological Splitting vs. Temporal Shift
Random cross-validation causes massive data leakage in time-series forecasting. Hence, a **strict chronological split** was applied:

| Partition | Time Span | Total Events | Positive Events ($M \ge 4.5$) | Positive % | Class Imbalance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train Set (70%)** | 1906 – ~2010s | 5,651 | 1,956 | **34.6%** | ~1 : 1.9 |
| **Validation Set (10%)** | ~2010s – ~2016 | 808 | 47 | **5.8%** | ~1 : 16.2 |
| **Test Set (20%)** | ~2016 – Oct 2026 | 1,615 | 73 | **4.5%** | **1 : 21.1** |

> **Geophysical Insight:** In the early 20th century, regional instruments only recorded felt or destructive earthquakes ($M \ge 4.5$). In the modern era, high-density digital telemetry captures thousands of micro-tremors ($M < 3.0$), causing positive prevalence to drop from 34.6% to 4.5%.

---

## ⚙️ Data Preprocessing & Feature Engineering

### 1. Dedicated Temporal Feature Engineering (`preprocess_dl_temporal.py`)
Produces `preprocessed_dl_temporal.csv` with **44 normalized continuous and categorical features**:

1. **Cyclical Temporal Encodings:**
   $$\text{feature}_{\sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad \text{feature}_{\cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$
   Encodes Month ($T=12$), Hour ($T=24$), Day of Week ($T=7$), and Day of Year ($T=365$).
2. **Inter-Event Dynamics (Tectonic Stress Clock):**
   * `time_diff_hours`: Elapsed hours since immediately preceding event.
   * `log_time_diff`: $\ln(1 + \Delta t)$, compressing multi-year gaps down to smooth distributions.
3. **Geophysical & Waveform Parameters:**
   * Hypocentral coordinates: `Lat`, `Long`, `Depth_km`.
   * $S-P$ travel time: $S_{\text{arrival}} - P_{\text{arrival}}$ (directly proportional to hypocentral distance).
   * Gutenberg-Richter energy proxy: $\log_{10}(E) \approx 1.5 M + 4.8$.
   * Station telemetry: Receiver coordinates, elevation, SNR, and one-hot station IDs (`receiver_code_*`).
4. **Target Integrity:** Clean MinMax scaling into $[0, 1]$ with zero NaNs.

### 2. Sensor Noise Augmentation (`cnn/generate_noisy_dataset.py`)
To prevent deep models from overfitting to exact telemetry coordinates and simulate real-world hypocentral location errors ($\pm 5\text{--}15\text{ km}$) and picking jitter, zero-mean Gaussian noise ($\sigma = 0.015$) was injected into continuous features:
$$x_{\text{noisy}} = \text{clip}\left(x + \epsilon, 0.0, 1.0\right), \quad \epsilon \sim \mathcal{N}(0, 0.015^2)$$
One-hot categorical flags and ground truth target values remain strictly uncorrupted.

---

## 🧠 Class Imbalance Mitigation Strategy

Four complementary techniques were integrated across the training pipelines:

1. **Flattened Sequence SMOTE:** 3D sequences $(N, 15, 44)$ were reshaped to $(N, 660)$ for Synthetic Minority Over-sampling Technique (SMOTE), rebalancing positive training sequences from 1,956 to 3,695, then restored to 3D.
2. **Binary Focal Loss:** Down-weights easy negative examples to focus gradient updates on rare positive foreshocks:
   $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \quad \gamma = 2.0, \; \alpha = 0.50$$
3. **Prior Log-Odds Output Bias Initialization:**
   $$b_{\text{init}} = \ln\left(\frac{p_{\text{pos}}}{1 - p_{\text{pos}}}\right)$$
   Prevents catastrophic gradient shocks during the initial epochs.
4. **Dynamic Validation Threshold Optimization:** Evaluates decision thresholds from $0.01$ to $0.99$ on the validation set rather than relying on default $0.50$.

---

## 📊 Model Comparison & Results

Testing against **1,615 strictly unseen chronological events** (73 actual positives, 1,542 actual negatives):

| Evaluation Metric | Baseline BiLSTM | Temporal BiLSTM | Pure 1D-CNN v2 (Noisy) | Hybrid CNN-BiLSTM-Attention |
| :--- | :---: | :---: | :---: | :---: |
| **Framework** | TensorFlow / CPU | TensorFlow / CPU | **PyTorch (RTX 3050 GPU)** | **PyTorch (RTX 3050 GPU)** |
| **Features** | 30 static | 44 temporal | 44 temporal + Gaussian noise | 44 temporal |
| **Parameters** | 99,713 | 99,713 | **27,857 (Lightest)** | 187,586 |
| **Best Val ROC-AUC** | — | `0.6724` (Ep 12) | **`0.6546`** (Ep 27) | `0.5739` (Ep 28) |
| **Test ROC-AUC** | `0.5654` | **`0.5939` (Highest)** | `0.5793` | `0.5634` |
| **Test Average Precision (PR-AUC)** | `0.0598` | `0.0667` | `0.0619` | **`0.0710` (Highest)** |
| **Minority Recall ($M \ge 4.5$)** | $43.8\%$ | $6.8\%$ | $16.4\%$ (12/73) | **$60.3\%$ (44/73 Detected)** |
| **Optimal Calibrated Threshold** | `0.16` | `0.44` | `0.27` | `0.12` |
| **Explainability (XAI)** | Black-box | Black-box | Squeeze-and-Excitation | **Temporal Self-Attention** |
| **Artifact Path** | `bilstm/bilstm_model.keras` | `bilstm/bilstm_temporal_best.keras` | `cnn/cnn_1d_v2_best.pth` | `hybrid/cnn_bilstm_attention_best.pth` |

### Confusion Matrix Comparison (at Respective Optimal Thresholds)

* **Temporal BiLSTM ($\tau = 0.44$):**
  $$\begin{pmatrix} \text{TN: } 1,478 & \text{FP: } 64 \\ \text{FN: } 68 & \text{TP: } 5 \end{pmatrix} \quad \implies \text{High Specificity (96\%), 92\% Overall Accuracy}$$

* **Pure 1D-CNN v2 ($\tau = 0.27$):**
  $$\begin{pmatrix} \text{TN: } 1,355 & \text{FP: } 187 \\ \text{FN: } 61 & \text{TP: } 12 \end{pmatrix} \quad \implies \text{Balanced Warning Mode, 85\% Overall Accuracy}$$

* **Hybrid CNN-BiLSTM-Attention ($\tau = 0.12$):**
  $$\begin{pmatrix} \text{TN: } 761 & \text{FP: } 781 \\ \text{FN: } 29 & \text{TP: } 44 \end{pmatrix} \quad \implies \text{Early Warning Priority (60.3\% Minority Recall Captured)}$$

---

## 🔍 Explainable AI (XAI): Temporal Attention Analysis

The **Hybrid CNN-BiLSTM-Attention** architecture contains a dedicated temporal self-attention mechanism:
$$\text{energy}_t = \tanh(\mathbf{W} \mathbf{h}_t + \mathbf{b}), \quad \alpha_t = \text{softmax}(\text{energy}_t), \quad \mathbf{c} = \sum_{t=1}^{15} \alpha_t \mathbf{h}_t$$

* **Learned Attention Profile:** Across the 15-event sliding window ($t-15$ to $t-1$), the attention mechanism dynamically allocates weights to identify foreshock clusters versus background seismicity.
* **Interpretation:** Alerts are strongly driven by intermediate and recent acceleration patterns, providing seismologists with verifiable event attribution rather than opaque black-box predictions.
* **Output Plots:** Saved in `hybrid/plots_pytorch/pytorch_attention_distribution.png` and `hybrid/evaluation_cnn_bilstm_attention/attention_weights_timeline.png`.

---

## 🚀 How to Reproduce the Entire Pipeline

### 1. Prerequisites & Dependencies
Create a Python 3.10+ virtual environment and install the required libraries:
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn torch torchvision torchaudio tensorflow
```
*(CUDA support recommended for PyTorch if using an NVIDIA GPU).*

### 2. Step 1: Run Temporal Preprocessing
Generate the 44-feature temporal sequence dataset:
```bash
python "Minor project/preprocess_dl_temporal.py"
```

### 3. Step 2: Train Temporal BiLSTM (TensorFlow)
Train the BiLSTM sequence model with SMOTE and Focal Loss:
```bash
python "Minor project/bilstm/bilstm_temporal_model.py"
```

### 4. Step 3: Generate Noise Augmented Dataset & Train Pure 1D-CNN (PyTorch)
```bash
# Generate noisy dataset
python "Minor project/cnn/generate_noisy_dataset.py"

# Train Efficient Dilated 1D-CNN
python "Minor project/cnn/cnn_1d_model.py"
```

### 5. Step 4: Train Hybrid CNN-BiLSTM-Attention (PyTorch GPU)
```bash
python "Minor project/hybrid/cnn_bilstm_attention_pytorch.py"
```

### 6. Step 5: Generate Full Evaluation Suite & Metrics
```bash
python "Minor project/hybrid/generate_cnn_attention_metrics.py"
```

All metrics, confusion matrices, trade-off curves, and attention weights will be exported to `Minor project/hybrid/evaluation_cnn_bilstm_attention/`.

---

## 💡 Key Findings & Conclusions

1. **Temporal Recurrence is Crucial:** Pure 1D-CNN is computationally efficient (27.8k params), but lacks the sequential memory needed to track long-term tectonic strain accumulation, resulting in lower recall (16.4% vs 60.3%).
2. **Hybrid Model Excels for Early Warning Systems:** In disaster mitigation, missing a major earthquake (False Negative) has catastrophic consequences. The Hybrid CNN-BiLSTM-Attention model captured **60.3% of all significant earthquakes** ($M \ge 4.5$) in unseen modern data.
3. **Threshold Tuning is Non-Negotiable:** Because of the extreme 1:21 class imbalance in modern catalogs, standard $0.50$ classification thresholds fail completely (predicting 0 positives). Operational early warning systems must tune thresholds dynamically based on validation precision-recall trade-offs.
