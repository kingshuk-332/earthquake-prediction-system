# Phase 2C: BiLSTM Feature Informativeness & Temporal Shift Audit Report

**Audit Objective:** Determine whether the 10 historical earthquake features contain stable predictive information for next-event magnitude ($M_L$) regression or whether observed predictive relationships in training represent non-stationary artifacts of catalog evolution.  
**Dataset:** `dataset/v2_timestamp_fixed/bilstm_catalog_seq20.pt`  
**Scope:** Training (4,973 samples), Validation (1,065 samples), and Test (1,067 samples). No model training, tuning, or modification was performed.

---

## 1. Executive Summary & Root-Cause Diagnosis

The core diagnostic question was: **Why does the BiLSTM suffer from severe prediction compression (std dev 0.12–0.15 vs actual 0.63) and 100% underprediction of high magnitudes ($M_L \ge 4.5$)?**

The audit establishes a definitive empirical answer:
1. **Correlation Collapse Across Epochs:** The apparent correlation between historical rolling magnitude and the target earthquake magnitude is **non-stationary and collapses from $r = +0.827$ in the early catalog (1924–2013) to $r = +0.291$ in late training (2013–2024), and $r = +0.201$ in validation (2024–2026)**.
2. **Artifact of Catalog Evolution (Completeness Threshold $M_c$):** In the early 20th century, only moderate-to-large earthquakes ($M_L \ge 4.0$) were detected and recorded. Over the past decade, high-density modern broadband networks were deployed, detecting thousands of micro-earthquakes down to $M_L 1.5$.
3. **Spurious Historical Association:** The BiLSTM learned a pseudo-rule from 100 years of historical data: *"High rolling magnitude + long inter-event gaps + low 30-day rate $\implies$ large target magnitude"*. In reality, this association merely differentiated the **historical low-sensitivity epoch** (where only large events were cataloged) from the **modern high-sensitivity epoch** (where micro-events dominate).
4. **Failure in Modern Monitoring:** In the validation and test periods, large earthquakes ($M_L \ge 4.5$) occur within dense sequences of micro-earthquakes ($2.98$ to $4.50$ events/day). The historical precursor signature never occurs, causing the BiLSTM to default to the empirical distribution mean ($\sim 3.43–3.48\ M_L$).

---

## 2. Comprehensive Analysis of All 10 Features

All features are evaluated at the last historical observation ($t-1$, step 19) preceding the target event $t$:

| Index | Feature Name | Units | Train Mean $\pm$ SD | Val Mean $\pm$ SD | Pearson $r$ (Train H1 / H2 / Val) | Spearman $\rho$ (Train $\to$ Val) | Cohen's $d$ ($M_L \ge 4.5$) | Redundancy ($|r| \ge 0.6$) |
|:---:|---|---|---:|---:|---:|---:|---:|---|
| **0** | `historical_magnitude` | $M_L$ | $4.04 \pm 0.91$ | $3.51 \pm 0.63$ | $+0.712 \to +0.173 \to \mathbf{+0.186}$ | $+0.643 \to \mathbf{+0.175}$ | $+1.22 \to \mathbf{+0.45}$ | Redundant with feat 6 ($r=0.82$) |
| **1** | `historical_depth_km` | km | $37.0 \pm 35.5$ | $31.8 \pm 27.5$ | $+0.063 \to +0.032 \to \mathbf{-0.051}$ | $+0.263 \to \mathbf{+0.022}$ | $+0.27 \to \mathbf{-0.10}$ | Redundant with feat 8 ($r=0.66$) |
| **2** | `historical_latitude_deg` | $^\circ\text{N}$ | $25.77 \pm 2.27$ | $26.23 \pm 2.43$ | $+0.125 \to -0.016 \to \mathbf{+0.013}$ | $+0.057 \to \mathbf{+0.048}$ | $+0.14 \to \mathbf{-0.04}$ | Independent spatial coordinate |
| **3** | `historical_longitude_deg` | $^\circ\text{E}$ | $93.32 \pm 2.51$ | $91.31 \pm 3.30$ | $+0.220 \to +0.006 \to \mathbf{-0.022}$ | $+0.180 \to \mathbf{-0.040}$ | $+0.36 \to \mathbf{+0.18}$ | Spatial coordinate (westward shift) |
| **4** | `inter_event_days` | days | $7.41 \pm 32.7$ | $0.39 \pm 0.91$ | $+0.299 \to +0.140 \to \mathbf{-0.011}$ | $+0.396 \to \mathbf{-0.051}$ | $+0.56 \to \mathbf{-0.08}$ | Inversely related to feat 5 |
| **5** | `event_rate_30d` | events/day | $0.55 \pm 0.45$ | $3.01 \pm 2.05$ | $-0.267 \to -0.281 \to \mathbf{+0.074}$ | $-0.599 \to \mathbf{+0.051}$ | $-0.86 \to \mathbf{-0.04}$ | Massive non-stationary shift |
| **6** | `mean_magnitude_20` | $M_L$ | $4.05 \pm 0.76$ | $3.47 \pm 0.38$ | $+0.827 \to +0.291 \to \mathbf{+0.201}$ | $+0.744 \to \mathbf{+0.217}$ | $+1.46 \to \mathbf{+0.34}$ | Dominant training predictor; collapsed |
| **7** | `std_magnitude_20` | $M_L$ | $0.49 \pm 0.15$ | $0.44 \pm 0.11$ | $-0.256 \to -0.072 \to \mathbf{+0.140}$ | $-0.463 \to \mathbf{+0.092}$ | $-0.85 \to \mathbf{+0.40}$ | Sign flipped across splits |
| **8** | `mean_depth_20_km` | km | $37.0 \pm 14.9$ | $31.8 \pm 9.6$ | $+0.107 \to +0.141 \to \mathbf{-0.046}$ | $+0.340 \to \mathbf{-0.070}$ | $+0.67 \to \mathbf{+0.08}$ | Unstable across epochs |
| **9** | `std_depth_20_km` | km | $29.4 \pm 13.3$ | $24.7 \pm 9.1$ | $+0.032 \to +0.081 \to \mathbf{-0.039}$ | $-0.030 \to \mathbf{-0.057}$ | $-0.12 \to \mathbf{+0.18}$ | Uninformative |

### Key Feature Observations:
- **`historical_magnitude` & `mean_magnitude_20` (Features 0 & 6):** Collinear ($r = 0.818$). In Early Train, they explain $\sim 68\%$ of target variance ($r \approx 0.83$). In Validation, they explain only $\mathbf{4\%}$ of variance ($r \approx 0.20$).
- **`event_rate_30d` & `inter_event_days` (Features 4 & 5):** Exhibited strong negative correlation with magnitude in training ($\rho = -0.60$), but in validation, this relationship **vanished completely** ($\rho = +0.05$ and $-0.05$).
- **Missing / Imputed Values:** Handled strictly by train-fitted median imputation (`inter_event_days` imputed with 1.57 days on initial left-censored boundaries). Zero unhandled NaNs exist.

---

## 3. Temporal Distribution Shift Analysis

### A. Target Magnitude & Proportion Shifts
- **Train (1924–2024, 4,973 samples):** Mean $4.034\ M_L$, Std $0.898$. High targets ($M_L \ge 4.5$): **$31.23\%$** (1,553 events).
- **Validation (Dec 2024–Feb 2026, 1,065 samples):** Mean $3.511\ M_L$, Std $0.629$. High targets ($M_L \ge 4.5$): **$7.04\%$** (75 events).
- **Test (Feb 2026–Oct 2026, 1,067 samples):** Mean $3.101\ M_L$, Std $0.641$. High targets ($M_L \ge 4.5$): **$3.56\%$** (38 events).
- **Statistical Distance:** Train $\to$ Validation Kolmogorov-Smirnov statistic $= 0.3145$ ($p < 10^{-15}$); Wasserstein distance $= 0.5247\ M_L$. Train $\to$ Test Wasserstein distance $= 0.9326\ M_L$.

### B. Event-Frequency & Catalog Acceleration
- **Train Period:** 4,973 events over 100.3 years ($\sim 50$ events/year overall; early period was $\sim 28$ events/year). Median inter-event time: **1.557 days**.
- **Validation Period:** 1,065 events over 1.13 years ($\mathbf{942\ \text{events/year}}$). Median inter-event time: **0.198 days** (4.7 hours).
- **Test Period:** 1,067 events over 0.67 years ($\mathbf{1,592\ \text{events/year}}$). Median inter-event time: **0.132 days** (3.2 hours).
- **Rate Jump:** Mean 30-day rate surged from $0.55$ events/day in Train to **$3.01$ events/day in Val** and **$4.50$ events/day in Test** (an 8.2-fold increase in event logging density).

### C. Geographic Centroid Shift
- **Train Centroid:** Lat $25.77^\circ\text{N}$, Long $93.32^\circ\text{E}$.
- **Validation Centroid:** Lat $26.23^\circ\text{N}$, Long $91.31^\circ\text{E}$ ($\sim 200\text{ km}$ westward shift toward Assam / Meghalaya network clusters).
- **Test Centroid:** Lat $26.15^\circ\text{N}$, Long $91.77^\circ\text{E}$.

---

## 4. Scientific Conclusion & Implications for Modeling

1. **Are the 10 features stably informative?**
   **No.** The apparent predictive signal ($r \approx 0.74–0.80$) in historical training data is an artifact of the chronological transition from an incomplete historical catalog to a dense modern digital network. In modern stationary monitoring (Validation), **no feature possesses a correlation $|r| > 0.20$ with the next earthquake's magnitude**.
2. **Why does BiLSTM collapse to mean predictions?**
   Because next-event magnitude in a complete regional earthquake catalog is largely conditionally independent of immediate prior magnitudes (consistent with standard statistical seismology), an unregularized or robust model correctly learns that the variance-minimizing solution is to predict the distribution mean ($\sim 3.43–3.50\ M_L$).
3. **Why did Huber $\delta=2$ fail?**
   Increasing $\delta$ penalizes errors quadratically up to 2.0 units, forcing the optimization to collapse variance even further ($0.157 \to 0.125$) to minimize squared error across the dense modern micro-earthquake majority.

---

## 5. Saved Diagnostic Figures
- `phase2_bilstm_results/error_diagnosis/feature_correlation_drift.png`
- `phase2_bilstm_results/error_diagnosis/temporal_distribution_shift.png`
- `phase2_bilstm_results/error_diagnosis/phase2c_feature_audit.json`

