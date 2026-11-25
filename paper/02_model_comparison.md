# Model Comparison & Evaluation Analysis

**Document:** `paper/02_model_comparison.md`  
**Source Data:** `final_comparison/four_model_eew_v1/final_metrics_table.csv`  
**Evaluation Cohort:** Identical common cohort of 1,200 eligible earthquakes and 1,214 noise records (2,414 records total; SHA256: `862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601`).

---

## 1. Master Evaluation Table

*Note: Best value in each horizon is shown in bold. Regression metrics are computed over all 1,200 eligible earthquakes independently of detection outcome.*

### Table 1: Comprehensive Model Comparison Across Horizons and Threshold Policies
<!-- source: final_comparison/four_model_eew_v1/final_metrics_table.csv -->

| Horizon | Model | Threshold Policy | Threshold | Accuracy | Precision | Recall | F1-Score | FPR | ROC-AUC | PR-AUC (AP) | MAE | RMSE | $R^2$ | Mean Bias | SD Ratio | $M \ge 5$ Underpred. % | $M \ge 5$ Bias |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0 s** | CNN | Fixed 0.5 | 0.50 | 0.661143 | 0.630643 | 0.768333 | 0.692712 | 0.444811 | 0.748724 | 0.747258 | 0.665020 | 0.821894 | 0.164370 | -0.101660 | 0.411175 | 97.71% | -1.381391 |
| 1.0 s | BiLSTM | Fixed 0.5 | 0.50 | 0.628418 | 0.572453 | **0.997500** | 0.727438 | 0.736409 | 0.675916 | 0.651856 | 0.670213 | 0.824230 | 0.159615 | -0.076714 | 0.342184 | 97.71% | -1.409562 |
| 1.0 s | BiLSTM | Val. Selected | 0.47 | 0.627175 | 0.571497 | 0.999167 | 0.727107 | 0.740527 | 0.675916 | 0.651856 | 0.670213 | 0.824230 | 0.159615 | -0.076714 | 0.342184 | 97.71% | -1.409562 |
| 1.0 s | Fusion | Fixed 0.5 | 0.50 | 0.652858 | 0.627645 | 0.741667 | 0.679908 | 0.434926 | 0.746584 | 0.744865 | **0.664240** | **0.817358** | **0.173570** | -0.074875 | 0.395727 | 98.47% | -1.364506 |
| 1.0 s | Fusion | Val. Selected | 0.42 | 0.632560 | 0.576453 | 0.983333 | 0.726825 | 0.714168 | 0.746584 | 0.744865 | **0.664240** | **0.817358** | **0.173570** | -0.074875 | 0.395727 | 98.47% | -1.364506 |
| 1.0 s | Attention | Fixed 0.5 | 0.50 | **0.663215** | **0.651290** | 0.694167 | 0.672045 | **0.367381** | **0.751284** | **0.748852** | 0.666359 | 0.818941 | 0.170365 | -0.076292 | 0.387569 | 98.47% | -1.373537 |
| 1.0 s | Attention | Val. Selected | 0.28 | 0.626761 | 0.571224 | 0.999167 | 0.726887 | 0.741351 | **0.751284** | **0.748852** | 0.666359 | 0.818941 | 0.170365 | -0.076292 | 0.387569 | 98.47% | -1.373537 |
| **2.0 s** | CNN | Fixed 0.5 | 0.50 | 0.724524 | 0.740774 | **0.685833** | 0.712246 | 0.237232 | 0.825527 | 0.845501 | 0.635296 | 0.783164 | 0.241270 | -0.065797 | 0.544796 | 94.66% | -1.162961 |
| 2.0 s | BiLSTM | Fixed 0.5 | 0.50 | 0.673571 | 0.700000 | 0.600833 | 0.646637 | 0.254530 | 0.763173 | 0.777534 | **0.634122** | **0.780973** | **0.245509** | -0.046810 | 0.487214 | 93.13% | -1.202630 |
| 2.0 s | BiLSTM | Val. Selected | 0.36 | 0.627589 | 0.571769 | 0.999167 | 0.727328 | 0.739703 | 0.763173 | 0.777534 | **0.634122** | **0.780973** | **0.245509** | -0.046810 | 0.487214 | 93.13% | -1.202630 |
| 2.0 s | Fusion | Fixed 0.5 | 0.50 | 0.734880 | 0.765152 | 0.673333 | 0.716312 | 0.204283 | 0.831996 | 0.849300 | 0.635879 | 0.783324 | 0.240961 | -0.061756 | 0.536968 | 94.66% | -1.171738 |
| 2.0 s | Fusion | Val. Selected | 0.41 | 0.721624 | 0.691860 | 0.793333 | 0.739130 | 0.349259 | 0.831996 | 0.849300 | 0.635879 | 0.783324 | 0.240961 | -0.061756 | 0.536968 | 94.66% | -1.171738 |
| 2.0 s | Attention | Fixed 0.5 | 0.50 | **0.735708** | **0.766098** | 0.674167 | **0.717199** | **0.203460** | **0.832168** | **0.849352** | 0.636489 | 0.782980 | 0.241627 | -0.056138 | 0.512419 | 95.42% | -1.182187 |
| 2.0 s | Attention | Val. Selected | 0.41 | 0.719138 | 0.689130 | 0.792500 | 0.737209 | 0.353377 | **0.832168** | **0.849352** | 0.636489 | 0.782980 | 0.241627 | -0.056138 | 0.512419 | 95.42% | -1.182187 |
| **3.0 s** | CNN | Fixed 0.5 | 0.50 | 0.766777 | 0.783615 | **0.733333** | **0.757641** | 0.200165 | 0.858474 | 0.882498 | 0.613005 | 0.757702 | 0.289803 | -0.090770 | 0.513718 | 96.95% | -1.164218 |
| 3.0 s | BiLSTM | Fixed 0.5 | 0.50 | 0.714167 | 0.740566 | 0.654167 | 0.694690 | 0.226524 | 0.813242 | 0.834604 | 0.615082 | 0.758421 | 0.288455 | -0.082112 | 0.516038 | 96.18% | -1.134616 |
| 3.0 s | BiLSTM | Val. Selected | 0.36 | 0.648302 | 0.589404 | 0.964167 | 0.731584 | 0.663921 | 0.813242 | 0.834604 | 0.615082 | 0.758421 | 0.288455 | -0.082112 | 0.516038 | 96.18% | -1.134616 |
| 3.0 s | Fusion | Fixed 0.5 | 0.50 | 0.772162 | 0.832311 | 0.678333 | 0.747475 | 0.135091 | 0.863415 | 0.886098 | 0.608215 | 0.753055 | 0.298488 | -0.061756 | 0.569917 | 92.37% | -1.047271 |
| 3.0 s | Fusion | Val. Selected | 0.36 | 0.751036 | 0.721033 | 0.814167 | 0.764775 | 0.311367 | 0.863415 | 0.886098 | 0.608215 | 0.753055 | 0.298488 | -0.061756 | 0.569917 | 92.37% | -1.047271 |
| 3.0 s | Attention | Fixed 0.5 | 0.50 | **0.776305** | **0.851812** | 0.665833 | 0.747428 | **0.114498** | **0.865393** | **0.887532** | **0.606601** | **0.748621** | **0.306725** | **-0.004881** | **0.600623** | **87.02%** | **-0.930820** |
| 3.0 s | Attention | Val. Selected | 0.41 | 0.761806 | 0.771503 | 0.740000 | 0.755423 | 0.216639 | **0.865393** | **0.887532** | **0.606601** | **0.748621** | **0.306725** | **-0.004881** | **0.600623** | **87.02%** | **-0.930820** |

---

## 2. Evaluation Metric Mathematical Formulations

Given binary confusion counts (True Positives $TP$, False Positives $FP$, True Negatives $TN$, False Negatives $FN$):

### 2.1 Detection Metrics
1. **Accuracy:**
   \begin{equation}
   \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
   \label{eq:accuracy}
   \end{equation}
2. **Precision (Positive Predictive Value):**
   \begin{equation}
   \text{Precision} = \frac{TP}{TP + FP}
   \label{eq:precision}
   \end{equation}
3. **Recall (True Positive Rate, Sensitivity):**
   \begin{equation}
   \text{Recall} = \frac{TP}{TP + FN}
   \label{eq:recall}
   \end{equation}
4. **False Positive Rate (FPR, Fall-out):**
   \begin{equation}
   \text{FPR} = \frac{FP}{FP + TN}
   \label{eq:fpr}
   \end{equation}
5. **$F_1$-Score (Harmonic Mean):**
   \begin{equation}
   F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}
   \label{eq:f1}
   \end{equation}
6. **Receiver Operating Characteristic Area Under Curve (ROC-AUC):**
   \begin{equation}
   \text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(u)) \, du
   \label{eq:roc_auc}
   \end{equation}
7. **Precision-Recall Area Under Curve (PR-AUC / Average Precision AP):**
   \begin{equation}
   \text{PR-AUC} = \sum_{k=1}^{K} (R_k - R_{k-1}) P_k
   \label{eq:pr_auc}
   \end{equation}

### 2.2 Magnitude Estimation Metrics
For eligible earthquakes $i \in \{1, \dots, N_{\text{EQ}}\}$ ($N_{\text{EQ}} = 1,200$) with true magnitude $M_i$ and predicted magnitude $\hat{M}_i$:

8. **Mean Absolute Error (MAE):**
   \begin{equation}
   \text{MAE} = \frac{1}{N_{\text{EQ}}} \sum_{i=1}^{N_{\text{EQ}}} |\hat{M}_i - M_i|
   \label{eq:mae}
   \end{equation}
9. **Root Mean Squared Error (RMSE):**
   \begin{equation}
   \text{RMSE} = \sqrt{ \frac{1}{N_{\text{EQ}}} \sum_{i=1}^{N_{\text{EQ}}} (\hat{M}_i - M_i)^2 }
   \label{eq:rmse}
   \end{equation}
10. **Coefficient of Determination ($R^2$):**
    \begin{equation}
    R^2 = 1 - \frac{\sum_{i=1}^{N_{\text{EQ}}} (M_i - \hat{M}_i)^2}{\sum_{i=1}^{N_{\text{EQ}}} (M_i - \bar{M})^2}, \quad \bar{M} = \frac{1}{N_{\text{EQ}}}\sum_{i=1}^{N_{\text{EQ}}} M_i
    \label{eq:r2}
    \end{equation}
11. **Prediction Bias (Signed Mean Error):**
    \begin{equation}
    \text{Bias} = \frac{1}{N_{\text{EQ}}} \sum_{i=1}^{N_{\text{EQ}}} (\hat{M}_i - M_i)
    \label{eq:bias}
    \end{equation}
12. **Standard Deviation Ratio:**
    \begin{equation}
    \text{SD Ratio} = \frac{\sigma_{\hat{M}}}{\sigma_M} = \frac{\sqrt{\frac{1}{N}\sum (\hat{M}_i - \bar{\hat{M}})^2}}{\sqrt{\frac{1}{N}\sum (M_i - \bar{M})^2}}
    \label{eq:sd_ratio}
    \end{equation}

---

## 3. Analysis: Why the Best Model Wins and Where the Others Fail

### 3.1 Detection Trade-Offs: Precision vs. Recall
- **At Fixed 0.5 Threshold:**
  - **Standalone 1D-CNN** achieves the highest recall ($0.733333$) and highest $F_1$-score ($0.757641$) at $3.0\text{ s}$. The raw convolutional kernels directly resolve high-frequency onset wave packets from continuous samples. However, CNN incurs a high False Positive Rate ($20.02\%$), generating $243$ false alarms out of $1,214$ non-seismic noise traces.
  - **Attention Fusion** achieves the highest precision ($0.851812$), highest overall accuracy ($0.776305$), highest ROC-AUC ($0.865393$), and PR-AUC ($0.887532$). Crucially, Attention suppresses false alarms by almost half, cutting FPR down to $11.45\%$ ($139$ false alarms vs. $243$ for CNN and $275$ for BiLSTM). By attending jointly to both raw convolutional features and temporal tabular statistics, Attention correctly identifies non-seismic transients that trick individual backbones.
- **Under Validation-Selected Operating Points:**
  - When thresholds are lowered to maximize validation $F_1$ or recall, **Standalone BiLSTM degrades precipitously**: its FPR increases to $66.39\%$ at $3\text{ s}$ and $74.05\%$ at $1\text{ s}$. Because the seven summary statistics discard continuous phase and waveform curvature, BiLSTM alone cannot separate non-stationary ambient noise from weak P-wave onsets.

### 3.2 Magnitude Regression: Variance Compression and Strong-Event Underprediction
- **Attention Dominance:** At $3.0\text{ s}$, Attention achieves the lowest MAE ($0.606601$), lowest RMSE ($0.748621$), highest $R^2$ ($0.306725$), and virtually zero aggregate bias ($-0.004881$). It maintains the highest variance preservation ($\text{SD Ratio} = 0.6006$ vs. $0.5137$ for CNN).
- **Universal Magnitude Compression Failure:** Despite Attention's relative victory, **all four models suffer from severe prediction compression**. Regression predictions cluster tightly around the catalog mean ($\approx 3.6\text{--}3.8$).
  - For destructive earthquakes ($M \ge 5.0$, $N = 131$), all models underpredict the majority of events:
    - CNN: $96.95\%$ underpredicted, bias = $-1.164$ magnitude units.
    - BiLSTM: $96.18\%$ underpredicted, bias = $-1.135$ magnitude units.
    - Fusion: $92.37\%$ underpredicted, bias = $-1.047$ magnitude units.
    - Attention: $87.02\%$ underpredicted, bias = $-0.931$ magnitude units.
  - This systematic underprediction of strong events is a fundamental physical limitation of short P-wave windows: within $1\text{--}3\text{ s}$ post-P, the earthquake rupture process has not completed for large fault dimensions, making true final magnitude physically ambiguous from initial P amplitudes alone.

### 3.3 Statistical Significance: Paired Bootstrap Analysis
While Attention shows nominal improvements over Fusion v1, paired stratified bootstrap analysis (500 replicates, seed 2026) reveals that the differences are not statistically distinguishable from zero at the 95% confidence level:
- $F_1$ Difference (Attention $-$ Fusion): $-0.000047$, $95\%\text{ CI} = [-0.0060, +0.0065]$.
- MAE Difference (Attention $-$ Fusion): $-0.001614$, $95\%\text{ CI} = [-0.0064, +0.0032]$.
Both confidence intervals span zero, establishing that Attention provides architectural refinement and noise suppression rather than a statistically significant breakthrough over simple feature concatenation.
