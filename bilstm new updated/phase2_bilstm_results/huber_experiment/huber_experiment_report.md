# Phase 2B: Controlled Huber Loss Experiment Report (δ=1 vs δ=2)

**Evaluation Protocol:** Validation set only (1,065 samples). Test set was strictly unaccessed.  
**Dataset:** `dataset/v2_timestamp_fixed/bilstm_catalog_seq20.pt`  
**Architecture & Optimization:** 2-layer BiLSTM (hidden 128, dropout 0.3145, 538,881 parameters), AdamW, Variant C weighted sampling, seed 42, early stopping patience 8.

---

## 1. Unit Verification
- Target transform verified: `target_definition["target_transform"] = {"name": "identity", "offset": 0.0, "scale": 1.0}`.
- Target magnitude $M_L$ retains its raw, unscaled Richter scale units (train mean 4.034, validation mean 3.511).
- Therefore, transition threshold $\delta$ is measured in **original magnitude units ($M_L$)**:
  - Baseline model: Huber $\delta = 1.0\ M_L$
  - New experiment: Huber $\delta = 2.0\ M_L$

---

## 2. Validation Metrics Comparison

| Metric | Baseline ($\delta=1.0$) | Experiment ($\delta=2.0$) | Difference ($\Delta$) | Outcome |
|---|---:|---:|---:|---|
| **Validation MAE** | 0.4724 | **0.4698** | -0.0026 | Marginal gain (-0.55%) |
| **Validation RMSE** | 0.6242 | 0.6243 | +0.0002 | Unchanged |
| **Validation $R^2$** | 0.0162 | 0.0158 | -0.0005 | Unchanged |
| **Validation Huber Loss** | 0.1754 ($\delta=1$) | 0.2312 ($\delta=2$) | - | Objective difference |
| **$M_L \ge 4.5$ MAE** | 1.4214 | 1.4658 | +0.0443 | **Worse (+3.1%)** |
| **$M_L \ge 4.5$ Underprediction Rate** | 100.0% (75/75) | 100.0% (75/75) | 0.0% | Unchanged (all 75 underpredicted) |
| **$M_L \ge 4.5$ Signed Bias** | -1.4214 | -1.4658 | -0.0443 | More negative bias |
| **Prediction Mean** | 3.4853 | 3.4336 | -0.0517 | Shifts closer to val center (3.511) |
| **Prediction Std Dev** | 0.1573 | **0.1248** | -0.0325 | **More compressed (-20.7%)** |
| **Prediction Range** | [3.180, 3.910] | [3.136, 3.707] | Span: 0.571 vs 0.730 | **Tighter band (-21.8% span)** |
| **Clean Train MAE** | 0.4309 | 0.4425 | +0.0116 | Slightly higher |
| **Generalization Gap (Val - Train MAE)** | +0.0415 | +0.0274 | -0.0141 | Smaller gap |
| **Best / Total Epochs** | 12 / 20 | 11 / 19 | -1 epoch | Stopped at epoch 19 |
| **Training Compute** | 10.8 s | 10.3 s | - | Identical efficiency |

---

## 3. Analysis & Research Findings

1. **Validation Performance:** Overall validation MAE slightly improved from 0.4724 to 0.4698 (-0.0026), but RMSE (0.6242 vs 0.6243) and $R^2$ (0.0162 vs 0.0158) remained flat.
2. **Prediction Variability & Tail Performance:** Increasing $\delta$ to 2.0 **did not improve magnitude prediction variability; it made compression worse**.
   - Prediction standard deviation dropped by 20.7% (0.1573 $\to$ 0.1248), covering only 19.8% of actual validation variance ($\sigma_{actual} = 0.6293$).
   - Prediction range narrowed from $[3.180, 3.910]$ to $[3.136, 3.707]$.
   - For $M_L \ge 4.5$, MAE worsened from 1.4214 to 1.4658 and signed bias became more negative (-1.4658). Underprediction remained 100.0% (75/75).
3. **Mechanism:** Expanding the quadratic region from $|\text{error}| \le 1.0$ to $|\text{error}| \le 2.0$ increases penalization for moderate errors ($1.0 < |\text{error}| \le 2.0$). Because the vast majority of validation events are concentrated between $M_L \in [2.5, 4.0]$, the model minimizes overall quadratic loss by collapsing even more tightly toward the empirical center of mass (~3.43 $M_L$), further shrinking upper-tail predictions.

---

## 4. Artifact Inventory

- Notebook: `phase2b_huber_loss_experiment.ipynb`
- Checkpoints: `best.pt`, `last.pt`
- Training history: `history.json`, `fit_summary.json`
- Validation predictions: `validation_predictions.npz`
- Metrics: `huber_delta2_metrics.json`, `huber_loss_comparison.csv`
- Figures: `plots/training_validation_curves.png`, `plots/prediction_distribution_comparison.png`

