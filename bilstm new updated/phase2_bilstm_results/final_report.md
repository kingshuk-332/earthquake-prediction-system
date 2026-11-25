# Phase 2: BiLSTM catalog magnitude regression

Completed on NVIDIA GeForce RTX 3050 Laptop GPU with corrected v2 dataset hash 97d357f71d6f83e3c7cbb5a163ca42b0ed40fbb2fc859c9e2076445c661c934f. No prior model weights, v1 arrays, waveform/CNN data or preprocessing reruns.

## Validation-selected configuration
Clean A only: 11 completed and 4 pruned TPE trials (15 total), seed 42, SQLite and persisted sampler RNG. Best tuning trial 8. Shared parameters: {"hidden_size": 128, "layers": 2, "dropout": 0.3144753602375439, "learning_rate": 0.0004549047669323766, "weight_decay": 0.00012155233771053111, "batch_size": 128}.
BiLSTM final forward/backward states are concatenated; dropout and a linear regression head produce one ML value. 538,881 trainable parameters. AdamW updates every weight and bias; Huber delta 1, clip 1, 50-epoch cap, patience 8 on validation MAE. Targets retain their saved identity transform.

## Controlled validation comparison
All four fits use fresh identical seeded initial weights and selected settings. A clean; B coordinate Gaussian noise (sigma 0.01); C approved regression-relevance complete-window sampling; D weighting plus identical noise. C/D share epoch draws. No interpolation, SMOTE or full SMOGN implementation.

| Variant | MAE | RMSE | R² | Best/run epochs | ML>=4.5 MAE | High underprediction | Val minus clean-train MAE | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.4783 | 0.6356 | -0.0200 | 14/22 | 1.5188 | 100.0% | +0.0670 | 11.8 |
| B | 0.4772 | 0.6350 | -0.0182 | 14/22 | 1.5221 | 100.0% | +0.0710 | 11.9 |
| C | 0.4724 | 0.6242 | 0.0162 | 12/20 | 1.4214 | 100.0% | +0.0415 | 10.8 |
| D | 0.4727 | 0.6245 | 0.0154 | 12/20 | 1.4239 | 100.0% | +0.0441 | 10.8 |

Winner **C** was locked using validation MAE before test-label values were accessed. High-magnitude metrics are tradeoff diagnostics, not an alternative post-hoc selection criterion. Per-epoch unique parents, repeated draws, effective high-magnitude proportions and losses are in variants/*/history.json.

Late-training overfitting pattern (validation MAE rising >0.01 while clean-training MAE falls after its best epoch): C, D. For winner C, post-best validation MAE change is +0.0118; clean-training MAE change -0.0175. Restoring the best checkpoint limits observed late-epoch degradation. A gap alone cannot distinguish overfitting from differences between chronological populations.

## Winner-only test and baselines
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| BiLSTM_C | 0.5477 | 0.6696 | -0.0899 |
| training_mean | 1.0032 | 1.1319 | -2.1142 |
| training_median | 0.9747 | 1.1042 | -1.9635 |
| recent_event_magnitude | 0.6739 | 0.8664 | -0.8245 |

Test ML>=4.5 count 38: MAE 1.5218, signed bias -1.5218, underprediction 100.0%. Full magnitude-range errors and residual quantiles are saved. Mean/median baselines use training labels only; recent-event baseline uses the final historical input magnitude with the saved inverse scaler.

Only the selected neural model was evaluated on test. Predictions and selection lock are cached; reruns verify/reuse them and do not retune. Metrics were independently checked with scikit-learn. Source and augmentation hashes remain unchanged.

## Compute, reproducibility and limits
Tuning fit compute: 2.38 minutes; four controlled fits: 0.75 minutes; total 3.14 minutes. Times include epoch preprocessing, training, validation and best-checkpoint saving; exclude notebook setup and some resume-checkpoint I/O. CUDA deterministic algorithms and TF32-off settings are recorded. All best checkpoints reload exactly on validation; clean refit reproduces the selected tuning trial. Same-seed comparison is not evidence of robustness across random seeds. Selection on a reused validation set can be optimistic.

This is next-retained-ML-catalog-event regression on retrospective historical context, not operational EEW or exact earthquake prediction. Reporting/revision availability and magnitude calibration remain UNKNOWN. Earlier observed targets can become later historical inputs; target splits are disjoint, but overlapping histories are not independent observations. Chronological population differences and source completeness may affect generalization; this experiment does not establish their causes.

## Research scope
[Graves & Schmidhuber (2005)](https://www.cs.toronto.edu/~graves/nn_2005.pdf), Sections II–III: bidirectional finite-segment processing and full-gradient BPTT motivate the architecture. PyTorch cells/head are implementation choices, not an exact reproduction.
[Chen & Fang (2025)](https://doi.org/10.1061/NHREFO.NHENG-2244): catalog sliding-window background from the accessible abstract; full methodology unavailable.
[Al Banna et al. (2021), including Andersson](https://doi.org/10.1109/ACCESS.2021.3071400): catalog seismicity-modeling background; different target/protocol, no exact reproduction.
[SMOTE (2002)](https://doi.org/10.1613/jair.953) and [SMOGN (2017)](https://proceedings.mlr.press/v74/branco17a.html): imbalance motivation only. Approved weighted-window/noise consumers are alternatives, not either algorithm.

## Artifacts
Notebook: ../phase2_bilstm_training.ipynb. Checkpoints/history: variants/A–D/best.pt and history.json. Hyperparameters: best_hyperparameters.json. Study: optuna_study.sqlite3; sampler state: optuna_sampler.pkl. Validation: validation_comparison.csv. Selection: selection_lock.json. Test: selected_test_metrics.json, selected_test_predictions.npz, test_baseline_comparison.csv, test_error_by_magnitude.csv. Figures: plots/. Verification: verification.json.
