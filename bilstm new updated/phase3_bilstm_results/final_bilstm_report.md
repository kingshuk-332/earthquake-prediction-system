# Final CSV-based BiLSTM research report — Phase 3

**Verdict C: Neither established as adequately generalizing; preserve original and modern as research baselines.** Selected by validation only: **clean**. No further tuning or original-test evaluation was performed.

## Design and provenance

Original dataset_2.csv hash matches verified provenance; retained ML eligibility/IDs and Magnitude_Value targets are unchanged. No old tensors, fitted scalers or weights were used to construct/train the modern experiment. The 10 historical feature definitions and sequence length20 remain fixed. Rolling features were recomputed in explicit time units. Train-only median imputation/standardization fits unique training-input events. Original artifacts remain unchanged.

The fixed newest-eligible training cutoff is2020, chosen only from pre-2023 targets using prespecified sample/representation and two-half mean/SD checks. It yields **692 training sequences,39 ML>=4.5**. These are exploratory sufficiency thresholds, not proof of stationarity or statistical power. Candidate details and all10 two-half feature correlations are saved in feasibility.json and training_feature_stability.csv; no validation/holdout performance selected the cutoff.

Validation is calendar2023 (**346 samples**); holdout is2024 through 2024-12-20T09:43:29+00:00 (**366 samples**). Boundary samples were purged until all inputs, targets and rolling-feature dependency events stayed within their partition. Event/dependency sets are disjoint across splits; every input precedes its target. Purged counts differ from unpurged calendar counts.

**Holdout limitation:** this newly frozen partition previously belonged to the original model's training catalog. It is an internal retrospective experiment, not newly collected or historically unseen external data. Fixed architecture/hyperparameters also inherit prior research choices. The original validation/test populations are excluded, and the old model is not evaluated on its own former training events. Old test MAE0.5477/R2-0.0899 and these new holdout scores are not directly comparable proof of improvement.

## Completeness and feature stability

See [completeness_report.md](completeness_report.md) for all periods, bin widths0.1/0.2/0.3,200 block-bootstrap intervals, and GFT failures. Modern2020–2022 estimates:
| Bin width | MAXC | MAXC95% interval | GFT | GFT95% interval | GFT success |
|---|---|---|---|---|---|
| 0.1 | 3.4000000000000004 | [2.90, 3.80] | 3.8000000000000003 | [3.40, 4.00] | 1.0 |
| 0.2 | 3.8000000000000003 | [3.20, 3.80] | 3.6 | [3.60, 3.83] | 0.98 |
| 0.3 | 3.5999999999999996 | [3.30, 3.90] | 3.9 | [3.60, 4.20] | 1.0 |

These are conditional FMD estimates in a geographically pooled, retained-ML catalog, not established physical detection thresholds. Reporting completeness, source magnitude calibration and current2026 Mc remain unverified. No below-Mc labels were removed, no declustering was applied, and no causal completeness explanation is claimed. Training half means for the selected cutoff differ by 0.068208 ML; univariate relationships remain descriptive and do not certify forecasting information.

## Fixed comparison and validation

Both variants use fresh identical initialization: hidden128,2 layers, dropout 0.3144753602375439, AdamW learning rate 0.0004549047669323766, weight decay 0.00012155233771053111, batch128, Huber delta1, seed42, max50 epochs, patience8 and clipping1.0. Weighted training draws complete windows with replacement using weights1+0.5*clip((ML-4)/1.5,0,1), then sorts parent indices. Labels are unchanged. No noise or interpolation was added. Full runtime, source hashes, sampler weights, per-epoch parent hashes and RNG state are saved for reproducibility.

| Validation model | MAE | RMSE | R2 | High n | High MAE | High underprediction | Prediction SD |
|---|---|---|---|---|---|---|---|
| clean | 0.465195 | 0.591350 | 0.013557 | 13 | 1.546616 | 100.0% | 0.055276 |
| weighted | 0.466662 | 0.592828 | 0.008621 | 13 | 1.549531 | 100.0% | 0.054338 |
| train_mean | 0.505082 | 0.622958 | -0.094713 | 13 | 1.316363 | 100.0% | 0.000000 |
| train_median | 0.506936 | 0.625563 | -0.103888 | 13 | 1.307692 | 100.0% | 0.000000 |
| recent5_mean | 0.514451 | 0.645124 | -0.174002 | 13 | 1.415385 | 100.0% | 0.292147 |
| recent20_mean | 0.477066 | 0.607974 | -0.042683 | 13 | 1.520000 | 100.0% | 0.165577 |

Selection was frozen in selection_lock.json before holdout array access. Baselines are training mean/median and historical last5/20-event magnitude means. Their formulas are fixed; the paired comparison baseline (recent20_mean) is selected by validation MAE.

## One frozen holdout evaluation

| Frozen holdout model | MAE | RMSE | R2 | High n | High MAE | High underprediction | Prediction SD |
|---|---|---|---|---|---|---|---|
| train_mean | 0.500706 | 0.626183 | -0.135333 | 10 | 1.308671 | 100.0% | 0.000000 |
| train_median | 0.502459 | 0.629229 | -0.146406 | 10 | 1.300000 | 100.0% | 0.000000 |
| recent5_mean | 0.512240 | 0.636275 | -0.172225 | 10 | 1.534000 | 100.0% | 0.280529 |
| recent20_mean | 0.478934 | 0.604886 | -0.059421 | 10 | 1.540500 | 100.0% | 0.157053 |
| selected_clean | 0.473052 | 0.589476 | -0.006128 | 10 | 1.545182 | 100.0% | 0.061853 |

Only the selected BiLSTM was evaluated on holdout; the losing model remains unevaluated there. Saved predictions and an access record prevent routine notebook reruns from reevaluating holdout. Selected MAE95% moving-block interval: [0.44, 0.50]. Paired MAE difference against recent20_mean: -0.005882,95% block20 interval [-0.02, 0.01]. This is conditional one-seed uncertainty; chronology, model selection and prior catalog exposure limit generality.

Holdout actual SD=0.587678; prediction SD=0.061853; ratio=0.1052. High-magnitude group n=10, MAE=1.5451816082000733, underprediction=1.0. A small high group limits precision. Error by magnitude range is saved separately.

## Overfitting and final interpretation

| Variant | Best/total epochs | Clean train MAE | Validation-train MAE | Saved duration seconds |
|---|---|---|---|---|
| clean | 10/18 | 0.514509 | -0.049314 | 3.831 |
| weighted | 8/16 | 0.508131 | -0.041469 | 1.709 |

[Training/validation curves](training_validation_curves.png) retain all epochs and best-epoch markers. Positive gaps and post-best validation deterioration can indicate overfitting, but chronology-induced distribution differences also contribute; gaps alone do not identify causes. Early stopping selects validation minima. Prediction compression is reported explicitly; greater variability alone would not prove improvement.

The prespecified adequacy screen requires better validation MAE than the best baseline, positive holdout R2, a paired MAE-difference interval wholly below zero and no worse high-magnitude MAE. It is a conservative research decision rule, not a universal standard. Result: **C**. Preserve original Variant C and both modern fits as research artifacts; stop here.

Selected modern checkpoint: [models/clean/best.pt](models/clean/best.pt). Original research baseline remains ../phase2_bilstm_results/variants/C/best.pt (unchanged). Dataset/scalers/manifests: modern_dataset_v1/. Training and evaluation code: ../phase3_modern_catalog_analysis.ipynb and ../phase3_modern_bilstm_training.ipynb.

This standalone catalog model is not independently validated seconds-ahead EEW. Historical publication times are unknown. Any future CNN fusion requires shared event IDs, causal cutoffs, compatible targets and common partitions; no CNN or fusion work was performed.
