# Phase 2A: BiLSTM error diagnosis
**No training, inference, checkpoint loading or experiment changes.** Selected C and its test results remain frozen.

## Confirmed prediction behavior
High-magnitude underprediction is systematic: validation **75/75**, test **38/38**. Test high MAE/bias are 1.5218/-1.5218.
For test ML<3, **490/490** are overpredicted; mean bias +0.7080.

Test predictions span 3.116–3.595, mean 3.272, SD 0.0779; actual SD is 0.6414. Prediction SD is only 12.1% of actual SD. Prediction/actual Pearson correlation is -0.018. This confirms strong compression toward common magnitudes. The global training mean is 4.034; predictions center -0.762 away from it, so literal collapse to that mean is unsupported.
Validation already shows prediction SD 0.1573 versus actual SD 0.6293.

| Test ML range | Count | MAE | Signed bias | Underpredicted |
|---|---:|---:|---:|---:|
| <3 | 490 | 0.7080 | +0.7080 | 0.0% |
| 3–<4 | 469 | 0.2514 | -0.0969 | 57.8% |
| 4–<4.5 | 70 | 0.8812 | -0.8812 | 100.0% |
| 4.5–<5 | 30 | 1.3852 | -1.3852 | 100.0% |
| >=5 | 8 | 2.0341 | -2.0341 | 100.0% |

Largest test error: NCS_EV_06727, actual 5.80, prediction 3.200, absolute error 2.600. Top-ten examples with IDs/times are in diagnostic_metrics.json; all validation/test residuals are in error_analysis.csv. These are errors, not evidence that source events are invalid.

## Chronological distribution shift
| Split | Targets | Mean ML | ML>=4.5 | Median target gap (days) |
|---|---:|---:|---:|---:|
| train | 4973 | 4.034 | 31.23% | 1.556 |
| val | 1065 | 3.511 | 7.04% | 0.196 |
| test | 1067 | 3.101 | 3.56% | 0.132 |

Train→validation target KS distance 0.315, Wasserstein 0.525 ML; train→test 0.457/0.933 ML. These are substantial descriptive population changes, not proof of a causal explanation for model error.

Largest absolute train→validation feature mean shifts, in training SD units: event_rate_30d +5.35, mean_depth_20_km -0.89, historical_longitude_deg -0.78.
Input latitude/longitude means (train→validation): 25.77/93.32 → 26.23/91.35.
Mean 30-day rate: 0.552 → 2.978 events/day.
Outside-training-range windows: validation 46.85%, test 100.00%. All ten features, tails, gap distributions and in/out-of-range errors are saved. Shift can occur inside training ranges, so range coverage alone is insufficient.

Feature comparisons deduplicate repeated historical event rows within each split. Boundary context still overlaps and all series are temporally dependent; no IID significance tests are claimed. Test comparisons are frozen description and are not used to rank interventions.

## Overfitting and sampling
| Variant | Best/run epoch | MAE gap at best | Later val MAE change | Later train MAE change | Mean unique/repeat draws per epoch |
|---|---:|---:|---:|---:|---:|
| A | 14/22 | +0.0670 | +0.0442 | +0.0628 | 4973/0 |
| B | 14/22 | +0.0710 | +0.0125 | +0.0093 | 4973/0 |
| C | 12/20 | +0.0415 | +0.0118 | -0.0175 | 3124/1849 |
| D | 12/20 | +0.0441 | +0.0100 | -0.0224 | 3124/1849 |

C/D select epoch 12 versus A/B 14, and show late validation degradation with training improvement. This is consistent with some late overfitting, not proof that weighted sampling caused memorization or that best-epoch error is primarily overfitting. All runs stopped eight nonimproving epochs after their best and restored that checkpoint.

Weighted C/D increase high-target exposure from 31.23% to an epoch-average 37.02%. Epoch zero has 3114 unique windows, 1859 repeated draws, and maximum 6 appearances of one parent. Weights span 1.00–1.50. Exposure changes are real; repetitions add no independent information. High-target coverage alone does not explain failure: all four variants underpredict all validation high targets.

Noise A→B validation MAE improvement is +0.001104; C→D is -0.000331. Twenty-step circular-block intervals: [-0.0004122639262060604, 0.0024775009703748097] and [-0.0006650636397616964, -3.3471013458682064e-05]. Fifty-step sensitivity is saved. These tiny single-seed effects do not establish a reliable noise benefit; bootstrap intervals omit training/selection uncertainty.

## Loss and feature informativeness
[Huber loss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.HuberLoss.html) is symmetric and limits prediction-space gradient magnitude beyond delta. With delta=1, 81.3% of validation high-target errors are in that region. Reduced outlier influence could contribute to conservative predictions, but no loss ablation exists; Huber does not inherently require downward bias.

Available inputs are past magnitude/depth/location, elapsed time, rate and rolling magnitude/depth summaries. They contain historical context, with no measured precursory or target-event information. Strongest last-step target |Spearman| is 0.744 in training (mean_magnitude_20) and 0.217 in validation (mean_magnitude_20). Full Pearson/Spearman, window-mean summaries, training halves and training-defined conditional bins are saved. Training feature pairs with |Spearman|>=0.8: 0.
Period changes and limited stable associations support investigating feature sufficiency; they do not establish causal importance, prove nonlinear signal is absent, or prove the task inherently unpredictable.

## Baseline review
| Test baseline | Overall MAE | Overall bias | High ML MAE | High underprediction |
|---|---:|---:|---:|---:|
| training_mean | 1.0032 | +0.9326 | 0.7556 | 100.0% |
| training_median | 0.9747 | +0.8988 | 0.7895 | 100.0% |
| recent_event_magnitude | 0.6739 | -0.0012 | 1.5605 | 100.0% |

C beats these baselines overall (MAE 0.5477), but training constants can have smaller upper-tail errors by predicting larger values everywhere. This is a tradeoff, not superior event discrimination. Constants were reconstructed from training values and checked against saved baseline metrics; recent-event test predictions were reused. No new model was fitted.

## Most plausible explanation and next experiments
**Hypothesis:** weak or nonstationary input–target information, combined with an average robust objective, produces a narrow common-magnitude prediction band; chronological distribution shift worsens transfer. Mild late overfitting and limited tail emphasis may contribute. Their causal contributions are not identified by this audit. Class imbalance alone is insufficient as an explanation.

Ranked below using **training/validation evidence only**; no proposals were implemented:
1. **Loss function adjustment** — Evidence: Validation C underpredicts all 75 high targets; 81.3% of their residuals exceed Huber delta 1. Expected benefit: Test whether increasing large-error influence reduces upper-tail shrinkage. Tradeoff: May raise overall MAE/low-magnitude overprediction and amplify questionable extreme labels. Next test: FIRST: fixed C parents, architecture, AdamW, seed and stopping; compare existing Huber delta=1 with delta=2 only. Report overall/high MAE, bias and prediction spread. No Optuna or test access; confirm across seeds before claiming a gain.
2. **Feature engineering** — Evidence: Strongest last-step |Spearman|: train 0.744, validation 0.217; correlations vary across periods. Expected benefit: Determine whether current historical summaries provide stable next-event information before adding complexity. Tradeoff: New regional/context indicators can overfit or require unavailable reporting information. Next test: Fit a small train-only regularized linear probe on existing last-step/window-mean features in a future task; compare recent-event baseline on validation, then test one justified causal feature at a time.
3. **Magnitude-aware weighting** — Evidence: C improves validation high MAE over A by 0.0973, yet high underprediction stays 100%. Expected benefit: Express upper-tail error cost directly instead of assuming sampling alone is sufficient. Tradeoff: Worse overall calibration/MAE and disproportionate influence of uncertain high ML values. Next test: After isolating the loss, keep the sampler fixed and compare no added loss weights versus a bounded high-target weight fixed from training only; evaluate the validation tradeoff.
4. **Sampling strategy** — Evidence: C averages 3124 unique parents and 1849 repeat draws per 4973-draw epoch. Expected benefit: Retain high-target exposure with less duplicate concentration. Tradeoff: Capping repeats changes effective target exposure and may reduce tail gains. Next test: Compare approved weighted draws with a conservative per-parent repeat cap, keeping epoch size fixed; record achieved exposure and validation results without changing loss simultaneously.
5. **Regularization** — Evidence: C/D show post-best train improvement while validation worsens; selected best checkpoints already limit this. Expected benefit: Reduce late memorization if it is material. Tradeoff: May further compress predictions or worsen underfitting; early stopping already helps. Next test: One fixed higher-dropout setting against current C, same loss/sampling; compare best-epoch validation and prediction spread.
6. **Alternative architecture** — Evidence: Current results show little stable explained validation variance; no evidence identifies BiLSTM structure as the cause. Expected benefit: Check whether a simpler sequence model generalizes as well with fewer parameters. Tradeoff: Architecture search can consume validation information without fixing weak inputs or calibration. Next test: One smaller unidirectional recurrent baseline under the same split/loss/sampling protocol, only after simpler ablations; no CNN/fusion.

## Scope, figures and preservation
Reuse [actual/predicted](../plots/selected_test_predictions_errors.png), [magnitude errors](../plots/test_error_by_magnitude.png), [MAE curves](../plots/training_validation_mae.png) and [Huber curves](../plots/training_validation_huber.png).
New: [residuals vs actual](residuals_vs_actual.png), [split targets](target_distribution_by_split.png).
Notebook: ../../phase2a_bilstm_error_diagnosis.ipynb. Original datasets, checkpoints, Optuna database, predictions, metrics and selection are unchanged. Reporting availability, magnitude calibration and catalog completeness remain UNKNOWN. This audit does not establish operational EEW.
