# Final integrated research report: synthetic earthquake early-warning benchmarks

## Abstract

Four models—CNN, tabular BiLSTM, CNN+BiLSTM fusion, and fusion with temporal Attention—were compared on verified synthetic oracle-P-aligned observations at 1s, 2s and 3s. No architecture consistently dominated detection and magnitude estimation. Attention showed no statistically supported advantage over non-Attention fusion in the available paired comparisons. A subsequent validation-only 12-trial 3s tuning study met none of its predeclared replacement criteria and retained the original Attention model. Diagnosis identified magnitude compression, opposite tail biases, a substantial false-positive/recall tradeoff, and highly correlated branch magnitude errors. These are controlled synthetic research findings, not evidence of real-world earthquake prediction or operational warning capability.

## 1. Methodology and evidence provenance

This report integrates four existing artifacts only. It performs no inference, training, new statistical testing, threshold change or dataset reconstruction. Independently verified model results remain unchanged. The tuning study and error diagnosis are incorporated as subsequent analyses, not represented as newly independently audited model results.

Catalog-informed simulation generated the waveform observations. CNN uses normalized waveform windows; BiLSTM uses seven numerical features from consecutive 0.1s bins of those same observations. Fusion combines frozen pretrained representations; Attention weights the 20/30/40 observed BiLSTM hidden states before combination with the CNN embedding. These branches are different representations of a shared source, not independent observational modalities.

At 100 Hz, windows contain 100 pre-reference samples plus 100/200/300 post-reference samples. Oracle P alignment, event-disjoint splits, training-only scaling, pre-S eligibility and earthquake-only magnitude regression define the benchmark. No true magnitude, future S observation or event identity is a model input.

The common test cohort is exactly **1,200 earthquakes + 1,214 noise records** at all horizons. Magnitude scores include all eligible earthquakes independently of detection decisions. The 3s validation cohort used for tuning/diagnosis is **1,203 earthquakes + 1,213 noise records**. Validation and test results are separated below.

## 2. Preserved four-model test comparison: threshold 0.5

| Horizon | Model | Accuracy | Precision | Recall | F1 | FPR | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | CNN | 0.661143 | 0.630643 | 0.768333 | 0.692712 | 0.444811 | 0.665020 | 0.821894 | 0.164370 |
| 1s | BiLSTM | 0.628418 | 0.572453 | 0.997500 | 0.727438 | 0.736409 | 0.670213 | 0.824230 | 0.159615 |
| 1s | Fusion | 0.652858 | 0.627645 | 0.741667 | 0.679908 | 0.434926 | 0.664240 | 0.817358 | 0.173570 |
| 1s | Attention | 0.663215 | 0.651290 | 0.694167 | 0.672045 | 0.367381 | 0.666359 | 0.818941 | 0.170365 |
| 2s | CNN | 0.724524 | 0.740774 | 0.685833 | 0.712246 | 0.237232 | 0.635296 | 0.783164 | 0.241270 |
| 2s | BiLSTM | 0.673571 | 0.700000 | 0.600833 | 0.646637 | 0.254530 | 0.634122 | 0.780973 | 0.245509 |
| 2s | Fusion | 0.734880 | 0.765152 | 0.673333 | 0.716312 | 0.204283 | 0.635879 | 0.783324 | 0.240961 |
| 2s | Attention | 0.735708 | 0.766098 | 0.674167 | 0.717199 | 0.203460 | 0.636489 | 0.782980 | 0.241627 |
| 3s | CNN | 0.766777 | 0.783615 | 0.733333 | 0.757641 | 0.200165 | 0.613005 | 0.757702 | 0.289803 |
| 3s | BiLSTM | 0.714167 | 0.740566 | 0.654167 | 0.694690 | 0.226524 | 0.615082 | 0.758421 | 0.288455 |
| 3s | Fusion | 0.772162 | 0.832311 | 0.678333 | 0.747475 | 0.135091 | 0.608215 | 0.753055 | 0.298488 |
| 3s | Attention | 0.776305 | 0.851812 | 0.665833 | 0.747428 | 0.114498 | 0.606601 | 0.748621 | 0.306725 |

ROC-AUC and average-precision PR-AUC are retained in the CSV. Rates are fractions, not percentages. FPR is FP/(FP+TN).

### Numerical leaders, not statistically established winners

| Horizon | Highest F1 | Lowest MAE | Lowest RMSE / highest R2 |
|---|---|---|---|
| 1s | BiLSTM | Fusion | Fusion |
| 2s | Attention | BiLSTM | BiLSTM |
| 3s | CNN | Attention | Attention |

At 1s, BiLSTM's highest F1 accompanies a 73.64% noise FPR. At 2s, Attention's numerical F1 lead over fusion is only 0.000887; BiLSTM has the lowest MAE and RMSE. At 3s, CNN leads F1, while Attention has the lowest MAE and RMSE. None of these rankings establishes a universally best architecture or an operational selection rule. Metric priorities and costs matter.

## 3. Test results at validation-selected thresholds: separate policy

| Horizon | Model | Threshold | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| 1s | BiLSTM | 0.47 | 0.627175 | 0.571497 | 0.999167 | 0.727107 | 0.740527 |
| 1s | Fusion | 0.42 | 0.632560 | 0.576453 | 0.983333 | 0.726825 | 0.714168 |
| 1s | Attention | 0.28 | 0.626761 | 0.571224 | 0.999167 | 0.726887 | 0.741351 |
| 2s | BiLSTM | 0.36 | 0.627589 | 0.571769 | 0.999167 | 0.727328 | 0.739703 |
| 2s | Fusion | 0.41 | 0.721624 | 0.691860 | 0.793333 | 0.739130 | 0.349259 |
| 2s | Attention | 0.41 | 0.719138 | 0.689130 | 0.792500 | 0.737209 | 0.353377 |
| 3s | BiLSTM | 0.36 | 0.648302 | 0.589404 | 0.964167 | 0.731584 | 0.663921 |
| 3s | Fusion | 0.36 | 0.751036 | 0.721033 | 0.814167 | 0.764775 | 0.311367 |
| 3s | Attention | 0.41 | 0.761806 | 0.771503 | 0.740000 | 0.755423 | 0.216639 |

CNN: no audited validation-selected common-cohort operating point is supplied in the inspected summaries; marked unavailable rather than substituting0.5 or another cohort. No threshold was fitted for this report. Fusion and Attention thresholds were selected by validation F1; high recall can be purchased at substantial FPR.

The entries above are **test outcomes at thresholds selected on validation**, not validation performance. No thresholds were refitted here. CNN's corresponding audited common-cohort operating point is unavailable; it is not invented. The fixed-0.5 table remains the primary controlled threshold-policy comparison.

## 4. Existing paired comparisons

Differences are candidate minus comparator. Positive F1 and negative MAE favor the candidate. These saved500-resample, seed2026, paired class-stratified percentile95% intervals use identical records; magnitude resamples earthquakes only. No new intervals or tests computed.

| Horizon | Candidate − comparator | F1 difference [95% CI] | MAE difference [95% CI] |
|---|---|---|---|
| 1s | Fusion − CNN | -0.012804 [-0.023792306807676085, -3.167606590949191e-05] | -0.000780 [-0.0034688999352511015, 0.0017248650634428485] |
| 1s | Fusion − BiLSTM | -0.047530 [-0.06428662499380051, -0.03231501238265269] | -0.005973 [-0.015197540121152996, 0.0032984839810524117] |
| 1s | Attention − Fusion | -0.007863 [-0.018330848456274, 0.001981389353165757] | 0.002119 [-0.0012190080509753896, 0.0054234042880125346] |
| 2s | Fusion − CNN | 0.004066 [-0.0051605001703711765, 0.014231114089397819] | 0.000583 [-0.004587645235005766, 0.005602469923906027] |
| 2s | Fusion − BiLSTM | 0.069675 [0.04497768064970756, 0.09491167683694436] | 0.001757 [-0.0046189094544388356, 0.008094231225550173] |
| 2s | Attention − Fusion | 0.000887 [-0.003969822205396578, 0.005982555642963445] | 0.000610 [-0.0018039557384327053, 0.002919959701830521] |
| 3s | Fusion − CNN | -0.010166 [-0.02269079653232397, -0.000400259480366555] | -0.004790 [-0.011949946149252355, 0.0021605384710710472] |
| 3s | Fusion − BiLSTM | 0.052784 [0.031601839949213034, 0.0756103546455666] | -0.006867 [-0.014054152299650013, 0.001418869048939083] |
| 3s | Attention − Fusion | -0.000047 [-0.006012117100356584, 0.0065387337626778815] | -0.001614 [-0.0063754469039849935, 0.003244647354586046] |

All Attention-versus-fusion F1 and MAE intervals include zero. There is no statistically supported Attention advantage under these saved comparisons, and absence of significance is not proof of equivalence. Every fusion-versus-standalone MAE interval also spans zero. Fusion F1 is lower than BiLSTM at1s and higher at2s/3s in the descriptive intervals. Against CNN, fusion has lower1s/3s point estimates; the1s interval barely excludes zero and is sensitive to bootstrap Monte Carlo uncertainty. The2s F1 interval crosses zero. No consistent fusion superiority is established.

Attention has **no statistically supported improvement over non-Attention fusion** at any horizon on F1 or MAE under the saved intervals. Lack of significance does not establish equivalence. No new confidence intervals, significance tests or claims of general superiority are added.

## 5. Controlled 3s Attention tuning: all 12 trials

The backbones remained frozen. Ten predefined trials varied AdamW learning rate (1e-4, 3e-4, 1e-3), dropout (0.1–0.4), fusion width (64/128) and regression weight (0.5/1/2). Trials 11–12 reused trial 10's configuration with training-only strong-bin weighting or asymmetric strong-event underprediction penalties. Magnitude was used as a loss target/weight, never as an input. Maximum 35 epochs, patience 7, seed 42 and gradient clipping 5 were fixed.

All checkpoint selection used unweighted validation BCE plus masked standardized Huber, even when the training loss was weighted. Operating points maximized validation F1 subject to FPR≤30% and recall≥70%, with declared fallbacks/ties. Replacement required feasible operation, overall MAE≤original+0.02 and strong-event MAE≤original+0.10, plus one of: FPR reduction≥0.02 with F1 loss≤0.01; F1 gain≥0.01 with FPR increase≤0.01; or MAE reduction≥0.02 with both FPR/F1 guards. Qualifying candidates would be ranked lexicographically by FPR, F1 and magnitude metrics. No hidden aggregate score or post-test selection was used.

All figures in the following trial table are validation results at each trial's validation operating point:

| Trial | LR | Dropout | Hidden | Reg weight | Variant | Best epoch | F1 | FPR | Recall | MAE | Strong MAE | Underprediction % |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0001 | 0.1 | 64 | 1.0 | standard | 33 | 0.765403 | 0.296785 | 0.805486 | 0.613366 | 1.094090 | 89.313 |
| 2 | 0.0003 | 0.2 | 64 | 1.0 | standard | 34 | 0.766929 | 0.299258 | 0.809643 | 0.614142 | 1.070109 | 88.550 |
| 3 | 0.001 | 0.3 | 64 | 1.0 | standard | 32 | 0.767812 | 0.220115 | 0.761430 | 0.613804 | 1.084246 | 88.550 |
| 4 | 0.0001 | 0.2 | 128 | 1.0 | standard | 35 | 0.765873 | 0.290190 | 0.802161 | 0.614128 | 1.075411 | 87.786 |
| 5 | 0.0003 | 0.3 | 128 | 1.0 | standard | 34 | 0.767645 | 0.289365 | 0.804655 | 0.613727 | 1.061949 | 87.786 |
| 6 | 0.001 | 0.4 | 128 | 1.0 | standard | 27 | 0.767031 | 0.216818 | 0.758105 | 0.613029 | 1.085261 | 88.550 |
| 7 | 0.0003 | 0.1 | 64 | 0.5 | standard | 34 | 0.766733 | 0.291838 | 0.804655 | 0.613956 | 1.077375 | 89.313 |
| 8 | 0.0003 | 0.3 | 128 | 0.5 | standard | 32 | 0.767311 | 0.270404 | 0.792186 | 0.614046 | 1.068495 | 87.023 |
| 9 | 0.0003 | 0.2 | 64 | 2.0 | standard | 34 | 0.767836 | 0.296785 | 0.809643 | 0.613902 | 1.067140 | 88.550 |
| 10 | 0.001 | 0.2 | 128 | 2.0 | standard | 27 | 0.766962 | 0.214345 | 0.756442 | 0.613496 | 1.086542 | 88.550 |
| 11 | 0.001 | 0.2 | 128 | 2.0 | strong_bin | 21 | 0.767221 | 0.291838 | 0.805486 | 0.616033 | 1.098798 | 88.550 |
| 12 | 0.001 | 0.2 | 128 | 2.0 | asymmetric | 21 | 0.766745 | 0.298434 | 0.808811 | 0.615486 | 1.077363 | 87.786 |

### Why the original model was retained

Original validation F1 was 0.768585, FPR 0.216818, overall MAE 0.615548 and M≥5 MAE 1.043461. All 12 trial operating-point F1 values were lower. The best overall MAE, 0.613029, improved by only 0.002518; the best FPR, 0.214345, improved by only 0.002473. Neither met the declared useful-improvement margin. Every trial worsened M≥5 MAE (range 1.061949–1.098798). Consequently, no candidate replaced the original hidden-32/dropout-0.2 Attention head. This is a negative tuning result, not an improved model claim.

Trial 12 improved strong-event MAE relative to trial 10 but remained worse than the original. Trial 11 worsened overall and strong-event MAE. Other settings changed jointly, so the study cannot isolate independent effects of learning rate, capacity or task weighting. No 1s/2s tuning followed.

## 6. Original 3s threshold tradeoff: validation versus exploratory test

| Split | Threshold | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Validation (2,416 records) | 0.50 | 0.837589 | 0.681629 | 0.751604 | 0.131080 |
| Validation (2,416 records) | 0.41 | 0.776740 | 0.760599 | 0.768585 | 0.216818 |
| Common test (2,414 records) | 0.50 | 0.851812 | 0.665833 | 0.747428 | 0.114498 |
| Common test (2,414 records) | 0.41 | 0.771503 | 0.740000 | 0.755423 | 0.216639 |

On validation, changing 0.50 to 0.41 recovers 95 earthquakes but adds 104 false positives: FP rises from 159 to 263 and FN falls from 383 to 288. The point meets the study's stated research constraint, but operational acceptability is unknown. The 0.41 threshold was chosen on validation and not changed after test inspection. The reused test cohort makes the later comparison exploratory, not an unbiased confirmation. Retained-model test MAE/RMSE remain 0.606601/0.748621; no tuning improvement is implied by repeating them.

## 7. Magnitude compression and strong-event errors

### Preserved common-test summaries

| Horizon | Model | Signed error | Predicted/actual SD | M>=5 N | Underprediction (%) | Strong signed error |
|---|---|---:|---:|---:|---:|---:|
| 1s | CNN | -0.101660 | 0.411175 | 131 | 97.710 | -1.381391 |
| 1s | BiLSTM | -0.076714 | 0.342184 | 131 | 97.710 | -1.409562 |
| 1s | Fusion | -0.074875 | 0.395727 | 131 | 98.473 | -1.364506 |
| 1s | Attention | -0.076292 | 0.387569 | 131 | 98.473 | -1.373537 |
| 2s | CNN | -0.065797 | 0.544796 | 131 | 94.656 | -1.162961 |
| 2s | BiLSTM | -0.054940 | 0.482220 | 131 | 93.130 | -1.202630 |
| 2s | Fusion | -0.067043 | 0.489833 | 131 | 93.893 | -1.218227 |
| 2s | Attention | -0.056138 | 0.512419 | 131 | 95.420 | -1.182187 |
| 3s | CNN | -0.090770 | 0.513718 | 131 | 96.947 | -1.164218 |
| 3s | BiLSTM | -0.082112 | 0.516038 | 131 | 96.183 | -1.134616 |
| 3s | Fusion | -0.061756 | 0.569917 | 131 | 92.366 | -1.047271 |
| 3s | Attention | -0.004881 | 0.600623 | 131 | 87.023 | -0.930820 |

SD ratios below1 show compressed magnitude predictions. Underprediction is the fraction of M>=5 records with predicted magnitude below truth; negative signed error means underestimation. All rows use the common-cohort magnitude summaries replayed during fusion/Attention audits.

### Validation diagnosis for original 3s Attention

| Range | N | MAE | RMSE | Signed error | Underprediction % | FN at0.50 | FN at0.41 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M<3 | 191 | 0.965733 | 1.009972 | 0.965733 | 0.000 | 96 | 74 |
| 3<=M<4 | 487 | 0.371700 | 0.457559 | 0.271721 | 23.614 | 180 | 136 |
| 4<=M<5 | 394 | 0.604918 | 0.690173 | -0.479795 | 84.518 | 95 | 71 |
| M>=5 | 131 | 1.043461 | 1.223419 | -0.945713 | 85.496 | 12 | 7 |

All 191 M<3 validation events were overestimated, with mean error +0.9657. M≥5 mean error was −0.9457 and 85.50% were underpredicted. Only 7/131 strong events were missed at threshold 0.41, demonstrating that successful detection is not accurate magnitude estimation. Validation aggregate signed error +0.0032 masks opposite tail errors. Predicted/actual SD ratio 0.60568, variance 0.30728 versus 0.83762 and regression slope 0.34575 quantify compression. On common test, M≥5 underprediction is 87.02%; that value must not be substituted for the validation rate.

| Split | Eligible EQ | Mean | SD | M<3 | 3–4 | 4–5 | M>=5 | Strong fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5593 | 3.865957 | 0.904796 | 886 | 2269 | 1829 | 609 | 10.8886% |
| validation | 1203 | 3.869742 | 0.915217 | 191 | 487 | 394 | 131 | 10.8894% |

Train/validation magnitude means, spreads and approximately 10.9% strong-event proportions are similar. A coarse magnitude-distribution shift is not supported as the main cause. Strong-event minority representation is observed, but imbalance alone is not a proven explanation; weighted-loss trials did not fix the problem.

## 8. Branch complementarity and interpretation of failed tuning

CNN and BiLSTM validation magnitude residual correlation is **0.982666 (approximately 0.983)**. Their errors are strongly shared, which limits the evidence for expecting another fusion-head search to remove them. Probability correlation is 0.79957. At threshold 0.5, BiLSTM correctly classifies 244 CNN errors, CNN correctly classifies 413 BiLSTM errors, and 306 errors are shared. Some classification complementarity exists, but an achievable learned selector is not established.

The retrospective oracle that chooses the smaller absolute magnitude error per event has MAE 0.568785. It uses true outcomes and is not an implementable prediction method or a validated gain. Correlation does not prove an irreducible limit, independence, or causality.

Frozen features plausibly constrain the head, but no unfreezing comparison or representation-sufficiency test was conducted. Gradient interference was not measured. Unweighted checkpoint selection could overlook tail-specific improvements, but epochwise strong-event predictions were unavailable. Near-cap selected epochs do not prove convergence. No trustworthy noise-family metadata was found in the inspected schema; noise-family shortcut effects remain **UNKNOWN**, with no fabricated labels.

The evidence supports retention of the original model and **NO FURTHER TUNING at this stage**, rather than an unsupported causal explanation or another broad search.

## 9. Limitations and future work

All findings remain synthetic and oracle-P-aligned. Both branches share simulated observations. No realistic online triggering, independent real-recording validation, operational false-alarm rate, warning lead time or real-world earthquake prediction was established. Single-seed studies, reused validation/test cohorts, limited strong-event counts and descriptive bootstrap intervals constrain statistical claims. Attention weights are diagnostics, not physical explanations.

Future work should prioritize independent real seismic recordings and verified observational metadata; realistic causal triggering and end-to-end timing; untouched event/station-separated evaluation; and adequately powered strong-event bias/calibration assessment. See [limitations and future work](limitations_and_future_work.md). These are proposals only; no experiment is initiated.

## 10. Existing figures (referenced, not duplicated)

- [Four-model F1](../four_model_eew_v1/plots/four_model_f1.png)
- [Four-model MAE](../four_model_eew_v1/plots/four_model_mae.png)
- [Four-model FPR](../four_model_eew_v1/plots/four_model_fpr.png)
- [Four-model RMSE](../four_model_eew_v1/plots/four_model_rmse.png)
- [Validation threshold tradeoffs](../../model_improvement/attention_tuning_v1/plots/validation_threshold_tradeoffs.png)
- [Validation trial tradeoffs](../../model_improvement/attention_tuning_v1/plots/validation_trial_tradeoffs.png)
- [Exploratory original/retained magnitude comparison](../../model_improvement/attention_tuning_v1/plots/exploratory_magnitude_comparison.png) — original retained, not evidence of improvement.

## 11. Sources and preservation

The four sources below are project evidence, not external literature. Their SHA256 hashes were checked unchanged. No datasets, models, notebooks or previous results were modified. Only the three requested Markdown files were created.

- `final_comparison/four_model_eew_v1/final_comparison_report.md` — SHA256 `68e1dcf02f973cb156039326c00530e6668fb4369a081191633d217e35bafef1`
- `final_comparison/four_model_eew_v1/final_metrics_table.csv` — SHA256 `cc8c413a7a2bc37d68176fd2451983690af54d8e37b6e14db045ff7d87599cfb`
- `model_improvement/attention_tuning_v1/tuning_report.md` — SHA256 `63a64f4902c0b066e5a362af0220198f3e810e483f8804c6ee4e387851ec5747`
- `model_improvement/error_diagnosis_v1/diagnostic_report.md` — SHA256 `f9832116f443eff288b2a535661a8ce0f5e12a83bd8dd85ae66a6513ec416f3f`
