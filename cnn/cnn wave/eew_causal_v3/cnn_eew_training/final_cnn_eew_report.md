# Standalone synthetic CNN EEW results

**Date:** 2026-11-13. **Completed:** three selected models and frozen-model test evaluation.

Scope: synthetic oracle-P-aligned same-event classification and magnitude regression. No operational EEW claim, BiLSTM, fusion, Attention, Optuna, dataset regeneration or cleanup.

## Fixed protocol

Architecture: EarlyCNN: 1-16-32-48 strided Conv1d, mean/max pool, 96-32 shared, two heads; 14,674 trainable parameters. Input is only the saved normalized waveform, transformed pointwise by sign(x) log(1+|x|). No magnitude, IDs, timing or eligibility metadata enters forward().

Settings: `{"seed": 42, "batch_size": 256, "max_epochs": 50, "patience": 8, "min_delta": 0.0001, "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip": 5.0, "magnitude_loss_weight": 1.0, "huber_delta": 1.0, "detection_threshold": 0.5}`. Target center and population SD fit separately on eligible training earthquakes. Huber is in these standardized magnitude units. Validation selection minimizes full-set mean BCE plus eligible-earthquake mean Huber. No test metric or threshold tuning. Seed 42 and identical initialization/settings across horizons; one seed, not a multi-seed uncertainty study. Train curves use epoch-end frozen weights, matching validation evaluation mode.

HDF5 classification masks exclude ineligible records before batching. Magnitude loss is evaluated only on the horizon magnitude mask; sentinel noise targets are excluded. An isolated poison-sentinel and empty-mask loss check passed. Saved masks define the benchmark subset; they are privileged evaluation metadata, not a runtime trigger. Existing train/validation/test assignments remain unchanged.

## Selected checkpoints

| Horizon | Best epoch / run | Validation loss | Train eligible / EQ | Val eligible / EQ | Fit seconds | Checkpoint |
|---|---:|---:|---|---|---:|---|
| 1s | 16 / 24 | 0.933796 | 11324 / 5662 | 2426 / 1213 | 20.01 | models/cnn_1s_best.pt |
| 2s | 15 / 23 | 0.817325 | 11302 / 5640 | 2422 / 1209 | 17.82 | models/cnn_2s_best.pt |
| 3s | 14 / 22 | 0.753892 | 11255 / 5593 | 2416 / 1203 | 18.58 | models/cnn_3s_best.pt |

## Eligible test detection

PR-AUC is average precision (stepwise PR integration). Fixed probability threshold 0.5. Confusion matrix order is actual rows/predicted columns [noise, earthquake].

| Horizon | Eligible / EQ / excluded | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1s | 2428 / 1214 / 0 | 0.66310 | 0.63415 | 0.77100 | 0.69591 | 0.75111 | 0.75287 | [[674, 540], [278, 936]] |
| 2s | 2421 / 1207 / 7 | 0.72532 | 0.74240 | 0.68766 | 0.71398 | 0.82643 | 0.84711 | [[926, 288], [377, 830]] |
| 3s | 2414 / 1200 / 14 | 0.76678 | 0.78362 | 0.73333 | 0.75764 | 0.85847 | 0.88250 | [[971, 243], [320, 880]] |

## Eligible-earthquake test magnitude

Regression metrics include every eligible earthquake, regardless of detection success; these are not end-to-end alarm-conditioned metrics. Horizon populations differ slightly because S eligibility changes.

| Horizon | MAE | RMSE | R² | Bias (pred−actual) | Pred/actual SD | Slope | Training-median baseline MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1s | 0.67368 | 0.83215 | 0.14585 | -0.08416 | 0.4238 | 0.1671 | 0.72323 |
| 2s | 0.64016 | 0.78908 | 0.23084 | -0.05687 | 0.5506 | 0.2690 | 0.72254 |
| 3s | 0.61301 | 0.75770 | 0.28980 | -0.09077 | 0.5137 | 0.2820 | 0.72208 |

| Horizon | Magnitude range | N | MAE | Bias |
|---|---|---:|---:|---:|
| 1s | [-inf,3.0) | 195 | 1.0358389616012573 | 1.0358389616012573 |
| 1s | [3.0,4.0) | 495 | 0.32894250750541687 | 0.2503887116909027 |
| 1s | [4.0,5.0) | 393 | 0.6895079612731934 | -0.6288540959358215 |
| 1s | [5.0,6.0) | 96 | 1.3883739709854126 | -1.379216194152832 |
| 1s | [6.0,inf) | 35 | 1.3934134244918823 | -1.3873580694198608 |
| 2s | [-inf,3.0) | 192 | 0.9667260050773621 | 0.9667260050773621 |
| 2s | [3.0,4.0) | 491 | 0.3295619487762451 | 0.225957453250885 |
| 2s | [4.0,5.0) | 393 | 0.6835809946060181 | -0.5416030883789062 |
| 2s | [5.0,6.0) | 96 | 1.2290449142456055 | -1.1942538022994995 |
| 2s | [6.0,inf) | 35 | 1.1030713319778442 | -1.0771276950836182 |
| 3s | [-inf,3.0) | 190 | 0.9233044981956482 | 0.9233044981956482 |
| 3s | [3.0,4.0) | 487 | 0.31272417306900024 | 0.1824447065591812 |
| 3s | [4.0,5.0) | 392 | 0.6486377120018005 | -0.5629839897155762 |
| 3s | [5.0,6.0) | 96 | 1.1839431524276733 | -1.17324960231781 |
| 3s | [6.0,inf) | 35 | 1.1416339874267578 | -1.13944673538208 |

Strong-event diagnostic is prespecified as catalog magnitude ≥5.0 for this experiment; it is not an operational severity threshold.

- 1s: {"threshold": 5.0, "n": 131, "underprediction_fraction": 0.9770992366412213, "bias": -1.3813914060592651, "mae": 1.3897204399108887}
- 2s: {"threshold": 5.0, "n": 131, "underprediction_fraction": 0.9465648854961832, "bias": -1.1629605293273926, "mae": 1.1953877210617065}
- 3s: {"threshold": 5.0, "n": 131, "underprediction_fraction": 0.9694656488549618, "bias": -1.1642183065414429, "mae": 1.1726391315460205}

## Timing

Hardware: {'python': '3.13.12', 'torch': '2.6.0+cu124', 'numpy': '2.3.4', 'device': 'cuda', 'gpu': 'NVIDIA GeForce RTX 3050 Laptop GPU'}. Batch-one latency uses 25 warmups and 200 timed GPU-synchronized forward passes, preloaded input; includes fixed transform, excludes disk/transfer/acquisition/trigger. Latency is subject to device load and is not a streaming benchmark.

| Horizon | Median / p95 inference ms | S minus cutoff min / p05 / median / max seconds |
|---|---|---|
| 1s | 0.6914 / 1.4843 | 0.0632 / 4.2883 / 19.4233 / 78.0442 |
| 2s | 0.9074 / 2.5258 | 0.0969 / 3.4192 / 18.6276 / 77.0442 |
| 3s | 0.7011 / 2.2285 | 0.2677 / 2.5557 / 17.7437 / 76.0442 |

S-minus-cutoff = saved continuous S time − (saved oracle P time + horizon), for eligible earthquakes only. It is a same-station synthetic interval, not operational warning lead time. The windows include 1 second of pre-P context; no streaming trigger delay was measured.

## Generalization and shortcut diagnostics

| Horizon | Train / validation / test F1 | Train / validation / test magnitude MAE | Post-P-zeroed F1 / ROC-AUC |
|---|---|---|---|
| 1s | 0.7175 / 0.6880 / 0.6959 | 0.6665 / 0.6741 / 0.6737 | 0.5855 / 0.6060 |
| 2s | 0.7340 / 0.7298 / 0.7140 | 0.6323 / 0.6463 / 0.6402 | 0.3879 / 0.6196 |
| 3s | 0.7752 / 0.7637 / 0.7576 | 0.6037 / 0.6199 / 0.6130 | 0.1973 / 0.5918 |

Post-P-zeroing is a distribution-shift diagnostic of a frozen model, not proof of leakage or a pure pre-P-trained baseline. High retained discrimination would suggest noise/background or alignment shortcuts. Low retained discrimination does not exclude synthetic source/amplitude shortcuts. No direct target metadata is used; prior causality approval is reused, not rerun.

- 1s noise-profile false-positive rates: {"colored_pink": {"n": 306, "false_positive_rate": 0.0196078431372549}, "gaussian_bandlimited": {"n": 292, "false_positive_rate": 0.636986301369863}, "modulated_nonstationary": {"n": 302, "false_positive_rate": 0.5960264900662252}, "transient_disturbances": {"n": 314, "false_positive_rate": 0.535031847133758}}
- 1s magnitude spread ratio 0.424, slope 0.167; actual range [2.0, 7.199999809265137], predicted range [3.2523856163024902, 6.801295280456543]. Values below one indicate prediction compression. Strong-event signed bias: -1.3813914060592651.
- 2s noise-profile false-positive rates: {"colored_pink": {"n": 306, "false_positive_rate": 0.006535947712418301}, "gaussian_bandlimited": {"n": 292, "false_positive_rate": 0.3253424657534247}, "modulated_nonstationary": {"n": 302, "false_positive_rate": 0.31788079470198677}, "transient_disturbances": {"n": 314, "false_positive_rate": 0.30254777070063693}}
- 2s magnitude spread ratio 0.551, slope 0.269; actual range [2.0, 7.199999809265137], predicted range [3.330890655517578, 7.094269752502441]. Values below one indicate prediction compression. Strong-event signed bias: -1.1629605293273926.
- 3s noise-profile false-positive rates: {"colored_pink": {"n": 306, "false_positive_rate": 0.006535947712418301}, "gaussian_bandlimited": {"n": 292, "false_positive_rate": 0.2910958904109589}, "modulated_nonstationary": {"n": 302, "false_positive_rate": 0.2781456953642384}, "transient_disturbances": {"n": 314, "false_positive_rate": 0.22929936305732485}}
- 3s magnitude spread ratio 0.514, slope 0.282; actual range [2.0, 7.199999809265137], predicted range [3.2762322425842285, 6.564357757568359]. Values below one indicate prediction compression. Strong-event signed bias: -1.1642183065414429.

Observed limitations: all three regressors show substantial prediction compression (predicted/actual SD 0.424–0.551; slopes 0.167–0.282), with many predictions near magnitude 3.5–3.8. They underpredict 94.7–97.7% of the 131 eligible magnitude≥5 events, with mean biases −1.16 to −1.38 magnitude units. The small training/validation/test MAE gaps do not establish external generalization: test R² remains only 0.146–0.290, and gains over a training-median predictor are modest.

Post-P-zeroed ROC-AUC remains 0.592–0.620, and false-positive rates differ substantially by noise family (colored pink is much easier than the other families). Together these are evidence consistent with background-family shortcuts; the distribution-shift probe does not isolate causation or prove future leakage. Direct target-feature leakage was not observed in the loader/model interface. No settings were changed in response to these results.

Results establish performance only on this generator and split. Event-disjoint synthetic testing cannot establish generalization to real instruments, noise, source physics or causal picked windows. Oracle alignment itself is privileged. Synthetic amplitude/distance assumptions can make magnitude easier or ambiguous. Tail estimates with small counts are uncertain; no confidence intervals or multiple seeds were evaluated. Test outputs were inspected only after all three checkpoint selections and did not change architecture, thresholds or settings.

## Essential plots (PNG, 300 DPI)

- [plots/training_validation_loss.png](plots/training_validation_loss.png)
- [plots/actual_predicted_magnitude.png](plots/actual_predicted_magnitude.png)
- [plots/confusion_matrices.png](plots/confusion_matrices.png)
- [plots/magnitude_residuals.png](plots/magnitude_residuals.png)
- [plots/magnitude_mae_comparison.png](plots/magnitude_mae_comparison.png)
- [plots/detection_f1_comparison.png](plots/detection_f1_comparison.png)

## Preservation and reproduction

Only the two notebooks, three best checkpoints, six plots and this report were created under cnn_eew_training/. No duplicate dataset or standalone Python script was written. Checkpoints contain model weights, configuration, selected epoch, complete small training history, train-only target transforms, input hashes and environment; no extra last/epoch checkpoints. Re-execution refuses to overwrite saved models/plots/report. Evaluation imports only the common-definition cell, never the training loop.

Before/after hashes match for all 4 opened original dependencies (three HDF5 files and readiness report). Other existing files were not opened for writing. The checkpoint records those hashes. Selected validation losses reproduced after checkpoint reload.

**STOP: standalone CNN training and evaluation complete.**
