# Independent standalone CNN performance audit

**Date:** 2026-11-13. **Verdict: APPROVED** — Suitable for standalone synthetic benchmark reporting and future controlled comparisons using identical IDs, cutoff/masks and targets. Not operational EEW; shortcut and magnitude-bias limitations must accompany reporting.

## Checks

| Check | Status | Evidence |
|---|---|---|
| train_1s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| train_2s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| train_3s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| validation_1s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| validation_2s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| validation_3s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| test_1s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| test_2s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| test_3s_masks | PASS | Direct mask reconstruction from labels and first-S grid; noise sentinel excluded. |
| event_identity_and_split_isolation | PASS | {'manifest_exact_row_joins': {'train': True, 'validation': True, 'test': True}, 'identity_intersections': {'train/test': {'source_id': 0, 'parent_id': 0, 'pair_id': 0, 'combined_index': 0}, 'validation/test': {'source_id': 0, 'parent_id': 0, 'pair_id': 0, 'combined_index': 0}, 'train/validation': {'source_id': 0, 'parent_id': 0, 'pair_id': 0, 'combined_index': 0}}} |
| checkpoint_1s_identity | PASS | {'path': 'C:\\Users\\choud\\Desktop\\Minor project\\cnn\\cnn wave\\eew_causal_v3\\cnn_eew_training\\models\\cnn_1s_best.pt', 'sha256': '43b2099c8a3f64e94a528ca32eb3874828bc5a0d3cc0b1c18b63d67c298399b4', 'selected_epoch': 16} |
| 1s_validation_selection_and_training_scaler | PASS | {'history_best_epoch': 16, 'saved_best_epoch': 16, 'saved_validation_loss': 0.9337961459493991, 'recomputed_validation_loss': 0.9337960779666901, 'delta': -6.798270901775538e-08, 'train_magnitude_center': 3.8584954738616943, 'train_magnitude_scale': 0.9050084948539734, 'training_band_counts': {'below3': 910, '3to4': 2305, '4to5': 1836, 'atleast5': 611}} |
| 1s_reported_metrics | PASS | {'recomputed_minus_reported': {'accuracy': -2.8006589786189195e-06, 'precision': -3.6585365853492036e-06, 'recall': 4.942339373981142e-06, 'f1': 7.806691449951941e-07, 'roc_auc': -4.690988440803068e-06, 'pr_auc_ap': -2.748634981619702e-06, 'mae': -1.9651908811368912e-06, 'rmse': 4.781870697745383e-06, 'r2': -4.584898604415022e-06}, 'tolerance': '5.1e-6 for five-decimal report; float64 independent regression aggregation', 'selected_row_identity_sha256': '4148a8586fcfa3a1076fd14fad89149601c57bfe258cbf5d09fd383bd92020c7', 'excluded_source_ids': []} |
| checkpoint_2s_identity | PASS | {'path': 'C:\\Users\\choud\\Desktop\\Minor project\\cnn\\cnn wave\\eew_causal_v3\\cnn_eew_training\\models\\cnn_2s_best.pt', 'sha256': 'cf414ddd4b13e61466b407dc7d5e2f998e3ee6f5d53f973b8e214103cac667d5', 'selected_epoch': 15} |
| 2s_validation_selection_and_training_scaler | PASS | {'history_best_epoch': 15, 'saved_best_epoch': 15, 'saved_validation_loss': 0.8173246371440724, 'recomputed_validation_loss': 0.8173246085643768, 'delta': -2.85796956100981e-08, 'train_magnitude_center': 3.860833168029785, 'train_magnitude_scale': 0.9051623344421387, 'training_band_counts': {'below3': 904, '3to4': 2292, '4to5': 1833, 'atleast5': 611}} |
| 2s_reported_metrics | PASS | {'recomputed_minus_reported': {'accuracy': 1.1565468815977198e-07, 'precision': -2.8622540250156092e-06, 'recall': -4.656172328143526e-06, 'f1': -1.5053763440242562e-06, 'roc_auc': 3.2579448003833278e-06, 'pr_auc_ap': 1.8182513870090844e-06, 'mae': -3.0470780291169675e-06, 'rmse': 5.990149082313323e-07, 'r2': -3.1081471225102852e-06}, 'tolerance': '5.1e-6 for five-decimal report; float64 independent regression aggregation', 'selected_row_identity_sha256': 'd80c037a136cb3577d88f3c0b9b5f17c8b3fceefc7722c097ca98af8b8a73b67', 'excluded_source_ids': ['NCS_EV_02506', 'NCS_EV_02561', 'NCS_EV_04805', 'NCS_EV_04848', 'NCS_EV_05056', 'NCS_EV_05065', 'NCS_EV_05658']} |
| checkpoint_3s_identity | PASS | {'path': 'C:\\Users\\choud\\Desktop\\Minor project\\cnn\\cnn wave\\eew_causal_v3\\cnn_eew_training\\models\\cnn_3s_best.pt', 'sha256': 'f7e7617552bc35cf5620f12b32e53bf2232efaf0d1a14e345cf220eb64ce287a', 'selected_epoch': 14} |
| 3s_validation_selection_and_training_scaler | PASS | {'history_best_epoch': 14, 'saved_best_epoch': 14, 'saved_validation_loss': 0.7538920510161912, 'recomputed_validation_loss': 0.7538920044898987, 'delta': -4.652629248536044e-08, 'train_magnitude_center': 3.865957498550415, 'train_magnitude_scale': 0.9047956466674805, 'training_band_counts': {'below3': 886, '3to4': 2269, '4to5': 1829, 'atleast5': 609}} |
| 3s_reported_metrics | PASS | {'recomputed_minus_reported': {'accuracy': -2.8666114333386616e-06, 'precision': -4.683882457667821e-06, 'recall': 3.3333333332441484e-06, 'f1': 9.81489453266704e-07, 'roc_auc': 4.052718286740031e-06, 'pr_auc_ap': -2.2236269680009357e-06, 'mae': -4.795203208951548e-06, 'rmse': 2.2528451373293734e-06, 'r2': 2.8117326790755293e-06}, 'tolerance': '5.1e-6 for five-decimal report; float64 independent regression aggregation', 'selected_row_identity_sha256': 'c47a7e1a16cafbea9dff96813a0ee7bf720e7ea2eed9e85e34b06f42f776dbcb', 'excluded_source_ids': ['NCS_EV_02506', 'NCS_EV_02523', 'NCS_EV_02561', 'NCS_EV_02912', 'NCS_EV_03078', 'NCS_EV_03626', 'NCS_EV_04353', 'NCS_EV_04805', 'NCS_EV_04848', 'NCS_EV_05056', 'NCS_EV_05065', 'NCS_EV_05658', 'NCS_EV_05775', 'NCS_EV_06475']} |
| cutoff_and_preS_interface | PASS | {'parent_rows_checked': 12, 'slice_comparisons': 36, 'max_difference': 0.0, 'last_sample_relative_P_seconds': {1: 0.99, 2: 1.99, 3: 2.99}, 'all_evaluated_EQ_preS': 'Verified by metadata masks and positive continuous S-minus-cutoff intervals'} |
| test_selection_protocol | PASS | Inspected 01_cnn_training.ipynb cells1–2: optimization reads train, selection reads validation, test accessed only for schema/hashing. Threshold .5 constant; target scaler recomputed from training only. All saved epochs reproduced from validation history and losses from frozen validation inference. 02 evaluation gates on all checkpoints existing. This verifies saved code/artifact consistency, not an independently authenticated historical execution log. |
| historical_execution_provenance | UNKNOWN | No external immutable execution log or saved original per-event predictions. Frozen recomputation and source/checkpoint consistency support reported protocol; unrecorded manual historical actions cannot be excluded. |
| timing_claim_scope | PASS | Original evaluation and this audit use identical warmup/repeat/batch/synchronization rules; latency naturally varies with device load. S-minus-cutoff is correctly called same-station interval, not operational lead time. Oracle P constructs windows; no autonomous streaming P picker is implemented. |
| noise_family_metadata | PASS | {'recorded_family_class_counts': {'0:colored_pink': 306, '0:gaussian_bandlimited': 292, '0:modulated_nonstationary': 302, '0:transient_disturbances': 314, '1:gaussian_bandlimited': 1214}, 'interpretation': 'Family labels present. Positive records are gaussian_bandlimited, negatives span four families. Lag/spectral statistics can encode family despite near-zero mean and unit variance. Family-dependent FPR and above-chance zeroed-post-P AUC support possible shortcuts, not future leakage.'} |
| causal_explanation_of_underprediction | UNKNOWN | Training magnitude imbalance, prediction compression and magnitude-dependent signed bias are measured associations. Source amplitude/distance/noise ambiguity and model/loss effects are possible; no controlled intervention here separates their causal contributions. |
| original_file_preservation | PASS | {'files': 11, 'scope': 'All opened original reports/notebooks/checkpoints/HDF5/manifest/readiness files unchanged; writes confined to new audit directory.'} |

## Independently recomputed test metrics

Rebuilt the architecture without executing training/evaluation cells. Loaded each horizon checkpoint strictly and inferred once per original test cohort; common-cohort metrics reuse those predictions. No original prediction archive was available. Detection counts are independently calculated; regression uses float64 arithmetic. Exact unrounded values and signed discrepancies against every reported five-decimal metric are in JSON; no discrepancy is hidden by rounding. PR-AUC means average precision.

| Horizon | N / EQ | Accuracy | Precision | Recall | F1 | ROC-AUC | AP | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | 2428 / 1214 | 0.663097199 | 0.634146341 | 0.771004942 | 0.695910781 | 0.751105309 | 0.752867251 | 0.673678035 | 0.832154782 | 0.145845415 |
| 2s | 2421 / 1207 | 0.725320116 | 0.742397138 | 0.687655344 | 0.713978495 | 0.826433258 | 0.847111818 | 0.640156953 | 0.789080599 | 0.230836892 |
| 3s | 2414 / 1200 | 0.766777133 | 0.783615316 | 0.733333333 | 0.757640981 | 0.858474053 | 0.882497776 | 0.613005205 | 0.757702253 | 0.289802812 |

## Common-cohort comparison

Exactly 1200 earthquakes and 1214 noise records (intersection of all horizon classification masks), identified by saved source/parent IDs and row-matched manifest. Magnitude uses only its earthquake subset; detection includes all common records.

| Horizon | Accuracy | F1 | MAE | RMSE | R² |
|---|---:|---:|---:|---:|---:|
| 1s | 0.66114333 | 0.69271225 | 0.66502001 | 0.82189444 | 0.16437028 |
| 2s | 0.72452361 | 0.71224578 | 0.63529553 | 0.78316398 | 0.24127009 |
| 3s | 0.76677713 | 0.75764098 | 0.61300520 | 0.75770225 | 0.28980281 |

Pairwise common-cohort differences (later minus earlier):

```json
{
  "2s_minus_1s": {
    "accuracy": 0.06338028169014087,
    "f1": 0.019533534615911008,
    "mae": -0.02972447971502934,
    "rmse": -0.0387304588473798,
    "r2": 0.07689981203759799
  },
  "3s_minus_2s": {
    "accuracy": 0.04225352112676051,
    "f1": 0.04539520044228751,
    "mae": -0.022290324767430647,
    "rmse": -0.02546172836261662,
    "r2": 0.04853271759900879
  },
  "3s_minus_1s": {
    "accuracy": 0.10563380281690138,
    "f1": 0.06492873505819852,
    "mae": -0.052014804482459986,
    "rmse": -0.06419218720999642,
    "r2": 0.12543252963660678
  }
}
```

These comparisons remove test-population differences. They compare three separately trained frozen models, not an isolated causal effect of duration; one seed and no uncertainty/significance analysis.

## Magnitude bias and compression

| Horizon | Band | N | MAE | Signed error | Underprediction % | Actual range | Predicted range |
|---|---|---:|---:|---:|---:|---|---|
| 1s | below3 | 195 | 1.035839 | 1.035839 | 0.000 | [2.0, 2.9000000953674316] | [3.36438250541687, 4.4527268409729] |
| 1s | 3to4 | 495 | 0.328942 | 0.250389 | 24.444 | [3.0, 3.9000000953674316] | [3.2523856163024902, 5.686954975128174] |
| 1s | 4to5 | 393 | 0.689508 | -0.628854 | 92.875 | [4.0, 4.900000095367432] | [3.360381603240967, 5.857227325439453] |
| 1s | atleast5 | 131 | 1.389720 | -1.381391 | 97.710 | [5.0, 7.199999809265137] | [3.4786057472229004, 6.801295280456543] |
| 2s | below3 | 192 | 0.966726 | 0.966726 | 0.000 | [2.0, 2.9000000953674316] | [3.3563308715820312, 4.013589382171631] |
| 2s | 3to4 | 491 | 0.329562 | 0.225957 | 29.124 | [3.0, 3.9000000953674316] | [3.330890655517578, 5.336948394775391] |
| 2s | 4to5 | 393 | 0.683581 | -0.541603 | 84.733 | [4.0, 4.900000095367432] | [3.3808720111846924, 5.759731292724609] |
| 2s | atleast5 | 131 | 1.195388 | -1.162961 | 94.656 | [5.0, 7.199999809265137] | [3.437469244003296, 7.094269752502441] |
| 3s | below3 | 190 | 0.923304 | 0.923304 | 0.000 | [2.0, 2.9000000953674316] | [3.2762322425842285, 4.037834167480469] |
| 3s | 3to4 | 487 | 0.312724 | 0.182445 | 31.211 | [3.0, 3.9000000953674316] | [3.2904162406921387, 4.614757537841797] |
| 3s | 4to5 | 392 | 0.648638 | -0.562984 | 86.480 | [4.0, 4.900000095367432] | [3.329251289367676, 5.3301777839660645] |
| 3s | atleast5 | 131 | 1.172639 | -1.164218 | 96.947 | [5.0, 7.199999809265137] | [3.356067657470703, 6.564357757568359] |

1s: predicted/actual SD=0.4238, slope=0.1671, train mean=3.8585, prediction mean=3.7865; 93.57% predictions within ±0.5 of training mean. Training band counts: {'below3': 910, '3to4': 2305, '4to5': 1836, 'atleast5': 611}.

2s: predicted/actual SD=0.5506, slope=0.2690, train mean=3.8608, prediction mean=3.8191; 88.57% predictions within ±0.5 of training mean. Training band counts: {'below3': 904, '3to4': 2292, '4to5': 1833, 'atleast5': 611}.

3s: predicted/actual SD=0.5137, slope=0.2820, train mean=3.8660, prediction mean=3.7895; 85.50% predictions within ±0.5 of training mean. Training band counts: {'below3': 886, '3to4': 2269, '4to5': 1829, 'atleast5': 609}.

The strong-event percentages independently verify the reported 94.7–97.7% underprediction. Predictions are compressed toward central training magnitudes. Large-magnitude records are a minority and errors vary systematically by magnitude. This demonstrates model bias and imbalance, but does not prove imbalance caused the bias. Simulator/source identifiability and loss/architecture limitations remain hypotheses, not verified causal explanations.

## Noise-family and pre-P audit

| Horizon | Label:family | N | Positive fraction (noise=FPR) | Confidence q05 / median / q95 |
|---|---|---:|---:|---|
| 1s | 0:colored_pink | 306 | 0.019608 | 0.00001 / 0.00404 / 0.35362 |
| 1s | 0:gaussian_bandlimited | 292 | 0.636986 | 0.34960 / 0.52678 / 0.75080 |
| 1s | 1:gaussian_bandlimited | 1214 | 0.771005 | 0.40190 / 0.60764 / 0.98011 |
| 1s | 0:modulated_nonstationary | 302 | 0.596026 | 0.36313 / 0.53606 / 0.75885 |
| 1s | 0:transient_disturbances | 314 | 0.535032 | 0.29327 / 0.50860 / 0.77791 |
| 2s | 0:colored_pink | 306 | 0.006536 | 0.00007 / 0.00449 / 0.22929 |
| 2s | 0:gaussian_bandlimited | 292 | 0.325342 | 0.18956 / 0.41037 / 0.75348 |
| 2s | 1:gaussian_bandlimited | 1207 | 0.687655 | 0.24584 / 0.68350 / 1.00000 |
| 2s | 0:modulated_nonstationary | 302 | 0.317881 | 0.17873 / 0.41947 / 0.75016 |
| 2s | 0:transient_disturbances | 314 | 0.302548 | 0.13127 / 0.38942 / 0.70297 |
| 3s | 0:colored_pink | 306 | 0.006536 | 0.00013 / 0.00583 / 0.09300 |
| 3s | 0:gaussian_bandlimited | 292 | 0.291096 | 0.19684 / 0.38089 / 0.72520 |
| 3s | 1:gaussian_bandlimited | 1200 | 0.733333 | 0.22534 / 0.80887 / 1.00000 |
| 3s | 0:modulated_nonstationary | 302 | 0.278146 | 0.17121 / 0.36896 / 0.73902 |
| 3s | 0:transient_disturbances | 314 | 0.229299 | 0.12579 / 0.34728 / 0.70724 |

Pre-P context is the first 100 samples. Recorded positive backgrounds all have the gaussian_bandlimited label, whereas negatives span four recorded families. Labels are taken from HDF5, never inferred. Descriptive distributions of raw/normalized mean, SD, lag-one correlation and 1–5 Hz power fraction appear in JSON.

| Label:family | Raw SD median | Normalized SD median | Normalized lag1 median | 1–5Hz fraction median |
|---|---:|---:|---:|---:|
| 0:colored_pink | 0.042435 | 1.000000 | 0.843433 | 0.588697 |
| 0:gaussian_bandlimited | 0.050063 | 1.000000 | 0.606497 | 0.191732 |
| 0:modulated_nonstationary | 0.049065 | 1.000000 | 0.614570 | 0.202889 |
| 0:transient_disturbances | 0.049880 | 1.000000 | 0.609721 | 0.203042 |
| 1:gaussian_bandlimited | 0.049853 | 1.000000 | 0.615184 | 0.197688 |

Pre-P normalization residuals: {'max_abs_mean': 2.772663719952106e-07, 'max_abs_std_minus1': 4.361544844178056e-07}.

Single-statistic pre-P AUC (positive orientation, no fitting): {'mean': {'all_classes_auc_positive_direction': 0.5077972120971966, 'gaussian_family_only_auc_positive_direction': 0.4936175554602695}, 'std': {'all_classes_auc_positive_direction': 0.5547314553710283, 'gaussian_family_only_auc_positive_direction': 0.5198229559251654}, 'lag1_correlation': {'all_classes_auc_positive_direction': 0.3846373582232548, 'gaussian_family_only_auc_positive_direction': 0.5327853129019882}, 'power_fraction_1to5Hz': {'all_classes_auc_positive_direction': 0.37412233443434506, 'gaussian_family_only_auc_positive_direction': 0.5138199318453657}}.

Frozen post-P-zeroed AUC: {1: {'post_P_zeroed_roc_auc': 0.6060099226758655}, 2: {'post_P_zeroed_roc_auc': 0.6195961504076304}, 3: {'post_P_zeroed_roc_auc': 0.5918005216913784}}.

Mean/variance normalization does not erase temporal/spectral family differences. Large family-conditioned FPR differences plus residual pre-P discrimination support an artificial-background shortcut risk. They do not prove acausal normalization or direct target leakage; zeroing post-P is itself out of distribution. No family-matched retraining or generator manipulation was performed.

## Timing and information cutoff

- 1s: same-station S-minus-cutoff min/median/max=0.063196/19.423283/78.044234 s; remeasured model-only latency median/p95=2.2457/3.7538 ms.
- 2s: same-station S-minus-cutoff min/median/max=0.096895/18.627559/77.044234 s; remeasured model-only latency median/p95=1.9336/2.7937 ms.
- 3s: same-station S-minus-cutoff min/median/max=0.267722/17.743726/76.044234 s; remeasured model-only latency median/p95=2.3165/4.0575 ms.

Both latency procedures use 25 warmups, 200 batch-one synchronized forwards on preloaded input. Timing values are workload-dependent, so numerical equality is not required. Acquisition, HDF5 IO, transfer and picking are excluded. Window samples end at P+h−0.01 s; the conservative cutoff is P+h. All evaluated earthquakes pass saved/reconstructed pre-S masks. Oracle P is a window alignment reference, never a feature; no continuous-stream picking capability is demonstrated.

## Leakage and scientific interpretation

Saved code, checkpoint history, train-only target scaling and identity-disjoint manifests show no holdout training or checkpoint-selection leakage. All checkpoints existed before original test evaluation; classification threshold is fixed at 0.5. Historical execution outside the saved code cannot be cryptographically proven. Conditional masks use privileged S metadata to define the research benchmark and cannot serve as deployed eligibility rules.

Original horizon-specific metrics use different populations and are correctly labeled; use the common-cohort table for cross-horizon claims. For future tabular BiLSTM comparisons, require the same event IDs, splits, horizons, eligible masks and same-event targets, plus verified availability of every tabular feature by the cutoff. Historical next-event forecasting targets are not comparable. Do not use labels, event IDs, future S times or oracle timing as model features. Keep noise sentinel excluded from magnitude loss.

The present models are valid but limited synthetic baselines. Approval is for honest reporting and controlled comparison, not accuracy sufficiency, real-world generalization or operational warning. No retraining is recommended or performed by this audit.

## Evidence paths

- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\final_cnn_eew_report.md`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\01_cnn_training.ipynb`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\02_cnn_evaluation.ipynb`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\configs\split_manifest.csv`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\codex_final_verification\hdf5_training_readiness.md`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\dataset\eew_dataset_train.h5`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\dataset\eew_dataset_validation.h5`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\dataset\eew_dataset_test.h5`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\models\cnn_1s_best.pt`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\models\cnn_2s_best.pt`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\models\cnn_3s_best.pt`

Original-file before/after hashes and checkpoint hashes are saved in the JSON. Only this notebook, report and JSON were created. No figures, dataset copies, training, cleanup or waveform regeneration.
