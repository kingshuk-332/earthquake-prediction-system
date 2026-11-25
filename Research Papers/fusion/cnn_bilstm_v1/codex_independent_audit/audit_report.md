# Independent CNN + BiLSTM fusion audit

**Date:** 2026-11-13. **Verdict: APPROVED**

Valid synthetic oracle-aligned fusion benchmark reporting and controlled comparison, not operational EEW or a claim of improved performance. Attention not implemented or started.

## Check results

| Check | Status | Evidence |
|---|---|---|
| train_pairing | PASS | {'records': 11324, 'mismatches': [], 'permutation_nonidentity_rows': 0} |
| train_1s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| train_1s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| train_2s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| train_2s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| train_3s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| train_3s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| validation_pairing | PASS | {'records': 2426, 'mismatches': [], 'permutation_nonidentity_rows': 0} |
| validation_1s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| validation_1s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| validation_2s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| validation_2s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| validation_3s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| validation_3s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| test_pairing | PASS | {'records': 2428, 'mismatches': [], 'permutation_nonidentity_rows': 0} |
| test_1s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| test_1s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| test_2s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| test_2s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| test_3s_masks_cutoff | PASS | Labels/first-S eligibility/noise sentinel; only observed horizon enters models. |
| test_3s_feature_scaling_metadata | PASS | Approved precomputed feature scaler provenance reused; feature extraction not rerun. |
| train/validation_parent_isolation | PASS | 0 |
| train/test_parent_isolation | PASS | 0 |
| validation/test_parent_isolation | PASS | 0 |
| 1s_backbone_checkpoint_preservation | PASS | {'source_hashes_match_training_snapshot': True, 'fusion_checkpoint': 'head-only weights; original backbone files used', 'scope': 'Saved training source uses inference-mode cached embeddings, eval frozen backbones, head-only optimizer and tensor-exact end checks. No historical optimizer trace exists beyond these artifacts.'} |
| 1s_train_only_fusion_scalers | PASS | {'embedding_mean_max_difference': 0.0, 'embedding_scale_max_difference': 0.0, 'target_center': 3.8584954738616943, 'target_scale': 0.9050084948539734} |
| 1s_validation_epoch_thresholds | PASS | {'epoch': 2, 'epochs_run': 9, 'validation_loss_delta': 1.1301788394035839e-09, 'threshold': 0.42, 'low_fpr_threshold': 0.6743693351745605, 'stale_epochs': 7} |
| 1s_common_cohort | PASS | {'source_order_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'parent_order_sha256': '6f42c966174ffef247ed80442342e89e2fa2a93d64d3f729312eb002327b804f', 'earthquakes': 1200, 'noise': 1214} |
| 1s_fixed_05_counts | PASS | [[686, 528], [310, 890]] |
| 1s_validation_selected_counts | PASS | [[347, 867], [20, 1180]] |
| 1s_validation_low_fpr_counts | PASS | [[1163, 51], [920, 280]] |
| 1s_fusion_metrics | PASS | {'max_difference': 0.0, 'tolerance': 1e-06} |
| 1s_CNN_policy | PASS | Identical parent-keyed records; fixed .5 for all three models. |
| 1s_paired_CNN | PASS | {'f1_difference_ci95': [-0.023792306807676085, -3.167606590949191e-05], 'mae_difference_ci95': [-0.0034688999352511015, 0.0017248650634428485]} |
| 1s_BiLSTM_policy | PASS | Identical parent-keyed records; fixed .5 for all three models. |
| 1s_paired_BiLSTM | PASS | {'f1_difference_ci95': [-0.06428662499380051, -0.03231501238265269], 'mae_difference_ci95': [-0.015197540121152996, 0.0032984839810524117]} |
| 2s_backbone_checkpoint_preservation | PASS | {'source_hashes_match_training_snapshot': True, 'fusion_checkpoint': 'head-only weights; original backbone files used', 'scope': 'Saved training source uses inference-mode cached embeddings, eval frozen backbones, head-only optimizer and tensor-exact end checks. No historical optimizer trace exists beyond these artifacts.'} |
| 2s_train_only_fusion_scalers | PASS | {'embedding_mean_max_difference': 0.0, 'embedding_scale_max_difference': 0.0, 'target_center': 3.860833168029785, 'target_scale': 0.9051623344421387} |
| 2s_validation_epoch_thresholds | PASS | {'epoch': 15, 'epochs_run': 22, 'validation_loss_delta': -2.5359695010607197e-08, 'threshold': 0.41, 'low_fpr_threshold': 0.7120206356048584, 'stale_epochs': 7} |
| 2s_common_cohort | PASS | {'source_order_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'parent_order_sha256': '6f42c966174ffef247ed80442342e89e2fa2a93d64d3f729312eb002327b804f', 'earthquakes': 1200, 'noise': 1214} |
| 2s_fixed_05_counts | PASS | [[966, 248], [392, 808]] |
| 2s_validation_selected_counts | PASS | [[790, 424], [248, 952]] |
| 2s_validation_low_fpr_counts | PASS | [[1159, 55], [642, 558]] |
| 2s_fusion_metrics | PASS | {'max_difference': 0.0, 'tolerance': 1e-06} |
| 2s_CNN_policy | PASS | Identical parent-keyed records; fixed .5 for all three models. |
| 2s_paired_CNN | PASS | {'f1_difference_ci95': [-0.0051605001703711765, 0.014231114089397819], 'mae_difference_ci95': [-0.004587645235005766, 0.005602469923906027]} |
| 2s_BiLSTM_policy | PASS | Identical parent-keyed records; fixed .5 for all three models. |
| 2s_paired_BiLSTM | PASS | {'f1_difference_ci95': [0.04497768064970756, 0.09491167683694436], 'mae_difference_ci95': [-0.0046189094544388356, 0.008094231225550173]} |
| 3s_backbone_checkpoint_preservation | PASS | {'source_hashes_match_training_snapshot': True, 'fusion_checkpoint': 'head-only weights; original backbone files used', 'scope': 'Saved training source uses inference-mode cached embeddings, eval frozen backbones, head-only optimizer and tensor-exact end checks. No historical optimizer trace exists beyond these artifacts.'} |
| 3s_train_only_fusion_scalers | PASS | {'embedding_mean_max_difference': 0.0, 'embedding_scale_max_difference': 0.0, 'target_center': 3.865957498550415, 'target_scale': 0.9047956466674805} |
| 3s_validation_epoch_thresholds | PASS | {'epoch': 27, 'epochs_run': 34, 'validation_loss_delta': -4.544512499649045e-08, 'threshold': 0.36, 'low_fpr_threshold': 0.6921972632408142, 'stale_epochs': 7} |
| 3s_common_cohort | PASS | {'source_order_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'parent_order_sha256': '6f42c966174ffef247ed80442342e89e2fa2a93d64d3f729312eb002327b804f', 'earthquakes': 1200, 'noise': 1214} |
| 3s_fixed_05_counts | PASS | [[1050, 164], [386, 814]] |
| 3s_validation_selected_counts | PASS | [[836, 378], [223, 977]] |
| 3s_validation_low_fpr_counts | PASS | [[1157, 57], [511, 689]] |
| 3s_fusion_metrics | PASS | {'max_difference': 0.0, 'tolerance': 1e-06} |
| 3s_CNN_policy | PASS | Identical parent-keyed records; fixed .5 for all three models. |
| 3s_paired_CNN | PASS | {'f1_difference_ci95': [-0.02269079653232397, -0.000400259480366555], 'mae_difference_ci95': [-0.011949946149252355, 0.0021605384710710472]} |
| 3s_BiLSTM_policy | PASS | Identical parent-keyed records; fixed .5 for all three models. |
| 3s_paired_BiLSTM | PASS | {'f1_difference_ci95': [0.031601839949213034, 0.0756103546455666], 'mae_difference_ci95': [-0.014054152299650013, 0.001418869048939083]} |
| saved_protocol_no_test_tuning | PASS | Reviewed training/evaluation cells: only training embeddings/scalers and validation losses/F1/negative-score quantiles select heads/thresholds; all three saved before test inference. Test bootstrap is descriptive only. No new threshold or training performed by audit. |
| historical_execution_certainty | UNKNOWN | Saved code, history, source hashes, scaler replay and frozen loss reproduction support protocol; no immutable independent log of every past action. Previously examined test cohort is exploratory, not untouched confirmatory evidence. |
| noise_family_shortcuts | UNKNOWN | No new family audit conducted; both branches share synthetic observations, so fusion does not establish independent information or shortcut removal. |
| preservation | PASS | {'original_files_hashed': 20, 'changed': []} |

## Frozen inference reproduction

Reused only backbone class declarations, independently implemented direct penultimate embedding extraction (without original forward hooks), parent-key joins, train-embedding scaler replay and frozen head inference. No original training/evaluation cells executed. Full-precision signed metric discrepancies are retained in JSON, including confusion matrices and strong-event bias. No feature generation or full dataset audit repeated.

| Horizon | Threshold policy | Precision | Recall | F1 | FPR | Accuracy | ROC-AUC | PR-AUC(AP) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1s | 0.5 | 0.62764457 | 0.74166667 | 0.67990833 | 0.43492586 | 0.65285833 | 0.74658361 | 0.74486519 |
| 1s | 0.42 | 0.57645335 | 0.98333333 | 0.72682476 | 0.71416804 | 0.63256007 | 0.74658361 | 0.74486519 |
| 1s | low-FPR 0.6743693351745605 | 0.84592145 | 0.23333333 | 0.36577400 | 0.04200988 | 0.59776305 | 0.74658361 | 0.74486519 |
| 2s | 0.5 | 0.76515152 | 0.67333333 | 0.71631206 | 0.20428336 | 0.73487987 | 0.83199616 | 0.84929952 |
| 2s | 0.41 | 0.69186047 | 0.79333333 | 0.73913043 | 0.34925865 | 0.72162386 | 0.83199616 | 0.84929952 |
| 2s | low-FPR 0.7120206356048584 | 0.91027732 | 0.46500000 | 0.61555433 | 0.04530478 | 0.71126761 | 0.83199616 | 0.84929952 |
| 3s | 0.5 | 0.83231084 | 0.67833333 | 0.74747475 | 0.13509061 | 0.77216239 | 0.86341502 | 0.88609770 |
| 3s | 0.36 | 0.72103321 | 0.81416667 | 0.76477495 | 0.31136738 | 0.75103563 | 0.86341502 | 0.88609770 |
| 3s | low-FPR 0.6921972632408142 | 0.92359249 | 0.57416667 | 0.70811922 | 0.04695222 | 0.76470588 | 0.86341502 | 0.88609770 |

| Horizon | MAE | RMSE | R2 | Bias | Pred/actual SD | Strong N | M>=5 underprediction % | Strong bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | 0.664239589 | 0.817357841 | 0.173569649 | -0.074875 | 0.39573 | 131 | 98.4733 | -1.364506 |
| 2s | 0.635878705 | 0.783323604 | 0.240960777 | -0.067043 | 0.48983 | 131 | 93.8931 | -1.218227 |
| 3s | 0.608215119 | 0.753054914 | 0.298488031 | -0.061756 | 0.56992 | 131 | 92.3664 | -1.047271 |

## Epoch 2 review

The 1s selected epoch2 was explicitly reviewed: replaying min_delta=0.0001 and patience7 selects epoch2 and ends at epoch9. Reloaded loss agrees within1e-6. Later epochs did not satisfy the recorded validation improvement rule. This is not a masking, checkpoint or selection error. Single-seed short head adaptation does not prove optimal convergence; no retraining or tuning is performed or recommended solely because the selected epoch is early. Full nine-epoch history is in JSON.

## Paired comparisons and interpretation

| Horizon | Baseline | Fusion-minus-baseline F1 [95% CI] | Fusion-minus-baseline MAE [95% CI] |
|---|---|---|
| 1s | CNN | -0.012804 [-0.023792306807676085, -3.167606590949191e-05] | -0.000780 [-0.0034688999352511015, 0.0017248650634428485] |
| 1s | BiLSTM | -0.047530 [-0.06428662499380051, -0.03231501238265269] | -0.005973 [-0.015197540121152996, 0.0032984839810524117] |
| 2s | CNN | 0.004066 [-0.0051605001703711765, 0.014231114089397819] | 0.000583 [-0.004587645235005766, 0.005602469923906027] |
| 2s | BiLSTM | 0.069675 [0.04497768064970756, 0.09491167683694436] | 0.001757 [-0.0046189094544388356, 0.008094231225550173] |
| 3s | CNN | -0.010166 [-0.02269079653232397, -0.000400259480366555] | -0.004790 [-0.011949946149252355, 0.0021605384710710472] |
| 3s | BiLSTM | 0.052784 [0.031601839949213034, 0.0756103546455666] | -0.006867 [-0.014054152299650013, 0.001418869048939083] |

The original 500-resample paired stratified bootstrap (seed2026) reproduces within numerical precision. Same event/noise draws are applied to both models; magnitude uses the paired earthquake draw. These are descriptive percentile intervals, not multiple-comparison-corrected or multi-seed confirmatory evidence. The 1s CNN F1 interval barely excludes zero and is particularly sensitive to bootstrap Monte Carlo uncertainty; do not overinterpret that boundary.

Fusion has no consistent advantage over CNN: its fixed-.5 F1 is lower at 1s/3s, its small 2s gain has an interval spanning zero, and every fusion-versus-CNN MAE interval spans zero. Magnitude compression and substantial M>=5 underprediction remain. Comparisons of validation-selected fusion F1 against fixed-.5 baselines are not equivalent policies; only the common fixed-.5 table supports that policy-matched comparison. Low-FPR thresholds are selected on validation and do not guarantee a fixed test FPR.

The common cohort matches exactly 1,200 earthquakes and 1,214 noise. All labels, magnitude targets, pre-S/regression masks and parent identities agree between modalities; source/parent metadata never enters embeddings. Fusion embedding and target scalers replay exactly from eligible training data. Approved tabular scaler provenance is reused rather than re-auditing extraction.

Backbone preservation is supported by original-checkpoint hashes matching fusion training snapshots, a head-only fusion checkpoint, cached inference-mode embeddings, head-only optimizer and original source tensor-equality guards. This audit loads unchanged weights; historical intermediate runtime states are not independently logged.

Both branches derive from the same synthetic observations and assume oracle P/reference alignment. Model capacity/representation/budget differences prevent an isolated architecture claim. No real-world EEW readiness, autonomous P picker, operational latency or warning lead time is demonstrated. Previously inspected test data must not be used to tune future Attention. No Attention, cleanup or further experiment began.

## Evidence hashes

- `C:\Users\choud\Desktop\Minor project\fusion\cnn_bilstm_v1\final_fusion_report.md` — `f9aa2d12a97cfa7edc3e231bf9bb7181bc0b044f747092b731ba0e30e2ca9fd7`
- `C:\Users\choud\Desktop\Minor project\fusion\cnn_bilstm_v1\01_fusion_training.ipynb` — `333b0b0bb04b01518a3ac85ea08b77a2f239ec2a9ab762247f6bafb2977fc318`
- `C:\Users\choud\Desktop\Minor project\fusion\cnn_bilstm_v1\02_fusion_evaluation.ipynb` — `9e93b36eb3e5c2fe4a44d15083fbb77e1fbab3a94e69dbe1a5c2d65c90c20e4a`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\causal_tabular_v1\tabular_features.h5` — `8905cfa29770a3a59d52fe26b766c5060be8dfcccc99c7c13caeef14525f08fe`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\01_cnn_training.ipynb` — `20003e98aa1718dc2d6f545b6353b203bb9e725114cf630667f4048bf6635e54`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\01_bilstm_training.ipynb` — `2df085de20bd88b75ab8dfcdd21de7fc5be8680edaaad4c54a901f74617b7f18`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\codex_performance_audit\cnn_performance_audit_results.json` — `58cadae483be48911671653fb714d168fa801b81078fde373b4bd02e4c9c815d`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\codex_independent_audit\audit_results.json` — `cb548791209a5b77cec7c05f66a75f10a9cc73d5b824370c09a67b88dc3b278f`
- `C:\Users\choud\Desktop\Minor project\fusion\cnn_bilstm_v1\models\fusion_1s_best.pt` — `e917397e277f48463d4396fa1f8a08e6b5f0007c7b136902267a88ed8d17f71a`
- `C:\Users\choud\Desktop\Minor project\fusion\cnn_bilstm_v1\models\fusion_2s_best.pt` — `0edb3c671ac65c3379b0f9c415b4b4d955ddb02e6057f8999d18bcbfe5b93f31`
- `C:\Users\choud\Desktop\Minor project\fusion\cnn_bilstm_v1\models\fusion_3s_best.pt` — `342835cac66598496aea8e67fa464e5f6275fa69e7eec054318339bc1c00ed4a`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\models\cnn_1s_best.pt` — `43b2099c8a3f64e94a528ca32eb3874828bc5a0d3cc0b1c18b63d67c298399b4`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\models\bilstm_1s_best.pt` — `6e58d0005b1f464345ec64b488197b371607c232e4a786919d3f9ba46bcb34ad`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\models\cnn_2s_best.pt` — `cf414ddd4b13e61466b407dc7d5e2f998e3ee6f5d53f973b8e214103cac667d5`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\models\bilstm_2s_best.pt` — `2b31d158df7b866b1049fd679dace6bfb26038f21d4f5b3ee0a0f4df0e94e681`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\models\cnn_3s_best.pt` — `f7e7617552bc35cf5620f12b32e53bf2232efaf0d1a14e345cf220eb64ce287a`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\models\bilstm_3s_best.pt` — `28a40646277f114e9a9bbb25763f765dadb47b6cfff602e250d4d8b5b743c1d0`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\dataset\eew_dataset_train.h5` — `2f88d8c1b50c1d52d0814360d7a32beabb5c90a822e121940a0db875ad49be15`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\dataset\eew_dataset_validation.h5` — `c983d0fa42aabf107204b1e7261e8f0b4ee93a86aca64fdf7ce0a270877ec1d1`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\dataset\eew_dataset_test.h5` — `9db5018d5cabf51a417d17c92344e4184148b87264361d538cf962f350184d43`

Only audit.ipynb, audit_report.md and audit_results.json were created. Existing inputs/checkpoints/notebooks/reports unchanged.
