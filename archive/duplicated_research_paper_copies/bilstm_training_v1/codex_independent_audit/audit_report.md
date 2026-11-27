# Independent BiLSTM performance audit

**Date:** 2026-11-13. **Verdict: APPROVED**

Verified standalone synthetic benchmark reporting and controlled comparison; suitable baseline for a separately designed fusion experiment, not evidence that fusion improves performance or authorization to begin it.

## Verification

Independent model reconstruction and read-only frozen checkpoint inference on validation/test; no training or original notebook execution. Exact numerical report JSON reused as comparison values, not accepted as proof. No standalone prediction archive existed. Feature HDF5 was read only as the required inference/mask/scaler dependency. The prior feature-extraction validation is reused, not repeated.

| Check | Status | Evidence |
|---|---|---|
| 1s_checkpoint | PASS | {'path': 'C:\\Users\\choud\\Desktop\\Minor project\\bilstm_eew\\bilstm_training_v1\\models\\bilstm_1s_best.pt', 'sha256': '6e58d0005b1f464345ec64b488197b371607c232e4a786919d3f9ba46bcb34ad'} |
| train_1s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| validation_1s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| test_1s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| 1s_feature_scaler | PASS | Checkpoint parameters exactly match approved HDF5 train-only scaler; no refit in training code. Reliance on prior feature validation explicitly retained. |
| 1s_target_scaler | PASS | {'training_mean': 3.8584954738616943, 'training_sd': 0.9050084948539734} |
| 1s_selection | PASS | {'epoch': 18, 'validation_loss_delta': 4.4126109655096e-08, 'saved_threshold': 0.47, 'validation_reproduced_threshold': 0.47, 'validation_f1': 0.7303980699638118} |
| 1s_common_cohort | PASS | {'earthquakes': 1200, 'noise': 1214, 'ordered_source_id_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'ordered_parent_id_sha256': '6f42c966174ffef247ed80442342e89e2fa2a93d64d3f729312eb002327b804f'} |
| 1s_detection_confusion | PASS | [[315, 899], [1, 1213]] |
| 1s_fixed_05_detection_confusion | PASS | [[320, 894], [3, 1211]] |
| 1s_strong_bias | PASS | {'n': 131, 'underprediction_percent': 97.70992366412213, 'mae': 1.4185908659723878, 'bias': -1.4095619434618767} |
| 1s_full_precision_metrics | PASS | {'max_abs_difference': 0.0, 'tolerance': 1e-06} |
| 2s_checkpoint | PASS | {'path': 'C:\\Users\\choud\\Desktop\\Minor project\\bilstm_eew\\bilstm_training_v1\\models\\bilstm_2s_best.pt', 'sha256': '2b31d158df7b866b1049fd679dace6bfb26038f21d4f5b3ee0a0f4df0e94e681'} |
| train_2s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| validation_2s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| test_2s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| 2s_feature_scaler | PASS | Checkpoint parameters exactly match approved HDF5 train-only scaler; no refit in training code. Reliance on prior feature validation explicitly retained. |
| 2s_target_scaler | PASS | {'training_mean': 3.860833168029785, 'training_sd': 0.9051623344421387} |
| 2s_selection | PASS | {'epoch': 27, 'validation_loss_delta': -2.9009174173921792e-08, 'saved_threshold': 0.36, 'validation_reproduced_threshold': 0.36, 'validation_f1': 0.7308507417499243} |
| 2s_common_cohort | PASS | {'earthquakes': 1200, 'noise': 1214, 'ordered_source_id_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'ordered_parent_id_sha256': '6f42c966174ffef247ed80442342e89e2fa2a93d64d3f729312eb002327b804f'} |
| 2s_detection_confusion | PASS | [[316, 898], [1, 1206]] |
| 2s_fixed_05_detection_confusion | PASS | [[905, 309], [479, 728]] |
| 2s_strong_bias | PASS | {'n': 131, 'underprediction_percent': 93.12977099236642, 'mae': 1.2312072906785338, 'bias': -1.2026304106675942} |
| 2s_full_precision_metrics | PASS | {'max_abs_difference': 0.0, 'tolerance': 1e-06} |
| 3s_checkpoint | PASS | {'path': 'C:\\Users\\choud\\Desktop\\Minor project\\bilstm_eew\\bilstm_training_v1\\models\\bilstm_3s_best.pt', 'sha256': '28a40646277f114e9a9bbb25763f765dadb47b6cfff602e250d4d8b5b743c1d0'} |
| train_3s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| validation_3s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| test_3s_masks | PASS | Existing pre-S masks reused; only eligible rows supplied; no beyond-cutoff feature bins. Original arrival/feature audits not repeated. |
| 3s_feature_scaler | PASS | Checkpoint parameters exactly match approved HDF5 train-only scaler; no refit in training code. Reliance on prior feature validation explicitly retained. |
| 3s_target_scaler | PASS | {'training_mean': 3.865957498550415, 'training_sd': 0.9047956466674805} |
| 3s_selection | PASS | {'epoch': 26, 'validation_loss_delta': -3.162160400194125e-08, 'saved_threshold': 0.36, 'validation_reproduced_threshold': 0.36, 'validation_f1': 0.7412140575079872} |
| 3s_common_cohort | PASS | {'earthquakes': 1200, 'noise': 1214, 'ordered_source_id_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'ordered_parent_id_sha256': '6f42c966174ffef247ed80442342e89e2fa2a93d64d3f729312eb002327b804f'} |
| 3s_detection_confusion | PASS | [[408, 806], [43, 1157]] |
| 3s_fixed_05_detection_confusion | PASS | [[939, 275], [415, 785]] |
| 3s_strong_bias | PASS | {'n': 131, 'underprediction_percent': 96.18320610687023, 'mae': 1.1527157521429863, 'bias': -1.1346164987287448} |
| 3s_full_precision_metrics | PASS | {'max_abs_difference': 0.0, 'tolerance': 1e-06} |
| train/validation_identity_isolation | PASS | No source/parent identity overlap in feature metadata. |
| train/test_identity_isolation | PASS | No source/parent identity overlap in feature metadata. |
| validation/test_identity_isolation | PASS | No source/parent identity overlap in feature metadata. |
| saved_protocol_no_test_tuning | PASS | Training notebook optimization uses train; epoch/threshold selection uses validation only. Test inference appears only after all three frozen checkpoints exist. Grid/tie rule reproduced independently; no new thresholds adopted. Target metadata is outside model.forward. This verifies artifacts and saved code, not unrecorded historical actions. |
| historical_execution_certainty | UNKNOWN | No immutable external execution log. Existing test set was already used in prior CNN research, so this is exploratory comparison rather than fresh confirmatory evaluation. |
| noise_family_shortcuts | UNKNOWN | Feature file lacks noise-family labels. No labels fabricated; upstream dataset/feature validation not rerun. |
| preservation | PASS | {'files': 9, 'changed': []} |

## Horizon-eligible detection at both thresholds

FPR=FP/(FP+TN). Magnitude metrics do not condition on successful detection.

| Horizon | Threshold | Precision | Recall | F1 | FPR | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 1s | 0.50 | 0.57529691 | 0.99752883 | 0.72973787 | 0.73640857 | 0.63056013 |
| 1s | 0.47 | 0.57433712 | 0.99917628 | 0.72940469 | 0.74052718 | 0.62932455 |
| 2s | 0.50 | 0.70202507 | 0.60314830 | 0.64884135 | 0.25453048 | 0.67451466 |
| 2s | 0.36 | 0.57319392 | 0.99917150 | 0.72848082 | 0.73970346 | 0.62866584 |
| 3s | 0.50 | 0.74056604 | 0.65416667 | 0.69469027 | 0.22652389 | 0.71416736 |
| 3s | 0.36 | 0.58940397 | 0.96416667 | 0.73158394 | 0.66392092 | 0.64830157 |

## Magnitude and compression

| Horizon | MAE | RMSE | R2 | Pred/actual SD | M>=5 N | Underprediction % | Strong bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1s | 0.678041925 | 0.833247047 | 0.143601660 | 0.35521 | 131 | 97.70992 | -1.409562 |
| 2s | 0.638255536 | 0.785525194 | 0.237752600 | 0.48719 | 131 | 93.12977 | -1.202630 |
| 3s | 0.615081756 | 0.758420880 | 0.288455030 | 0.51604 | 131 | 96.18321 | -1.134616 |

## Identical CNN common cohort

Each horizon matches the audited ordered source-ID hash for exactly 1,200 earthquakes and 1,214 noise records. Parent-ID hashes are recorded too; source-parent association relies on unchanged validated feature provenance, not a rerun of waveform joins.

| Horizon | CNN F1 .5 | BiLSTM F1 .5 | BiLSTM F1 selected | CNN MAE | BiLSTM MAE |
|---|---:|---:|---:|---:|---:|
| 1s | 0.69271225 | 0.72743847 | 0.72710734 | 0.66502001 | 0.67021296 |
| 2s | 0.71224578 | 0.64663677 | 0.72732787 | 0.63529553 | 0.63412197 |
| 3s | 0.75764098 | 0.69469027 | 0.73158394 | 0.61300520 | 0.61508176 |

Full precision metrics, signed differences versus saved results, confusion matrices and both threshold policies on the common cohort are in audit_results.json. Differences up to 1e-6 are treated as numerical precision; exact threshold choices and confusion counts must agree.

## Interpretation and later fusion boundaries

All reported numbers are valid synthetic baseline results if the checks above pass; weak performance is not an evaluation-integrity failure. Validation-F1 selection produces high false-positive rates (approximately 66–74%). Strong-event underprediction and compressed variance persist. The BiLSTM is not uniformly superior to CNN: threshold policies must be matched, and tiny magnitude differences are not significance evidence. Feature representation, model capacity and budget also differ.

No test-driven optimization is present in saved code, and validation threshold selection reproduces without test labels. This cannot prove absence of unrecorded manual experimentation. The shared test cohort has already been viewed and must not guide future fusion architecture, calibration, threshold or checkpoint selection. For a later authorized fusion experiment, use existing train/validation roles, identical event/parent alignment, per-horizon masks and train-only transforms; define the protocol before any further test evaluation. Fresh evidence would be needed for strong confirmatory claims. No fusion was implemented or begun.

Noise-family shortcut status remains UNKNOWN with this feature-only interface. Both branches originate from the same synthetic observations; their errors may be correlated. No complementary information or fusion gain has been established. Operational trigger capability, real-data validity and warning lead time remain unverified. Bidirectional processing covers only the observed 20/30/40 bins. Stored pre-S masks are offline eligibility metadata, not model features or deployable gates.

Minor documentation issue: the source Markdown contains encoding artifacts in some symbols. The embedded numerical JSON is readable and metrics are unaffected; no protected report was repaired.

## Evidence and preservation

- `C:\Users\choud\Desktop\Minor project\bilstm_eew\causal_tabular_v1\tabular_features.h5` — SHA256 `8905cfa29770a3a59d52fe26b766c5060be8dfcccc99c7c13caeef14525f08fe`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\codex_performance_audit\cnn_performance_audit_results.json` — SHA256 `58cadae483be48911671653fb714d168fa801b81078fde373b4bd02e4c9c815d`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\final_bilstm_eew_report.md` — SHA256 `e6d42302de5fa8306f47003ebaa820d251e5a206bb13b31a3005a9d191741756`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\01_bilstm_training.ipynb` — SHA256 `2df085de20bd88b75ab8dfcdd21de7fc5be8680edaaad4c54a901f74617b7f18`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\02_bilstm_evaluation.ipynb` — SHA256 `33957d8c75fc0825746fedad06d42cef9e0b8076d9b1887b9a375745e6b195b6`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\causal_tabular_v1\feature_validation_report.md` — SHA256 `685028d898d7487729a8d67a007c065e9c8679e6e3d47bb38882e55d61fd13a8`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\models\bilstm_1s_best.pt` — SHA256 `6e58d0005b1f464345ec64b488197b371607c232e4a786919d3f9ba46bcb34ad`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\models\bilstm_2s_best.pt` — SHA256 `2b31d158df7b866b1049fd679dace6bfb26038f21d4f5b3ee0a0f4df0e94e681`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\models\bilstm_3s_best.pt` — SHA256 `28a40646277f114e9a9bbb25763f765dadb47b6cfff602e250d4d8b5b743c1d0`

Only audit.ipynb, audit_report.md and audit_results.json created. No retraining, tuning, new thresholds, dataset regeneration, figures, cleanup or changes to existing files.
