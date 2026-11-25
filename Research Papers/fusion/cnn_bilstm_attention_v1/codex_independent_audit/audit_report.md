# Independent temporal Attention audit

**Verdict: APPROVED** — synthetic oracle-aligned research reporting only.

## Evidence

Only original backbone class declarations reused. Attention inference independently implemented with functional linear/tanh/softmax/weighted-sum operations; original Attention forward and evaluation functions not executed. Exact parent joins, frozen checkpoint replay, training-moment recomputation and validation selection replay performed. Existing validated tabular feature-scaler provenance reused; preprocessing audit not repeated.

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
| 1s_original_training_dependencies | PASS | All saved training dependency hashes match, including frozen backbones and datasets. |
| 1s_train_fitted_scalers | PASS | {'cmean': 0.0, 'cscale': 0.0, 'smean': 0.0, 'sscale': 0.0} |
| 1s_validation_selection | PASS | {'epoch': 3, 'epochs_run': 10, 'loss_delta': 4.913820461638352e-10, 'threshold': 0.28, 'stale_epochs': 7} |
| 1s_common_cohort | PASS | {'source_order_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'earthquakes': 1200, 'noise': 1214} |
| 1s_temporal_attention | PASS | {'shape': [2414, 20], 'minimum': 0.0017231086967512965, 'maximum': 0.5373079180717468, 'max_sum_error': 2.384185791015625e-07, 'mean_entropy': 2.8978333473205566, 'mean_by_bin': [0.06772282719612122, 0.06350068747997284, 0.06162021681666374, 0.06125397980213165, 0.061122968792915344, 0.06088811904191971, 0.06020008772611618, 0.05927509069442749, 0.05784759297966957, 0.056113362312316895, 0.05372684821486473, 0.05099615082144737, 0.04782627522945404, 0.0440465472638607, 0.039983149617910385, 0.03587128221988678, 0.03206879273056984, 0.029146980494260788, 0.027851495891809464, 0.02893754094839096]} |
| 1s_fixed_05_confusion | PASS | [[768, 446], [367, 833]] |
| 1s_validation_selected_confusion | PASS | [[314, 900], [1, 1199]] |
| 1s_metrics_reproduced | PASS | {'maximum_absolute_difference': 0.0} |
| 1s_paired_fusion | PASS | {'f1_difference_ci95': [-0.018330848456274, 0.001981389353165757], 'mae_difference_ci95': [-0.0012190080509753896, 0.0054234042880125346]} |
| 2s_original_training_dependencies | PASS | All saved training dependency hashes match, including frozen backbones and datasets. |
| 2s_train_fitted_scalers | PASS | {'cmean': 0.0, 'cscale': 0.0, 'smean': 0.0, 'sscale': 0.0} |
| 2s_validation_selection | PASS | {'epoch': 17, 'epochs_run': 24, 'loss_delta': -7.387565448269129e-09, 'threshold': 0.41, 'stale_epochs': 7} |
| 2s_common_cohort | PASS | {'source_order_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'earthquakes': 1200, 'noise': 1214} |
| 2s_temporal_attention | PASS | {'shape': [2414, 30], 'minimum': 0.006347933784127235, 'maximum': 0.14386127889156342, 'max_sum_error': 2.384185791015625e-07, 'mean_entropy': 3.3388986587524414, 'mean_by_bin': [0.034181032329797745, 0.036955464631319046, 0.038333818316459656, 0.039078038185834885, 0.039408065378665924, 0.03944091126322746, 0.03927401453256607, 0.039034660905599594, 0.0387437641620636, 0.0383807010948658, 0.03802083805203438, 0.03765295445919037, 0.037233226001262665, 0.03675151243805885, 0.03617092967033386, 0.03545841574668884, 0.03462877869606018, 0.03366730734705925, 0.03257378563284874, 0.03135201334953308, 0.03001449629664421, 0.02865603379905224, 0.027333032339811325, 0.026104198768734932, 0.02509211003780365, 0.024436283856630325, 0.024193547666072845, 0.02452157996594906, 0.025671884417533875, 0.027636433020234108]} |
| 2s_fixed_05_confusion | PASS | [[967, 247], [391, 809]] |
| 2s_validation_selected_confusion | PASS | [[785, 429], [249, 951]] |
| 2s_metrics_reproduced | PASS | {'maximum_absolute_difference': 0.0} |
| 2s_paired_fusion | PASS | {'f1_difference_ci95': [-0.003969822205396578, 0.005982555642963445], 'mae_difference_ci95': [-0.0018039557384327053, 0.002919959701830521]} |
| 3s_original_training_dependencies | PASS | All saved training dependency hashes match, including frozen backbones and datasets. |
| 3s_train_fitted_scalers | PASS | {'cmean': 0.0, 'cscale': 0.0, 'smean': 0.0, 'sscale': 0.0} |
| 3s_validation_selection | PASS | {'epoch': 32, 'epochs_run': 35, 'loss_delta': 6.868284163630278e-09, 'threshold': 0.41, 'stale_epochs': 3} |
| 3s_common_cohort | PASS | {'source_order_sha256': '862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601', 'earthquakes': 1200, 'noise': 1214} |
| 3s_temporal_attention | PASS | {'shape': [2414, 40], 'minimum': 0.0037591434083878994, 'maximum': 0.10318470746278763, 'max_sum_error': 1.7881393432617188e-07, 'mean_entropy': 3.638434886932373, 'mean_by_bin': [0.020057033747434616, 0.022769922390580177, 0.024680867791175842, 0.026150725781917572, 0.027299221605062485, 0.028172247111797333, 0.028807613998651505, 0.029232796281576157, 0.029489221051335335, 0.02961382083594799, 0.029546022415161133, 0.029399722814559937, 0.029205722734332085, 0.028917105868458748, 0.028563745319843292, 0.028167329728603363, 0.027726173400878906, 0.02721422351896763, 0.026705356314778328, 0.0261541735380888, 0.025545351207256317, 0.024926362559199333, 0.024322593584656715, 0.023739799857139587, 0.023172900080680847, 0.022654255852103233, 0.02218114398419857, 0.021767325699329376, 0.021407583728432655, 0.02111164480447769, 0.02091832645237446, 0.020836181938648224, 0.02088187262415886, 0.021045733243227005, 0.021323222666978836, 0.021751387044787407, 0.022387977689504623, 0.023247621953487396, 0.0241392869502306, 0.02476630173623562]} |
| 3s_fixed_05_confusion | PASS | [[1075, 139], [401, 799]] |
| 3s_validation_selected_confusion | PASS | [[951, 263], [312, 888]] |
| 3s_metrics_reproduced | PASS | {'maximum_absolute_difference': 0.0} |
| 3s_paired_fusion | PASS | {'f1_difference_ci95': [-0.006012117100356584, 0.0065387337626778815], 'mae_difference_ci95': [-0.0063754469039849935, 0.003244647354586046]} |
| frozen_backbone_protocol | PASS | Inspected training cells: inference-mode cached embeddings; requires_grad False/eval; optimizer receives only AttentionHead parameters; end-of-training tensor equality asserts. Original backbone hashes unchanged. No original training/evaluation cell executed by audit. |
| observed_window_only | PASS | Independent loader supplies exactly200/300/400 samples and20/30/40 feature bins; all hidden states derive only from these inputs. IDs, true magnitudes and arrival metadata are excluded from model inputs. Bidirectionality within observed window allowed; no within-window prefix invariance claim. |
| validation_only_protocol | PASS | Saved source selects checkpoints using validation loss and thresholds using validation F1. All horizons selected before test evaluation. Replayed histories, selected loss and thresholds agree. No retraining/tuning during audit. |
| historical_unlogged_actions | UNKNOWN | Saved source/history/hash evidence supports protocol; no immutable runtime log proves absence of all unrecorded prior tuning. Previously inspected test set is exploratory. |
| original_integrity | PASS | {'files': 23, 'changed': []} |

## Detection

| Horizon | Threshold | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC(AP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | 0.5 | 0.66321458 | 0.65129007 | 0.69416667 | 0.67204518 | 0.36738056 | 0.75128364 | 0.74885225 |
| 1s | 0.28 | 0.62676056 | 0.57122439 | 0.99916667 | 0.72688694 | 0.74135091 | 0.75128364 | 0.74885225 |
| 2s | 0.5 | 0.73570837 | 0.76609848 | 0.67416667 | 0.71719858 | 0.20345964 | 0.83216776 | 0.84935186 |
| 2s | 0.41 | 0.71913836 | 0.68913043 | 0.79250000 | 0.73720930 | 0.35337727 | 0.83216776 | 0.84935186 |
| 3s | 0.5 | 0.77630489 | 0.85181237 | 0.66583333 | 0.74742750 | 0.11449753 | 0.86539264 | 0.88753171 |
| 3s | 0.41 | 0.76180613 | 0.77150304 | 0.74000000 | 0.75542322 | 0.21663921 | 0.86539264 | 0.88753171 |

## Magnitude

| Horizon | MAE | RMSE | R2 | Signed error | Pred/actual SD | M>=5 N | Underprediction % | Strong bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | 0.6663588509956996 | 0.818940840992192 | 0.1703654081451904 | -0.07629210571448009 | 0.3875688444820556 | 131 | 98.47328244274809 | -1.3735367378205743 |
| 2s | 0.6364890929063162 | 0.7829799640130151 | 0.24162660426249727 | -0.05613822380701701 | 0.5124193578543239 | 131 | 95.41984732824427 | -1.1821874917008495 |
| 3s | 0.6066006402174632 | 0.7486207347412538 | 0.30672507114512293 | -0.004880607922871908 | 0.6006228625923995 | 131 | 87.02290076335878 | -0.9308202175693657 |

## Attention versus approved fusion

| Horizon | F1 difference [95% CI] | MAE difference [95% CI] |
|---|---|---|
| 1s | -0.00786315 [-0.018330848456274, 0.001981389353165757] | 0.00211926 [-0.0012190080509753896, 0.0054234042880125346] |
| 2s | 0.00088652 [-0.003969822205396578, 0.005982555642963445] | 0.00061039 [-0.0018039557384327053, 0.002919959701830521] |
| 3s | -0.00004725 [-0.006012117100356584, 0.0065387337626778815] | -0.00161448 [-0.0063754469039849935, 0.003244647354586046] |

All six intervals include zero: no statistically supported Attention improvement over fusion on either metric at any horizon under this paired bootstrap. 500 paired stratified resamples, seed2026; same eligible earthquake/noise indices per model; regression on all eligible earthquakes independent of detection. Fixed .5 for paired F1. These are descriptive percentile intervals, not equivalence tests, multi-seed evidence or multiplicity-adjusted confirmatory results. Tiny changes must not be interpreted as meaningful gains.

## Interpretation and limitations

Epochs3/17/32 reproduce. The3s model reached the35-epoch cap with3 stale epochs; it did not stop by patience7. Best observed validation checkpoint remains epoch32. Attention is genuinely temporal over20/30/40 states with normalized, finite nonnegative weights. Weight entropy, extrema and mean-by-bin are in JSON. Weights are diagnostics, not physical feature importance; bidirectional states mix context across the already observed window.

Synthetic oracle-P alignment; shared synthetic signal source across branches. No demonstrated operational warning latency, autonomous P detection or real-world EEW readiness. Magnitude compression, M>=5 underprediction and false-positive rates remain concerns. Validation-selected thresholds are reported separately and not compared against fixed-.5 baseline policies. Dataset upstream causality/scaling validation is inherited from approved artifacts, not repeated. Historical unlogged actions and noise-family shortcut removal are not independently established.

All original dependencies retain their hashes; hashes recorded in JSON. Only audit.ipynb, audit_report.md and audit_results.json created. No training, tuning, new architecture or cleanup.
