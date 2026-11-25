# Final four-model synthetic EEW comparison

## Methodology

This synthesis uses only saved final reports and independent audit results. No model inference, training, threshold fitting or preprocessing was repeated. All four audits approve controlled synthetic research reporting; approval does not establish operational capability.

Catalog-informed simulation generated the seismic waveforms. CNN processes early waveform windows; BiLSTM processes seven numerical features in consecutive0.1s bins extracted from those same synthetic observations, not independent catalog measurements or historical next-event sequences. Fusion joins frozen learned embeddings; Temporal Attention pools20/30/40 frozen BiLSTM hidden states before fusion with the CNN embedding.

Sampling100Hz; windows contain100 pre-reference samples plus100/200/300 post-reference samples. Oracle P alignment is assumed. Existing event-disjoint partitions, pre-S eligibility and training-only normalization are preserved. Common test cohort: exactly1,200 earthquakes and1,214 noise records for every model/horizon. Regression covers all1,200 earthquakes regardless of detection; noise sentinel magnitudes never contribute.

Ordered source-ID cohort SHA256: `862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601`. Matching was established by the existing audits; this synthesis does not re-audit datasets.

Fixed0.5 results are the primary policy-matched comparison. Validation-selected operating points appear separately; optimized objectives/grids need not be identical across experiments, so that table describes operating tradeoffs rather than a controlled architecture ranking. CSV repeats unconditional magnitude metrics across operating points; these are not detection-conditioned estimates.

## Results: fixed threshold 0.5

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

## Validation-selected thresholds (separate cohort-matched table)

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

## Magnitude compression and strong earthquakes

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

## Existing paired uncertainty

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

## Best architecture by horizon: qualified point estimates

| Horizon | Highest F1 | Lowest MAE | Lowest RMSE / highest R2 |
|---|---|---|---|
| 1s | BiLSTM | Fusion | Fusion |
| 2s | Attention | BiLSTM | BiLSTM |
| 3s | CNN | Attention | Attention |

These are descriptive rankings, not statistically established winners or a deployment selection rule. At1s, BiLSTM F1 benefits from near-total earthquake recall with73.64% noise FPR. At2s, Attention leads F1 by only0.000887 over fusion, while BiLSTM has the lowest MAE. At3s, CNN has the highest F1; Attention has the lowest MAE/RMSE, but its gain over fusion is unsupported by the available intervals. Direct paired CNN-versus-BiLSTM winner tests are not available here.

## Discussion

Longer observed windows generally reduce magnitude error, but architecture complexity does not yield a consistent benefit. Frozen fusion and attention compare different representations and trainable parameter budgets; they do not isolate an architecture effect. Shared synthetic inputs mean the two branches are complementary representations, not independent sensor evidence. Low aggregate error must be interpreted alongside magnitude compression and systematic underprediction of stronger events.

## Limitations

The benchmark uses synthetic catalog-informed waveforms and oracle P-wave alignment. No operational P-wave triggering, measured streaming detection delay, real-world warning lead time or earthquake forecasting ability was established. Pre-S cohort masks use offline annotations and are not a deployed eligibility rule. High false positives and strong-event underprediction remain substantial limitations. Attention weights are diagnostics rather than physical explanations.

Single-seed models and a previously inspected test cohort limit generalization and confirmatory inference. Bootstrap intervals are descriptive, not multi-seed or multiplicity-adjusted tests; no missing significance tests or confidence intervals were invented. Calibration, real sensor/instrument variability and performance under distribution shift are not established. Audit provenance supports the saved protocol but cannot prove absence of every unlogged historical action.

## Future work

1. Preregister evaluation and reserve a fresh untouched cohort before any further model selection.
2. Evaluate multiple seeds and paired uncertainty with explicit false-alarm/recall objectives.
3. Test real station waveforms, causal online triggering, instrument calibration and out-of-distribution noise.
4. Investigate strong-event bias and calibration using training/validation data only.
5. Measure end-to-end latency including feature extraction and triggering; keep inference timing separate from physical warning time.

These are proposals only; no additional experiment was started.

## Figures and provenance

- [four_model_f1.png](plots/four_model_f1.png) —300 DPI; audited point estimates only, no implied uncertainty bands.
- [four_model_fpr.png](plots/four_model_fpr.png) —300 DPI; audited point estimates only, no implied uncertainty bands.
- [four_model_mae.png](plots/four_model_mae.png) —300 DPI; audited point estimates only, no implied uncertainty bands.
- [four_model_rmse.png](plots/four_model_rmse.png) —300 DPI; audited point estimates only, no implied uncertainty bands.

All source report/JSON SHA256 hashes were checked unchanged after consolidation. Datasets, notebooks, checkpoints and original plots were not opened for inference or altered. Writes confined to this new output directory.

- `cnn/cnn wave/eew_causal_v3/cnn_eew_training/final_cnn_eew_report.md` — SHA256 `eaad5e6f58bfab30ac57a6918c8af2370639f10a98c183d0995d0c003a68ce18`
- `cnn/cnn wave/eew_causal_v3/cnn_eew_training/codex_performance_audit/cnn_performance_audit_report.md` — SHA256 `97db09ed439a4f97497be373cda0b2e287e0fee92705759b1b66e9d62e565773`
- `bilstm_eew/bilstm_training_v1/final_bilstm_eew_report.md` — SHA256 `e6d42302de5fa8306f47003ebaa820d251e5a206bb13b31a3005a9d191741756`
- `bilstm_eew/bilstm_training_v1/codex_independent_audit/audit_report.md` — SHA256 `12b187d1fa3e4ce6b5898afdf503e213cbb771ce8c8b2d03c92c6c74c4214a77`
- `fusion/cnn_bilstm_v1/final_fusion_report.md` — SHA256 `f9aa2d12a97cfa7edc3e231bf9bb7181bc0b044f747092b731ba0e30e2ca9fd7`
- `fusion/cnn_bilstm_v1/codex_independent_audit/audit_report.md` — SHA256 `a885730e5fd6e1515f82da0bae80046b17611ece86c57441f48b76361d3c83ca`
- `fusion/cnn_bilstm_attention_v1/final_attention_report.md` — SHA256 `d980e25c9a1eac847ddbdbade33635bf030856dab198699963a411dc6c550e41`
- `fusion/cnn_bilstm_attention_v1/codex_independent_audit/audit_report.md` — SHA256 `7b03236668d4451ff35cfdf6a4cff9db775626784939706694a8eeb608153fd1`
- `cnn/cnn wave/eew_causal_v3/cnn_eew_training/codex_performance_audit/cnn_performance_audit_results.json` — SHA256 `58cadae483be48911671653fb714d168fa801b81078fde373b4bd02e4c9c815d`
- `bilstm_eew/bilstm_training_v1/codex_independent_audit/audit_results.json` — SHA256 `cb548791209a5b77cec7c05f66a75f10a9cc73d5b824370c09a67b88dc3b278f`
- `fusion/cnn_bilstm_v1/codex_independent_audit/audit_results.json` — SHA256 `8f117cc56cae05471d405b49518fe477d1742ef1aaa6fc7e7c3078f91a0201cd`
- `fusion/cnn_bilstm_attention_v1/codex_independent_audit/audit_results.json` — SHA256 `fe67e94f2ef0fdd33ea6e8fca18afa959da6d0b7b4aaf9cefef02e929ad94fc0`
