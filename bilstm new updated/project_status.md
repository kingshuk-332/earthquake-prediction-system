## Phase 3 CSV BiLSTM branch finalized — 2026-11-10

- Final verdict C: Neither established as adequately generalizing; preserve original and modern as research baselines.
- Fixed training-only period selection; no Optuna, CNN changes or original-test tuning. Phase 3 report: phase3_bilstm_results/final_bilstm_report.md.
- Split counts: {'train': 692, 'val': 346, 'holdout': 366}; all rolling-feature dependencies purged across boundaries; train-only new scaling.
- Holdout is an internal 2024 partition previously in original-model training, not unseen external validation. Publication availability and independently calibrated Mc remain unverified.
- Code: phase3_modern_catalog_analysis.ipynb and phase3_modern_bilstm_training.ipynb. Original artifacts preserved. No further tuning or fusion started.

## Phase 2E report corrections and provenance — 2026-11-10

- READY WITH LIMITATIONS for next-experiment planning; no training, inference or dataset changes.
- Versioned Phase 2B/C revisions correct all 19 feature summaries plus Huber loss, variance, mean shift, runtime, labels and scientific interpretations. Original artifacts preserved.
- Saved checkpoint configurations/AdamW groups and validation-selected epochs match. Complete historical execution equivalence and absolute test non-access remain UNKNOWN.
- Original Phase 2C notebook not recovered; verified statistics preserved separately. Historical/current Mc UNKNOWN; completeness causation remains a hypothesis.
- Current documentation: phase2_bilstm_results/gemini_verification/correction_report.md. Earlier Gemini Phase 2B/C narrative entries below are superseded by these revisions and retained only as historical notes.

## Phase 2D independent Gemini verification — 2026-11-10

- Verification only completed; no model loading, training, tuning, preprocessing or selection.
- PASS: Phase2B core validation metrics; matching recorded initialization and all19 common sampler epochs; all340 Phase2C JSON fields; all29 registered original Phase2/v2 hashes.
- FAIL: delta2 Huber/runtime/mean-shift/variance prose and19 Phase2C feature-table summaries; unsupported causal completeness/independence conclusions.
- UNKNOWN: exact historical control equivalence, absolute historical test non-access, Mc values, missing Phase2C notebook/execution provenance.
- Audit: phase2d_gemini_verification.ipynb; phase2_bilstm_results/gemini_verification/verification_report.md. Existing reports retained; no subsequent experiment started.

<!-- PHASE2C_CURRENT_START -->
# Phase 2C complete — feature informativeness & temporal shift audit (2026-11-09)
- No models trained/modified. Pure diagnostic data audit across Train (4,973), Val (1,065), Test (1,067).
- Empirical finding: Apparent training predictive signal is non-stationary and collapses in modern data.
  * Rolling magnitude correlation collapses: r = +0.827 (Train 1924–2013) -> +0.291 (Train 2013–2024) -> +0.201 (Val 2024–2026).
  * 30-day rate correlation with magnitude flips: r = -0.480 (Train) -> +0.074 (Val).
  * Inter-event days correlation collapses: r = +0.337 (Train) -> -0.011 (Val).
- Cause: 100-year catalog completeness evolution (Mc shift). Historical catalogue only reported M>=4.0 events (mean 4.03 ML, 28 events/yr); modern high-density broadband detects M1.5–3.0 microseismicity (mean 3.10–3.51 ML, 942–1592 events/yr; rate jumped 8.2x).
- Explains BiLSTM behavior: Model learned to associate historical low rate/high magnitude with large targets. In modern data, large quakes occur within dense micro-seismicity sequences, so BiLSTM correctly predicts distribution mean (~3.43 ML) to minimize error.
- Report: phase2_bilstm_results/error_diagnosis/phase2c_feature_informativeness_report.md; plots: feature_correlation_drift.png, temporal_distribution_shift.png.
<!-- PHASE2C_CURRENT_END -->

<!-- PHASE2B_CURRENT_START -->
# Phase 2B complete — controlled Huber loss experiment (2026-11-09)
- Unit verification: target magnitude ML is unscaled identity; delta is measured in original magnitude units.
- Evaluated on validation only (1,065 samples); test set remains frozen and unaccessed. Notebook: phase2b_huber_loss_experiment.ipynb.
- All settings fixed: Variant C weighted sampling, hidden 128, 2 layers, dropout 0.3145, AdamW (lr 0.0004549, wd 0.0001216), seed 42.
- Delta=2 vs Delta=1: Validation MAE 0.4698 vs 0.4724 (-0.0026); RMSE 0.6243 vs 0.6242 (+0.0002); R2 0.0158 vs 0.0162.
- Prediction variability worsened: std dev compressed from 0.1573 to 0.1248 (range [3.136, 3.707] vs [3.180, 3.910]).
- High-magnitude (ML>=4.5) underprediction worsened: MAE 1.4658 vs 1.4214; underprediction rate 75/75 (100.0%); signed bias -1.4658 vs -1.4214.
- Checkpoint, metrics, curves and report saved: phase2_bilstm_results/huber_experiment/.
- STOP after Phase 2B. No further interventions or test evaluation.
<!-- PHASE2B_CURRENT_END -->

<!-- PHASE2A_CURRENT_START -->
# Phase 2A diagnosis complete (2026-11-09)
- Saved-output audit only; no checkpoint inference/training/tuning or source changes. Report: phase2_bilstm_results/error_diagnosis/diagnosis_report.md.
- C predictions concentrate at mean 3.272, SD 0.0779 vs actual 0.6414; training mean 4.034. High underprediction 38/38; low (<3) overprediction 490/490.
- Target means train/val/test: 4.034/3.511/3.101. Feature/relationship shifts and late C/D overfitting documented; causal contributions remain hypotheses.
- First proposed future ablation: fixed-C Huber delta 1 vs 2 using validation only. No intervention implemented; test remains frozen. Ranked alternatives and tradeoffs saved.
- STOP after diagnosis; wait for user instruction. Operational availability/calibration remain UNKNOWN.

## Earlier notes
<!-- PHASE2A_CURRENT_END -->

<!-- PHASE2_CURRENT_START -->
# Phase 2 COMPLETE - corrected catalog BiLSTM (2026-11-09)
- Only approved v2 data; run 1bd0faa814c714ba. Fifteen clean-A TPE trials (11 complete, 4 pruned), SQLite/sampler/epoch resume saved.
- A/B/C/D trained independently with identical selected settings and fresh seeded initial weights. Winner C selected by validation MAE 0.4724.
- Winner-only test: MAE 0.5477, RMSE 0.6696, R2 -0.0899; high-magnitude underprediction 100.0%.
- Notebook phase2_bilstm_training.ipynb; report phase2_bilstm_results/final_report.md. Checkpoints variants/A-D/best.pt; selection lock, caches and independent verification saved.
- Total fit compute 3.14 minutes. No CNN/attention/hybrid processing. Phase 1 artifacts unchanged.
- STOP after Phase 2. No further selection using test. Operational availability/calibration remain UNKNOWN; no EEW capability claim.

## Archived notes
<!-- PHASE2_CURRENT_END -->

<!-- PHASE1C_CURRENT_START -->
# Phase 1C - timestamp-fixed v2 APPROVED (2026-11-09)
- Explicit ns conversion replaces native-unit integer arithmetic. Rates, target gaps, dependency intervals and time-derived summaries rebuilt from original CSV; no tensor rescaling shortcut.
- Corrected dataset: dataset/v2_timestamp_fixed/. New train-only scalers; previous artifacts unchanged. Targets, schema, event identities/timestamps and chronological splits preserved.
- Shapes train/val/test: (4973,20,10)/(1065,20,10)/(1067,20,10). Independent known-time/boundary and full-catalog checks passed.
- Corrected augmentation: augmentation_experiment/v2_timestamp_fixed/. Four variants, seed 42, coordinate sigma 0.01; complete-window weighting, no interpolation.
- ML>=4.5 proportions: A/B 31.23%; C/D 36.64%.
- Source feature rate/gap violations now 0/0. Parent, label, split, scaler and validation/test preservation checks pass.
- Read dataset/v2_timestamp_fixed/reports/catalog_preprocessing_report.md and augmentation_experiment/v2_timestamp_fixed/final_report.md.
- Both corrected notebooks executed. No training/Optuna/fusion performed. Operational availability remains UNKNOWN; approval is for retrospective research preprocessing only.
- Earlier v1 notes below are archived; use v2 exclusively for future authorized work.

## Archived notes
<!-- PHASE1C_CURRENT_END -->

<!-- PHASE1B_CURRENT_START -->
# Phase 1B — implemented, BLOCKED by source consistency (2026-11-09)
- CSV-only weighted complete-window sampling and coordinate Gaussian noise implemented; four diagnostic variants, seed 42, sigma 0.01. No interpolation or model training.
- Independent audit found 99459 incorrect 30-day rate cells across 4973 training windows and 4973 incorrect target-gap records. Original datetime resolution was not normalized before nanosecond arithmetic.
- Earlier Phase 1 preprocessing PASS was insufficient for these semantics; actual origin-time ordering still passes. No original preprocessing/dataset/scaler/provenance was changed.
- Training-ready acceptance: False. A separate authorized correction of source preprocessing is required if false; do not train from diagnostic plans.
- Notebook: phase1b_imbalance_noise_preprocessing.ipynb. Outputs/report: augmentation_experiment/final_report.md. Detailed source evidence: augmentation_experiment/source_feature_audit.json.
- STOP after preprocessing. Operational availability remains UNKNOWN.

## Earlier project notes
<!-- PHASE1B_CURRENT_END -->

<!-- CSV_CONTEXT_CURRENT_START -->
# Active task: CSV-only historical-context preprocessing — COMPLETE
User's latest request establishes a separate catalog branch. The earlier waveform branch below remains archived and unchanged; its gate does not block this CSV-only task.
- Run: f0230d622ef74023; input only ../dataset_2.csv. No waveform/CNN/old model data loaded.
- ML-only Magnitude_Value: 7125 records, 7105 seq20 samples, 10 features.
- Shapes: train [4973, 20, 10]; val [1065, 20, 10]; test [1067, 20, 10].
- Strict origin-time ordering, target-disjoint chronological splits, permitted historical overlap, train-only scalers and .pt/.npz equality verified.
- Operational availability and magnitude calibration UNKNOWN. Not an operational EEW target.
- Optional σ=0.01 coordinate augmentation implemented, disabled by default; validation/test clean.
- Notebook: phase1_bilstm_csv_preprocessing.ipynb. Report: reports/catalog_preprocessing_report.md. Arrays and manifests: dataset/bilstm_catalog_seq20.* and dataset/split_manifest.csv.
- STOP after preprocessing. No BiLSTM, Optuna, CNN, fusion or attention training authorized in this task.

## Archived earlier branch notes
<!-- CSV_CONTEXT_CURRENT_END -->

# BiLSTM EEW project status

**Stage A: BLOCKED — future-signal-dependent synthetic background noise.**

- Work is confined to this new project. Earlier BiLSTM datasets, weights and scalers were not reused. Original CSV and CNN waveform sources are unchanged.
- Verified Phase 11 raw waveform identity using source IDs and trace names through explicit manifests to catalog magnitudes and P/S annotations. Event/noise pairs are split-disjoint; source hashes and exact cross-split duplicate checks passed.
- Saved quarantined 1/3/5-second post-P windows: 11,324 train, 2,426 validation and 2,428 test parents, with 100/300/500 samples and one amplitude channel. All durations retain the same partitions. Noise magnitudes are missing, and S-containing windows are labeled.
- Reproduced source trace NCS_EV_07168 exactly. Increasing only its future S-wave amplitude changed early noise while its clean prefix stayed identical. See `reports/causal_noise_counterexample.json`.
- No scaling fit, augmentation, BiLSTM training, Optuna trials, model checkpoint or test performance evaluation. Stage B/C notebooks record the blocked state and do not train or predict.
- Catalog origin-time ordering is retrospective. Reporting/revision availability is UNVERIFIED. Oracle P picks and synthetic arrivals do not establish operational detection or target-site warning lead time.

Next: obtain independently generated ambient noise without target/future-peak calibration in a separate source version, or verified real waveforms; then repeat Stage A before implementing training. Do not use the quarantined archive for training. CNN and hybrid work remain outside scope.

Entry points: `phase1_eew_dataset_preparation.ipynb`, `phase2_eew_bilstm_training.ipynb`, `phase3_eew_bilstm_evaluation.ipynb`. Full findings: `reports/final_research_report.md`.
