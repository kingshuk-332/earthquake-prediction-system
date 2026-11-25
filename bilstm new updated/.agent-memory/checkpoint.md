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

<!-- CSV_CONTEXT_CURRENT_START -->
# Current checkpoint: CSV-only preprocessing complete
- Latest user request replaces the active waveform task with historical CSV context. Read reports/catalog_preprocessing_report.md and dataset/dataset_validation.json; do not rescan earlier branches.
- Input parsed once this run: ../dataset_2.csv; SHA-256 5cfebe7268a87940ec9489d4f72b45381e2c940cc3f38d96d60b4575355d8bf3; run f0230d622ef74023.
- New notebook phase1_bilstm_csv_preprocessing.ipynb owns all executable preprocessing/validation. Original sources and earlier EEW artifacts unchanged.
- Chosen definition: Magnitude_Value with Magnitude_Type=ML, no conversion; next ML event target. 7125 retained / 8089 source rows, seq20, 10 features.
- Train/val/test: 4973/1065/1067; 70/15/15 chronological target order. Historical overlap is allowed and recorded; no target-ID overlap.
- StandardScaler plus median imputer fit only on 4992 unique train input events. Parameters JSON plus exact fit-event manifest saved; target identity transform.
- Optional seeded coordinate Gaussian augmentation σ=0.01 defaults off. Labels/derived features/times unaffected. No permanent noisy data.
- Verification PASS for retrospective chronology, provenance, train-only fit, finite arrays and exact PT/NPZ contents. Operational availability, independent source calibration and fuzzy duplicate absence remain UNKNOWN.
- Stop: no training/tuning/fusion. Next work requires a new user request. Earlier branch notes below are archival, not the current objective.

## Archived earlier checkpoint
<!-- CSV_CONTEXT_CURRENT_END -->

# Checkpoint: new BiLSTM EEW project

- Objective supersedes catalog forecasting: initial waveform earthquake detection, then earthquake-only magnitude estimation. No CNN/attention/hybrid training.
- Stage A executed; run ID `e64115e91298ee59`. Stage B/C intentionally blocked. Read `reports/stage_a_validation.json` and `reports/final_research_report.md` first; do not rescan waveform phases.
- Original `../dataset_2.csv` was parsed once into this project's `.cache/source_snapshot.json.gz`; `catalog_preprocessing.ipynb` is source inspection only. No catalog feature sequences were built. Snapshot hash is verified against original CSV by Stage A.
- Source: `../cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/` raw `waveforms`, not full-trace standardized arrays. Explicit source_id + trace_name joins verify Phase 11 → Phase 10 → Phase 9 → P/S manifest → catalog. Files/hashes recorded in `dataset/source_manifest.json`.
- Quarantined `dataset/eew_candidate_windows.npz`: split counts 11324/2426/2428; durations 1/3/5 seconds, lengths 100/300/500, one raw amplitude channel. `dataset/window_manifest.json` has `training_allowed: false`. Noise target magnitudes NaN. All versions of a parent/pair share a split.
- Proven source defect: `improved_waveform_generator.py:224-243` sets early noise scale from full future clean-signal peak. Exact source reproduction plus same-seed S-amplitude counterfactual established prefix dependence, not merely a hypothesis. Evidence and figure in `reports/causal_noise_counterexample.*`.
- Need separate independent-noise source or real recordings, then new validation. Never silently repair/overwrite old sources or train on current candidates. Stage B/C notebooks are executable blocked-state guards, not implemented training/evaluation pipelines.
- No scaler, augmentation, checkpoint, Optuna trial or inference executed. Research notes distinguish catalog background from Graves' bidirectional principle. Real-time reporting availability and target-site warning lead times remain unverified.
