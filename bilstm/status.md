# BiLSTM status — 2026-11-09

Phase 2 completed in `bilstm/`, as requested. All six code cells in [phase2_bilstm.ipynb](phase2_bilstm.ipynb) executed successfully on the RTX 3050 GPU. All implementation and training code remains in the notebook.

- Dataset: original `bilstm_dataset/bilstm_dataset_seq20.pt`; existing 5629/1191/1192 splits, input 20×36. No data regeneration or substitution.
- Baseline and tuned BiLSTM trained with AdamW; 15 Optuna TPE trials (10 complete, 5 pruned) selected by validation MAE. Baseline/tuned validation MAE: 0.289308/0.271818; RMSE: 0.378674/0.358284.
- Frozen tuned test: MAE 0.434094, RMSE 0.655644, R² −0.004341. Training-mean baseline: MAE 1.021378, RMSE 1.153770. Despite beating this baseline, negative test R² indicates weak generalization: squared error slightly exceeds the test-mean reference used by R². The test mean is a descriptive reference, not a fitted prediction baseline.
- Both fits show late overfitting; best epochs 23/8 were retained. Total training/development evaluation: 170.9 seconds, excluding setup and checkpoint I/O.
- Strict checkpoint reload and validation-prediction equality passed. Independent metrics reconstructed from saved test predictions match; all 15 trial outcomes and checkpoint hash verified. Test inference is cached for reruns.
- No scaler object was supplied; the affine inverse was recovered from saved training scaled/raw targets and verified (approximately `raw = 7.1 × scaled + 1.5`). Event IDs, feature semantics, scaler-fitting provenance and target timing remain unknown. These results do not establish future earthquake forecasting.

See [final_report.md](phase2_bilstm_results/final_report.md) for hyperparameters, detailed comparisons and artifact inventory. Rerun the notebook to reuse completed fits and the SQLite study. Do not run concurrent notebook copies. No CNN+BiLSTM hybrid work was performed.

## Generalization audit — 2026-11-09

Completed and executed [phase2_bilstm_generalization_audit.ipynb](phase2_bilstm_generalization_audit.ipynb), five code cells. See the [audit report](phase2_bilstm_results/generalization_audit/audit_report.md) for evidence and the baseline comparison. No BiLSTM checkpoint was loaded, no retraining or Optuna rerun occurred, and all protected original artifacts passed before/after hash checks.

- Substantial descriptive target/input shift: magnitude means train/validation/test 4.086/3.474/3.135; M≥4.5 proportions 34.3%/6.1%/3.9%. Input range violations affect 4.3% of validation and 31.5% of test windows. Test MAE outside/inside training feature ranges: 0.694/0.314. Association does not establish causality or an incorrect input scaler.
- Late overfitting exists; post-selection divergence starts at epoch 9. Best epoch 8 and patience-8 stopping at epoch 16 worked. Train–validation MAE gap 0.033 is much smaller than test–validation gap 0.162.
- No constant prediction collapse (predicted/actual SD ratio 0.928), but conditional bias exists: low magnitudes are overpredicted; 89.4% of M≥4.5 cases are underpredicted. M<3 has the worst well-supported range MAE (0.593); the top 5% of errors contribute 52.5% of squared error.
- BiLSTM test MAE 0.434 beats training mean 1.021, median 1.033 and validation-selected Ridge 0.693. Ridge alpha 0.1 was selected from a fixed grid using validation only. Target inverse discrepancies remain below 4.2e-7.
- Leading hypothesis: split mismatch and associated biased/outlier predictions, with overfitting contributing. Highest priority next experiment: a provenance-verified split/preprocessing protocol with representative validation and a fresh event-disjoint holdout, keeping BiLSTM configuration fixed and fitting transforms on training data. This is a recommendation, not work begun. No Phase 3 training.

## Phase 2B provenance gate — 2026-11-09

[phase2b_generalization_experiment.ipynb](phase2b_generalization_experiment.ipynb) executed successfully (four code cells). **Controlled training is blocked by insufficient seq20 provenance.** See [Phase 2B final report](phase2b_generalization_results/final_report.md) and its PASS/FAIL/UNKNOWN assessment.

The raw catalog has 8,089 parseable timestamps and unique source IDs, but neither seq20 archive maps windows/targets to those IDs or records the 36-feature schema. Available legacy documentation is 15×44. The legacy preprocessing script fits clipping/imputation/scaling globally; its connection to seq20 is unverified, so actual seq20 preprocessing remains UNKNOWN. Input-before-target ordering also remains UNKNOWN.

Exact checks found no shared full windows, step vectors or contiguous 19-step blocks across existing splits; this does not prove event independence. Candidate catalog identities were saved explicitly as **unlinked source records**, not a new split manifest. No split, fitted preprocessing, new baseline, model checkpoint or holdout evaluation was created; no improvement is claimed. Original artifacts passed hash-preservation checks.

Next: recover the exact seq20 constructor, ordered feature schema, per-step/target IDs and timestamps, split/purge rules and preprocessing fit records. Phase 3 feasibility is not established; Phase 3/4 work has not begun.

## Seq20 source recovery — 2026-11-09

**PARTIAL: packaging verified; constructor remains UNKNOWN.** Windows download metadata on all three datasets points to `C:/Users/choud/Desktop/bilstm_dataset.7z`. Each member exactly matches the current dataset by SHA-256. The archive contains only those three datasets and no source code, schema, scaler or manifest. A bounded search of 16 named sources found only downstream dataset consumers; no original seq20 writer was recovered. Dataset and source hashes remained unchanged; no training or regeneration occurred. See [seq20_provenance_report.md](seq20_provenance_report.md) and [executed investigation](seq20_provenance_investigation.ipynb). The original producing notebook/script and identity/preprocessing manifests are still required.
