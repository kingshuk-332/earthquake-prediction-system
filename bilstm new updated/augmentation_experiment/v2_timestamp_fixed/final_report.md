# Phase 1C: corrected-data augmentation revalidation
**APPROVED for retrospective research preprocessing. No training performed.**

## Corrected source
Only dataset/v2_timestamp_fixed/bilstm_catalog_seq20.pt supplies numerical model data. All source artifact hashes match the corrected registry. Original v1 dataset and augmentation artifacts are preserved.

The source preprocessing now explicitly converts datetime integers to nanoseconds before rates and intervals. This notebook independently checks corrected training features using time-index rolling windows and timedelta arithmetic. Rate mismatches: 0; target-gap mismatches: 0; other feature violations: 0. Actual/saved target-gap ratio: 1 to 1.

## Prepared variants
A clean; B coordinate-only Gaussian noise; C bounded regression-relevance weighted complete-window sampling; D the same weighted draws with the same noise. Seed 42; sigma=0.01 standardized units, latitude/longitude only. No ordinary SMOTE interpolation; weighted sampling is not full SMOGN. No new target magnitudes, IDs or timestamps.

| Variant | Draws | Unique parents | Repeats | Synthetic feature views | ML>=4.5 |
|---|---:|---:|---:|---:|---:|
| A | 4973 | 4973 | 0 | 0 | 31.23% |
| B | 4973 | 4973 | 0 | 4973 | 31.23% |
| C | 4973 | 3114 | 1859 | 0 | 36.64% |
| D | 4973 | 3114 | 1859 | 4973 | 36.64% |

Each shape is (4973,20,10). C/D have identical parents; B/D apply identical per-event noise within an epoch. No 50/50 quota. Repeated draws and synthetic feature views are explicitly marked in parent_provenance.csv and are not new independent earthquake records.

## Acceptance checks
Source feature semantics, shape/finite constraints, training-only parents, coordinate bounds, unchanged target labels and non-coordinate features, chronological parent/window order, deterministic replay, preserved timestamps and split membership all pass when approval is APPROVED. The saved training consumer was exercised against its deterministic preview. Validation/test tensors and fitted scalers are unchanged and never augmented/refitted here. The v1 augmentation output registry also remains unchanged.

The reproducible plan plus notebook functions materializes accepted epochs on demand; no redundant full augmented tensor files are needed. Sampling/noise config, content hashes, parent records, distribution tables, independent source audit and validation results are saved beside this report.

Operational catalog availability, magnitude calibration and real-world EEW relevance remain unverified. There is no claim that augmentation improves model performance. Training, optimization and fusion are outside this task.

Notebook: ../../phase1b_imbalance_noise_preprocessing.ipynb.
Corrected dataset report: ../../dataset/v2_timestamp_fixed/reports/catalog_preprocessing_report.md.
