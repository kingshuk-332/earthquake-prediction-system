# CSV-only BiLSTM preprocessing validation

**Completed retrospective construction; operational feature availability UNKNOWN.** Run f0230d622ef74023. No model fitting, Optuna, waveform loading or modality fusion.

## Source and target

The original CSV has 8,089 rows and 28 columns. Its SHA-256 is 5cfebe7268a87940ec9489d4f72b45381e2c940cc3f38d96d60b4575355d8bf3. Parsing, identity, missingness, ranges and discrepancy evidence are in catalog_input_inspection.json and dataset/catalog_record_disposition.csv.

Use Magnitude_Value only for records whose Magnitude_Type is ML: 7,125 retained, 964 excluded, 7,105 supervised samples after 20 warm-up events. The target is the **next retained ML event**, not the next event of any type in the complete CSV. Binary labels are reported ML≥4.5. No magnitude conversions, averaging, alternate-column substitutions, declustering or high-magnitude clipping were applied.

There are 540 numeric disagreements and 553 type disagreements between magnitude columns; maximum numeric discrepancy 2.1. Exact rows are preserved in catalog_magnitude_conflicts.csv. Choosing a column/type does not establish its accuracy; unusually high ML values remain a calibration limitation. Original types: {'ML': 7125, 'MB': 458, 'MW': 398, 'MC': 107, 'MS': 1}. Source-alternative types: {'ML': 7629, 'MB': 298, 'MC': 132, 'MW': 30}.

## Features and chronology

Ten ordered features: historical magnitude, depth, latitude, longitude, inter-event days, 30-day ML event rate, 20-event magnitude mean/std and depth mean/std. Rolling statistics include only already observed historical events; population standard deviations use ddof=0. No target attributes, IDs, absolute timestamps or synthetic fields enter X. See feature_schema.json.

Every sample preserves all 20 input IDs/original and UTC timestamps, target ID/time/magnitude, context cutoff and target time gap. Feature dependency intervals reference catalog_events.csv and can extend older than the explicit 20 input rows. Early rolling summaries have partial catalog history; the initial 30-day rate is left-censored. Any tied origin-time group is excluded rather than arbitrarily ordered; 0 records required this rule here.

The cutoff is the latest input origin time. Historical reporting/revision times are absent, so it is not an asserted operational availability time. Targets' times/gaps are provenance only. Prior validation/test events may become observed history for later samples; evaluation is sequential observed-context, not fixed-origin forecasting.

## Frozen chronological partitions

| Split | X shape | Target dates (UTC) | ML mean ± std | ML range | ML≥4.5 |
|---|---|---|---|---|---|
| train | [4973, 20, 10] | 1924-08-13 to 2024-12-20 | 4.034 ± 0.898 | 1.5–8.6 | 1553 (31.23%) |
| val | [1065, 20, 10] | 2024-12-20 to 2026-02-05 | 3.511 ± 0.629 | 2.0–7.5 | 75 (7.04%) |
| test | [1067, 20, 10] | 2026-02-05 to 2026-10-06 | 3.101 ± 0.641 | 1.5–5.8 | 38 (3.56%) |

No target IDs overlap. Shared explicit historical event counts: {'train__val': 19, 'train__test': 0, 'val__test': 19}. Earlier targets can appear as later historical inputs; later target information never enters earlier samples. These permitted overlaps and exact IDs are separately recorded in catalog_context_overlap.json.

Full bin counts, quantiles, target gaps and per-split missing/invalid counts are in dataset_validation.json. Test labels were inspected once for this requested descriptive audit, not for feature, preprocessing or model selection. The policy was fixed before those summaries.

## Training-only transforms and augmentation

Median imputation and StandardScaler were fitted on 4,992 unique events appearing in training inputs. Repeated window appearances do not overweight events. The exact fitted-event manifest and JSON parameters are saved. No target-only/validation/test event enters the fit. Targets use an identity transform; inverse transformation and float32 precision were verified. No outlier clipping was applied. The first undefined inter-event gap is the only expected missing feature value; all saved arrays are finite.

Optional on-the-fly Gaussian augmentation defaults **disabled**. With explicit enablement, σ=0.01 standardized units applies only to historical latitude/longitude, bounded to physical coordinate limits. Seed 20261109. Time/rate, magnitudes, depths, derived summaries, labels and provenance stay unchanged. Validation/test calls are rejected. This is an experimental coordinate sensitivity option, not established seismic measurement noise; no noisy duplicates were saved.

## Validation

| Check | Finding | Evidence |
|---|---|---|
| strict_input_before_target | PASS | All 20 input origin times strictly precede each target origin time. |
| correct_next_event_alignment | PASS | Targets are immediate next events in the retained ML catalog. |
| chronological_target_splits | PASS | Chronological contiguous 70/15/15 target partitions; no shuffling. |
| unique_target_ids | PASS | Every retained target ID occurs in exactly one sample/partition. |
| past_only_feature_prefixes | PASS | Recomputing three catalog prefixes reproduces existing feature rows; summaries never use target/future rows. |
| train_only_fit_ids | PASS | Saved exact unique input-event fit manifest; no validation/test target contributes. |
| train_only_fit_parameters | PASS | Independent training-only median/mean/variance calculation agrees with saved estimators. |
| target_inverse_transform | PASS | Identity target transform; float32 values recover source target magnitudes within serialization tolerance. |
| numerical_artifact_equality | PASS | Every .pt tensor and .npz array exactly matches the in-memory clean array. |
| finite_model_arrays | PASS | All model input and target arrays finite. |
| saved_preprocessing_reconstruction | PASS | Reloaded JSON parameters reproduce all saved X exactly. |
| complete_event_provenance | PASS | Reloaded every row's 20 IDs/times, target identity/time, cutoff and feature dependency interval against retained source catalog. |
| augmentation_label_and_split_safety | PASS | Optional augmentation deterministic; modifies only eligible coordinates in a copy; preserves labels; rejects validation/test. |
| source_csv_unchanged | PASS | Original file SHA-256 unchanged after preprocessing. |
| operational_feature_availability | UNKNOWN | Publication/revision times absent; origin-time causality is retrospective only. |
| source_magnitude_calibration | UNKNOWN | ML labels are taken as reported; conflicting alternatives and unusually high values require independent catalog verification. |
| duplicate_events_under_different_metadata | UNKNOWN | IDs/exact rows/exact physical keys checked; fuzzy duplicates and catalog completeness not independently established. |

Saved .pt/.npz numerical arrays agree exactly; reloaded scaler JSON reproduces inputs, and every provenance row was checked against source records. Original CSV hash is unchanged. Numeric ranges and cross-column conflicts do not establish source calibration, catalog completeness, independent observations or operational availability.

## Methodological scope

[Chen & Fang (2025)](https://doi.org/10.1061/NHREFO.NHENG-2244): the [accessible university abstract](https://scholars.ncu.edu.tw/en/publications/development-of-attention-based-bidirectional-long-short-term-memo/) describes multivariate catalog sliding windows. Full methodology was inaccessible; no exact implementation is attributed to it.

[Al Banna et al. (2021), including Andersson](https://irep.ntu.ac.uk/id/eprint/42763/1/1432454_Mahmud.pdf), Sections III-B/C, supports historical seismicity summaries including elapsed days and mean magnitude, using a different monthly/50-event protocol. Our settings, ordinary rolling standard deviations and split/scaling rules are implementation choices. GR indicators, scale conversion, declustering, HCTSA, attention and energy formulas are excluded.

This branch provides retrospective historical seismicity context. It does not demonstrate real-time P-wave detection, validated earthquake forecasting or seconds-ahead EEW. Future waveform alignment must respect both event identity and actual catalog availability.

## Artifacts and execution

Notebook: ../phase1_bilstm_csv_preprocessing.ipynb (all executable preprocessing and checks).
Dataset files: ../dataset/bilstm_catalog_seq20.pt and ../dataset/bilstm_catalog_seq20.npz.
Schema, target definition, split manifest, scalers, configurations and validation: ../dataset/.
Source/output hashes: ../dataset/catalog_artifact_hashes.json.
The notebook was executed sequentially with one Python namespace, no magics; it is runnable top-to-bottom. Previous artifacts are preserved.
