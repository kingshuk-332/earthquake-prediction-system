# CNN versus BiLSTM dataset comparison

Audit date: 2026-11-06. **Verdict: independent numerical training is feasible; direct output comparison and fusion readiness FAIL.** Scientific validity of the BiLSTM task remains UNKNOWN until its provenance is recovered. No training, model changes or dataset modifications were performed.

## Scope and evidence

Read project `AGENTS.md`, checkpoint, context, task graph and dataset index first. Reused `phase1_waveform_cnn_results/reports/waveform_audit.json`, `verified_metadata.csv`, and the status provenance audit. Fresh SHA-256 checks of all three Phase 11 HDF5 files match that audit; its full-waveform finite/nonzero and exact-duplicate checks therefore remain applicable. Fresh inspection used HDF5 schemas, labels and small metadata arrays, without loading waveform arrays. BiLSTM tensors total about 23 MB: inspected all numerical values and compared all 12 arrays with NPZ; every array is identical. Loaded PT with `weights_only=True`.

Paths below are relative to the workspace root unless explicitly stated. Define **P11** as `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/` and **B** as `bilstm/bilstm_dataset/`.

## Side-by-side comparison

| Property | CNN Phase 11 | BiLSTM seq20 |
|---|---|---|
| Files / format / exact size | P11 `train.h5`: 1,264,629,942 bytes; `validation.h5`: 270,991,791; `test.h5`: 271,200,372; HDF5 | B `bilstm_dataset_seq20.pt`: 23,175,273 bytes, PyTorch dictionary (primary); `bilstm_dataset_seq20.npz`: 887,733 bytes, compressed NumPy equivalent |
| Train / validation / test counts | 11,324 / 2,426 / 2,428 | 5,629 / 1,191 / 1,192 |
| Input storage shape | `(N,10000)`; model receives `(N,1,10000)` | `(N,20,36)` |
| Input dtype | float32 | float32 |
| Input meaning | Single synthetic seismic waveform, 100 Hz, 100 seconds; model uses `waveforms_standardized` | 20 ordered feature vectors, 36 channels; channel names, units and event/station ordering UNKNOWN |
| Targets | `labels`, int8 `(N,)`: 1 synthetic earthquake, 0 synthetic noise; model casts to float32 | float32 `(N,1)` arrays: `y_*`, `y_*_raw`, `y_*_bin`; numeric relationships below |
| Label distribution, train / val / test | Earthquake/noise: 5662/5662; 1213/1213; 1214/1214 | Binary positive/negative: 1932/3697; 73/1118; 47/1145; positives 34.32%, 6.13%, 3.94% |
| Scaling | Linear detrending, then per-trace z-score with epsilon 1e-8; alternate `waveforms_global_norm` uses training mean/std | X train range [0,1], val [-0.002616,1], test [-0.077151,12.735738]; transform and fitting population UNKNOWN. Target scaling numerically recovered below |
| Missing / invalid inputs and targets | PASS: cached full raw/standardized waveform checks finite and nonzero; fresh labels valid | PASS: all 12 tensors contain no NaN or infinity; range excursions alone are not invalid values |
| Event metadata | `source_id`, `trace_name`, `pair_id`, `sample_type`; magnitude, depth, hypocentral distance, P/S indices, SNR, QC | No IDs, timestamps, names, schema, scaler or split manifest in either archive |
| Time / location / station | HDF5 has `receiver_code`, but no absolute origin time or event latitude/longitude; cached earthquake joins provide origin time, Lat/Long and receiver latitude/longitude | Explicit event time/location/station metadata absent; whether unnamed feature columns encode these is UNKNOWN |
| Split evidence | Pair/source-disjoint classification splits; not chronological | Separate arrays; adjacent windows overlap by 19 feature vectors within every split. Chronological order and boundary policy UNKNOWN |

CNN metadata fields `magnitude`, `depth_km`, `hypocentral_distance_km`, `snr_db` have 5662/1213/1214 nonfinite values respectively across train/val/test, matching noise counts. The cached audit identifies these as intentional noise missingness; a fresh train check confirms NaNs occur exactly on noise rows. Noise uses receiver `NONE`, synthetic noise IDs and arrival sentinel -1. Ten real receiver codes plus `NONE` occur in each split. These metadata must not become classifier inputs: their missingness and sentinels reveal the label. They are not the CNN's inputs.

BiLSTM archive keys are `X_train`, `X_val`, `X_test` and, for each split suffix, `y_*`, `y_*_raw`, `y_*_bin`. A separate station-grouped NPZ exists but is outside this requested comparison and was not substituted.

### BiLSTM target distributions

| Split | Raw min / Q1 / median / Q3 / max | Raw mean ± population SD | Scaled mean ± population SD |
|---|---|---|---|
| Train | 1.5 / 3.4 / 4.1 / 4.7 / 8.6 | 4.08584 ± 0.87907 | 0.36420 ± 0.12381 |
| Validation | 1.5 / 3.0 / 3.5 / 3.8 / 7.5 | 3.47380 ± 0.62033 | 0.27800 ± 0.08737 |
| Test | 1.5 / 2.7 / 3.0 / 3.5 / 6.0 | 3.13549 ± 0.65423 | 0.23035 ± 0.09214 |

Every binary target equals `y_raw >= 4.5`. Every scaled target agrees with `(y_raw - 1.5) / 7.1` within 1.2e-7. These facts strongly suggest magnitude regression and significant-magnitude classification, but **physical target meaning is UNKNOWN for this exact archive** without a schema/generator. The denominator matches the training target range; this does not prove training-only fitting of either targets or features.

## Target compatibility and objectives

- **PASS — CNN objective established:** probability that the observed synthetic trace contains an earthquake, evaluated with detection accuracy, precision, recall, F1 and related classification metrics. This is existing-event detection, not future forecasting.
- **UNKNOWN — exact BiLSTM objective:** the archive supports scaled continuous regression or binary classification. No linked seq20 model/config establishes which target a model actually uses.
- **UNKNOWN — future versus existing event:** no target event IDs, sequence member IDs, issue timestamps, horizon or target offset are stored. Window overlap proves sequence construction, not chronological ordering or next-event prediction.
- **FAIL — equivalent targets:** the supplied CNN earthquake/noise labels and BiLSTM raw-threshold labels have different definitions. No evidence establishes equivalent endpoints.
- **FAIL — fair direct metric comparison:** regression errors cannot be compared with detection accuracy; even classification metrics would measure different labels, prevalence and evaluation examples. Do not rank models using their existing scores.

Legacy `status.md` and `data wave/README.md` describe 15-event, 44-feature next-event magnitude classification, not this 20-by-36 archive. The root `preprocess_dl_temporal.py` produces 44 features with full-catalog MinMax fitting. Neither establishes seq20 feature names, preprocessing or target timing. Their statements must not be transferred to this archive.

## Split compatibility and event alignment

| Check | Result | Evidence / limit |
|---|---|---|
| CNN pair/source separation | PASS | Hash-validated prior audit; earthquake/noise pairs stay together |
| CNN chronological forecasting split | FAIL | Train earthquake dates span 1908–2026, validation 1906–2026, test 1922–2026; historical periods overlap |
| Exact BiLSTM feature-vector overlap across partitions | PASS | SHA-256 comparison of float32 36-value vectors found zero shared vectors for each split pair; this is not proof of event disjointness |
| BiLSTM chronological separation / embargo / training-only preprocessing | UNKNOWN | No dated manifest or scaler provenance |
| Overlapping earthquake origins across the two datasets | UNKNOWN | CNN catalog lineage is documented; seq20 lineage and event identifiers are absent |
| Common usable event IDs | FAIL | CNN has IDs; neither BiLSTM archive supplies them |
| Verified waveform-to-sequence join | FAIL | No member/target IDs or timing contract; no defensible join currently exists |
| Shared split assignment for fusion | UNKNOWN | Cannot compare assignments without event-level joins; current partitions are not certified compatible |

The CNN's existing `verified_metadata.csv` provides joined earthquake origins and coordinates, with noise metadata intentionally missing. This is useful for a future manifest but does not recover BiLSTM identities. Similar magnitude distributions, sequence counts, or positional ordering are not linkage evidence. **No alignment by row number or sample count was performed.**

## Leakage risks

1. **Future-information leakage:** pairing a future-event target with that same event's synthetic waveform would reveal post-event information and generator inputs. Such a branch cannot support a pre-event forecast. For existing-event detection, observing the trace is legitimate, but a clear observation cutoff is still required for an early-warning claim.
2. **Label leakage through metadata:** CNN noise sentinels/NaNs and earthquake-only magnitude/location availability expose class identity. BiLSTM feature meanings are unknown, so direct or derived target inclusion cannot be ruled out.
3. **Split and pretraining contamination:** CNN random historical splits may include events from a future fused validation/test period in CNN training. Reusing its frozen weights does not remove this exposure. All supervised pretraining must respect the final evaluation exclusions.
4. **Unverified preprocessing:** BiLSTM scalers, clipping/imputation statistics and their fitting sets are absent. Full-catalog preprocessing is a demonstrated legacy practice, but its use here remains UNKNOWN.
5. **Correlated windows and synthetic shortcuts:** all adjacent BiLSTM windows share 19/20 vectors within splits; they are not independent examples. Cross-split identity and temporal leakage require event manifests. CNN exact duplicate checks pass, but exhaustive near-duplicate checks and real-data generalization remain unresolved; generator-specific cues can inflate synthetic scores.

## Hybrid feasibility and minimum changes

1. **Can they train independently? PASS for numerical usability.** CNN already has an established synthetic detection protocol. BiLSTM tensors can be supplied to a model, but meaningful scientific training/evaluation is **UNKNOWN** until feature and target provenance is restored. No training is recommended yet.
2. **Can outputs be compared directly? FAIL.** They address different supplied endpoints.
3. **Can learned features be combined? PASS in architectural principle; FAIL for these datasets as presently documented.** Attention or concatenation cannot establish sample identity, compatible labels or causality.
4. **Missing alignment information:** stable sequence IDs; ordered member event IDs; target event ID; source-catalog version/hash; feature names/units/order; station and waveform IDs; origin and data-availability times; forecast issue time; target offset/horizon; scaler parameters and fitting rows; split-generation policy and assignments.
5. **Minimum scientific repair, before any hybrid training:**
   - Recover the exact seq20 builder/schema and produce a verifiable manifest for each sequence and target. Join by explicit event/station IDs and timestamps, validate cardinality and unmatched records, and retain waveform IDs. If provenance cannot be recovered, prepare a new documented dataset later; preserve these archives.
   - Choose one common endpoint. For detection/estimation, create paired observed-event examples with a defined causal catalog history and compatible labels; detection also needs legitimate negative examples. For forecasting, define issue time and horizon and use only historical waveforms/events available before issue time, never the target-event waveform. Fixed-horizon occurrence forecasts also require no-event windows.
   - Establish one shared split policy before preprocessing or supervised pretraining. Forecasting requires chronological cutoffs and boundary checks for history, target intervals and availability; use appropriate purge/embargo rules. Keep each event, waveform variants and earthquake/noise pair in one partition where relevant. Document any permitted historical context reuse.
   - Fit preprocessing on training data, audit target-derived features, and retrain encoders if existing weights have seen held-out events. Compare CNN-only, BiLSTM-only and fusion on the same endpoint, paired evaluation set, metrics and fixed test protocol.

**Recommended next step:** recover the seq20 generation provenance and event/target manifest, then decide detection versus forecasting. Do not begin BiLSTM or CNN+BiLSTM+Attention training from the current unaligned archives.

## BiLSTM fingerprints

- PT SHA-256: `aa35f5d28fd1c84391940a16bad83ad1e35a6efb8d846b46df2c470c9049f3e3`
- NPZ SHA-256: `8b7f455cd434e6118c4921dcfc4150459bf7ce4300cfecd2ecd3a148285008ae`

CNN fingerprints and prior exhaustive checks remain recorded in `phase1_waveform_cnn_results/reports/waveform_audit.json`.
