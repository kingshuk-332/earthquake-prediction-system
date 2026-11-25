# CNN research status
## EEW feasibility & causality audit — 2026-11-10
- Completed read-only EEW audit. Created `cnn_eew_feasibility_report.md` and compact `cnn_eew_opus_handoff.md`.
- Overall EEW Capability: **FAIL**. The 99.84% test accuracy does not reflect early warning performance.
- Key findings:
  1. Input is full 100-second trace (10,000 samples) with global average pooling. No streaming, localized, or early alert capability.
  2. Model is strictly offline binary classifier (Noise vs Earthquake). Does not predict P onset, S arrival, magnitude, or lead time.
  3. Simulator noise scaling defect verified (`improved_waveform_generator.py:224-243`): pre-P noise is scaled by full-trace peak (dominated by future S-wave), causing acausal future leakage (confirmed by counterfactual $\Delta=0.5076$ on trace NCS_EV_07168).
  4. Full-trace linear detrending and z-score standardization leak future S-wave energy into early samples.
  5. P-arrival is artificially fixed at $t \approx 6.00$ s across all earthquakes.
- Minimum corrections identified: short causal post-P windowing ($1–3$ s), pre-P independent noise calibration, running pre-event normalization, and real-seismogram benchmarking.
- STOP: No models retrained, no datasets regenerated, no fusion attempted.

## Frozen CNN overfitting audit — 2026-11-06
- Executed and validated `phase1_cnn_overfitting_audit.ipynb`; outputs: `phase1_waveform_cnn_results/overfitting_audit/` (split metrics, gaps, errors, stress results, report, verification and four figures). Reproduce with `python run_overfitting_audit.py` from this directory. No training, threshold tuning or new trained experiment; registry unchanged.
- Best epoch 22/25, evaluated in eval mode on all splits: train/validation/test accuracy **99.7351% / 99.7115% / 99.8353%**, BCE **0.007772 / 0.010031 / 0.005562**. Train-minus-validation gap **+0.023617 percentage points**; train-minus-test **−0.100179 points**. No clear ordinary overfitting or underfitting within the synthetic distribution; late validation fluctuations do not show sustained divergence.
- Leakage: none detected by reused exact-duplicate, pair/source/event and alignment checks; exhaustive near-duplicate checking remains unresolved. Phase 11 paths and standardized input confirmed; global normalization is not used.
- Test: 1 false positive (transient-noise subgroup), 3 false negatives (all M<3, SNR<10). Earthquake-only accuracy/recall: M<3 98.4615% (195); M3–5 100% (888); M>=5 100% (131); SNR<10 99.5708% (699); SNR>=10 100% (515).
- Cached phase-randomization accuracy 94.1928%, reversal 99.7117%, shift 99.7529% support spectral/background shortcut concerns without proving a specific mechanism. New fixed-seed white, low-pass and impulsive noise controls: each 0/256 false positives (Wilson 95% FPR interval 0–1.4784%); negative controls do not establish independent-earthquake or field generalization.
- Ready for BiLSTM comparison: **Yes, as a controlled synthetic baseline only**, with fixed inputs/splits and validation-only selection; reserve an untouched holdout for broader claims. No BiLSTM implemented.
- Verification: original notebook/model hashes unchanged, HDF5 read-only and size/mtime unchanged, saved validation/test predictions reproduced. Initial audit-only syntax issue corrected; existing runtime needed approved sandbox escalation; a maximum 0.0000166 validation-probability discrepancy with zero class changes was resolved by matching original batch composition and deterministic CUDA settings. Final notebook has no execution errors.

## Dataset provenance verification — 2026-11-06

**Confirmed:** the saved `phase1_waveform_cnn.ipynb` routes all CNN training, validation and held-out testing waveform inputs exclusively through the three Phase 11 HDF5 files below, using `waveforms_standardized`. This finding follows the notebook's actual loader and model calls, the audit's resolved paths, the execution runner's working directory, saved configuration/preprocessing, and fresh read-only HDF5 schema/label/hash checks. No notebook cells were executed, no model was modified or retrained, and no datasets were written during this verification. Historical execution is supported by the saved notebook/artifacts; this is not a new execution trace.

Resolved directory: `C:/Users/choud/Desktop/Minor project/cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/`.

| File in that exact directory | CNN key / dtype | Array shape | Labels shape / stored dtype | Earthquake (1) | Noise (0) |
|---|---|---|---|---:|---:|
| `train.h5` | `waveforms_standardized` / float32 | (11324, 10000) | (11324,) / int8 | 5662 | 5662 |
| `validation.h5` | `waveforms_standardized` / float32 | (2426, 10000) | (2426,) / int8 | 1213 | 1213 |
| `test.h5` | `waveforms_standardized` / float32 | (2428, 10000) | (2428,) / int8 | 1214 | 1214 |

All three files have the same 16 dataset keys. For split size N:

- `waveforms`, `waveforms_standardized`, `waveforms_global_norm`: (N, 10000), float32. Only `waveforms_standardized` enters the CNN.
- `labels`: (N,), int8; loaded as float32 targets, never concatenated into the waveform inputs. First N/2 rows are earthquake (1); remaining rows are synthetic noise (0), freshly checked in every file.
- `depth_km`, `hypocentral_distance_km`, `magnitude`, `snr_db`: (N,), float32.
- `p_arrival_sample`, `s_arrival_sample`, `pair_id`: (N,), int32.
- `qc_pass`: (N,), bool.
- `receiver_code`, `sample_type`, `source_id`, `trace_name`: (N,), HDF5 strings (h5py object dtype).

Preprocessing: the upstream `phase11_split_and_preprocess.py`, lines 177–197, linearly detrends each entire 10,000-sample trace, casts to float32, and applies `(detrended - per_trace_mean) / (per_trace_std + 1e-8)` with NumPy's population standard deviation. These already processed arrays are stored in `waveforms_standardized`; the notebook does not fit or apply another scaler to CNN inputs. The input is one vertical component, 100 Hz, 100 seconds. The loader adds the channel dimension, yielding float32 `(batch, 1, 10000)` and labels `(batch, 1)`. It reads two class blocks of up to 100 traces, shuffles for training, and emits minibatches of at most 32. No cropping, resampling, metadata concatenation or training augmentation appears in that loader. `waveforms_global_norm` is an alternative stored representation; its training mean/std are recorded in preprocessing metadata but are **not applied to the selected CNN input**.

Exact cell references (one-based notebook position includes Markdown; JSON index is zero-based; execution counters refer to the saved notebook):

| Notebook cell | JSON index | Saved execution | Role |
|---|---:|---:|---|
| 2 | 1 | 1 | Resolve BASE, load waveform_audit.json, define PATHS and waveform_key |
| 4 | 3 | 2 | Check source hashes, load labels from each PATHS file, assert shape/class order; read metadata/features |
| 6 | 5 | 3 | Plot training examples from the same standardized key |
| 8 | 7 | 4 | Define the sole waveform minibatch loader and evaluate() |
| 9 | 8 | 5 | Train via batches('train', epoch, True); select checkpoint using evaluate('validation') |
| 11 | 10 | 6 | Fit separate signal-feature logistic/STA-LTA baselines, not the CNN |
| 13 | 12 | 7 | evaluate(split) for validation/test; join noise mapping for subgroup reports |
| 15 | 14 | 8 | Frozen-model diagnostic transformations of Phase 11 test inputs |
| 17 | 16 | 9 | Plot test examples from the same standardized key |
| 19 | 18 | 10 | Reload saved weights and evaluate('test') for reproducibility |

The runner `execute_waveform_notebook.py` sets the kernel working directory to `cnn/cnn wave`; the notebook also supports starting at the repository root. PATHS are taken from `phase1_waveform_cnn_results/reports/waveform_audit.json`, whose three split paths are exactly `data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/{train,validation,test}.h5`. No later PATHS or waveform_key reassignment appears in the notebook.

Files read for other purposes must not be confused with CNN waveform inputs:

- `phase1_waveform_cnn_results/reports/waveform_audit.json`: selects split paths and supplies cached provenance, checks and normalization documentation.
- `verified_metadata.csv` in that reports directory: labels cross-checks, reporting, prediction tables and subgroup analysis; metadata never enters model(x).
- `signal_features.csv` in that reports directory: plots and separate signal/RMS logistic and STA/LTA baselines. These are inputs to comparison models, not CNN features.
- Phase 10 `earthquake_noise_mapping.csv`: notebook subgroup reporting and source hashing; Phase 9 `synthetic_waveforms_mapping.csv`, Phase 2 `metadata_with_theoretical_arrivals.csv`, Phase 11 `split_indices.csv` and `normalization_parameters.json`: provenance hash checks in the notebook, with joins/preprocessing checks performed by the prior audit. The Phase 2 join supplies origin time/coordinates for reporting.
- Phase 9 `synthetic_seismic_waveforms.h5` and Phase 10 `earthquake_noise_dataset.h5`: upstream generation/provenance and sampled alignment checks in `audit_waveform_data.py`; neither is opened by the notebook as a CNN input. Phase 11 arrays were historically derived from upstream simulator data, so exclusivity refers to the files loaded by this CNN, not an absence of upstream ancestry.
- Earlier catalog CSVs, forecasting notebook/results and simulator documentation: separate experiment/context and preservation checks, not waveform CNN inputs.
- Test shortcut probes shift, reverse, phase-randomize or repeat/renormalize prefixes from the same Phase 11 test array after training; they do not introduce another input dataset or change the ordinary held-out evaluation.

Fresh SHA256 checks matched all eight entries in the saved waveform audit, including:

| File | SHA256 |
|---|---|
| train.h5 | `2a80c55bfd2d990ebfb737cbd55d0f2b4136cb5e4c6d016712021a6b2de861db` |
| validation.h5 | `2fd21debed351636ce56e6979fbf75bf9d77f5f2ad4dc8c1a0773685c93b7936` |
| test.h5 | `e45958510d32224154cb078d09b001e3cea58e7507d2b7e7cdf058461cf23e5d` |

Inspected notebook SHA256: `cd0a7e1dcbe61ca88b7cec890e9bb6b9fd36c06ce020528b76b3c9eda1c58469`. The exact loading excerpts are recorded at the end of this status file.

## Current Phase — corrected waveform baseline complete
The intended waveform Experiment A is complete and verified: `wavecnn_ccb3b06b3d37`. It is earthquake/noise detection, **not forecasting**. The earlier catalog forecasting study remains preserved below; its notebook and every fingerprinted artifact are unchanged. No BiLSTM or hybrid model has been started.

## Waveform dataset and audit
Files: `data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/train.h5`, `validation.h5`, `test.h5`; key `waveforms_standardized`. Shapes 11324/2426/2428 x 10000; 100 Hz /100 seconds; one vertical component. Each split is exactly 50/50 earthquake/noise, label1/0. All source and pair sets are disjoint; no exact raw/standardized trace duplicates or physical-event-key duplicates. All traces were checked for numerical integrity and normalized moments; global training normalization was independently recomputed. Formula/upstream alignment checks used 12 seeded traces per split. Exhaustive near-duplicate checks remain unperformed.
Magnitude, depth, distance, P/S indices, SNR, QC and IDs are stored in HDF5. Origin timestamps and coordinates are absent; verified joins to Phase 2 metadata supply them for reporting only. P onsets range 592–607 samples across splits, reflecting the approximately 6-second anchoring convention. No event metadata enters the CNN.

## Waveform training configuration
Input (10000,1), internally PyTorch (batch,1,10000). Conv8 k9/s2 +BN/ReLU/pool4; Conv16 k7/s2 +BN/ReLU/pool4; Conv32 k5/ReLU/global average; dense16/ReLU/dropout0.3; one sigmoid probability. 4177 parameters, seed42, deterministic CUDA on RTX 3050, AdamW lr0.001/weight decay0.0001, batch32, max25 epochs, validation-BCE early stopping patience5, LR-halving patience2. Two 100-trace class blocks form a small CPU buffer; minibatches alone go to GPU. Each training trace appears once per epoch. Threshold0.5 fixed before test.

## Verified waveform findings — 2026-11-05
The CNN has 4,177 trainable parameters. It completed 25 epochs and restored epoch 22 (validation BCE 0.010031).
Held-out test: accuracy 0.998353, precision 0.999175, recall 0.997529, F1 0.998351, ROC-AUC 0.999995, average precision 0.999995, trapezoidal PR-AUC 0.999995.
Confusion matrix (rows true noise/EQ, columns predicted noise/EQ): [[1213, 1], [3, 1211]].
The five-signal-feature logistic baseline has accuracy 0.915568 and F1 0.913902. This is an in-simulator comparison, not an external field benchmark.

### What the shortcut probes show
- Circular 25 s shift: accuracy 0.997529, AUC 0.999996.
- Time reversal: accuracy 0.997117, AUC 0.999988.
- Spectrum-preserving phase randomization: accuracy 0.941928, AUC 0.973689.
- Pre-onset-noise-only: overall false-positive rate 0.067957; noise taken from earthquake traces 0.071664; noise taken from synthetic-noise traces 0.064250. Its AUC against original source labels is 0.563610; those original labels are diagnostic group labels, not correct event labels after signal removal.

These probes do not establish exactly which physical features are learned. The CNN separates the supplied generated classes; disruption of phase/ordering tests sensitivity to waveform structure, while remaining discrimination can expose generator-specific spectral/background differences. Noise-only probes are derived after full-trace detrending, which itself can leave a class-associated linear-trend imprint; this is another preprocessing shortcut risk. Repetition introduces artificial boundaries. Treat all probes as diagnostics, not causal proof or realistic out-of-distribution performance estimates.

Experiment B is **not trained**. Earthquake magnitude/P/S labels are verified; origin times and coordinates require source-ID joins. Standardization removes absolute amplitude, and waveform shape is generated from known metadata formulas. Absolute origin time and unique epicentral coordinates cannot be recovered from this single clock-free vertical trace. Separate magnitude-inversion/picking studies need explicit baselines and independent data.

Paired-bootstrap accuracy difference (CNN minus signal logistic): 0.082784, 95% interval [0.071664, 0.093904]. This interval covers sampled pairs, not uncertainty across generator families or seeds.


## Waveform outputs and verification
- Notebook: `phase1_waveform_cnn.ipynb` — 10 code cells executed sequentially in a fresh kernel without errors.
- Best model: `phase1_waveform_cnn_results/models/wavecnn_ccb3b06b3d37/best.pt`; full resumable state: `last.pt`; normalization: `preprocessing.json`.
- Metrics, predictions, subgroup results, confusion matrix, baselines and probes: `phase1_waveform_cnn_results/metrics/wavecnn_ccb3b06b3d37/`.
- Six figures: `phase1_waveform_cnn_results/plots/wavecnn_ccb3b06b3d37/`.
- Audit: `phase1_waveform_cnn_results/reports/waveform_audit.json`; final report/checks: `phase1_waveform_cnn_results/reports/wavecnn_ccb3b06b3d37/`.
- Reloaded model reproduces test predictions; independent saved-file metrics match; original source hashes unchanged; full optimizer/scheduler/RNG checkpoint verified. No essential waveform verification remains incomplete.

## Waveform errors and fixes
Initial sandbox execution could not read the previously installed notebook runtime (PermissionError). Approved execution outside the sandbox succeeded. No additional packages were installed or global environment changed. Existing torch/sympy environment mismatch remains documented from the catalog experiment; actual eager CUDA training passed. Jupyter emitted its existing Windows selector/local TCP warnings; execution completed.

## Waveform limitations and next steps
High synthetic accuracy does not establish physical feature learning or real-field generalization. Fixed onset, narrow synthetic harmonics, designed noise profiles, artificial class balance and full-trace normalization create shortcut opportunities. Pair-disjoint splitting prevents direct event/pair overlap but not shared generator bias. Optional parameter estimation was audited, not implemented. Before BiLSTM, retain this exact detection protocol for a fair comparison and add real recordings, independent generators, varied onsets and station/generalization tests. Keep forecasting separate and pre-event only.

## Dated waveform milestones — 2026-11-05
W1 streaming dataset audit and split verification completed; W2 detection/parameter-feasibility decision completed; W3 actual GPU CNN training completed; W4 held-out evaluation and shortcut probes completed; W5 fresh-kernel and external artifact verification completed. Memory, task graph and registry updated. Reproduce via `python "cnn/cnn wave/execute_waveform_notebook.py"`; see `WAVEFORM_REPRODUCE.md`.

---
## Preserved catalog forecasting experiment — original report
## Project Overview
Investigate probabilistic future recorded seismicity in Northeast India and adjacent catalog regions. This is retrospective research, not deterministic prediction or an operational warning system.

## Model Development Roadmap
- [x] Phase 1: CNN baseline — trained, evaluated and verified; no superiority established.
- [ ] Phase 2: BiLSTM.
- [ ] Phase 3: CNN + BiLSTM.
- [ ] Phase 4: CNN + BiLSTM + Attention.
- [ ] Final model comparison.

## Current Phase
Phase 1 completed on 2026-11-05. Experiment `cnn_b86050c74334`. Phases 2–4 were not implemented.

## Dataset Investigation
Raw `../../dataset_2.csv`: 8,089 x 28, UTC 1906-08-31 14:57:30 through 2026-10-06 09:03:32; sorted, no missing values, duplicate rows, source IDs or event-key duplicates. Coordinates 20–30.5 N / 87–98.5 E; depth 1–430 km; magnitude 1.5–8.6. Magnitude types: ML 7125, MB 458, MW 398, MC 107, MS 1. No verified publication-time or coverage metadata.
Selected 5,603 records from 2011-01-01 through 2026-10-06 exclusive, removing the partial last day. Start year is exploratory, not a demonstrated completeness threshold. Source SHA256 `5cfebe7268a87940ec9489d4f72b45381e2c940cc3f38d96d60b4575355d8bf3`.
Inspected 28 datasets and 19 relevant sources. Full paths, shapes, columns, missing values and purposes: [.agent-memory/dataset_index.md](.agent-memory/dataset_index.md). HDF5 phase 6/9: 8089x10000; phase 10: 16178x10000; phase 11 train/validation/test: 11324/2426/2428x10000 with earthquake/noise labels. These are excluded from forecasting.

## Research References
[Perol, Gharbi & Denolle (2018), Science Advances](https://doi.org/10.1126/sciadv.1700578), [author site](https://groups.csail.mit.edu/graphics/convnetquake/), [author preprint](https://arxiv.org/pdf/1702.02073). Adopt local convolutional feature extraction and regularization. Do not transfer waveform detection labels, detection performance claims, balanced sampling or full-waveform normalization into forecasting.

## Decisions and Justifications
Historical daily bins retain negative windows and forecast time exposure. Use 90 days of 12 features: counts (all/M4/M4.5), max/mean magnitude, mean depth/latitude/longitude, four cell counts. Fixed empty-bin zeros and log1p counts; standardize unique training-history days only.
Predict seven marginal probabilities for M>=4.5: whole next 7 days, days [0,3), days [3,7), and four cells within 7 days. Bounds [20,32) N x [86,100) E, split at 26 N/93 E. Not a mutually exclusive location distribution. Training support: 212/562 positive windows versus 75/562 for M5; coarser horizons lose temporal resolution. SW has only 15 training positives; spatial estimates remain exploratory.
Calendar 70/15/15 with full history+horizon contained in each partition and weekly issue spacing. Model selection uses validation mean BCE only. Test does not fit model, scaler or threshold.

| Split | Calendar partition (end exclusive) | Forecast issues | Samples | Primary positives | Events in target windows / M4.5 events |
|---|---|---|---:|---:|---:|
| Train | 2011-01-01 to 2022-01-12 | 2011-04-01 to 2021-12-31 | 562 | 212 | 2373 / 296 |
| Validation | 2022-01-12 to 2024-05-25 | 2022-04-15 to 2024-05-17 | 110 | 29 | 761 / 33 |
| Test | 2024-05-25 to 2026-10-06 | 2024-08-23 to 2026-09-25 | 110 | 62 | 2257 / 114 |

## Tasks Completed
- 2026-11-05 initial inspection: raw catalog audit, GPU check and memory system.
- 2026-11-05 audit milestone: all eleven simulator stages, prior model/preprocessing, 28 dataset schemas and 19 source hashes indexed. Existing full-catalog normalization and same-event waveform risks documented.
- 2026-11-05 implementation: 24-cell notebook with all 20 requested sections and exact-label/date/scaler assertions.
- 2026-11-05 training: 21 epochs on RTX 3050, best epoch 6 restored; complete resumable state saved each epoch.
- 2026-11-05 verification: fresh Jupyter kernel executed all 12 code cells without errors; model reload reproduces test predictions; saved test scores independently recomputed; all artifact and source-preservation checks passed. Six figures saved and calibration plot visually inspected.

## Tasks Remaining
No essential Phase 1 implementation/verification task remains. Research improvements are listed below and later-phase models remain unstarted.
- [ ] Verify catalog provenance/completeness and event reporting delays.
- [ ] Harmonize magnitude types and validate outage handling.
- [ ] Pre-register rolling-origin/multiple-seed comparisons and a fresh prospective holdout.
- [ ] Assess a stronger triggering-process baseline and coherent output formulation.

## Training Configuration
PyTorch 2.6.0+cu124, Python 3.13.12, NVIDIA GeForce RTX 3050 Laptop GPU; complete versions in `phase1_cnn_results/reports/cnn_b86050c74334/environment.json` and `requirements-phase1.txt`.
Seed 42; deterministic algorithms enabled. Conv1d 12->16 (k5,pad2), BN/ReLU, Conv1d 16->32 (k3,pad1), ReLU/global average, dense32->16/ReLU/dropout0.25, dense16->7. **3,223 trainable parameters**.
Unweighted BCEWithLogitsLoss; AdamW lr0.001/weight decay0.001; batch32; maximum100 epochs; early-stop patience15; LR-halving patience5. No SMOTE, test calibration or test threshold tuning. `last.pt` includes optimizer, scheduler and all RNG states; repeated completed configurations restore weights rather than retrain.

## Training Results
Actual epochs **21**, restored epoch **6**, best mean validation BCE 0.357905.

| Model | Test Brier ↓ | Test log loss ↓ | Average precision ↑ |
|---|---:|---:|---:|
| CNN | 0.281609 | 0.758084 | 0.567793 |
| training_Poisson | 0.269457 | 0.732576 | 0.563636 |
| recent_Poisson | 0.242390 | 0.680387 | 0.649873 |
| training_prevalence | 0.280619 | 0.755973 | 0.563636 |

CNN precision/recall at 0.5 are both 0: all predicted probabilities fall below 0.5. Primary CNN predictions range 0.3725–0.3768, versus test frequency 0.5636, showing poor adaptation/calibration and weak discrimination. Validation prevalence is 0.2636; training 0.3772. The recent-rate baseline performs better in this split. CNN-minus-recent Brier difference +0.039219, four-week block-bootstrap 95% interval [-0.011912,0.089982]; small dependent samples prevent a strong superiority claim. CNN spatial mean Brier 0.131242 vs recent 0.125200; time-window mean Brier 0.234552 vs recent 0.225823.
Saved model: `phase1_cnn_results/models/cnn_b86050c74334/best.pt`; resume: `last.pt`; preprocessing: `preprocessing.json`.
Metrics/predictions/calibration/history: `phase1_cnn_results/metrics/cnn_b86050c74334/`; six figures: `phase1_cnn_results/plots/cnn_b86050c74334/`; report and verification: `phase1_cnn_results/reports/cnn_b86050c74334/`.

## Errors and Fixes
- PMC full-text endpoint returned a browser challenge; retrieved the author project and accessible author preprint, with published-paper metadata/search content for cross-checking.
- Jupyter execution packages were absent. Network sandbox blocked PyPI; approved installation to `.notebook-runtime` succeeded. A premature runner invocation failed before installation finished; rerun after completion.
- Sandbox could not read installed runtime files (PermissionError). Approved notebook execution outside that sandbox succeeded.
- pip reported a pre-existing torch/sympy version mismatch (torch wants 1.13.1; global sympy is 1.14.0). No global packages were changed; actual eager training and checks passed. Exact environment recorded; resolve in a separate environment before broader replication.
- Windows Jupyter emitted selector/TCP warnings; local kernel execution completed. No training/data/verification failure remains.
- The first external artifact check expected seven PNG files; the notebook intentionally produces six multi-panel figures covering all requested plots. Corrected the check to six and reran independent verification.

## Scientific Limitations
Unknown completeness, catalog revisions/reporting delays and mixed magnitude scales; modern start is heuristic. Quiet bins are absence of recorded events, not proven seismic inactivity. Geographic domain includes neighboring countries. Single seed and exploratory temporal holdout, sparse cells and correlated histories; no prospective validation. Marginal outputs have no enforced joint coherence. Poisson baselines omit triggering. Global pooling discards exact timing; near-constant forecasts suggest insufficient learned signal. Mild validation deterioration is controlled by early stopping, but this does not solve distribution shift. No deterministic/operational earthquake-prediction claim is justified.

## Next Steps
Improve data provenance/completeness, magnitude consistency and availability assumptions before BiLSTM. Freeze identical inputs, targets, split manifests and baselines for fair comparison; document test reuse and reserve a new holdout. No Phase 2 work has begun.

## Reproduction and session resume
Read `AGENTS.md`, then `.agent-memory/project_context.md`, `checkpoint.md`, `task_graph.md`. Run from repository root: `python "cnn/cnn wave/execute_phase1_notebook.py"`. The runner uses the existing project-local runtime if present; install notebook dependencies in a suitable environment on another machine. It executes a clean kernel and restores completed matching runs. Change the experiment version for protocol changes; do not overwrite prior runs. See `REPRODUCE.md`.


## Exact data-loading excerpts — verification 2026-11-06

Verbatim excerpts from the inspected notebook; cell positions include Markdown.

### Cell 2 / JSON index 1 / In [1]: path and key selection

```python
BASE=Path.cwd()
if not (BASE/'data wave').is_dir():BASE=BASE/'cnn'/'cnn wave'
RES=BASE/'phase1_waveform_cnn_results';MEM=BASE/'.agent-memory'
AUDIT=json.loads((RES/'reports'/'waveform_audit.json').read_text())
PATHS={s:BASE/AUDIT['splits'][s]['path'] for s in ['train','validation','test']}
CONFIG=dict(version='waveform-detection-v1',seed=42,waveform_key='waveforms_standardized',
    input_samples=10000,sampling_hz=100,batch_size=32,hdf5_block_rows=100,filters=[8,16,32],
    kernels=[9,7,5],strides=[2,2,1],pool_size=4,dropout=.3,learning_rate=.001,
    weight_decay=.0001,max_epochs=25,patience=5,threshold=.5)
```

### Cell 4 / JSON index 3 / In [2]: label loading and metadata checks

```python
for rel,digest in AUDIT['source_hashes'].items():assert sha(BASE/rel)==digest,'Source changed: refresh audit'
META=pd.read_csv(RES/'reports'/'verified_metadata.csv')
FEATURES=pd.read_csv(RES/'reports'/'signal_features.csv')
assert all(AUDIT['checks'][k] for k in ['all_waveforms_finite_nonzero','all_labels_and_mapping_aligned','pairs_and_sources_disjoint','no_duplicate_event_keys','no_exact_raw_or_standardized_duplicates'])
labels={};metas={}
for s,p in PATHS.items():
    with h5py.File(p,'r') as f:
        labels[s]=f['labels'][:].astype(np.float32)
        assert f[CONFIG['waveform_key']].shape==(len(labels[s]),10000)
        half=len(labels[s])//2
        assert (labels[s][:half]==1).all() and (labels[s][half:]==0).all()
    metas[s]=META[META.split.eq(s)].sort_values('row').reset_index(drop=True)
    assert np.array_equal(labels[s],metas[s].labels)
```

### Cell 8 / JSON index 7 / In [4]: complete minibatch loader

```python
def batches(split,epoch=0,shuffle=False):
    n=len(labels[split]);half=n//2;block=CONFIG['hdf5_block_rows'];rng=np.random.default_rng(CONFIG['seed']+epoch)
    order=np.arange(0,half,block)
    if shuffle:rng.shuffle(order)
    with h5py.File(PATHS[split],'r',rdcc_nbytes=32*1024*1024) as f:
        for start in order:
            stop=min(start+block,half)
            rows=np.r_[np.arange(start,stop),np.arange(half+start,half+stop)]
            x=np.concatenate([f[CONFIG['waveform_key']][start:stop],f[CONFIG['waveform_key']][half+start:half+stop]])
            order2=np.arange(len(rows))
            if shuffle:rng.shuffle(order2)
            for j in range(0,len(rows),CONFIG['batch_size']):
                sel=order2[j:j+CONFIG['batch_size']];rr=rows[sel]
                yield torch.from_numpy(x[sel,None,:].copy()),torch.from_numpy(labels[split][rr,None].copy()),rr
```

### Cell 8 / JSON index 7 / In [4]: evaluation input path

```python
def evaluate(split,transform=None):
    model.eval();probs=np.empty(len(labels[split]),dtype=np.float64);total=0.;seen=np.zeros(len(probs),dtype=int)
    with torch.no_grad():
        for x,y,rows in batches(split):
            if transform is not None:x=transform(x,rows)
            logits=model(x.to(DEVICE));total+=loss_fn(logits,y.to(DEVICE)).item()*len(rows)
            probs[rows]=torch.sigmoid(logits).cpu().numpy().ravel();seen[rows]+=1
    assert (seen==1).all() and np.isfinite(probs).all()
    return total/len(probs),probs
```


## Dataset comparison audit — 2026-11-06

Completed `dataset_comparison_report.md`. Phase 11 HDF5 hashes match the prior audit; CNN remains balanced synthetic earthquake/noise detection (11324/2426/2428). BiLSTM PT and NPZ arrays are exactly equivalent, finite float32, with X=(N,20,36), splits 5629/1191/1192; binary targets equal raw target >=4.5. Exact feature meanings, event lineage, target timing and preprocessing provenance are UNKNOWN. Independent numerical training: PASS; equivalent targets/direct metric comparison and current fusion readiness: FAIL. Recover seq20 builder/schema and event/target manifest, choose a common causal endpoint, and establish shared leakage-safe splits before training. No data/model changes or training performed.
