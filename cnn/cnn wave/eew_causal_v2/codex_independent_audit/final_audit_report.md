# Final independent causal EEW v2 dataset audit

**Decision: BLOCKED. Classification C — not causally valid.** No model training, full dataset regeneration, repairs or protected-file changes. This audit does not approve operational EEW.

## 1. Actual inspection

All **16,178 records**: train11324, validation2426, test2428. Every numeric HDF5 array was checked for finiteness; all labels, IDs, arrival annotations, eligibility masks and nested horizons were compared. All30 NPZ members were compared exactly against HDF5 counterparts. Lengths200/300/400 contain100 pre-reference samples plus100/200/300 post-reference samples, at100Hz; axis order is(samples,time), one waveform channel. Models would need a separately added channel axis.

Stratified replay: **21 saved records**, including15 earthquake parameter/seed cases spanning magnitude, depth and short S-P intervals, plus noise. Exact raw/normalized replay matches: **21/21**. Saved-window reproduction links the counterfactual code path to actual data; it does not verify unsaved100-second parents exhaustively.

## 2. Audits A–F

| Audit | Status | Observed evidence |
|---|---|---|
| A | UNKNOWN | Saved numeric/label checks PASS; regression masking not implemented/verified. |
| B | PASS | Cross-split membership intersections 0; exact window duplicate hits 0. |
| C | FAIL | Future-noise and future-template normalization change already-observed prefixes. |
| D | PASS | P/S metadata, physical interval units and strict eligibility independently checked. |
| E | FAIL | Oracle-aligned inputs plus upstream future dependence; classification C. |
| F | PASS | 17/17 original baseline matches; 41/41 protected files unchanged during audit. |

Detailed per-split checks, every metadata comparison and measured exceptions are in audit_results.json. Sentinel-1 noise magnitudes and P/S labels are stored correctly where checks pass, but no explicit regression mask or training consumer exists. Noise must be excluded from magnitude regression with labels==1; true pre-S masks on noise mean classification eligibility, not physical arrivals.

## 3. Independent causal counterfactuals

Only class definitions from notebook02 JSON code-cell index2 were executed in memory. The bulk generation loop was not executed. The different pilot class from notebook01 was not treated as proof for notebook02.

| Perturbation | Maximum earlier/raw or saved3s difference | Maximum normalized difference |
|---|---|---|
| Future S amplitude x4, before S | 0 | 0 |
| Future noise innovations x4 after P+8s, saved3s | 0.154729534 | 361.790329 |
| P coda envelope x4 only after P+4s, saved3s | 5.18156576 | 99.5730591 |
| S coda envelope x4 only after S+5s, earlier samples | 2.52475429 | 50.5555248 |

**Critical:** notebook02 computes raw_std=np.std(filtered_noise) over the full100-second earthquake background, and cur_std=np.std(filtered) for noise parents. Modifying only future innovations preserves earlier innovations and causal-filter output, but changes global scaling and earlier observed windows. Pre-P normalization can cancel common noise scaling approximately for pure noise; it does **not** cancel the changed signal/noise ratio in earthquake windows.

The P/S envelopes are also divided by their full-record maxima. Perturbing only the future coda envelope can therefore change earlier signal amplitude. These tests change future-tail values before the generator's existing normalization; the earlier unnormalized envelope and ambient realization are held fixed. They do not conflate changing a whole rupture-duration parameter with a future-only intervention.

The narrower S-amplitude-only test passes, so the original future-S-to-noise coupling was removed. It is insufficient to certify the complete generator. Downstream pre-context-only normalization itself passes a direct future-waveform perturbation test (maximum prefix change 0); upstream defects remain.

## 4. Physics, eligibility and discrepancies

| Post-P horizon | Eligible earthquake records | Fraction | Minimum physical S-P seconds |
|---|---|---|---|
| 1s | 8089/8089 | 100.0000% | 1.063196 |
| 2s | 8056/8089 | 99.5920% | 1.063196 |
| 3s | 7996/8089 | 98.8503% | 1.063196 |

P indices vary across parent records, but oracle cropping places P at window index100 for every earthquake example. Noise has no P/S label and uses a random reference segment. Validation/test therefore require oracle P timing; they are not a trigger-based streaming benchmark. S timing is needed to filter/evaluate pre-S eligibility, not to calculate the recorded waveform channels.

The completion report's minimum S-P3.89s and100% eligibility at every horizon are contradicted: the actual minimum is1.063195884s; eligible counts are8089/8056/7996 of8089 at1/2/3s. The earlier verification's1.06s and99.6%/98.8% figures are consistent after rounding. Both reports' complete causality PASS is contradicted by this audit. The completion report does not contain the explicit oracle acknowledgment attributed to it by the earlier verification.

Magnitude/depth/distance join by source ID; station/source coordinates are available in the linked Phase2 metadata, not input channels. Arrival indices describe simulated source-aligned times, not detected arrivals. At fixed geometry/noise, changing magnitude changes normalized P windows in the numerical probes; this supports simulator sensitivity only, not unique magnitude identifiability across distance/source uncertainty or real calibrated ground-motion units.

**Annotation precision:** NCS_EV_03038 at2s: physical S-P=2.001841556s, stored difference=200 samples. NCS_EV_03572 at3s: physical S-P=3.002951118s, stored difference=300 samples. These two validation masks are correct using continuous source times but differ from strict comparisons on rounded integer S indices. The saved windows end before the nominal cutoff, so this is not observed S contamination; consumers must retain the continuous-time definition.

## 5. Leakage/split severity and data-interface limits

Observed cross-split source/pair/trace/background-key intersections and exact duplicates are listed above. Noise parent mapping uses Phase10 seed/profile and combined index; earthquake ambient seeds use42+combined_index. No learned global scaler is fitted on train/validation/test. Full parent waveforms are absent, so exhaustive whole-parent content uniqueness remains UNKNOWN even when mapped identities and all saved windows are disjoint.

NPZ arrays contain waveforms and labels separately, not extra label channels, but omit IDs, P/S arrays and1s/2s masks. Their verified row identity requires retaining HDF5/manifest association; noise sentinel-1 is not a ready-to-use regression target. Exact labels and arrival values are not directly concatenated into waveform arrays. The critical issue is upstream future-dependent synthesis, not an observed event-split overlap.

## 6. Evidence and protection

- Actual arrays: ../dataset/eew_dataset_{train,validation,test}.h5 and ../dataset/eew_windows_multitask.npz.
- Exact generator and seed/export path: ../02_generate_causal_eew_dataset.ipynb, JSON cells2–4; pilot differences inspected in notebook01 and verification scope in notebook03.
- Source/label association: ../configs/causal_eew_dataset_manifest.csv; ../../data wave/synthetic_waveform_simulator/output/phase2_travel_time/metadata_with_theoretical_arrivals.csv; Phase10 earthquake_noise_mapping.csv; Phase11 split_indices.csv.
- Parameters and registered hashes: ../configs/causal_simulation_parameters.json and causal_dataset_hashes.json.
- Preservation: ../original_files_manifest_initial.json plus older ../../phase1_waveform_cnn_results/reports/waveform_audit.json. The older audit corroborates3/3 Phase11 hashes. The17-file claim is checked against that supplied historical baseline, not an assertion about every repository file or an externally authenticated history.
- This audit: [dataset_audit.ipynb](dataset_audit.ipynb), [audit_results.json](audit_results.json), containing all before/after hashes, source-matched replay errors and per-seed counterfactuals.

All41 inspected protected files are compared before/after; unchanged=True. Existing status, memory, reports, datasets and checkpoints were not edited. **Stop after audit; no repairs or training.**
