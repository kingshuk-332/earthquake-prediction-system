# HDF5 training readiness — 2026-11-13

**Decision: APPROVED for standalone synthetic oracle-aligned CNN training.**

**NPZ: NOT REQUIRED.** The missing NPZ was an unfulfilled format-specific check, not missing training information. This addendum resolves that condition for HDF5-only training; the original verification report remains unchanged. No duplicate archive should be created.

## Evidence and scope

Read `final_verification_report.md`; retain its completed 183 counterfactual tests and exhaustive 16,178-record audit as scientific evidence. This review did not repeat either. Opened all three actual HDF5 files read-only, inspected dataset shapes/dtypes and attributes, and successfully read two rows of each required input/target/identity/mask field. Checked the horizon shapes and singleton CNN channel conversion for every split. Inspected the three v3 generation/validation notebooks' source without executing generation or validation cells.

| Required information | Direct HDF5 interface | Result |
|---|---|---|
| Split assignments | `dataset/eew_dataset_{train,validation,test}.h5`, matching `split` attribute; 11,324 / 2,426 / 2,428 rows | PASS |
| Observation inputs | `windows_{1,2,3}s_norm`, float32, `(N,200/300/400)`; corresponding `_raw` arrays also available | PASS |
| CNN tensor layout | Read selected rows and insert one channel axis: `(batch,1,200/300/400)` | PASS |
| Classification target | `labels`, int8, `(N,)` | PASS |
| Magnitude target | `magnitude`, float32, `(N,)` | PASS |
| Event/parent identity and row association | `source_id`, `parent_id`, `pair_id`, `combined_index`; aligned on the record axis | PASS |
| Horizon eligibility | `classification_mask_{h}s`, `magnitude_mask_{h}s`, `pre_s_valid_{h}s`; boolean `(N,)` | PASS |
| Normalization and timing provenance | `context_mean`, `context_std`, `reference_sample`, P/S sample and second annotations; 100 Hz attribute | PASS |

Additional saved inputs include 10,000-sample raw parents and pure-post-P windows. They are not needed to reconstruct a missing format. Existing split manifests remain available for provenance; split membership is already represented by the three files.

## Intended loader contract

Use `h5py.File(split_path, 'r')` and load `windows_{h}s_norm` plus aligned targets and masks directly. Select classification-eligible rows with `classification_mask_{h}s`. Compute magnitude loss only where `magnitude_mask_{h}s` is true, handling batches with no eligible regression rows without evaluating an empty mean. Noise magnitude -1 is a masked sentinel, never a regression target. The base `magnitude_mask` alone does not impose pre-S eligibility. Preserve the existing splits and row identities.

No v3 CNN training notebook/DataLoader implementation exists in the inspected v3 entry points. Thus this is verified **dataset/interface readiness**, not certification of an already implemented trainer. The intended HDF5 interface is supported by actual successful reads, and the inspected v3 generation/validation path uses HDF5 without an NPZ dependency. A future trainer must implement this contract; it does not need an NPZ file.

No essential dataset field is missing for the stated classification and magnitude tasks. Feed waveform arrays only into the CNN; identities, targets, eligibility masks and P/S annotations are supervision/filtering/provenance, not input channels. Use the saved pre-context normalization rather than introducing whole-parent normalization.

## Approval boundaries

Approval covers controlled synthetic same-event classification and magnitude research at 1/2/3-second observation horizons with the saved masks. Earthquake windows are oracle-aligned with P at index 100; oracle arrival and S-based eligibility do not demonstrate an autonomous real-time trigger. This approval does not establish operational EEW, real-data generalization, or readiness of tabular/fusion models.

No training, regeneration, cleanup, deletion or existing-file modification was performed. Only this readiness addendum was created.
