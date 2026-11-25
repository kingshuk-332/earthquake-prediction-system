# Independent final CNN EEW v3 verification

**Date:** 2026-11-13

**Decision: CONDITIONAL — NPZ artifact verification remains outstanding**

All 16,178 HDF5 records were checked directly. All parent samples and numeric arrays were checked for finiteness; every saved raw, normalized and pure-post-P window was recomputed from its parent. Labels/magnitudes were joined to catalog and split sources. No previously reported PASS was used as a substitute for execution.

| Check | Status | Evidence |
|---|---|---|
| v2 regression controls | PASS | {'noise': 361.7903289794922, 'coda': 99.57305908203125} |
| v3 earthquake causality | PASS | {'cases': 15, 'tests': 135, 'max_difference': 0.0} |
| v3 noise causality | PASS | {'tests': 48} |
| saved HDF5 arrays and source alignment | PASS | {'failures': [], 'counts': {'train': {'total': 11324, 'earthquakes': 5662, 'noise': 5662}, 'validation': {'total': 2426, 'earthquakes': 1213, 'noise': 1213}, 'test': {'total': 2428, 'earthquakes': 1214, 'noise': 1214}}, 'records': 16178, 'subnormal_values': 0} |
| eligible counts | PASS | {1: 8089, 2: 8056, 3: 7996} |
| split disjointness | PASS | {'train/validation': {'source_id': 0, 'pair_id': 0, 'combined_index': 0, 'parent_id': 0, 'background_id': 0, 'background_sha256': 0, 'actual_parent_hash': 0}, 'train/test': {'source_id': 0, 'pair_id': 0, 'combined_index': 0, 'parent_id': 0, 'background_id': 0, 'background_sha256': 0, 'actual_parent_hash': 0}, 'validation/test': {'source_id': 0, 'pair_id': 0, 'combined_index': 0, 'parent_id': 0, 'background_id': 0, 'background_sha256': 0, 'actual_parent_hash': 0}} |
| saved simulator replay | PASS | {'records': 66, 'max_difference': 0.0} |
| NPZ verification | UNKNOWN | No NPZ archive exists in the v3 dataset directory. No v2 archive substituted. |
| model input provenance | PASS | Inspected generation path and exact saved-parent replay: inputs are waveform-only (record,time) arrays. Labels, magnitudes, IDs, P/S annotations and masks are separate datasets. Simulator source parameters affect physical synthetic signals; no target channels are concatenated. This is source-path evidence, not a guarantee against arbitrary statistical shortcuts. |
| oracle alignment and scientific scope | PASS | P reference is theoretical/oracle and occurs at window index 100. Context normalization uses only the preceding 100 samples. Horizon windows are half-open; eligible examples have reference+100*horizon < ceil(S_seconds*100). S timing is privileged eligibility metadata, never an input feature. This is synthetic post-trigger analysis, not a demonstrated streaming trigger or operational EEW system. |
| regression masking contract | PASS | All saved magnitude masks equal label==1; per-horizon masks additionally require pre-S eligibility. Noise targets are -1 and masked out. Ineligible EQ rows remain stored and MUST be excluded using horizon masks. No training consumer exists in this audit, so runtime enforcement by a future trainer is unverified. |
| existing input preservation | PASS | {'files': 26, 'changed': [], 'scope': 'All read dependencies and v3 top-level/config/dataset/Gemini artifacts hashed before and after; no project-wide scan.'} |

## Causality evidence

Original v2 regression maxima: noise 361.790328979; P coda 99.573059082. Independently injected future-only changes reproduce both reported defects. V3: 135 earthquake tests across 15 configurations/seeds and 48 noise tests across four profiles; strict zero tolerance. Non-duration earthquake interventions also require a changed future tail so a no-op cannot pass.

Saved simulator linkage: 66 stratified full-parent replays; individual results and source hashes are in the JSON. All observed prefix differences and failures are recorded rather than repaired.

## Scientific interpretation

Oracle P alignment is explicit. A real deployment would need a causal trigger and an evaluation without oracle timing. Use waveform arrays only; do not feed labels, masks, IDs or P/S metadata into the CNN. Apply classification_mask_{h}s for classification and magnitude_mask_{h}s for regression at each horizon. The base magnitude_mask alone does not enforce pre-S eligibility. All windows contain 100 pre-P context samples and 100/200/300 post-P samples. Noise references are synthetic random anchors. Physical amplitudes are relative simulator amplitudes, not calibrated instrument displacement. Real-noise/domain validity, magnitude identifiability, false-alert rates and operational latency remain outside this dataset verification.

## Remaining check

The requested NPZ verification is UNKNOWN: no v3 NPZ artifact is present. Supply its existing location for an independent array/row/mask comparison, or explicitly scope approval to the verified HDF5 artifacts. No archive was created or repaired by this verifier.

## Preservation and reproducibility

All executable verification logic is in final_verification.ipynb. The notebook executes only isolated simulator definitions; original write/generation cells are never executed. Input hashes were checked before/after. Only this new verification directory contains outputs. No CNN training or Optuna was started.
