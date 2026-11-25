# Independent-review handoff: causal EEW v3

**Requested decision: independent dataset verification. Implementation status: READY FOR INDEPENDENT DATASET VERIFICATION; not finally approved.**

## Exact v2 evidence and v3 comparison

| Intervention | Reproduced v2 maximum normalized prefix difference | v3 maximum |
|---|---:|---:|
| Future noise only, after P+8s | 361.790328979 | 0 |
| Future P-coda only, after P+4s, x4 | 99.573059082 | 0 |
| Future S amplitude x4 before S | 0 | 0 |
| Future S coda +50%, future P coda +50%, record extension/truncation | Independently specified v3 gates | 0 |

The exact per-source v2 values are reproduced from the protected audit's instrument definition using the original saved-data generator. V3 tests preserve past source parameters and innovations; no tolerance was loosened (pilot tolerance0). Separate full-parent replay uses predetermined float32 tolerances atol1e-7/rtol1e-6, reported separately from the causality gate.

## Review targets

1. Inspect01_causal_simulator_v3.ipynb's narrow transformations of the v2 earthquake class: fixed filter gains, analytic gamma peaks, separate RNG domains and causal background profiles. Pilot code is in the same notebook.
2. Independently check02_generate_dataset_v3.ipynb's gate and immutable original assignments, source/magnitude/arrival metadata, seeds, full parents and explicit masks.
3. Re-run independent tests against saved HDF5 parents/windows.03_dataset_validation_v3.ipynb records exhaustive checks and a separate earthquake-equation replay, but was written by the implementation agent.
4. Check the two arrival-boundary records, source IDs NCS_EV_03038/NCS_EV_03572: v2 rounded differences200/300 become first-grid201/301 without changing continuous travel times. Verify strict integer masks; not all3s earthquake windows qualify.
5. Confirm magnitude loss uses magnitude_mask or magnitude_mask_{horizon} and never treats noise sentinel-1 as a magnitude target. Classification masks are separately provided.
6. Verify all before/after protected hashes and frozen generation hashes. Exact window and full-parent duplicate checks now cover saved parent arrays; background identities/digests are available, with full backgrounds reproducible by seeds rather than separately stored.

## Files and boundaries

- Executed notebooks: ../01_causal_simulator_v3.ipynb, ../02_generate_dataset_v3.ipynb, ../03_dataset_validation_v3.ipynb.
- Evidence: v3_counterfactual_results.json, v3_validation_results.json, v3_validation_report.md.
- Data: ../dataset/eew_dataset_{train,validation,test}.h5.
- Provenance: ../configs/split_manifest.csv; simulation_v3.json; generation_manifest.json; protected_sources_before.json; protected_sources_after.json; artifact_hashes.json.
- Protected v2 findings: ../../eew_causal_v2/codex_independent_audit/final_audit_report.md and audit_results.json.

Oracle P knowledge is still required to construct validation/test windows. No causal trigger has been implemented. Noise spectral realism and real-instrument magnitude calibration are not established. Review approval, if granted, can cover synthetic oracle-aligned research only—not operational EEW. No training or fusion was performed.
