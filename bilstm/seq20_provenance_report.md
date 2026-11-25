# Seq20 provenance recovery

**PARTIAL — the referenced packaging archive was recovered and verified; the constructor and scientific provenance remain UNKNOWN.**

This report records new evidence. Prior schema, numeric-target and overlap findings remain in the [Phase 2B assessment](phase2b_generalization_results/split_provenance_report.md).

## Verified new evidence
All three dataset files have Windows Zone.Identifier metadata pointing to [Desktop/bilstm_dataset.7z](<C:/Users/choud/Desktop/bilstm_dataset.7z>). That archive exists, and its only file members are the three dataset artifacts:

| Existing dataset | Referenced archive member |
|---|---|
| bilstm_dataset_seq20.npz | Exact SHA-256 match |
| bilstm_dataset_seq20.pt | Exact SHA-256 match |
| bilstm_dataset_seq20_station_grouped.npz | Exact SHA-256 match |

Archive SHA-256: 55cfb1389c27d8956e1f8375a908617939b530367de62637dd3c4d61ce7f50e0.
Members were streamed to a hash function; no files were extracted or changed. This verifies a packaging reference, not the original author or producing program.

The archive contains **no notebook, script, README, schema, scaler, manifest or generation log**. It has no Zone.Identifier stream to identify an earlier download location. PT metadata was inspected by ZIP listing and pickle disassembly without unpickling or loading tensors; serialized strings identify array/storage keys, not feature names, source paths or a constructor.

A bounded search read **16 named files**: relevant catalog/BiLSTM notebooks, the hybrid sequence script, root preprocessing sources/log, dataset indexes/registry and existing manifests. It covered seq20/final filenames, 36-feature references, sequence constructors, torch.save and np.savez. The only files combining seq20 references with writer calls are the two downstream Phase 2 notebooks: they load existing data and save model/prediction artifacts. Those calls do **not** construct seq20. No verified original constructor or ordered 36-feature definition was recovered.

No exact-filename generation cells were found in the two explicitly checked project-local IPython history locations; availability is recorded in the evidence. The three checked notebook-checkpoint directories are absent. No whole-project or waveform-generation scan was performed.

## Verified versus unverified
**Verified:** the referenced archive contains exact copies of all three current datasets.
**Unverified:** any connection between the legacy 15×44 preprocessing pipeline and these 20×36 tensors. A 36-channel model, loading call, file timestamp or matching sample count does not establish construction history.

Still **UNKNOWN**: exact 36-feature order/derivations; meanings and ordering/grouping of the 20 steps; physical target alignment/horizon; input/target event and station IDs/timestamps; split/cutoff/purge rules; and scaler/imputer/encoder fit rows and parameters. Existing numeric target-encoding checks do not resolve those semantics.

## Minimum missing artifacts
- Actual producing notebook/script or exported execution history, tied to the source-catalog version/hash and these output hashes.
- Ordered feature specification and window-to-input/target identity/time manifest, including target alignment and split assignments.
- Saved preprocessing parameters and fit-record IDs, or reproducible code/logs establishing where each transformation was fitted.

Recover these from the original dataset producer before reconstructing any split. No row-order mapping is assumed.

[Executed investigation notebook](seq20_provenance_investigation.ipynb) · [bounded search evidence](.agent-memory/seq20_recovery_evidence.json).
All three datasets, the referenced archive and searched sources passed final hash checks. No training, Optuna, regeneration or dataset modification occurred.
