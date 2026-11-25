# Strict EEW cleanup audit â€” 2026-11-13

Dataset verification completed separately: v3 HDF5 checks PASS; overall approval remains CONDITIONAL because no v3 NPZ exists. No training started.

## Inventory

Inventoried and SHA256-hashed 24412 existing files, 4,787,941,418 bytes (4.459 GiB). Per-file classifications, sizes, hashes, references and duplicate matches are in cleanup_manifest.csv/json.

| Classification | Files | Bytes |
|---|---:|---:|
| KEEP_CORE | 285 | 4,215,188,305 |
| KEEP_REPRODUCIBILITY | 23384 | 542,018,459 |
| ARCHIVE_CANDIDATE | 1 | 16,322,280 |
| UNKNOWN | 742 | 14,412,374 |

## Retained

All original catalogs, CSVs, HDF5/NPZ/NPY data, manifests, scalers, arrival metadata and current v3 artifacts remain in place. Original waveform CNN notebook, best/last checkpoints, configuration, history, performance curves and source datasets are retained. Forecasting notebooks/checkpoints remain historical evidence, not authorized EEW training inputs.

V3 loads the v2 generator definitions directly. Archiving v2 would break active paths. Failed v2 and Phase11 datasets/reports retain scientific evidence; newer v3 does not waive old integrity requirements. Duplicate notebooks/plots are listed by exact hash but retained where paths, provenance or reproducibility remain relevant. The hybrid ZIP has one differing report and is retained.

## Proposed archive

- `bilstm new updated/phase3_bilstm_results.zip` â€” 16,322,280 bytes; all 41 members exactly match the retained extracted result directory; no source reference or integrity constraint found. Destination: `C:\Users\choud\Desktop\Minor project archive\cleanup_2026-11-13\bilstm new updated\phase3_bilstm_results.zip`.

## Actual archive and integrity

Actually archived by agent: **0 files, 0 bytes**. The user deleted the proposed ZIP during the audit and explicitly confirmed that it was no longer needed. The move command stopped at its source-existence/hash check before any move or archive directory creation. User-deleted size: **16,322,280 bytes (15.57 MiB)**. This is not an archive operation, and no archive rollback copy exists. All 41 extracted ZIP members remain in the project and were included in post-cleanup hashing. The original path, SHA256 and proposed destination remain documented.

Historical protected-baseline checks: 61/61 matching entries (44 unique paths). Existing integrity-controlled memory/status files were deliberately not updated. Post-cleanup SHA256 verification: **24411/24411 retained files unchanged**, excluding only the explicitly user-deleted ZIP. Integrity status: **PASS**. All protected originals and current v3 files remain retained.

## Unresolved dependencies and future work

Static references cannot prove absence of every dynamic dependency; uncertain files stay. Legacy seq20 input timing/provenance remains unresolved. Tabular BiLSTM must use information available by the same observation cutoff and comparable same-event targets/splits as CNN; historical next-event models are not interchangeable. S-arrival masks and oracle alignment are evaluation metadata, not operational trigger capabilities.

Future outputs: notebook-only executable ML code, reuse verified datasets, retain essential checkpoints and curves, concise validation reports, no redundant large datasets. No training until dataset approval; no permanent deletion.

Unknown items were retained: mostly training-runtime compiled/cache files, the compressed source snapshot, a temporary-looking CSV and a log. Names alone do not establish dispensability. No newer replacement was assumed. No source data or experiment artifact was deleted by the agent.
