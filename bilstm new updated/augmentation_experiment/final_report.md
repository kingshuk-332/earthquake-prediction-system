# Phase 1B: sampling and noise preprocessing

**BLOCKED for training-ready use.** No model training, Optuna, waveform processing or fusion occurred.

## Implementation

A: clean control. B: Gaussian noise only. C: regression-relevance weighted sampling of complete historical windows. D: the identical C draws plus B's noise policy. Each preview contains 4973 draws of shape (4973,20,10).

Interpolation is rejected: unrelated chronological histories do not provide justified sequence-aware neighbors; rolling statistics cannot simply be blended. Full SMOGN is not claimed. Weights range from 1 to 1.5 using training targets only; no 50/50 quota. Replacement sampling preserves labels and within-window ordering, with parent draws sorted chronologically. No new earthquake IDs, timestamps or target magnitudes are created.

Noise is switchable, sigma=0.01 standardized units, seed=42, historical latitude/longitude only. Shared historical events receive identical noise within an epoch. All other features, labels and provenance remain unchanged. Coordinates respect physical bounds. Perturbed inputs are marked SYNTHETIC_FEATURE_PERTURBATION.

## Diagnostic training distributions

| Variant | Draws | Unique parents | Repeated draws | Synthetic feature views | ML mean ± SD | ML≥4.5 |
|---|---:|---:|---:|---:|---|---|
| A | 4973 | 4973 | 0 | 0 | 4.034 ± 0.898 | 1553 (31.23%) |
| B | 4973 | 4973 | 0 | 4973 | 4.034 ± 0.898 | 1553 (31.23%) |
| C | 4973 | 3114 | 1859 | 0 | 4.144 ± 0.941 | 1822 (36.64%) |
| D | 4973 | 3114 | 1859 | 4973 | 4.144 ± 0.941 | 1822 (36.64%) |

C/D are one seeded epoch, not extra independent observations. Full magnitude bins/ranges, violations and content hashes are in distribution_comparison.json. These are preprocessing diagnostics, not performance results.

## Source validation finding

Independent checks found incorrect original 30-day event rates and target-gap units. The earlier preprocessing used native datetime integers with nanosecond constants. This audit explicitly normalizes timestamp resolution and checks rates with Timedelta rolling windows. The earlier PASS checks reused the same feature implementation, so they missed this error.

- Incorrect rate cells in original training X: 99459; affected training windows: 4973/4973.
- Incorrect training provenance gaps: 4973/4973; actual/stored gap ratio: 1000–1000.
- Example rate mismatch: {"training_parent_row": 0, "input_step": 1, "source_id": "NCS_EV_07168", "event_time": "1908-12-12 12:54:09+00:00", "saved_rate_per_day": 0.06666656309526786, "expected_rate_per_day": 0.03333333333333333}.
- Other independently audited feature mismatch counts: {'historical_magnitude': 0, 'historical_depth_km': 0, 'historical_latitude_deg': 0, 'historical_longitude_deg': 0, 'inter_event_days': 0, 'mean_magnitude_20': 0, 'std_magnitude_20': 0, 'mean_depth_20_km': 0, 'std_depth_20_km': 0}.
- Actual input-before-target chronological order still passes. The confirmed problem is feature/timing semantics, not evidence that future target events entered X.

No candidate passed full source-feature validity. Synthetic feature views and weighted draws were produced only as diagnostic previews; no accepted augmented tensor dataset was saved. The training consumer raises an error.

## Leakage and preservation

Training-only parents, no cross-split neighbors, unchanged labels and derived features, bounded coordinates, deterministic replay, parent provenance and original file/tensor hashes passed. Original validation/test data and scalers remain unchanged; nothing was refitted. Operational catalog availability remains UNKNOWN.

The original preprocessing and artifacts were not regenerated or corrected, as requested. The issue requires an explicitly authorized source correction in a separate version before augmentation can be approved. Repetition/noise cannot repair it.

## Saved work

Executed notebook: ../phase1b_imbalance_noise_preprocessing.ipynb.
Configuration and reproducible plan: augmentation_config.json, sampler_manifest.json, sampling_plan.npz.
Parents: parent_provenance.csv (variant/draw identifiers are not event IDs).
Audit: source_feature_audit.json, validation_results.json.
Distributions: distribution_comparison.csv/json and training_distribution_comparison.png.
Methods and future comparison: research_adaptation_notes.md.
No improvement claim or model performance metrics.
