# BiLSTM EEW research report — Stage A blocked

## Outcome
Identity and split checks passed, but **the upstream causality gate failed**. No BiLSTM baseline, Optuna search, clean/augmented experiment or test evaluation was run. The saved waveform crops are quarantined, not an approved training dataset.

## Verified sources and labels
Phase 11 train/validation/test raw waveforms were verified against existing audit hashes, then joined to explicit split, Phase 10, Phase 9 and P/S manifests by **source_id + trace_name**. Catalog identities and magnitude labels also matched these keys. Pairs remain split-disjoint; full-parent duplicate checks were reused only after hash verification, and exact cross-split crop duplicates were independently excluded.
Catalog origin timestamps and IDs are preserved. Publication/revision times are absent; real-time catalog availability is UNVERIFIED and no reporting delay was invented. Catalog values are not model inputs.

## Candidate observations
100 Hz, 100-second single-channel synthetic parents. Extracted raw post-P windows of 100/300/500 samples; no pre-arrival context. No full-trace detrending/standardization is reused. Noise-only magnitudes are NaN, never zero. Magnitude targets preserve the original catalog's reported values/types; mixed-scale calibration would need explicit treatment before a magnitude experiment.

| Split | Duration | Shape | Cutoff before S (earthquakes only) |
|---|---:|---|---:|
| train | 1 s | (11324, 100, 1) | 100.00% |
| train | 3 s | (11324, 300, 1) | 98.80% |
| train | 5 s | (11324, 500, 1) | 95.44% |
| validation | 1 s | (2426, 100, 1) | 100.00% |
| validation | 3 s | (2426, 300, 1) | 99.09% |
| validation | 5 s | (2426, 500, 1) | 96.13% |
| test | 1 s | (2428, 100, 1) | 100.00% |
| test | 3 s | (2428, 300, 1) | 98.85% |
| test | 5 s | (2428, 500, 1) | 95.96% |

Windows with S motion are explicitly labeled; they cannot be represented as pre-S-only observations. Oracle P annotations anchor the windows, and noise uses explicitly paired anchors; this is not evidence of continuous detection or operational false alarms per hour.

## Confirmed blocker
improved_waveform_generator.py lines 224–243 sets background-noise scale using the maximum clean-signal amplitude over **P through the end of the full trace**.
For keyed event **NCS_EV_07168**, the current generator reproduced the stored source trace (max absolute error 0). Keeping P, magnitude, S arrival, distance, SNR and seed fixed while multiplying only the later S-wave amplitude by four:
- Clean samples before the five-second cutoff stayed identical.
- Noise sigma changed from 0.0542097 to 0.216836.
- The observed early window changed by up to 0.521841.
- S arrives at 33.86 seconds in the trace, after the latest observation cutoff at 10.96 seconds.

Thus slicing early samples does not remove a future-signal-dependent synthetic noise cue. [Counterfactual evidence](causal_noise_counterexample.json) and [figure](causal_noise_counterexample.png) document the failure. It is a confirmed generator dependency, not a measured amount of model performance inflation.

## Training and evaluation status
Noise augmentation, fitted scalers, baseline/tuned hyperparameters, detection and magnitude metrics, processing latency and overfitting assessment are **NOT RUN / unavailable**, because Stage A failed. No checkpoint or Optuna study exists.
Observation durations are 1/3/5 seconds after an oracle P pick, not guaranteed warning times. Relative synthetic S-minus-cutoff intervals are retained, but no validated target-site arrival estimate exists; usable target-site warning lead time is UNKNOWN. A chronological holdout is deferred until causality is repaired.

## Research and next step
Graves & Schmidhuber supplies the forward/backward sequence-processing principle; a BiLSTM may use both directions only within the already observed prefix. The two earthquake-catalog papers are background, not evidence for P-wave EEW. See research_notes.md.
Create a separately versioned source with independent ambient noise (no target/future-peak calibration), or supply real recorded traces with verified P/S picks and identities. Re-run Stage A before any clean-versus-augmented BiLSTM training. Do not silently repair existing traces or reuse these quarantined crops.

All work is inside this new project. Previous data/models are unchanged; no CNN or hybrid code was modified or trained.
