# Decisions
## Waveform correction — 2026-11-05
- User explicitly authorizes synthetic waveform earthquake/noise detection. Earlier catalog-only restrictions apply only to the preserved forecasting experiment.
- Verified Phase 11 split integrity; use waveforms_standardized at 100 Hz / 10000 samples. Metadata is reporting-only and excluded from classifier inputs.
- Random paired/stratified splits are appropriate for this in-generator detection benchmark, not for forecasting or unseen-generator generalization.
- Streaming 100-row HDF5 blocks from both classes, shuffled into batches of32; deterministic seed42. No whole dataset loaded on GPU.
- Optional parameter-estimation experiment is not trained: magnitude inversion would reflect simulator formulas; absolute origin time and unique location lack appropriate input information; relative arrival picking needs a separate benchmark against the fixed-onset prior.
- Evaluate frozen waveform CNN against signal-feature logistic, raw-RMS diagnostic, and max STA/LTA baselines, plus shift/reversal/phase/noise-only shortcut probes. Do not tune on test outcomes.

- 2026-11-05: Use raw catalog, not synthetic target-event waveforms. Simulator takes same-event magnitude/location/depth to synthesize traces.
- Rebuild features rather than reuse globally scaled temporal CSV: `preprocess_dl_temporal.py:153` fits MinMaxScaler before splitting.
- Favor fixed calendar windows to retain negative forecasts and define meaningful time horizons.
- Keep model selection independent of held-out test metrics.

- 2011 modern start is a coverage heuristic; old sparse years cannot support reliable negative physical-event labels. No claim of completeness.
- Freeze M>=4.5 / 7 days / four broad cells; training-only threshold/horizon support table must pass before fitting.
- Purge full 90-day histories across partitions to provide conservative disjoint inputs/targets; retain negative windows.
- Probability scoring with unweighted BCE; do not balance or optimize thresholds on test.
- Isolated project notebook runtime installed to avoid changing existing global training packages.

- Verified waveform detection experiment wavecnn_ccb3b06b3d37 complete; no parameter-estimation model or BiLSTM implemented. Read shortcut_probes.csv and final report before interpreting accuracy as physical signal learning. Original catalog experiment preserved byte-for-byte.
