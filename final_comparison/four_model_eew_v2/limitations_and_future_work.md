# Limitations and future work

## Scope of inference

This project is a synthetic oracle-P-aligned early-observation benchmark. Catalog-informed simulation produced the waveforms; BiLSTM inputs are seven time-ordered numerical features extracted from those same synthetic observations. They are not independent catalog measurements, historical next-event predictors or a second independent sensor modality.

Oracle P arrival defines the observation windows. No operational P-wave triggering, continuous-stream detection delay, real-world warning lead time or deployment readiness was established. Offline pre-S eligibility annotations define the research cohort and are not a deployable decision rule. Bidirectional processing is confined to the already observed window.

## Statistical and measurement limitations

The four-model comparison uses the same 1,200 earthquakes and 1,214 noise records. The 3s validation diagnosis instead uses 1,203 earthquakes and 1,213 noise records. Neither cohort should be substituted for the other. Fixed-0.5 comparisons and validation-selected operating points answer different questions.

Models use a single training seed. Saved paired bootstrap intervals are exploratory percentile intervals, not multi-seed or multiplicity-adjusted confirmatory tests. The test cohort had already been examined before the controlled tuning study; its later evaluation is exploratory. Twelve validation trials also create selection uncertainty. No statistically supported Attention advantage over non-Attention fusion was found; this is not proof of equivalence.

Magnitude compression, overestimation of small events and underestimation of stronger events remain important. A nearly zero aggregate signed error masks opposite tail biases. High noise-window false-positive rates remain problematic; rates cannot be translated to false alarms per hour without continuous recordings and a trigger/aggregation policy. Realistic event prevalence and alarm costs were not measured.

CNN–BiLSTM validation magnitude residual correlation was 0.982666. This indicates shared errors but does not prove a causal bottleneck or rule out all possible fusion improvements. Frozen-feature limitations, gradient interference and irreducible information limits were not established experimentally. Several trials approached the epoch cap, so optimal convergence is not established. Trustworthy noise-family labels were unavailable in the inspected metadata: family-shortcut effects remain UNKNOWN. Discarded trial weights and per-record predictions were unavailable, limiting trial-specific diagnosis.

## Future work: evidence before further tuning

1. Acquire independent real seismic recordings with verified event/station identities, instrument response, calibration, magnitude definitions and documented provenance. Preserve actual observed P/S picks and their uncertainty separately from theoretical annotations. Include genuine non-earthquake recordings with trustworthy noise labels.
2. Reserve an untouched evaluation set separated by event and, where feasible, station, region and recording period. Predeclare objectives and thresholds using training/validation data only. Report repeated-seed uncertainty and paired comparisons without tuning on the reserved set.
3. Evaluate realistic causal P triggering and continuous-stream preprocessing. Measure false alarms per unit time, missed events, feature-extraction and inference latency, and physically justified warning intervals separately. Oracle alignment remains an upper-level research assumption until this is done.
4. Strengthen evaluation of large events using adequately sized, independently verified observations. Report magnitude-band errors, bias, underprediction severity and calibration with uncertainty; do not fabricate strong events or equate detection success with accurate magnitude estimation.

These are future research directions, not experiments authorized or started by this integration. The completed diagnosis recommends NO FURTHER TUNING at this stage and retention of the original 3s Attention model.
