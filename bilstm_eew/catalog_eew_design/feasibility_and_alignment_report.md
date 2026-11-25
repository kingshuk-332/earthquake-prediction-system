# Historical-catalog BiLSTM: EEW feasibility and alignment

Date: 2026-11-13

**Verdict: BLOCKED — the CSV-only standalone EEW objectives are unsupported.**

A historical temporal-context model is conceptually possible, but the present catalog does not establish as-of feature availability, current-event observations or earthquake/noise examples. This report does not authorize training. It supersedes neither the CNN results nor the earlier waveform-derived tabular proposal: that proposal is a different information source and is not implemented here.

## 1. Evidence and checks

Read only the CSV header/representative rows, previous readiness report, documented historical feature schema, CNN common-cohort summary and event/split manifest. Narrow column-level metadata queries additionally checked complete ID/magnitude alignment, magnitude types and split date ranges; no sequence dataset was constructed.

- `dataset_2.csv`: 8,089 records and 8,089 unique source IDs.
- CNN manifest: 8,089 earthquake records; all join to catalog IDs. No magnitude-value join discrepancies at absolute tolerance 1e-6 against `Magnitude_Value`.
- Catalog magnitude types: ML 7,125; MB 458; MW 398; MC 107; MS 1. Numeric agreement does not make these scientifically interchangeable magnitude scales.
- Previously verified common test cohort: 1,200 earthquakes and 1,214 noise records. Catalog noise counterparts do not exist.
- Historical feature documentation explicitly marks operational feature availability UNKNOWN.
- CSV timestamps are origin times, not event publication, field issuance, revision, acquisition or trigger-delivery times.

| CNN earthquake split | Count | Earliest origin UTC | Latest origin UTC |
|---|---:|---|---|
| Train | 5,662 | 1908-12-12 12:54:09 | 2026-10-06 09:03:32 |
| Validation | 1,213 | 1906-08-31 14:57:30 | 2026-10-05 22:57:25 |
| Test | 1,214 | 1922-12-24 00:06:26 | 2026-10-05 22:53:47 |

These are event-disjoint benchmark partitions, not a chronological prospective evaluation. No complete historical-dependency audit or temporal sample construction was performed; absence of contamination is not asserted.

## 2. Historical features: conditionally available versus unsafe

| Candidate | Causality requirement | Present status |
|---|---|---|
| Previous magnitudes and magnitude types | Value/version actually issued before the current decision; retain scale identity | Values exist, availability/version history absent |
| Previous depths, latitude and longitude | Historical estimates available by cutoff, not later revised catalog values | Values exist, availability unverified |
| Inter-event intervals | Historical origin estimates as known at cutoff, UTC and explicit units | Can be calculated retrospectively; real-time availability unverified |
| Rolling magnitude/depth statistics | Only eligible preceding records and their as-of versions | Proposed features, not validated causal inputs yet |
| Historical P/S picks or S–P intervals | Genuine measured picks with sampling/record-origin metadata, pick timestamps and publication provenance | CSV sample indices do not establish these conditions; exclude initially |
| Historical event rates | Defined catalog coverage, eligible report history and left-censoring policy | Coverage/reporting completeness unverified; omit from minimal design |

An S pick from an older event is not intrinsically future information: it may be legitimate once observed and published before the current cutoff. However, the existing P/S columns do not prove this provenance. They must not be assumed to be historical real-time station observations merely because they contain numbers.

Exclude the current event's true magnitude and aliases, depth, location, distance, final origin time as a predictive feature, future S arrival and true S–P interval. Exclude theoretical arrivals computed from unknown source geometry, unverified full-record SNR, source/trace IDs as embeddings, synthetic noise-family labels and split/eligibility markers. IDs remain bookkeeping only. Do not compute historical statistics over a window containing the target or any later event.

## 3. Exact proposed sequence specification — not implemented

This is a specification for a future availability-audited context branch, not a ready dataset.

1. **Decision time:** use an externally observed station/network P-trigger delivery timestamp, plus h=1, 2 or 3 seconds. Do not obtain absolute trigger time by adding the synthetic trace-relative P offset to catalog origin time: a trace start/acquisition-time mapping is not supplied. Oracle alignment can be disclosed for retrospective research, but cannot establish a real-time clock or detection capability.
2. **As-of history:** reconstruct each prior record's field versions known by the decision time. A field's release time must be no later than that cutoff. Never use a final revised value merely because the event's origin precedes the target. Exclude the current event and its duplicate reports through event association. This requires missing publication/revision and association information.
3. **Preceding-event rule:** for retrospective identity accounting, require historical origin strictly earlier than the target origin; do not resolve equal-time events by arbitrary ID ordering. Operational use instead needs a documented past-event association rule based on information actually available at trigger time, not the target's unknown final origin. Do not equate these two rules without validation.
4. **Window:** after availability and split restrictions, select the latest at most 20 eligible preceding earthquakes, order oldest to newest and preserve their IDs separately. Preserve actual unequal inter-event gaps; do not interpolate fictional events. Same-time ambiguity is excluded or explicitly resolved using trustworthy availability records.
5. **Minimal numeric features per historical event j:** reported magnitude, depth, latitude, longitude, interval since the preceding selected historical event, and four rolling statistics: magnitude mean/SD and depth mean/SD. Rolling statistics use only the selected-window prefix ending at j, at most 20 events; population SD uses ddof=0, singleton SD=0. This is nine numeric features. The first selected interval is missing, not a fabricated elapsed time. Keep historical magnitude type as a documented categorical attribute if mixed scales are retained; do not silently treat mixed values as homogeneous ML or MW. Historical row j may contribute its own attributes because it precedes the target and must already be available.
6. **Short histories:** use a recorded sequence length and masked left padding to 20, with explicit missing-feature masks. Impute undefined features from training-only fits. Zero history means no evaluable context prediction under this specification; record exclusion and its reason. Do not claim all 1,200 common earthquakes remain evaluable before constructing and validating histories.
7. **Bidirectionality:** both directions may traverse these already available historical rows. Neither direction may include the target event or a later event. No CNN features, embeddings, P-wave extraction or fabricated observations are proposed here.
8. **Normalization:** fit imputation, scaling and any magnitude-type encoding only on historical entries permitted for training samples. Prefer unique contributing history IDs to avoid accidental over-weighting from repeated windows; document the chosen weighting. Freeze transforms for validation/test. Fit any target transform only to training target magnitudes. Padding must be excluded from fit statistics and sequence summaries.

Do not reuse previous saved scalers or next-event sequence tensors without proving their exact fit population, availability and dependency rules. The specified raw fields and operations are candidate context features; no predictive sufficiency is implied.

## 4. Current-event magnitude is not established by historical context alone

Given the same history, a model without current-event observations supplies the same contextual magnitude prior before and after a P trigger. Calling it at P+1, P+2 or P+3 does not add evidence about this earthquake's amplitude, source or distance. The fact that a trigger occurred may condition a prior on event occurrence; it does not establish useful event-specific early magnitude estimation. Any newly arriving historical catalog update would need its own verified release time and would still not constitute a current P-wave measurement.

Training preceding-event sequences against the following catalog event's magnitude would reproduce the earlier next-event regression formulation. Aligning that target's ID to CNN does not change its scientific interpretation. A history-only estimate could be reported as an externally triggered contextual-prior baseline, but not as demonstrated P-wave EEW magnitude estimation or an equivalent observation-based standalone model. No such baseline is trained here.

Minimum event-specific evidence would be authentic cutoff-limited measurements or issued estimates: for example timestamped early P amplitude/energy descriptors, causal picker confidence, or early network estimates with uncertainty and issuance times. These need not be fabricated or extracted from CNN waveforms in this task. Acquiring such tabular observations would change the proposed CSV-only input definition and require an explicitly documented protocol.

## 5. Earthquake/noise classification is unsupported

The original catalog contains earthquakes, not station trigger streams with noise/false-trigger examples. Historical catalog context alone does not tell whether a particular current station signal is an earthquake or noise. Assuming a perfect P trigger would already supply the positive-event condition and would not test CNN's detection task.

Noise has no catalog event magnitude or source origin. Do not create negative rows, copy paired earthquake history into invented trigger timestamps, or use sentinel-filled station/source fields to fabricate class separation. The CNN's 1,214 common-test noise records cannot be represented as current catalog events. Valid detection would need real negative trigger/reference times, observed current-signal tabular features and verified labels with equivalent acquisition/availability rules for both classes.

## 6. Temporal leakage and split restrictions

Origin ordering alone is insufficient. With overlapping CNN split date ranges, a held-out target can precede a training target; naïve full-catalog windows can then expose that held-out event's magnitude/depth/location to optimization as historical features. Conversely, a record preceding a test target is not necessarily available by its decision cutoff. The presence of these risks is established; actual window overlap counts remain UNKNOWN because no sequences were built.

For the smallest conservative retrospective comparison, keep CNN target assignments fixed and restrict history inputs for **all** partitions to eligible training-ID catalog records only, additionally enforcing as-of time and strict predecessor conditions. This prevents validation/test target attributes from entering training histories or preprocessing. It produces sequences of up to 20 permitted events, not necessarily the 20 immediately preceding events in the complete catalog; state that restriction. History may be shared across samples when genuinely past/available, but its dependency IDs must be recorded.

This restriction does not convert the existing CNN split into prospective deployment: model fitting and global train-only scalers can still involve dates later than an early validation/test event. A genuinely prospective test requires chronological model-fitting cutoffs and availability-respecting evaluation for both branches. Do not silently resplit or alter the completed CNN to manufacture that claim. An online protocol permitting past held-out catalog labels as they become publicly available would require separate explicit design; it is not assumed here.

Required future validation: target absent from its history; no future/tied ambiguous history rows; field issuance before cutoff; no validation/test IDs in conservative training dependencies; preprocessing fit IDs confined to allowed training entries; valid padding/missingness; magnitude-scale handling; and source-ID/target alignment. These are requirements, not completed PASS claims.

## 7. Common-cohort comparison and magnitude consistency

ID correspondence and numeric target agreement are established. The CNN's common earthquake cohort offers up to **1,200 matched targets** for a future context analysis, subject to history availability/exclusions. Its full detection cohort additionally contains **1,214 noise examples**, which CSV-only data cannot support. Match by source/parent ID and preserve the existing horizon masks; never assume catalog and HDF5 row order agree.

For consistency with the frozen CNN, use its exact numeric same-event targets and identify them honestly as mixed catalog magnitude values. An ML-only restriction would change the cohort and require explicitly matched CNN subset metrics; no such restriction or recomputation is performed here. Unknown cross-scale calibration cannot be repaired by relabeling magnitude values as MW.

Even if matching earthquake targets becomes possible, historical-prior and P-wave-observation models have different information budgets. Report that distinction rather than asserting identical EEW capability. With fixed historical inputs and no authenticated new reports, the history branch has no genuine 1s/2s/3s observation evolution. Existing CNN cohort metrics are reused as evidence only; no audit or inference was rerun.

## 8. Minimum missing information and next step

1. Obtain catalog first-publication times and field-level revision/availability histories, or a genuinely frozen, timestamped pre-decision catalog archive with a defensible completeness policy. Do not assume a fixed latency that the data cannot support.
2. Obtain genuine current trigger/acquisition timestamps, event association and cutoff-limited current-event tabular observations if an EEW estimator, rather than a contextual prior, is required. For detection, include authentic negative-trigger observations and timing as well.
3. Fix magnitude-scale semantics and decide between a disclosed retrospective common-ID comparison and a prospective protocol; the existing interleaved CNN split only supports the former.
4. Only after these choices and data are available, implement an availability/dependency manifest and the proposed history builder in a notebook, validate it, then reconsider readiness. Do not start another next-catalog-event experiment to bypass the missing EEW evidence.

If the project retains the strict original-CSV-only restriction, stop at this BLOCKED verdict. A historical context branch may later be useful as a prior, but the intended standalone detection-plus-current-magnitude EEW task is not established today.

## Evidence paths and preservation

- `dataset_2.csv`
- `bilstm_eew/readiness_report.md`
- `bilstm new updated/dataset/v2_timestamp_fixed/feature_schema.json`
- `cnn/cnn wave/eew_causal_v3/configs/split_manifest.csv`
- `cnn/cnn wave/eew_causal_v3/cnn_eew_training/codex_performance_audit/cnn_performance_audit_report.md` — common-cohort section only.

Only this report was created. No existing file was modified, moved or deleted. No training, feature extraction, synthetic/noise generation, data duplication, fusion, CNN rework, repository scan or cleanup occurred.
