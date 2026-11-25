# Tabular EEW BiLSTM readiness

Date: 2026-11-13

**Verdict: CONDITIONAL. CSV-only EEW modeling is BLOCKED; a comparable synthetic tabular benchmark is feasible after causal feature extraction from the existing verified v3 waveform windows. No training is authorized by this readiness finding.**

The CSV supplies event identity, supervision and retrospective metadata. It does not supply an observed within-event sequence for earthquake/noise detection or early magnitude estimation. Replacing waveform observations with true source parameters would invalidate the comparison. The proposed features below are requirements, not newly generated data or claims that these columns already exist.

## Evidence inspected

- `dataset_2.csv`: header, first three and last two records; source-ID column across the catalog solely to resolve complete event alignment.
- `bilstm new updated/dataset/v2_timestamp_fixed/feature_schema.json`: relevant existing feature definitions and availability statement.
- `cnn/cnn wave/eew_causal_v3/codex_final_verification/hdf5_training_readiness.md`.
- `cnn/cnn wave/eew_causal_v3/cnn_eew_training/final_cnn_eew_report.md`: protocol, eligible populations, results and limitations.
- `cnn/cnn wave/eew_causal_v3/configs/split_manifest.csv`: schema and identity/split/mask columns needed for alignment and common-cohort counts.

No CNN checkpoint, waveform array, generator, earlier phase implementation or unrelated project directory was inspected. No CNN audit was repeated. Read-only metadata queries were used; no executable artifact or dataset was created.

## Existing fields and availability

Availability below means information actually established as known by the decision cutoff, not merely a column present in a retrospective catalog. The inspected sources provide no arrival-pick delivery times, source-estimate issuance times, catalog publication times or revision histories.

| Field group | Existing fields | At P+1 / P+2 / P+3 seconds | Treatment |
|---|---|---|---|
| P annotation | `p_arrival_sample` | Not verified as a causal detected pick at any horizon; CSV example indices differ from v3 annotations | Oracle/legacy annotation only. Use the existing v3 reference for benchmark alignment, never as an input feature or demonstrated trigger |
| S annotation | `s_arrival_sample` | Future in the eligible pre-S cohorts | Exclude from inputs. Use v3 eligibility masks only for offline filtering |
| Magnitude | `Magnitude_Value`, `Magnitude`, `Magnitude_Type`, `source_magnitude`, `source_magnitude_type` | Final target value/type is not established as an issued early estimate | Magnitude value is supervision only; do not input its duplicates or type as shortcuts |
| Source depth and location | `Depth_km`, `Lat`, `Long`, `source_depth_km`, `source_latitude`, `source_longitude`; `Country`, `State`, `Region`, `Location` | No demonstrated real-time availability at any horizon | Exclude; final source geometry and descriptions are privileged metadata |
| Distance | `epicentral_dist_km`; manifest `hypocentral_distance_km` | Depends on unknown source location/depth | Exclude true distances. A future causal network estimate would require issuance-time provenance and a separately defined information budget |
| Station metadata | `receiver_code`, `receiver_latitude`, `receiver_longitude`, `receiver_elevation_m` | Real station inventory can be known before P | Potentially causal in principle, but exclude from this first comparison: noise has placeholder station metadata and CNN receives no station features |
| Time | `year`, `origin_time_ist`, `origin_time_utc` | Final source origin time is not the receiver acquisition clock or trigger delivery time | Exclude these catalog timestamps as model inputs. Relative elapsed time from the reference is legitimate sequence indexing, not a true-origin feature |
| Signal quality | `snr_db` | Its observation window and calculation provenance are not established as cutoff-limited | Exclude. Do not assume the supplied SNR is an early-P measurement |
| Identity | `source_id`, `trace_name`; manifest `parent_id`, `pair_id`, `combined_index` | Useful offline for joins, not seismic measurements | Keep for provenance/splitting only; ID strings can reveal class |
| Simulator/evaluation metadata | Manifest P/S seconds, `sp_seconds`, theoretical travel times if supplied elsewhere, seed, background ID, noise profile, labels, masks | Derived from generation, true source conditions or supervision | Exclude from forward inputs. Even a theoretical S arrival known to the simulator is not an observed or available station measurement |

**No dynamic early-P input feature is presently established from the CSV alone at any of the three cutoffs.** There are P/S index annotations, but no sampled P-wave amplitude/energy evolution, causal trigger confidence, polarization sequence or timestamped early source estimates. There are likewise no observed S-wave measurements suitable for these pre-S tasks.

The previous ten-feature BiLSTM schema uses historical magnitude/depth/latitude/longitude, inter-event days, 30-day event rate and rolling 20-event magnitude/depth means and standard deviations. Its axis is `historical_events`, sequence length 20, and operational availability is explicitly UNKNOWN. It describes next-retained-event catalog context, not this event's first seconds of P-wave observations. Final catalog values require publication/revision availability before they could be called causal historical context. Do not reuse its tensors, scaler, target definition or event-order sequence for the EEW benchmark.

## Event and noise alignment

Full source-ID set comparison found **8,089 catalog rows, 8,089 unique catalog IDs, 8,089 CNN earthquake rows and 8,089 unique CNN earthquake IDs; symmetric ID difference = 0**. This verifies identity coverage, not the timing or availability of every catalog field. Join by ID, not row position, date ordering or approximate coordinates. Use v3 magnitude targets and v3 arrival/mask metadata rather than overwriting them with legacy CSV columns.

| Split | Earthquakes | Noise | Total |
|---|---:|---:|---:|
| Train | 5,662 | 5,662 | 11,324 |
| Validation | 1,213 | 1,213 | 2,426 |
| Test | 1,214 | 1,214 | 2,428 |

The manifest contains **8,089 noise examples with zero catalog-ID matches**. An inspected noise row has `source_id=NOISE_P1`, `receiver_code=NOISE_STA`, station coordinates -1, empty origin time and P/S indices -1. Thus station identity, missing timestamps, coordinate sentinels, source-ID prefixes and the presence of earthquake catalog attributes would directly reveal class if passed to a model. Do not manufacture noise catalog rows or fill them with paired earthquake source parameters.

Noise examples can instead be represented by applying exactly the same signal-feature extractor to their existing v3 observation windows. Preserve their existing reference alignment and IDs as bookkeeping. This supplies genuine observed synthetic-noise features without requiring nonexistent event magnitude, origin or source geometry.

## Feasible targets and sequence definition

- **Detection:** existing v3 `labels` (earthquake versus noise), restricted to `classification_mask_{h}s`. This is classification of oracle/reference-aligned windows, not autonomous P picking or next-earthquake forecasting.
- **Early magnitude:** existing v3 `magnitude` of the same event, restricted to `magnitude_mask_{h}s`. Noise sentinel -1 never contributes to regression. Compute regression metrics over all eligible earthquakes, independent of whether detection succeeds, matching the CNN report.
- **Sequence:** successive feature snapshots within a single parent waveform's observed prefix. Do not concatenate different catalog earthquakes. A static event row repeated over time is not an observed temporal sequence.

Both targets are **conditionally feasible using existing waveform-derived tabular sequences**, but neither has a defensible comparable implementation from the CSV alone. Feature availability does not guarantee good magnitude identifiability or predictive accuracy. Existing CNN bias/compression and background-family concerns remain relevant.

A BiLSTM can process both directions within the already observed prefix at the decision cutoff. It must not process a full 3-second sequence and then report an intermediate hidden state as a 1-second prediction. Build/truncate the input separately for each horizon; this restriction applies to pooling, sequence normalization and padding as well.

## Smallest valid preprocessing plan — proposed only

1. Reuse `dataset/eew_dataset_{train,validation,test}.h5` in place. Load `windows_{h}s_norm` and the corresponding labels, magnitudes, masks and row identities. Preserve train/validation/test assignments and parent pairing. Do not create another waveform dataset or NPZ copy.
2. Preserve the CNN's information budget: one normalized waveform channel, including its 100-sample pre-reference context. For direct comparability, apply the same fixed pointwise `sign(x) * log1p(abs(x))` transform before extracting summary features. This uses no future samples or fitted transform. Do not add station/source context unavailable to the CNN.
3. Use non-overlapping **10-sample (0.1-second) frames**. The 200/300/400 observed samples yield **20/30/40 feature timesteps**, respectively, with 10 pre-reference frames. Define a frame's timestamp as its right boundary; use only samples in that completed frame. Last included samples are P+h−0.01 seconds. Do not pad shorter horizons with observed samples from longer ones.
4. A minimal explicit feature set is frame signed mean, RMS amplitude, maximum absolute amplitude and RMS of the nine within-frame first differences. These four deterministic summaries use only observed samples and form a tabular time sequence; compute them identically for positives and noise. They are not current CSV columns. Avoid adding redundant catalog-derived geometry or final-event features. Ten-sample frames are a proposed fixed design, not a tuned optimum or validated feature set.
5. If standardization is used, fit feature means/scales on eligible **training** frames only and freeze them for validation/test. Preserve per-row eligibility and fit target scaling only on eligible training earthquakes. Do not normalize a short horizon using later observations, nor introduce whole-parent detrending or full-record SNR. Use explicit validity masks if later extensions require padding/missing measurements; missingness must not encode class.
6. Document feature order, equations, units, frame boundaries, observation cutoff, source HDF5 keys, transform and scaler fit population. Preserve source/parent IDs, split and horizon-mask linkage outside the model inputs. Derive the small feature sequences in memory from the verified HDF5 when needed; no feature dataset was produced here.
7. Before training, perform only targeted extractor checks: frame sample bounds, ID/label/mask correspondence, identical positive/noise processing, train-only scaler fit, and prefix invariance when unseen future samples are changed. Check that no S observations, metadata sentinels or ID channels enter the tensor. Do not repeat the full CNN/dataset audits.

Physical peak displacement, characteristic-period or instrument-unit features would additionally require reliable channel units, instrument response/calibration, and documented causal filtering/integration. These cannot be manufactured from catalog magnitudes, theoretical arrivals or arbitrary synthetic amplitudes. Multi-station location/distance or trigger-confidence features require actual causal estimates with issuance times and uncertainty; they are unnecessary for the smallest fair benchmark above and would change the input-information comparison.

## Fair evaluation protocol

- Use the same event-disjoint HDF5 splits and horizon-specific masks as CNN, and match predictions by split, row, `source_id` and `parent_id`. Do not introduce chronological next-event splits or restrict to an ML-only subset silently; reuse the exact v3 targets.
- Compare each horizon on precisely the same eligible records for both architectures. For across-horizon comparisons, the manifest intersection contains **1,200 test earthquakes and 1,214 test noise records (2,414 total)**. Detection uses all 2,414; magnitude uses the 1,200 earthquakes. Retain the original horizon-specific results as separately labeled analyses.
- Select checkpoints and any thresholds using training/validation only. A fixed 0.5 detection threshold most directly matches the CNN. Freeze the protocol before accessing test outputs. The test set has already been inspected during CNN research; do not describe later model design as a fresh untouched confirmatory test. Stronger confirmatory claims would require separately reserved evidence, not resplitting or regenerating anything here.
- Report the same detection metrics, eligible-earthquake MAE/RMSE/R², magnitude bands, strong-event underprediction and confidence/FPR by existing noise family. Noise profile is diagnostic metadata only. Report magnitude compression and background shortcuts rather than assuming different architecture removes them.
- Include feature-extraction time when comparing end-to-end compute, or clearly separate it from network-only latency. Observation horizons and same-station S-minus-cutoff intervals remain synthetic/oracle-aligned quantities, not operational warning lead times or measured P detection delays.

## Conditions before readiness

The remaining work is a small documented causal waveform-to-tabular extractor and its targeted validation, not another dataset generator. The raw CSV cannot provide the missing observed features. A future synthetic BiLSTM may operate standalone on the resulting tabular frame sequences, without CNN embeddings, shared weights or fusion. An operational system would further need causal picks/acquisition timestamps and real-data validation; those are outside this feasibility result.

Only this report was created. No existing file was changed, deleted or moved; no data was duplicated, feature sequence manufactured, model trained, CNN audit repeated, or fusion/cleanup begun.
