# Causal EEW v3 implementation validation

**READY FOR INDEPENDENT DATASET VERIFICATION.** This is implementation validation, not final dataset approval. No CNN/BiLSTM training, Optuna or fusion was performed.

## Root causes and repairs

V2's full-record background SD made early noise amplitude depend on future random innovations. Its finite-record P/S envelope maxima made earlier signal depend on future coda values. Exact v2 audit reproduction recovered normalized early-window differences **361.790328979** and **99.573059082**. Altering a future tail while preserving earlier innovations/envelopes is a valid intervention; altering shared magnitude/distance/decay parameters is not an invariance test.

V3 reuses the actual v2 earthquake implementation's velocities, Q(f), attenuation, source scaling, carriers and gamma time constants. Noise calibration now uses fixed impulse-response energy, independent of realized waveform samples. Gamma normalization uses the analytic peak at2*tau_d. Independent random streams separate onset, noise and background-profile parameters. Causal FIR pink-like noise replaces length-dependent FFT synthesis; transient disturbances have explicit causal onsets. These background changes are required causal repairs, not a claim of real-noise validation.

## Pilot gate

Fifteen stratified earthquake cases reproduce the exact v2 defects, then exercise S amplitude x4, future S coda +50%, future noise perturbation, future P coda +50% and x4,12/100/140-second durations, and postprocessing prefix tests. Sixteen noise cases cover four families and four seeds. **All v3 pilot prefix differences are exactly0 raw and normalized**, with a fixed zero tolerance. Full generation started only after this gate passed.

## Exhaustive saved-data checks

All **16,178 records**: train11,324, validation2,426, test2,428. HDF5 files save full100-second parents (N,10000), raw/normalized windows (N,200/300/400), pure-post windows (N,100/200/300),100Hz, float32 waveform arrays, separate metadata and explicit masks. Every numeric array is finite. Every window equals its parent slice and independently recomputed pre-context normalization; maximum normalization error=0.0.

All class/magnitude/source/station labels, original source IDs and split assignments, seeds, timestamps, P/S alignment and mask formulas pass. Parent waveform hashes and all saved window hashes show no cross-split duplicates; event, pair, parent and background identities are disjoint. Background digests are checked across every record and reproduced in stratified examples; full background arrays are not separately stored. 29 stratified replays pass, including a separate earthquake-equation implementation and each noise family. Maximum parent replay error=0; maximum normalized replay error=0. Saved normalization's future-perturbation prefix difference=0.

Noise magnitude=-1 is explicitly excluded by magnitude_mask=(label==1); per-horizon magnitude masks also require pre-S eligibility. Poisoning noise targets leaves masked regression diagnostics unchanged. Noise has no fictional P/S prediction target: arrival labels remain-1. Noise pre-S flags are false; separate classification masks admit noise records. These are verified dataset contracts; a future training consumer must use the supplied masks.

## Arrival precision and pre-S eligibility

Continuous travel times are unchanged. S sample denotes the first grid sample at/after physical arrival (ceil), not a rounded-nearest index. Strict integer eligibility is reference_sample+100*tau < S_sample, with continuous-time agreement checked.

| Source | Unchanged S-P seconds | v2 rounded difference | v3 first-grid difference | 2s/3s eligibility |
|---|---|---|---|---|
| NCS_EV_03038 | 2.001841556 | 200 | 201 | True/False |
| NCS_EV_03572 | 3.002951118 | 300 | 301 | True/True |

| Horizon | Eligible earthquakes | Percentage |
|---|---|---|
| 1s | 8089/8089 | 100.0000% |
| 2s | 8056/8089 | 99.5920% |
| 3s | 7996/8089 | 98.8503% |

No real arrival was moved to force eligibility. Raw parent P positions vary; oracle cropping deliberately puts P at index100 of positive windows. Negative references use the same3–7s range without a phase target. Future S is used only for offline eligibility labels; source/phase/magnitude metadata are not input channels.

## Limitations and preservation

This remains a **synthetic oracle-aligned benchmark**, not a streaming trigger/picker or operational EEW system. The1-second pre-P context is extra observed history beyond each1/2/3-second post-P horizon. Synthetic magnitudes affect early amplitudes under the source model, but single-station magnitude/distance tradeoffs, relative amplitude units, fixed propagation parameters, low-SNR examples and artificial background families limit identifiability and transfer to real data.

All **44 protected source/v2/audit files** have identical before/after SHA256, including original Phase11 data and CNN checkpoints. No existing status, memory, BiLSTM or original dataset was changed. New work is confined to eew_causal_v3.

Evidence: v3_counterfactual_results.json, v3_validation_results.json; ../configs/simulation_v3.json, split_manifest.csv, generation_manifest.json and protected_sources_before/after.json; ../dataset/eew_dataset_{train,validation,test}.h5; the three executed notebooks. No model training is authorized by this implementation result.
