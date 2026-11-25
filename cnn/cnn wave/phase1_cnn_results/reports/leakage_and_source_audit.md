# Source and leakage audit — 2026-11-05
## Simulator stages and actual provenance
1. phase1_inspect_metadata.py: metadata validity and hypocentral-distance EDA.
2. phase2_travel_time.py: R=sqrt(epicentral_distance^2+depth^2), P=R/6, S=R/3.5; origin and window-aligned arrivals differ.
3. phase3_clean_synthesis.py plus waveform_generator.py: clean damped P/S packets.
4. phase4_attenuation_noise.py: same-event magnitude amplitude, distance attenuation and ambient noise.
5. phase5_prototype_evaluation.py: ten diverse metadata records, waveform diagnostics.
6. phase6_generate_full_dataset.py: compressed 8089 x 10000 vertical-component traces and mapping.
7. waveform_quality_control.py: seeded 50-trace numerical, SNR, energy and arrival QC.
8. improved_waveform_generator.py and improved_simulation_qc.py: Q(f), gamma envelopes dependent on magnitude/distance, multi-harmonic carriers, pre-P SNR calibration and separate numerical/structural checks.
9. phase9_generate_full_dataset.py: reads output/phase2_travel_time/metadata_with_theoretical_arrivals.csv; improved 8089 x 10000 traces.
10. phase10_generate_classification_dataset.py: 8089 earthquakes + 8089 noise traces, labels 1/0; paired noise-floor calibration.
11. phase11_split_and_preprocess.py: magnitude/QC stratified pair splitting; detrending, per-trace and global training-only normalization; 11324/2426/2428 samples. Pair leakage precautions do not establish temporal forecasting validity.
config.py contains a legacy `data wave/Minor project/preprocessed_eda.csv` path which is absent here. The actual source metadata output remains available. No simulator was rerun or modified.

## Prior catalog model
cnn/cnn_1d_model.py uses 15-event sequences to classify next-event M>=4.5, prefers a noisy derived CSV, uses a 70/10/20 event-index split, SMOTE and focal loss. Checkpoints exist (cnn_1d_v2_best.pth, cnn_1d_v2_final.pth); not imported or changed. Historical reported results are not new experiment results.
preprocess_dl_temporal.py computes 99th-percentile gap clipping, imputation, categorical encoding and MinMax scaling on the whole catalog (fit_transform at line 153). These transformations must be rebuilt. The same-event features of *past* events are not intrinsically future leakage; the new design nevertheless excludes questionable synthetic station/trace metadata.

## New protocol
Raw metadata only, daily left-closed/right-open aggregation, past 90 days, 7-day targets at weekly issue times. No event-conditioned selection of forecast windows. Calendar splits with whole history and horizon confined to each partition. Train-only unique-day scaler; no future clipping or learned grid boundaries. Final day excluded as partial; no gap-filling of event observations. Target membership verified independently from raw timestamps.
Historical origin time is not ingestion time. Real-time availability and completeness are unverified; this is a retrospective catalog experiment only.
All-catalog descriptive EDA was seen during protocol development. Test values are excluded from fitting/model selection; independent prospective confirmation is still necessary.

## Inspection evidence
dataset_inventory.json records 28 file schemas/HDF5 structures; source_inventory.json records 19 source hashes, module/function descriptions and line numbers. annual_catalog_counts.csv includes zero-record years. Large waveform values were not exhaustively reread, because these datasets are excluded from training.
