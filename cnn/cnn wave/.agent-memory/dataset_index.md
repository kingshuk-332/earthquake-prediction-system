# Dataset index — inspected 2026-11-05
## Prepared forecasting dataset (new Phase 1 output)
- Path relative to this directory's parent: `phase1_cnn_results/reports/cnn_b86050c74334/forecast_dataset.npz`.
- Format/shape: compressed NPZ; X=(782,12,90), Y=(782,7); partition indices 562/110/110. Feature/target names and scaling are in the matching `models/.../preprocessing.json`.
- Features: count, m4, m45, max_mag, mean_mag, mean_depth, mean_lat, mean_lon, cell_0_count through cell_3_count; log1p counts, training-only standardization.
- Targets: M45_any_7d, M45_days_0_3, M45_days_3_7, M45_SW_7d, M45_SE_7d, M45_NW_7d, M45_NE_7d; binary recorded-event occurrence with negative windows retained.
- Coverage: 2011-01-01 through 2026-10-06 exclusive; forecast issues and purged intervals are in split_manifest.csv. Raw input contains 5,603 complete-day events in this period.
- Purpose: baseline future catalog forecasting. Risks: unknown coverage/availability, mixed magnitude types, sparse cells, nonstationarity and exploratory holdout. No numerical missing values; all labels and full-history split separation verified.

Paths below are relative to repository root. Full dtypes/missing values are in `phase1_cnn_results/reports/dataset_inventory.json`. Refresh changed sources only.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase10_classification_dataset/earthquake_noise_dataset.h5`
- Format: .h5; shape: HDF5 datasets below.
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields HDF5 numerical arrays not exhaustively scanned; duplicate rows not assessed for trace arrays.
  - `depth_km`: [16178], float32
  - `hypocentral_distance_km`: [16178], float32
  - `labels`: [16178], int8, label counts {'1': 8089, '0': 8089}
  - `magnitude`: [16178], float32
  - `p_arrival_sample`: [16178], int32
  - `qc_pass`: [16178], bool
  - `receiver_code`: [16178], object
  - `s_arrival_sample`: [16178], int32
  - `sample_type`: [16178], object
  - `snr_db`: [16178], float32
  - `source_id`: [16178], object
  - `trace_name`: [16178], object
  - `waveforms`: [16178, 10000], float32

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase10_classification_dataset/earthquake_noise_mapping.csv`
- Format: .csv; shape: [16178, 10].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: combined_index, label, sample_type, source_id, trace_name, receiver_code, qc_pass, original_phase9_index, noise_profile_type, seed.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase10_classification_dataset/phase10_dataset_qc.csv`
- Format: .csv; shape: [16178, 13].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: combined_index, label, sample_type, is_finite, not_all_zero, min_amplitude, max_amplitude, peak_amplitude, rms_amplitude, pre_p_rms, std_amplitude, qc_pass, noise_type.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/phase11_leakage_audit.csv`
- Format: .csv; shape: [5, 5].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: audit_check, description, target_expectation, measured_result, pass_fail.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/split_indices.csv`
- Format: .csv; shape: [16178, 7].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: combined_index, split, label, sample_type, pair_id, source_id, trace_name.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/test.h5`
- Format: .h5; shape: HDF5 datasets below.
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields HDF5 numerical arrays not exhaustively scanned; duplicate rows not assessed for trace arrays.
  - `depth_km`: [2428], float32
  - `hypocentral_distance_km`: [2428], float32
  - `labels`: [2428], int8, label counts {'1': 1214, '0': 1214}
  - `magnitude`: [2428], float32
  - `p_arrival_sample`: [2428], int32
  - `pair_id`: [2428], int32
  - `qc_pass`: [2428], bool
  - `receiver_code`: [2428], object
  - `s_arrival_sample`: [2428], int32
  - `sample_type`: [2428], object
  - `snr_db`: [2428], float32
  - `source_id`: [2428], object
  - `trace_name`: [2428], object
  - `waveforms`: [2428, 10000], float32
  - `waveforms_global_norm`: [2428, 10000], float32
  - `waveforms_standardized`: [2428, 10000], float32

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/train.h5`
- Format: .h5; shape: HDF5 datasets below.
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields HDF5 numerical arrays not exhaustively scanned; duplicate rows not assessed for trace arrays.
  - `depth_km`: [11324], float32
  - `hypocentral_distance_km`: [11324], float32
  - `labels`: [11324], int8, label counts {'1': 5662, '0': 5662}
  - `magnitude`: [11324], float32
  - `p_arrival_sample`: [11324], int32
  - `pair_id`: [11324], int32
  - `qc_pass`: [11324], bool
  - `receiver_code`: [11324], object
  - `s_arrival_sample`: [11324], int32
  - `sample_type`: [11324], object
  - `snr_db`: [11324], float32
  - `source_id`: [11324], object
  - `trace_name`: [11324], object
  - `waveforms`: [11324, 10000], float32
  - `waveforms_global_norm`: [11324, 10000], float32
  - `waveforms_standardized`: [11324, 10000], float32

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/validation.h5`
- Format: .h5; shape: HDF5 datasets below.
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields HDF5 numerical arrays not exhaustively scanned; duplicate rows not assessed for trace arrays.
  - `depth_km`: [2426], float32
  - `hypocentral_distance_km`: [2426], float32
  - `labels`: [2426], int8, label counts {'1': 1213, '0': 1213}
  - `magnitude`: [2426], float32
  - `p_arrival_sample`: [2426], int32
  - `pair_id`: [2426], int32
  - `qc_pass`: [2426], bool
  - `receiver_code`: [2426], object
  - `s_arrival_sample`: [2426], int32
  - `sample_type`: [2426], object
  - `snr_db`: [2426], float32
  - `source_id`: [2426], object
  - `trace_name`: [2426], object
  - `waveforms`: [2426, 10000], float32
  - `waveforms_global_norm`: [2426, 10000], float32
  - `waveforms_standardized`: [2426, 10000], float32

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase1_eda/phase1_descriptive_statistics.csv`
- Format: .csv; shape: [8, 11].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: Unnamed: 0, count, mean, std, min, 25%, 50%, 75%, max, median, iqr.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase2_travel_time/metadata_with_theoretical_arrivals.csv`
- Format: .csv; shape: [8089, 42].
- Dates: 1906-08-31 14:57:30+00:00 to 2026-10-06 09:03:32+00:00.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: year, origin_time_ist, origin_time_utc, Lat, Long, Depth_km, Magnitude_Value, Magnitude_Type, Country, State, Region, Location, source_magnitude_type, source_id, receiver_code, receiver_latitude, receiver_longitude, receiver_elevation_m, p_arrival_sample, s_arrival_sample, snr_db, trace_name, epicentral_dist_km, month, day_of_week, hour, day_of_year, decade, magnitude_category, depth_class, sp_travel_time_samples, log_magnitude, log_depth, src_recv_dist_km, hypocentral_distance_km, theoretical_p_arrival_seconds, theoretical_s_arrival_seconds, theoretical_sp_time_seconds, theoretical_p_arrival_sample, theoretical_s_arrival_sample, theoretical_sp_time_samples, window_aligned_theoretical_s_sample.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase5_prototype_10/prototype_10_metadata.csv`
- Format: .csv; shape: [10, 12].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: source_id, trace_name, receiver_code, magnitude, depth, epicentral_distance_km, hypocentral_distance_km, p_arrival_sample, s_arrival_sample, waveform_filename, sampling_rate, waveform_length.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase5_prototype_10/prototype_evaluation_metrics.csv`
- Format: .csv; shape: [10, 20].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: record_index, source_id, trace_name, receiver_code, magnitude, depth_km, epicentral_dist_km, hypocentral_distance_km, p_arrival_sample, s_arrival_sample, sp_sample_difference, sp_time_seconds, peak_amplitude, rms_amplitude, target_snr_db, measured_snr_db, dominant_frequency_hz, has_nan, has_inf, p_before_s.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase6_full_dataset/failed_records.csv`
- Format: .csv; shape: HDF5 datasets below.
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields HDF5 numerical arrays not exhaustively scanned; duplicate rows not assessed for trace arrays.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase6_full_dataset/synthetic_seismic_waveforms.h5`
- Format: .h5; shape: HDF5 datasets below.
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields HDF5 numerical arrays not exhaustively scanned; duplicate rows not assessed for trace arrays.
  - `depth_km`: [8089], float32
  - `hypocentral_distance_km`: [8089], float32
  - `magnitude`: [8089], float32
  - `p_arrival_sample`: [8089], int32
  - `receiver_code`: [8089], object
  - `s_arrival_sample`: [8089], int32
  - `snr_db`: [8089], float32
  - `source_id`: [8089], object
  - `trace_name`: [8089], object
  - `waveforms`: [8089, 10000], float32

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase6_full_dataset/synthetic_waveforms_mapping.csv`
- Format: .csv; shape: [8089, 13].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: hdf5_index, catalog_row_index, source_id, trace_name, receiver_code, magnitude, depth_km, epicentral_dist_km, hypocentral_distance_km, p_arrival_sample, s_arrival_sample, snr_db, status.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase7_quality_control/qc_measurements_50_traces.csv`
- Format: .csv; shape: [50, 32].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: qc_sample_id, hdf5_index, source_id, trace_name, receiver_code, magnitude, depth_km, hypocentral_distance_km, theoretical_p_arrival_sample, theoretical_s_arrival_sample, theoretical_p_time_seconds, theoretical_s_time_seconds, theoretical_sp_interval_seconds, target_snr_db, min_amplitude, max_amplitude, mean_amplitude, std_amplitude, peak_amplitude, rms_amplitude, pre_p_rms, p_window_rms, s_window_rms, measured_snr_db, p_in_bounds, s_in_bounds, p_before_s, is_finite, not_all_zero, p_energy_jump, s_energy_jump, overall_qc_passed.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase8_improved_simulation/old_vs_new_qc_comparison.csv`
- Format: .csv; shape: [50, 17].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: sample_id, hdf5_index, source_id, receiver_code, magnitude, distance_km, target_snr_db, old_measured_snr_db, new_measured_snr_db, old_snr_error_db, new_snr_error_db, old_p_energy_jump, new_p_energy_jump, old_s_energy_jump, new_s_energy_jump, old_overall_passed, new_overall_passed.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase8_improved_simulation/qc_measurements_improved_50_traces.csv`
- Format: .csv; shape: [50, 35].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: qc_sample_id, hdf5_index, source_id, trace_name, receiver_code, magnitude, depth_km, hypocentral_distance_km, p_arrival_sample, s_arrival_sample, theoretical_p_sec, theoretical_s_sec, target_snr_db, min_amplitude, max_amplitude, std_amplitude, peak_amplitude, rms_amplitude, pre_p_rms, p_window_rms, s_window_rms, p_to_noise_energy_ratio, s_to_noise_energy_ratio, signal_peak_amplitude, measured_snr_db, p_in_bounds, s_in_bounds, p_before_s, is_finite, not_all_zero, p_energy_jump, s_energy_jump, mandatory_numerical_passed, structure_passed, overall_qc_passed.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase9_full_dataset/full_dataset_qc_metrics.csv`
- Format: .csv; shape: [8089, 32].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: hdf5_index, source_id, trace_name, receiver_code, magnitude, depth_km, hypocentral_distance_km, p_arrival_sample, s_arrival_sample, target_snr_db, measured_snr_db, snr_error_db, abs_snr_error_db, min_amplitude, max_amplitude, peak_amplitude, rms_amplitude, pre_p_rms, p_window_rms, s_window_rms, p_to_noise_ratio, s_to_noise_ratio, p_in_bounds, s_in_bounds, p_before_s, is_finite, not_all_zero, mandatory_numerical_passed, p_energy_jump, s_energy_jump, structure_passed, overall_qc_passed.

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase9_full_dataset/synthetic_seismic_waveforms.h5`
- Format: .h5; shape: HDF5 datasets below.
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields HDF5 numerical arrays not exhaustively scanned; duplicate rows not assessed for trace arrays.
  - `depth_km`: [8089], float32
  - `hypocentral_distance_km`: [8089], float32
  - `magnitude`: [8089], float32
  - `p_arrival_sample`: [8089], int32
  - `receiver_code`: [8089], object
  - `s_arrival_sample`: [8089], int32
  - `snr_db`: [8089], float32
  - `source_id`: [8089], object
  - `trace_name`: [8089], object
  - `waveforms`: [8089, 10000], float32

## `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase9_full_dataset/synthetic_waveforms_mapping.csv`
- Format: .csv; shape: [8089, 11].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Synthetic waveform simulation or quality control; excluded from forecasting.
- Risks: Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.
- Quality: missing fields {}; duplicate rows 0.
- Columns: hdf5_index, source_id, trace_name, receiver_code, magnitude, depth_km, hypocentral_distance_km, p_arrival_sample, s_arrival_sample, target_snr_db, trace_seed.

## `cnn/preprocessed_dl_temporal_noisy.csv`
- Format: .csv; shape: [8089, 45].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Historical catalog/preprocessing audit.
- Risks: Previously scaled/encoded versions may use full-catalog statistics; only raw dataset_2.csv is used.
- Quality: missing fields {}; duplicate rows 0.
- Columns: Lat, Long, Depth_km, Magnitude_Value, receiver_latitude, receiver_longitude, receiver_elevation_m, p_arrival_sample, s_arrival_sample, snr_db, epicentral_dist_km, sp_travel_time, year_raw, month_sin, month_cos, dow_sin, dow_cos, hour_sin, hour_cos, doy_sin, doy_cos, time_diff_hours, log_time_diff, depth_mag_ratio, log_energy, Magnitude_Type_MB, Magnitude_Type_MC, Magnitude_Type_ML, Magnitude_Type_MS, Magnitude_Type_MW, source_magnitude_type_MB, source_magnitude_type_MC, source_magnitude_type_ML, source_magnitude_type_MW, receiver_code_AGT, receiver_code_DBR, receiver_code_GHY, receiver_code_IMP, receiver_code_ITAN, receiver_code_KOH, receiver_code_PASI, receiver_code_SHL, receiver_code_SIL, receiver_code_TEZ, target.

## `dataset_2.csv`
- Format: .csv; shape: [8089, 28].
- Dates: 1906-08-31 14:57:30+00:00 to 2026-10-06 09:03:32+00:00.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Historical catalog/preprocessing audit.
- Risks: Previously scaled/encoded versions may use full-catalog statistics; only raw dataset_2.csv is used.
- Quality: missing fields {}; duplicate rows 0.
- Columns: year, origin_time_ist, origin_time_utc, Lat, Long, Depth_km, Magnitude_Value, Magnitude_Type, Magnitude, Country, State, Region, Location, source_latitude, source_longitude, source_depth_km, source_magnitude, source_magnitude_type, source_id, receiver_code, receiver_latitude, receiver_longitude, receiver_elevation_m, p_arrival_sample, s_arrival_sample, snr_db, trace_name, epicentral_dist_km.

## `preprocessed_cleaned.csv`
- Format: .csv; shape: [8089, 23].
- Dates: 1906-08-31 14:57:30+00:00 to 2026-10-06 09:03:32+00:00.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Historical catalog/preprocessing audit.
- Risks: Previously scaled/encoded versions may use full-catalog statistics; only raw dataset_2.csv is used.
- Quality: missing fields {}; duplicate rows 0.
- Columns: year, origin_time_ist, origin_time_utc, Lat, Long, Depth_km, Magnitude_Value, Magnitude_Type, Country, State, Region, Location, source_magnitude_type, source_id, receiver_code, receiver_latitude, receiver_longitude, receiver_elevation_m, p_arrival_sample, s_arrival_sample, snr_db, trace_name, epicentral_dist_km.

## `preprocessed_dl.csv`
- Format: .csv; shape: [8089, 30].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Historical catalog/preprocessing audit.
- Risks: Previously scaled/encoded versions may use full-catalog statistics; only raw dataset_2.csv is used.
- Quality: missing fields {}; duplicate rows 41.
- Columns: Lat, Long, Depth_km, Magnitude_Value, receiver_latitude, receiver_longitude, receiver_elevation_m, p_arrival_sample, s_arrival_sample, snr_db, epicentral_dist_km, Magnitude_Type_MB, Magnitude_Type_MC, Magnitude_Type_ML, Magnitude_Type_MS, Magnitude_Type_MW, source_magnitude_type_MB, source_magnitude_type_MC, source_magnitude_type_ML, source_magnitude_type_MW, receiver_code_AGT, receiver_code_DBR, receiver_code_GHY, receiver_code_IMP, receiver_code_ITAN, receiver_code_KOH, receiver_code_PASI, receiver_code_SHL, receiver_code_SIL, receiver_code_TEZ.

## `preprocessed_dl_temporal.csv`
- Format: .csv; shape: [8089, 44].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Historical catalog/preprocessing audit.
- Risks: Previously scaled/encoded versions may use full-catalog statistics; only raw dataset_2.csv is used.
- Quality: missing fields {}; duplicate rows 0.
- Columns: Lat, Long, Depth_km, Magnitude_Value, receiver_latitude, receiver_longitude, receiver_elevation_m, p_arrival_sample, s_arrival_sample, snr_db, epicentral_dist_km, sp_travel_time, year_raw, month_sin, month_cos, dow_sin, dow_cos, hour_sin, hour_cos, doy_sin, doy_cos, time_diff_hours, log_time_diff, depth_mag_ratio, log_energy, Magnitude_Type_MB, Magnitude_Type_MC, Magnitude_Type_ML, Magnitude_Type_MS, Magnitude_Type_MW, source_magnitude_type_MB, source_magnitude_type_MC, source_magnitude_type_ML, source_magnitude_type_MW, receiver_code_AGT, receiver_code_DBR, receiver_code_GHY, receiver_code_IMP, receiver_code_ITAN, receiver_code_KOH, receiver_code_PASI, receiver_code_SHL, receiver_code_SIL, receiver_code_TEZ.

## `preprocessed_eda.csv`
- Format: .csv; shape: [8089, 34].
- Dates: 1906-08-31 14:57:30+00:00 to 2026-10-06 09:03:32+00:00.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Historical catalog/preprocessing audit.
- Risks: Previously scaled/encoded versions may use full-catalog statistics; only raw dataset_2.csv is used.
- Quality: missing fields {}; duplicate rows 0.
- Columns: year, origin_time_ist, origin_time_utc, Lat, Long, Depth_km, Magnitude_Value, Magnitude_Type, Country, State, Region, Location, source_magnitude_type, source_id, receiver_code, receiver_latitude, receiver_longitude, receiver_elevation_m, p_arrival_sample, s_arrival_sample, snr_db, trace_name, epicentral_dist_km, month, day_of_week, hour, day_of_year, decade, magnitude_category, depth_class, sp_travel_time_samples, log_magnitude, log_depth, src_recv_dist_km.

## `preprocessed_ml.csv`
- Format: .csv; shape: [8089, 28].
- Dates: Not encoded as origin_time_utc; no independent coverage claim to not applicable.
- Labels: No forecast target; catalog or QC metadata.
- Purpose: Historical catalog/preprocessing audit.
- Risks: Previously scaled/encoded versions may use full-catalog statistics; only raw dataset_2.csv is used.
- Quality: missing fields {}; duplicate rows 0.
- Columns: year, Lat, Long, Depth_km, Magnitude_Value, Magnitude_Type, Country, State, source_magnitude_type, receiver_code, receiver_latitude, receiver_longitude, receiver_elevation_m, p_arrival_sample, s_arrival_sample, snr_db, epicentral_dist_km, month, day_of_week, hour, day_of_year, decade, magnitude_category, depth_class, sp_travel_time_samples, log_magnitude, log_depth, src_recv_dist_km.

## Selected raw catalog findings
8,089 x 28, sorted UTC 1906-08-31 through 2026-10-06; no missing values, duplicate rows, source IDs or physical event keys. Magnitude scales: ML 7125, MB 458, MW 398, MC 107, MS 1. Reporting shifts are visible, not proven instrumentation changes. No publication timestamps or outage information.
Forecast inputs: origin_time_utc, Magnitude_Value, Lat, Long, Depth_km only. Final partial day excluded. Modern start 2011 is exploratory, not a completeness guarantee.

## Verified waveform detection use — wavecnn_ccb3b06b3d37
Phase11 train/validation/test HDF5: (11324,10000)/(2426,10000)/(2428,10000), float32 at100Hz. Detection labels1/0 balanced; not forecast labels. All raw/standardized traces finite/nonzero, exactly unique; pair/source sets disjoint; formulas checked on12/split and all standardized moments verified. Full keys/attributes and per-split historical date coverage: `phase1_waveform_cnn_results/reports/waveform_audit.json`.
HDF5 has magnitude/P/S/depth/distance/SNR but no origin_time_utc/Lat/Long. Earthquake source-ID join to Phase2 metadata complete; noise has no corresponding event metadata. `verified_metadata.csv` contains reporting-only joined fields; `signal_features.csv` holds seven derived signal descriptors for simple comparators. Neither supplies event metadata to the CNN.
The prior exclusion of waveforms applies to future forecasting, not the now-completed same-trace detection experiment. Synthetic shortcut and single-component identifiability limitations remain.
