"""Persist the completed inspection without repeatedly loading original datasets."""
from pathlib import Path
import json
BASE=Path(__file__).resolve().parent
MEM=BASE/'.agent-memory'
REPORTS=BASE/'phase1_cnn_results'/'reports'
data=json.loads((REPORTS/'dataset_inventory.json').read_text())
lines=['# Dataset index — inspected 2026-11-05','Paths below are relative to repository root. Full dtypes/missing values are in `phase1_cnn_results/reports/dataset_inventory.json`. Refresh changed sources only.','']
for d in data:
    p=d['path']
    synthetic='synthetic_waveform_simulator' in p
    labels='No forecast target; catalog or QC metadata.'
    purpose='Historical catalog/preprocessing audit.'
    risk='Previously scaled/encoded versions may use full-catalog statistics; only raw dataset_2.csv is used.'
    if synthetic:
        purpose='Synthetic waveform simulation or quality control; excluded from forecasting.'
        risk='Same-event magnitude, distance, arrivals and SNR inform waveform generation; future-information leakage if used as forecast inputs.'
    if 'phase10' in p or 'phase11' in p:
        labels='Detection: 1 synthetic earthquake, 0 synthetic noise; paired stratified splits are not chronological forecasting splits.'
    if 'evaluation_cnn' in p: labels='Old next-event M>=4.5 classification; not fixed-horizon forecasts.'
    lines += [f'## `{p}`',f'- Format: {Path(p).suffix}; shape: {d.get("shape", "HDF5 datasets below")}.',
              f'- Dates: {d.get("start","Not encoded as origin_time_utc; no independent coverage claim")} to {d.get("end","not applicable")}.',
              f'- Labels: {labels}',f'- Purpose: {purpose}',f'- Risks: {risk}',
              f'- Quality: missing fields {d.get("missing", "HDF5 numerical arrays not exhaustively scanned")}; duplicate rows {d.get("duplicate_rows", "not assessed for trace arrays")}.']
    if 'columns' in d: lines.append('- Columns: '+', '.join(d['columns'])+'.')
    if 'datasets' in d:
        lines.extend(f'  - `{k}`: {v["shape"]}, {v["dtype"]}'+(f', label counts {v["counts"]}' if 'counts' in v else '') for k,v in d['datasets'].items())
    lines.append('')
lines += ['## Selected raw catalog findings','8,089 x 28, sorted UTC 1906-08-31 through 2026-10-06; no missing values, duplicate rows, source IDs or physical event keys. Magnitude scales: ML 7125, MB 458, MW 398, MC 107, MS 1. Reporting shifts are visible, not proven instrumentation changes. No publication timestamps or outage information.','Forecast inputs: origin_time_utc, Magnitude_Value, Lat, Long, Depth_km only. Final partial day excluded. Modern start 2011 is exploratory, not a completeness guarantee.']
(MEM/'dataset_index.md').write_text('\n'.join(lines),encoding='utf-8')
(REPORTS/'leakage_and_source_audit.md').write_text('''# Source and leakage audit — 2026-11-05
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
''',encoding='utf-8')
(MEM/'architecture.md').write_text('''# Frozen baseline design (phase1-v1)
- Raw catalog, start 2011-01-01; discard last partial day; predeclared [20,32) N x [86,100) E, 2x2 cells split at 26 N / 93 E.
- 90 daily bins x 12 features: count, M4 count, M4.5 count, max/mean magnitude, mean depth/latitude/longitude, four cell counts. Empty bins zero; log1p counts; StandardScaler fit on unique training-history days.
- Seven marginal Bernoulli outputs: M>=4.5 any in 7 days, first 3 days, last 4 days, four cells in 7 days. Not a joint coherent distribution.
- Conv1d 12->16 k5/pad2; BN/ReLU; Conv1d 16->32 k3/pad1; ReLU/global-average; Dense 32->16/ReLU/dropout .25; Dense 16->7; sigmoid at inference.
- Unweighted BCE; AdamW lr .001/decay .001; batch32; max100 epochs; patience15; ReduceLROnPlateau patience5/factor.5; seed42; CUDA when available.
- Calendar 70/15/15; entire history+horizon inside partition; weekly nonoverlapping targets. Validation mean BCE chooses checkpoint.
- Baselines: training Poisson, smoothed trailing90-day Poisson (30-day training-rate prior), smoothed training prevalence. Brier/log loss/AP/precision/recall, calibration and paired 4-week block bootstrap.
- Version must change for architectural/protocol changes; identical completed configurations restore saved weights. last.pt contains full optimizer/scheduler/RNG state.
''',encoding='utf-8')
with (MEM/'decisions.md').open('a',encoding='utf-8') as f:
    f.write('\n- 2011 modern start is a coverage heuristic; old sparse years cannot support reliable negative physical-event labels. No claim of completeness.\n- Freeze M>=4.5 / 7 days / four broad cells; training-only threshold/horizon support table must pass before fitting.\n- Purge full 90-day histories across partitions to provide conservative disjoint inputs/targets; retain negative windows.\n- Probability scoring with unweighted BCE; do not balance or optimize thresholds on test.\n- Isolated project notebook runtime installed to avoid changing existing global training packages.\n')
text=(MEM/'task_graph.md').read_text()
for task in ['T1','T2','T3','T4']:
    lines=text.splitlines()
    text='\n'.join(line.replace('in_progress','completed').replace('pending','completed') if line.startswith('| '+task+' |') else line for line in lines)+'\n'
text=text.replace('| T5 | Dataset preparation | pending','| T5 | Dataset preparation | in_progress')
(MEM/'task_graph.md').write_text(text,encoding='utf-8')
with (BASE/'cnn wave status.md').open('a',encoding='utf-8') as f:
    f.write('\n## Audit milestone — 2026-11-05\nInspected 28 dataset files and 19 relevant Python sources. All eleven simulator stages are indexed; see `phase1_cnn_results/reports/leakage_and_source_audit.md`. Raw event-key duplicates: zero. Magnitude types are mixed. Created the 24-cell forecasting notebook with 20 labeled sections and independent label/split assertions. Frozen protocol is in `.agent-memory/architecture.md`. Training and end-to-end execution remain pending. Notebook kernel packages are installing in a project-local directory; the initial runner attempt occurred before installation finished and raised ModuleNotFoundError for nbclient.\n')
print('Audit findings and design persisted; training not yet marked complete.')
