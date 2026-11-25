# Phase 1 dependency graph
## Waveform correction: W1 -> W2 -> W3 -> W4 -> W5
| ID | Task | Status | Dependencies | Relevant files | Completion criteria | Outputs |
|---|---|---|---|---|---|---|
| W1 | Verify waveform splits/metadata | completed | cached T1/T2 audit | phase9/10/11 HDF5 and mappings | shapes, labels, pairs, source IDs, normalization, finite data and exact duplicates checked | waveform audit |
| W2 | Define detection and parameter feasibility | completed | W1 | waveform audit, generator | detection objective and optional estimation limits documented | notebook Markdown |
| W3 | Implement and train waveform CNN | completed | W2 | phase1_waveform_cnn.ipynb | real minibatch training, validation stopping, checkpoints | model, history |
| W4 | Evaluate and probe synthetic shortcuts | completed | W3 | frozen model, test HDF5 | metrics/baselines/confusion/ablation reports | metrics and plots |
| W5 | Verify and update memory | completed | W4 | executed notebook, status, registry | clean execution, artifacts, preserved catalog outputs | final status/checkpoint |
| W6 | Frozen CNN overfitting audit (2026-11-06) | completed | W5 | saved best checkpoint/history, Phase 11 splits, cached probes | all-split eval metrics/gaps, errors, fixed noise controls, executed notebook, four figures; no retraining | phase1_cnn_overfitting_audit.ipynb; phase1_waveform_cnn_results/overfitting_audit/ |

## Preserved catalog experiment
T1 -> T2 -> T3 -> T4 -> T5 -> T6 -> T7 -> T8 -> T9

| ID | Task | Status | Dependencies | Relevant files | Completion criteria | Outputs |
|---|---|---|---|---|---|---|
| T1 | Project inspection | completed | none | simulator/*.py, cnn_1d_model.py, preprocessing | inspect all 11 simulator stages and prior pipeline | source audit |
| T2 | Dataset audit | completed | T1 | raw/derived CSV, HDF5 | schemas, quality, coverage and EDA verified | inventory, plots |
| T3 | Leakage assessment | completed | T2 | source audit | input availability and split risks documented | leakage report |
| T4 | Forecast target design | completed | T3 | raw catalog | train-period support and exact labels defined | configuration |
| T5 | Dataset preparation | completed | T4 | notebook | chronological boundaries, shapes and preprocessing assertions pass | split manifest, scaler |
| T6 | CNN implementation | completed | T5 | notebook | readable architecture and resumable loop | valid notebook |
| T7 | CNN training | completed | T6 | notebook | actual epochs and best restoration | model, history |
| T8 | Model evaluation | completed | T7 | best model, test partition | unseen test scores, baselines and calibration | metrics, figures |
| T9 | Results and documentation | completed | T8 | status, memory, notebook | end-to-end execution and artifact verification | final report, registry |

Verified 2026-11-05: T1–T9 complete for experiment cnn_b86050c74334. Completion means the experiment is executed and checked, not that forecasting skill is demonstrated. Future research tasks are in the status file.
