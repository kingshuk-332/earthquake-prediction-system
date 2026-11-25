# EEW Audit & Feasibility Checkpoint — 2026-11-10
- Completed read-only EEW feasibility audit: created `cnn_eew_feasibility_report.md` and `cnn_eew_opus_handoff.md`.
- Overall EEW status: **FAIL**. The 99.84% accuracy reflects offline whole-trace binary detection on synthetic data, not real-time EEW.
- Verified: input is full 100-second trace (10,000 samples) with global average pooling. No P-onset picking, magnitude regression, S-arrival prediction, or lead time estimation is implemented.
- Proven acausal leakage: (1) simulator scales pre-P noise from future S-peak (`improved_waveform_generator.py:224-243`, counterfactual $\Delta=0.5076$), (2) full-trace detrending/standardization divides early P-waves by future S-wave energy, (3) P-arrival is pinned to 6.00 s.
- Minimum corrections documented: short post-P causal windowing ($1–3$ s), independent ambient noise calibration, running pre-event normalization, and real-seismogram benchmarking.
- STOP: no retraining, no Phase 11 modification, no fusion.

# Final waveform checkpoint — 2026-11-05
## Latest audit — 2026-11-06
Executed `phase1_cnn_overfitting_audit.ipynb`, six code cells passed; results in `phase1_waveform_cnn_results/overfitting_audit/`. Frozen epoch 22 train/val/test accuracy 0.997350759/0.997114592/0.998352554; gaps +0.023617/−0.100179 percentage points. No clear ordinary overfitting; cached leakage checks pass, exhaustive near-duplicates unresolved, synthetic shortcuts remain plausible. Three fixed noise families each 0/256 false positives. Original notebook/models/data preserved, no training or threshold tuning. Ready only for controlled synthetic BiLSTM comparison when authorized; registry unchanged. See status/report for details and resolved execution issues.

- Current: waveform Experiment A completed and verified; no later-phase model authorized yet.
- Last completed: W5 saved notebook, metric reconstruction, input/catalog hashes, resume state and documentation.
- Experiment wavecnn_ccb3b06b3d37; Phase11 train/val/test 11324/2426/2428; input(1,10000), output earthquake probability.
- 25 epochs, best 22, 4177 parameters; accuracy 0.998353, precision 0.999175, recall 0.997529, F1 0.998351.
- Best/resume: phase1_waveform_cnn_results/models/wavecnn_ccb3b06b3d37/best.pt and last.pt (optimizer/scheduler/RNG included).
- Files: phase1_waveform_cnn.ipynb, phase1_waveform_cnn_results/*, WAVEFORM_REPRODUCE.md, waveform audit/runner/finalizer, status and memory. Catalog notebook/results unchanged by SHA256 checks.
- Successful commands: audit_waveform_data.py; build_waveform_notebook.py; execute_waveform_notebook.py (approved fresh kernel); finalize_waveform.py.
- Errors: sandbox runtime PermissionError resolved by approved execution; no essential incomplete checks. Near-duplicate exhaustive correlation and real-data generalization were not attempted.
- Experiment B: investigated, not trained; see notebook feasibility/identifiability discussion.
- Next: strengthen real-data/generator/onset tests and freeze fair BiLSTM comparison when requested. Do not retrain completed configurations unintentionally.

## Read-only provenance verification — 2026-11-06
Confirmed notebook CNN inputs exclusively use Phase 11 train.h5, validation.h5 and test.h5 / waveforms_standardized. Fresh schemas, label counts/order and all eight audit source hashes match. Exact loader cells, full keys/shapes, per-trace detrending/standardization and metadata/baseline distinctions recorded in cnn wave status.md. No notebook execution, retraining or model/data changes. Existing completed experiment state remains unchanged.


## Dataset comparison checkpoint — 2026-11-06
Completed read-only CNN versus seq20 BiLSTM audit; see `dataset_comparison_report.md` for paths, sizes, hashes, statistics and evidence limits. CNN hashes unchanged; PT/NPZ all 12 arrays identical and finite. BiLSTM 5629/1191/1192 samples, 20x36 float32; y=(raw-1.5)/7.1 and binary=(raw>=4.5). No seq20 schema, event IDs or target timing found; legacy 15x44 documentation does not establish them. No row-based alignment. Fusion not ready; recover provenance/manifest and define shared causal target/splits. No training or data/model changes.
