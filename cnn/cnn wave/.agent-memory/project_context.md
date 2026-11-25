# Context
## Completed correction — waveform CNN
User now authorizes `phase1_waveform_cnn.ipynb` and `phase1_waveform_cnn_results/`: train earthquake/noise detection from Phase 11 synthetic HDF5 waveforms after verifying splits, numerical contents and metadata. Investigate optional parameter estimation feasibility; do not claim forecasting. Preserve the catalog notebook and all its artifacts. No BiLSTM yet. Cached simulator audit remains relevant; inspect changed task-specific arrays only.

Develop and actually train a simple catalog-based 1D CNN forecasting baseline for Northeast India and adjacent catalog region. Only Phase 1; BiLSTM and hybrids remain future work.
Main deliverables: executable `phase1_cnn_forecasting.ipynb`, saved model/preprocessing/results under `phase1_cnn_results/`, and `cnn wave status.md`.
Raw catalog: `../../dataset_2.csv`, 8,089 x 28, UTC origins 1906-08-31 through 2026-10-06. No raw missing values or duplicate source IDs. CUDA PyTorch 2.6.0+cu124 and RTX 3050 available.
Synthetic target-event traces are detection inputs, not forecasting inputs. Existing full-catalog scaling must not be reused.
Client date: 2026-11-05. Preserve all prior outputs. Use compact memory and targeted source inspection.

## Final Phase 1 state
Verified experiment `cnn_b86050c74334`: 562/110/110 samples, 3,223 parameters, 21 epochs (best 6). CNN Brier 0.281609 versus recent Poisson 0.242390; no model advantage. Main notebook and all artifacts exist. Read checkpoint/status for results; do not restart inspection. New model phases are not authorized yet.

## Verified waveform result
`wavecnn_ccb3b06b3d37`: 25 epochs, best 22, accuracy 0.998353, F1 0.998351; waveform detection only. Notebook/outputs verified and catalog artifacts unchanged. W1–W5 complete. Parameter estimation investigated but not trained; no BiLSTM. Read the final checkpoint/status rather than repeating the audit.
