# Waveform detection experiment

Open `phase1_waveform_cnn.ipynb`, or run from repository root:

```powershell
python "cnn/cnn wave/execute_waveform_notebook.py"
```

The runner starts a clean Jupyter kernel and saves executed cells to the notebook. It uses the existing `.notebook-runtime` when present. This workspace's sandbox cannot read some installed runtime files, so execution required approval outside the sandbox. No new packages or global environment changes were needed for this experiment.

The notebook trains only on Phase 11 synthetic HDF5 waveforms: `data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/{train,validation,test}.h5`, key `waveforms_standardized`. Metadata never enters the classifier. The raw catalog forecasting dataset is not used for training. Experiment artifacts are isolated under `phase1_waveform_cnn_results/{models,metrics,plots,reports}/<experiment_id>/`.

`audit_waveform_data.py` performs the initial streaming audit and writes `reports/waveform_audit.json`, `verified_metadata.csv` and `signal_features.csv`. These compact artifacts support metadata reporting and signal baselines; the notebook checks source SHA256 hashes before using them. Do not rerun this expensive audit unless sources change or a check fails. Numerical raw/standardized arrays were scanned in full; normalization formulas and upstream waveform alignment were checked on 12 seeded examples per split. Exact duplicates were checked exhaustively, but near-duplicate correlations were not.

Training visits every sample once per epoch. At most 200 full traces are held in a loading buffer before 32-sample GPU minibatches. Block order and within-buffer order are reproducible from seed plus epoch. The full dataset is never copied to GPU memory. Paired examples remain within their provided partition.

`last.pt` stores model/optimizer/scheduler, all random states, history and early-stopping state. An interrupted notebook resumes from the next completed epoch; a completed matching configuration restores its best weights without retraining. `best.pt` contains the selected model weights/configuration. `preprocessing.json` records normalization, waveform layout and labels. `signal_baselines.pkl` contains fitted signal comparators. Load only trusted project checkpoints/pickles.

On another machine, use a compatible PyTorch environment plus NumPy, pandas, scipy, h5py, scikit-learn, matplotlib, nbformat, nbclient and ipykernel. Exact verified versions are saved in the experiment's `environment.json`. The existing environment has the previously documented torch/sympy mismatch; actual eager-mode GPU execution was tested, but cross-machine bitwise identity is not promised.

Change `CONFIG['version']` when changing model/protocol code, or choose a documented new seed for intentional replication. Never overwrite the preserved catalog experiment. `build_waveform_notebook.py` is a one-time authoring utility and refuses to overwrite the notebook; edit the notebook directly for subsequent changes.

Experiment B is a feasibility audit only. Same-event parameter estimation is distinct from detection and from forecasting. The notebook explains why absolute origin time and unique location are not identifiable from the selected input, and why magnitude/arrival regression would need dedicated simulator-inversion/picking baselines.
