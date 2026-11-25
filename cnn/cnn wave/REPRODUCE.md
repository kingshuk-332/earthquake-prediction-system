# Reproduce or resume Phase 1

The experiment implementation is in `phase1_cnn_forecasting.ipynb`. The notebook contains all preprocessing, architecture, training and evaluation code. It can run sequentially from its directory or the repository root. Python 3.13 and PyTorch 2.6.0+cu124 were used; CUDA was available on an RTX 3050 Laptop GPU. Exact direct versions are in `requirements-phase1.txt` and the experiment's `environment.json`.

From repository root:

```powershell
python "cnn/cnn wave/execute_phase1_notebook.py"
```

The runner launches a fresh Jupyter kernel, saves outputs after each code cell and validates the result. It discovers the installed project-local `.notebook-runtime` automatically. On another machine, install the requirements in an isolated Python environment with a compatible PyTorch build and notebook kernel. The saved environment has a pre-existing torch/sympy dependency mismatch; it passed this eager-mode experiment, but should not be treated as a clean universal installation recipe. A CUDA wheel may require the corresponding PyTorch package source; CPU execution is supported when CUDA is unavailable.

No dataset downloads are needed: the raw source is `../../dataset_2.csv`. The notebook reads the existing audit inventory. `audit_project.py` refreshes source/data inspection only if inputs change; it does not generate or modify original waveforms.

Each experiment ID hashes raw CSV bytes and its declared configuration. Do not change code while retaining the same version: increment `CONFIG['version']` for a different architecture/protocol. The registry also records the executed notebook code hash. Keep existing results. For an intentional replication, use a distinct version/seed and document why; do not delete checkpoints.

`models/<experiment>/last.pt` saves model, optimizer, scheduler, epoch/history, stopping state and Python/NumPy/PyTorch/CUDA/DataLoader RNG states. The notebook resumes at the next epoch if interrupted, or restores an already completed model without retraining. `best.pt` is restored for evaluation. Load only trusted local checkpoints. Exact numerical replay is expected only within a compatible environment, not guaranteed across hardware or library versions.

`preprocessing.json` defines the ordered 12 features, log transforms, means/scales, empty-bin convention, history, targets and region. `forecast_dataset.npz` contains the compact prepared arrays and partition indices; `split_manifest.csv` records all issue/history/target dates. Saved test predictions permit independent score reconstruction.

Utilities `build_phase1_notebook.py` and `update_audit_memory.py` are one-time authoring/initial-audit helpers; do not rerun them to resume completed work. The builder refuses to overwrite an existing notebook. `finalize_phase1.py` checks the initial verified experiment and refreshes its documentation; it includes run-specific assertions and is not a general runner for future experiments. Read `AGENTS.md` and `.agent-memory/checkpoint.md` first.
