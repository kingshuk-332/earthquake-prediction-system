"""Independent saved-artifact checks and final human/agent documentation."""
from pathlib import Path
import hashlib
import json
import re
import importlib.metadata
import numpy as np
import pandas as pd
import nbformat
from sklearn.metrics import brier_score_loss, log_loss

BASE=Path(__file__).resolve().parent
MEM=BASE/'.agent-memory'
registry=json.loads((MEM/'experiment_registry.json').read_text())
entry=registry['experiments'][-1]
eid=entry['experiment_id']
RES=BASE/'phase1_cnn_results'
D={k:RES/k/eid for k in ['models','metrics','plots','reports']}
nb_path=BASE/'phase1_cnn_forecasting.ipynb'
nb=nbformat.read(nb_path,as_version=4)
nbformat.validate(nb)
code_cells=[c for c in nb.cells if c.cell_type=='code']
assert len(code_cells)==12
assert [c.execution_count for c in code_cells]==list(range(1,13))
assert all(not any(o.output_type=='error' for o in c.outputs) for c in code_cells)
headers='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
assert all(re.search(r'## '+str(i)+r'\.',headers) for i in range(1,21))
source_sha=hashlib.sha256((BASE.parent.parent/'dataset_2.csv').read_bytes()).hexdigest()
assert source_sha==entry['dataset_version']
for item in json.loads((RES/'reports'/'source_inventory.json').read_text()):
    assert hashlib.sha256((BASE.parent.parent/item['path']).read_bytes()).hexdigest()==item['sha256']
for item in json.loads((RES/'reports'/'dataset_inventory.json').read_text()):
    assert (BASE.parent.parent/item['path']).stat().st_size==item['bytes']
scores=pd.read_csv(D['metrics']/'scores.csv')
summary=pd.read_csv(D['metrics']/'results_summary.csv')
split=json.loads((D['reports']/'split_summary.json').read_text())
verification=json.loads((D['reports']/'verification.json').read_text())
hist=pd.read_csv(D['metrics']/'training_history.csv')
for model in summary.model:
    frame=pd.read_csv(D['metrics']/f'test_{model}_predictions.csv')
    for target in frame.filter(regex='^p_').columns:
        label=target[2:]; y=frame['y_'+label]; p=frame[target]
        row=scores[(scores.split=='test')&(scores.model==model)&(scores.target==label)].iloc[0]
        assert np.isclose(brier_score_loss(y,p),row.brier,atol=1e-7)
        assert np.isclose(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),row.log_loss,atol=1e-7)
cnn=scores[(scores.split=='test')&(scores.model=='CNN')&(scores.target=='M45_any_7d')].iloc[0]
assert len(hist)==21 and verification['best_epoch']==6
assert all((D['models']/p).is_file() for p in ['best.pt','last.pt','preprocessing.json'])
assert len(list(D['plots'].glob('*.png')))==6

# Add run-specific interpretation as Markdown only; preserve all executed code/outputs.
marker='### Observed result from the verified run'
if marker not in headers:
    nb.cells[-2].source += '''

### Observed result from the verified run
**2026-11-05:** 5,603 complete-day records from 2011 onward produced 562 training, 110 validation and 110 test samples. Training-only support for M≥4.5 is 212 positive and 350 negative seven-day windows (296 qualifying events); M≥5 has only 75 positive windows, while M≥4 has 392. Fourteen-day M4.5 windows give 171/281 positives and 30-day windows 114/131. Seven days preserves temporal resolution, useful positive/negative balance and more samples. Spatial training positives are SW=15, SE=134, NW=51, NE=40: keep the grid coarse and treat sparse-cell results as exploratory.

The 3,223-parameter model trained for 21 epochs and restored epoch 6. Primary test Brier=0.281609, log loss=0.758084, average precision=0.567793. The recent Poisson baseline is better on all three (0.242390, 0.680387, 0.649873). The CNN forecasts are nearly constant (0.3725–0.3768), whereas observed test occurrence is 62/110=0.5636; this indicates poor adaptation/calibration and weak learned discrimination. At threshold 0.5 the CNN predicts no positives, yielding precision=recall=0 (zero-division convention). The threshold was not tuned after seeing these results.

Validation prevalence was 29/110=0.2636 versus training 212/562=0.3772 and test 0.5636. These changes explain why low validation loss need not imply good test calibration. Training mean BCE changes only slightly while validation worsens after epoch 6; there is mild validation deterioration and a largely climatological solution, not evidence that more epochs would fix the problem. The CNN-minus-recent-rate Brier difference is +0.039219; its four-week block-bootstrap 95% interval [-0.011912, 0.089982] is wide and crosses zero. No robust model-superiority claim is supported. The baseline itself is not validated for operational forecasting.
'''
nbformat.write(nb,nb_path)
nbformat.validate(nb)
source_code_sha=hashlib.sha256('\n'.join(c.source for c in code_cells).encode()).hexdigest()
verification.update(saved_notebook_valid=True,executed_code_cells=12,required_sections=20,
                    independently_recomputed_test_scores=True,original_source_hashes_unchanged=True,
                    inventoried_dataset_file_sizes_unchanged=True,notebook_code_sha256=source_code_sha)
(D['reports']/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
environment=json.loads((D['reports']/'environment.json').read_text())
package_names=['torch','numpy','pandas','scikit-learn','matplotlib','h5py','nbformat']
versions={p:importlib.metadata.version(p) for p in package_names}
versions.update({p:environment[p] for p in ['nbclient','ipykernel']})
(BASE/'requirements-phase1.txt').write_text('# Snapshot of verified direct dependencies; Python '+environment['python']+'\n'+'\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf-8')
prefix=f'phase1_cnn_results'
artifact_root=f'{prefix}/reports/{eid}'
status=f'''# CNN forecasting research status

## Project Overview
Investigate probabilistic future recorded seismicity in Northeast India and adjacent catalog regions. This is retrospective research, not deterministic prediction or an operational warning system.

## Model Development Roadmap
- [x] Phase 1: CNN baseline — trained, evaluated and verified; no superiority established.
- [ ] Phase 2: BiLSTM.
- [ ] Phase 3: CNN + BiLSTM.
- [ ] Phase 4: CNN + BiLSTM + Attention.
- [ ] Final model comparison.

## Current Phase
Phase 1 completed on 2026-11-05. Experiment `{eid}`. Phases 2–4 were not implemented.

## Dataset Investigation
Raw `../../dataset_2.csv`: 8,089 x 28, UTC 1906-08-31 14:57:30 through 2026-10-06 09:03:32; sorted, no missing values, duplicate rows, source IDs or event-key duplicates. Coordinates 20–30.5 N / 87–98.5 E; depth 1–430 km; magnitude 1.5–8.6. Magnitude types: ML 7125, MB 458, MW 398, MC 107, MS 1. No verified publication-time or coverage metadata.
Selected 5,603 records from 2011-01-01 through 2026-10-06 exclusive, removing the partial last day. Start year is exploratory, not a demonstrated completeness threshold. Source SHA256 `{source_sha}`.
Inspected 28 datasets and 19 relevant sources. Full paths, shapes, columns, missing values and purposes: [.agent-memory/dataset_index.md](.agent-memory/dataset_index.md). HDF5 phase 6/9: 8089x10000; phase 10: 16178x10000; phase 11 train/validation/test: 11324/2426/2428x10000 with earthquake/noise labels. These are excluded from forecasting.

## Research References
[Perol, Gharbi & Denolle (2018), Science Advances](https://doi.org/10.1126/sciadv.1700578), [author site](https://groups.csail.mit.edu/graphics/convnetquake/), [author preprint](https://arxiv.org/pdf/1702.02073). Adopt local convolutional feature extraction and regularization. Do not transfer waveform detection labels, detection performance claims, balanced sampling or full-waveform normalization into forecasting.

## Decisions and Justifications
Historical daily bins retain negative windows and forecast time exposure. Use 90 days of 12 features: counts (all/M4/M4.5), max/mean magnitude, mean depth/latitude/longitude, four cell counts. Fixed empty-bin zeros and log1p counts; standardize unique training-history days only.
Predict seven marginal probabilities for M>=4.5: whole next 7 days, days [0,3), days [3,7), and four cells within 7 days. Bounds [20,32) N x [86,100) E, split at 26 N/93 E. Not a mutually exclusive location distribution. Training support: 212/562 positive windows versus 75/562 for M5; coarser horizons lose temporal resolution. SW has only 15 training positives; spatial estimates remain exploratory.
Calendar 70/15/15 with full history+horizon contained in each partition and weekly issue spacing. Model selection uses validation mean BCE only. Test does not fit model, scaler or threshold.

| Split | Calendar partition (end exclusive) | Forecast issues | Samples | Primary positives | Events in target windows / M4.5 events |
|---|---|---|---:|---:|---:|
| Train | 2011-01-01 to 2022-01-12 | 2011-04-01 to 2021-12-31 | 562 | 212 | 2373 / 296 |
| Validation | 2022-01-12 to 2024-05-25 | 2022-04-15 to 2024-05-17 | 110 | 29 | 761 / 33 |
| Test | 2024-05-25 to 2026-10-06 | 2024-08-23 to 2026-09-25 | 110 | 62 | 2257 / 114 |

## Tasks Completed
- 2026-11-05 initial inspection: raw catalog audit, GPU check and memory system.
- 2026-11-05 audit milestone: all eleven simulator stages, prior model/preprocessing, 28 dataset schemas and 19 source hashes indexed. Existing full-catalog normalization and same-event waveform risks documented.
- 2026-11-05 implementation: 24-cell notebook with all 20 requested sections and exact-label/date/scaler assertions.
- 2026-11-05 training: 21 epochs on RTX 3050, best epoch 6 restored; complete resumable state saved each epoch.
- 2026-11-05 verification: fresh Jupyter kernel executed all 12 code cells without errors; model reload reproduces test predictions; saved test scores independently recomputed; all artifact and source-preservation checks passed. Six figures saved and calibration plot visually inspected.

## Tasks Remaining
No essential Phase 1 implementation/verification task remains. Research improvements are listed below and later-phase models remain unstarted.
- [ ] Verify catalog provenance/completeness and event reporting delays.
- [ ] Harmonize magnitude types and validate outage handling.
- [ ] Pre-register rolling-origin/multiple-seed comparisons and a fresh prospective holdout.
- [ ] Assess a stronger triggering-process baseline and coherent output formulation.

## Training Configuration
PyTorch {environment['torch']}, Python {environment['python']}, {environment['gpu']}; complete versions in `{artifact_root}/environment.json` and `requirements-phase1.txt`.
Seed 42; deterministic algorithms enabled. Conv1d 12->16 (k5,pad2), BN/ReLU, Conv1d 16->32 (k3,pad1), ReLU/global average, dense32->16/ReLU/dropout0.25, dense16->7. **3,223 trainable parameters**.
Unweighted BCEWithLogitsLoss; AdamW lr0.001/weight decay0.001; batch32; maximum100 epochs; early-stop patience15; LR-halving patience5. No SMOTE, test calibration or test threshold tuning. `last.pt` includes optimizer, scheduler and all RNG states; repeated completed configurations restore weights rather than retrain.

## Training Results
Actual epochs **21**, restored epoch **6**, best mean validation BCE {hist.validation_bce.min():.6f}.

| Model | Test Brier ↓ | Test log loss ↓ | Average precision ↑ |
|---|---:|---:|---:|
'''
for row in summary.itertuples(): status+=f'| {row.model} | {row.primary_brier:.6f} | {row.primary_log_loss:.6f} | {row.primary_AP:.6f} |\n'
status+=f'''
CNN precision/recall at 0.5 are both 0: all predicted probabilities fall below 0.5. Primary CNN predictions range 0.3725–0.3768, versus test frequency 0.5636, showing poor adaptation/calibration and weak discrimination. Validation prevalence is 0.2636; training 0.3772. The recent-rate baseline performs better in this split. CNN-minus-recent Brier difference +0.039219, four-week block-bootstrap 95% interval [-0.011912,0.089982]; small dependent samples prevent a strong superiority claim. CNN spatial mean Brier 0.131242 vs recent 0.125200; time-window mean Brier 0.234552 vs recent 0.225823.
Saved model: `{prefix}/models/{eid}/best.pt`; resume: `last.pt`; preprocessing: `preprocessing.json`.
Metrics/predictions/calibration/history: `{prefix}/metrics/{eid}/`; six figures: `{prefix}/plots/{eid}/`; report and verification: `{artifact_root}/`.

## Errors and Fixes
- PMC full-text endpoint returned a browser challenge; retrieved the author project and accessible author preprint, with published-paper metadata/search content for cross-checking.
- Jupyter execution packages were absent. Network sandbox blocked PyPI; approved installation to `.notebook-runtime` succeeded. A premature runner invocation failed before installation finished; rerun after completion.
- Sandbox could not read installed runtime files (PermissionError). Approved notebook execution outside that sandbox succeeded.
- pip reported a pre-existing torch/sympy version mismatch (torch wants 1.13.1; global sympy is 1.14.0). No global packages were changed; actual eager training and checks passed. Exact environment recorded; resolve in a separate environment before broader replication.
- Windows Jupyter emitted selector/TCP warnings; local kernel execution completed. No training/data/verification failure remains.
- The first external artifact check expected seven PNG files; the notebook intentionally produces six multi-panel figures covering all requested plots. Corrected the check to six and reran independent verification.

## Scientific Limitations
Unknown completeness, catalog revisions/reporting delays and mixed magnitude scales; modern start is heuristic. Quiet bins are absence of recorded events, not proven seismic inactivity. Geographic domain includes neighboring countries. Single seed and exploratory temporal holdout, sparse cells and correlated histories; no prospective validation. Marginal outputs have no enforced joint coherence. Poisson baselines omit triggering. Global pooling discards exact timing; near-constant forecasts suggest insufficient learned signal. Mild validation deterioration is controlled by early stopping, but this does not solve distribution shift. No deterministic/operational earthquake-prediction claim is justified.

## Next Steps
Improve data provenance/completeness, magnitude consistency and availability assumptions before BiLSTM. Freeze identical inputs, targets, split manifests and baselines for fair comparison; document test reuse and reserve a new holdout. No Phase 2 work has begun.

## Reproduction and session resume
Read `AGENTS.md`, then `.agent-memory/project_context.md`, `checkpoint.md`, `task_graph.md`. Run from repository root: `python "cnn/cnn wave/execute_phase1_notebook.py"`. The runner uses the existing project-local runtime if present; install notebook dependencies in a suitable environment on another machine. It executes a clean kernel and restores completed matching runs. Change the experiment version for protocol changes; do not overwrite prior runs. See `REPRODUCE.md`.
'''
(BASE/'cnn wave status.md').write_text(status,encoding='utf-8')
graph=(MEM/'task_graph.md').read_text()
graph=re.sub(r'\| (in_progress|pending|blocked) \|','| completed |',graph)
graph+='\nVerified 2026-11-05: T1–T9 complete for experiment '+eid+'. Completion means the experiment is executed and checked, not that forecasting skill is demonstrated. Future research tasks are in the status file.\n'
(MEM/'task_graph.md').write_text(graph,encoding='utf-8')
entry.update(experiment_status='completed_verified',notebook_path='phase1_cnn_forecasting.ipynb',
             notebook_code_sha256=source_code_sha,verification_path=f'{artifact_root}/verification.json',
             interpretation='No CNN advantage over simple baselines; recent Poisson has lower primary test Brier/log loss and higher AP.')
(MEM/'experiment_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
(MEM/'checkpoint.md').write_text(f'''# Final checkpoint — 2026-11-05
- Current: Phase 1 complete, verified; no active implementation task.
- Last completed: T9 results/documentation and independent saved-score verification.
- Experiment: {eid}; 562/110/110 samples; 3,223 parameters; 21 epochs, best epoch 6.
- Best: phase1_cnn_results/models/{eid}/best.pt; full resume state: last.pt alongside it.
- Primary test Brier: CNN 0.281609; recent Poisson 0.242390. No CNN advantage established.
- Files: phase1_cnn_forecasting.ipynb, cnn wave status.md, AGENTS.md, .agent-memory/*, REPRODUCE.md, requirements-phase1.txt, audit/runner/finalization utilities, phase1_cnn_results/*.
- Successful commands: audit_project.py; build_phase1_notebook.py; update_audit_memory.py; execute_phase1_notebook.py (fresh kernel, all 12 code cells); finalize_phase1.py.
- Verification: valid notebook, all labels/raw timestamps, purged partitions, train-only scaler, model reload, independent score reconstruction, source hashes unchanged, six plots.
- Resolved errors: missing notebook packages and sandbox read/network restrictions. Existing torch/sympy mismatch remains an environment caveat; training passed.
- Next: await Phase 2 scope; improve catalog completeness/provenance and freeze comparison protocol. Do not retrain the same configuration unintentionally or implement later models without authorization.
''',encoding='utf-8')
context=(MEM/'project_context.md').read_text()
if '## Final Phase 1 state' not in context:
    context+=f'\n## Final Phase 1 state\nVerified experiment `{eid}`: 562/110/110 samples, 3,223 parameters, 21 epochs (best 6). CNN Brier 0.281609 versus recent Poisson 0.242390; no model advantage. Main notebook and all artifacts exist. Read checkpoint/status for results; do not restart inspection. New model phases are not authorized yet.\n'
(MEM/'project_context.md').write_text(context,encoding='utf-8')
files=[p for folder in D.values() for p in folder.rglob('*') if p.is_file() and p.name!='artifact_checksums.json']
checksums={p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(D['reports']/'artifact_checksums.json').write_text(json.dumps(checksums,indent=2),encoding='utf-8')
print('PASS: executed notebook, 28 original dataset sizes, 19 source hashes, saved predictions/scores, all artifacts.')
print('Phase 1 status, task graph, registry and checkpoint finalized:',eid)
