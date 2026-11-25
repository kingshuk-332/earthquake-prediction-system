"""Independent artifact/metric verification and waveform research-memory update."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
import nbformat
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,precision_recall_curve,auc,confusion_matrix
BASE=Path(__file__).resolve().parent;MEM=BASE/'.agent-memory';RES=BASE/'phase1_waveform_cnn_results'
registry=json.loads((MEM/'experiment_registry.json').read_text())
entry=next(e for e in reversed(registry['experiments']) if e['model_type']=='1D waveform CNN detection')
eid=entry['experiment_id'];D={k:RES/k/eid for k in ['models','metrics','plots','reports']}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')
npath=BASE/'phase1_waveform_cnn.ipynb';n=nbformat.read(npath,as_version=4);nbformat.validate(n)
cc=[c for c in n.cells if c.cell_type=='code']
assert [c.execution_count for c in cc]==list(range(1,len(cc)+1))
assert all(not any(o.output_type=='error' for o in c.outputs) for c in cc)
audit=json.loads((RES/'reports'/'waveform_audit.json').read_text())
for p,h in audit['source_hashes'].items():assert sha(BASE/p)==h
preserved=json.loads((RES/'reports'/'preserved_catalog_hashes.json').read_text())
for p,h in preserved.items():assert sha(BASE/p)==h
pred=pd.read_csv(D['metrics']/'test_predictions.csv');scores=pd.read_csv(D['metrics']/'detection_metrics.csv')
test=scores[scores.split.eq('test')].set_index('model');cnn=test.loc['CNN'];y=pred.labels.to_numpy()
for name,row in test.iterrows():
    p=pred['score_'+name].to_numpy();yy=p>=row.threshold;pr,rc,_=precision_recall_curve(y,p)
    measured=dict(accuracy=accuracy_score(y,yy),precision=precision_score(y,yy,zero_division=0),recall=recall_score(y,yy,zero_division=0),
        f1=f1_score(y,yy,zero_division=0),roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),pr_auc_trapezoidal=auc(rc,pr))
    for k,v in measured.items():assert np.isclose(v,row[k],atol=1e-9)
    assert np.array_equal(confusion_matrix(y,yy,labels=[0,1]).ravel(),row[['tn','fp','fn','tp']].to_numpy())
last=torch.load(D['models']/'last.pt',map_location='cpu',weights_only=False)
assert {'optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng','model','history'}.issubset(last)
assert last['complete'] and last['optimizer']['state']
best=torch.load(D['models']/'best.pt',map_location='cpu',weights_only=False)
hist=pd.read_csv(D['metrics']/'training_history.csv')
assert len(hist)==last['epoch'] and np.isclose(hist.set_index('epoch').loc[best['epoch'],'validation_loss'],best['val_loss'])
v=json.loads((D['reports']/'verification.json').read_text())
assert v['saved_model_predictions_match'] and v['catalog_artifacts_unchanged']
code_hash=hashlib.sha256('\n'.join(c.source for c in cc).encode()).hexdigest()
v.update(executed_notebook_valid=True,executed_code_cells=len(cc),source_files_rehashed_unchanged=True,
    original_catalog_files_verified=len(preserved),external_test_metrics_recomputed=True,
    full_resume_state_verified=True,notebook_code_sha256=code_hash)
probes=pd.read_csv(D['metrics']/'shortcut_probes.csv').set_index('probe')
prefix=probes.loc['pre_onset_noise_only'];signal=test.loc['signal_logistic']
interpretation=f'''## Verified waveform findings — 2026-11-05
The CNN has {v['parameters']:,} trainable parameters. It completed {len(hist)} epochs and restored epoch {best['epoch']} (validation BCE {best['val_loss']:.6f}).
Held-out test: accuracy {cnn.accuracy:.6f}, precision {cnn.precision:.6f}, recall {cnn.recall:.6f}, F1 {cnn.f1:.6f}, ROC-AUC {cnn.roc_auc:.6f}, average precision {cnn.average_precision:.6f}, trapezoidal PR-AUC {cnn.pr_auc_trapezoidal:.6f}.
Confusion matrix (rows true noise/EQ, columns predicted noise/EQ): [[{int(cnn.tn)}, {int(cnn.fp)}], [{int(cnn.fn)}, {int(cnn.tp)}]].
The five-signal-feature logistic baseline has accuracy {signal.accuracy:.6f} and F1 {signal.f1:.6f}. This is an in-simulator comparison, not an external field benchmark.

### What the shortcut probes show
- Circular 25 s shift: accuracy {probes.loc['shift_25s','accuracy']:.6f}, AUC {probes.loc['shift_25s','roc_auc']:.6f}.
- Time reversal: accuracy {probes.loc['time_reversal','accuracy']:.6f}, AUC {probes.loc['time_reversal','roc_auc']:.6f}.
- Spectrum-preserving phase randomization: accuracy {probes.loc['phase_randomized','accuracy']:.6f}, AUC {probes.loc['phase_randomized','roc_auc']:.6f}.
- Pre-onset-noise-only: overall false-positive rate {prefix.all_noise_false_positive_rate:.6f}; noise taken from earthquake traces {prefix.original_EQ_noise_false_positive_rate:.6f}; noise taken from synthetic-noise traces {prefix.original_noise_false_positive_rate:.6f}. Its AUC against original source labels is {prefix.roc_auc:.6f}; those original labels are diagnostic group labels, not correct event labels after signal removal.

These probes do not establish exactly which physical features are learned. The CNN separates the supplied generated classes; disruption of phase/ordering tests sensitivity to waveform structure, while remaining discrimination can expose generator-specific spectral/background differences. Noise-only probes are derived after full-trace detrending, which itself can leave a class-associated linear-trend imprint; this is another preprocessing shortcut risk. Repetition introduces artificial boundaries. Treat all probes as diagnostics, not causal proof or realistic out-of-distribution performance estimates.

Experiment B is **not trained**. Earthquake magnitude/P/S labels are verified; origin times and coordinates require source-ID joins. Standardization removes absolute amplitude, and waveform shape is generated from known metadata formulas. Absolute origin time and unique epicentral coordinates cannot be recovered from this single clock-free vertical trace. Separate magnitude-inversion/picking studies need explicit baselines and independent data.
'''
# Paired bootstrap respects the designed earthquake/noise pairing.
pairs=pred.pair_id.unique();pairrows=np.stack([np.flatnonzero(pred.pair_id.to_numpy()==p) for p in pairs]);assert pairrows.shape==(1214,2)
rng=np.random.default_rng(42);diff=[]
cnn_ok=(pred.score_CNN.to_numpy()>=.5)==y;sig_ok=(pred.score_signal_logistic.to_numpy()>=.5)==y
for _ in range(2000):
    ix=pairrows[rng.integers(0,len(pairs),len(pairs))].ravel();diff.append(cnn_ok[ix].mean()-sig_ok[ix].mean())
ci=dict(comparison='CNN minus signal-feature logistic accuracy',mean=float(cnn.accuracy-signal.accuracy),
        ci95_low=float(np.quantile(diff,.025)),ci95_high=float(np.quantile(diff,.975)),resampling_unit='matched earthquake/noise pair',replicates=2000)
save(D['metrics']/'paired_accuracy_bootstrap.json',ci)
interpretation+=f"\nPaired-bootstrap accuracy difference (CNN minus signal logistic): {ci['mean']:.6f}, 95% interval [{ci['ci95_low']:.6f}, {ci['ci95_high']:.6f}]. This interval covers sampled pairs, not uncertainty across generator families or seeds.\n"
report=D['reports']/'final_report.md'
text=report.read_text(encoding='utf-8').split('## Verified waveform findings')[0]
report.write_text(text+'\n'+interpretation,encoding='utf-8')
if not any('## Verified waveform findings' in c.source for c in n.cells if c.cell_type=='markdown'):
    n.cells[-2].source+='\n\n'+interpretation
nbformat.write(n,npath);nbformat.validate(n)
save(D['reports']/'verification.json',v)
entry.update(experiment_status='completed_verified',notebook_path='phase1_waveform_cnn.ipynb',notebook_code_sha256=code_hash,
             verification_path=str((D['reports']/'verification.json').relative_to(BASE)),shortcut_probe_metrics=probes.reset_index().replace({np.nan:None}).to_dict(orient='records'))
save(MEM/'experiment_registry.json',registry)
status=(BASE/'cnn wave status.md').read_text(encoding='utf-8')
old_catalog='## Project Overview'+status.split('## Project Overview',1)[1]
wave=f'''# CNN research status
## Current Phase — corrected waveform baseline complete
The intended waveform Experiment A is complete and verified: `{eid}`. It is earthquake/noise detection, **not forecasting**. The earlier catalog forecasting study remains preserved below; its notebook and every fingerprinted artifact are unchanged. No BiLSTM or hybrid model has been started.

## Waveform dataset and audit
Files: `data wave/synthetic_waveform_simulator/output/phase11_preprocessed_dataset/train.h5`, `validation.h5`, `test.h5`; key `waveforms_standardized`. Shapes 11324/2426/2428 x 10000; 100 Hz /100 seconds; one vertical component. Each split is exactly 50/50 earthquake/noise, label1/0. All source and pair sets are disjoint; no exact raw/standardized trace duplicates or physical-event-key duplicates. All traces were checked for numerical integrity and normalized moments; global training normalization was independently recomputed. Formula/upstream alignment checks used 12 seeded traces per split. Exhaustive near-duplicate checks remain unperformed.
Magnitude, depth, distance, P/S indices, SNR, QC and IDs are stored in HDF5. Origin timestamps and coordinates are absent; verified joins to Phase 2 metadata supply them for reporting only. P onsets range 592–607 samples across splits, reflecting the approximately 6-second anchoring convention. No event metadata enters the CNN.

## Waveform training configuration
Input (10000,1), internally PyTorch (batch,1,10000). Conv8 k9/s2 +BN/ReLU/pool4; Conv16 k7/s2 +BN/ReLU/pool4; Conv32 k5/ReLU/global average; dense16/ReLU/dropout0.3; one sigmoid probability. {v['parameters']} parameters, seed42, deterministic CUDA on RTX 3050, AdamW lr0.001/weight decay0.0001, batch32, max25 epochs, validation-BCE early stopping patience5, LR-halving patience2. Two 100-trace class blocks form a small CPU buffer; minibatches alone go to GPU. Each training trace appears once per epoch. Threshold0.5 fixed before test.

{interpretation}

## Waveform outputs and verification
- Notebook: `phase1_waveform_cnn.ipynb` — {len(cc)} code cells executed sequentially in a fresh kernel without errors.
- Best model: `phase1_waveform_cnn_results/models/{eid}/best.pt`; full resumable state: `last.pt`; normalization: `preprocessing.json`.
- Metrics, predictions, subgroup results, confusion matrix, baselines and probes: `phase1_waveform_cnn_results/metrics/{eid}/`.
- Six figures: `phase1_waveform_cnn_results/plots/{eid}/`.
- Audit: `phase1_waveform_cnn_results/reports/waveform_audit.json`; final report/checks: `phase1_waveform_cnn_results/reports/{eid}/`.
- Reloaded model reproduces test predictions; independent saved-file metrics match; original source hashes unchanged; full optimizer/scheduler/RNG checkpoint verified. No essential waveform verification remains incomplete.

## Waveform errors and fixes
Initial sandbox execution could not read the previously installed notebook runtime (PermissionError). Approved execution outside the sandbox succeeded. No additional packages were installed or global environment changed. Existing torch/sympy environment mismatch remains documented from the catalog experiment; actual eager CUDA training passed. Jupyter emitted its existing Windows selector/local TCP warnings; execution completed.

## Waveform limitations and next steps
High synthetic accuracy does not establish physical feature learning or real-field generalization. Fixed onset, narrow synthetic harmonics, designed noise profiles, artificial class balance and full-trace normalization create shortcut opportunities. Pair-disjoint splitting prevents direct event/pair overlap but not shared generator bias. Optional parameter estimation was audited, not implemented. Before BiLSTM, retain this exact detection protocol for a fair comparison and add real recordings, independent generators, varied onsets and station/generalization tests. Keep forecasting separate and pre-event only.

## Dated waveform milestones — 2026-11-05
W1 streaming dataset audit and split verification completed; W2 detection/parameter-feasibility decision completed; W3 actual GPU CNN training completed; W4 held-out evaluation and shortcut probes completed; W5 fresh-kernel and external artifact verification completed. Memory, task graph and registry updated. Reproduce via `python "cnn/cnn wave/execute_waveform_notebook.py"`; see `WAVEFORM_REPRODUCE.md`.

---
## Preserved catalog forecasting experiment — original report
'''
(BASE/'cnn wave status.md').write_text(wave+old_catalog,encoding='utf-8')
graph=(MEM/'task_graph.md').read_text(encoding='utf-8')
graph='\n'.join(re.sub(r'\| (pending|in_progress) \|','| completed |',line) if re.match(r'\| W[1-5] \|',line) else line for line in graph.splitlines())+'\n'
(MEM/'task_graph.md').write_text(graph,encoding='utf-8')
(MEM/'checkpoint.md').write_text(f'''# Final waveform checkpoint — 2026-11-05
- Current: waveform Experiment A completed and verified; no later-phase model authorized yet.
- Last completed: W5 saved notebook, metric reconstruction, input/catalog hashes, resume state and documentation.
- Experiment {eid}; Phase11 train/val/test 11324/2426/2428; input(1,10000), output earthquake probability.
- {len(hist)} epochs, best {best['epoch']}, {v['parameters']} parameters; accuracy {cnn.accuracy:.6f}, precision {cnn.precision:.6f}, recall {cnn.recall:.6f}, F1 {cnn.f1:.6f}.
- Best/resume: phase1_waveform_cnn_results/models/{eid}/best.pt and last.pt (optimizer/scheduler/RNG included).
- Files: phase1_waveform_cnn.ipynb, phase1_waveform_cnn_results/*, WAVEFORM_REPRODUCE.md, waveform audit/runner/finalizer, status and memory. Catalog notebook/results unchanged by SHA256 checks.
- Successful commands: audit_waveform_data.py; build_waveform_notebook.py; execute_waveform_notebook.py (approved fresh kernel); finalize_waveform.py.
- Errors: sandbox runtime PermissionError resolved by approved execution; no essential incomplete checks. Near-duplicate exhaustive correlation and real-data generalization were not attempted.
- Experiment B: investigated, not trained; see notebook feasibility/identifiability discussion.
- Next: strengthen real-data/generator/onset tests and freeze fair BiLSTM comparison when requested. Do not retrain completed configurations unintentionally.
''',encoding='utf-8')
context=(MEM/'project_context.md').read_text(encoding='utf-8')
context=context.replace('## Active correction — waveform CNN','## Completed correction — waveform CNN')
context=context.replace('No model phases are authorized','No later model phases are authorized')
context+=f'\n## Verified waveform result\n`{eid}`: {len(hist)} epochs, best {best["epoch"]}, accuracy {cnn.accuracy:.6f}, F1 {cnn.f1:.6f}; waveform detection only. Notebook/outputs verified and catalog artifacts unchanged. W1–W5 complete. Parameter estimation investigated but not trained; no BiLSTM. Read the final checkpoint/status rather than repeating the audit.\n'
(MEM/'project_context.md').write_text(context,encoding='utf-8')
index=(MEM/'dataset_index.md').read_text(encoding='utf-8')
index+=f'''\n## Verified waveform detection use — {eid}
Phase11 train/validation/test HDF5: (11324,10000)/(2426,10000)/(2428,10000), float32 at100Hz. Detection labels1/0 balanced; not forecast labels. All raw/standardized traces finite/nonzero, exactly unique; pair/source sets disjoint; formulas checked on12/split and all standardized moments verified. Full keys/attributes and per-split historical date coverage: `phase1_waveform_cnn_results/reports/waveform_audit.json`.
HDF5 has magnitude/P/S/depth/distance/SNR but no origin_time_utc/Lat/Long. Earthquake source-ID join to Phase2 metadata complete; noise has no corresponding event metadata. `verified_metadata.csv` contains reporting-only joined fields; `signal_features.csv` holds seven derived signal descriptors for simple comparators. Neither supplies event metadata to the CNN.
The prior exclusion of waveforms applies to future forecasting, not the now-completed same-trace detection experiment. Synthetic shortcut and single-component identifiability limitations remain.
'''
(MEM/'dataset_index.md').write_text(index,encoding='utf-8')
with (MEM/'architecture.md').open('a',encoding='utf-8') as f:
    f.write(f'\n# Completed waveform detection architecture\nExperiment {eid}: (1,10000) -> Conv8 k9/s2/BN/ReLU/pool4 -> Conv16 k7/s2/BN/ReLU/pool4 -> Conv32 k5/ReLU/global average -> dense16/ReLU/dropout.3 -> dense1/sigmoid; {v["parameters"]} parameters. Full config/checkpoints under phase1_waveform_cnn_results; catalog design above preserved.\n')
with (MEM/'decisions.md').open('a',encoding='utf-8') as f:
    f.write(f'\n- Verified waveform detection experiment {eid} complete; no parameter-estimation model or BiLSTM implemented. Read shortcut_probes.csv and final report before interpreting accuracy as physical signal learning. Original catalog experiment preserved byte-for-byte.\n')
artifact_files=[p for folder in D.values() for p in folder.rglob('*') if p.is_file() and p.name!='artifact_checksums.json']
save(D['reports']/'artifact_checksums.json',{p.relative_to(BASE).as_posix():sha(p) for p in artifact_files})
print('PASS: waveform notebook, independent metrics, resume state, source/catalog preservation and documentation.')
print(test[['accuracy','precision','recall','f1','roc_auc','average_precision']].to_string())
print('Epochs',len(hist),'best',best['epoch'],'parameters',v['parameters'])
