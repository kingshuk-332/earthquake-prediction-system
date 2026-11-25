# BiLSTM earthquake early-warning research

**Blocked before training:** verified synthetic early noise depends on the future S-wave peak. The prepared candidate windows are quarantined and must not be used for model fitting. Read [the research report](reports/final_research_report.md) and [validation gate](reports/stage_a_validation.json).

1. `catalog_preprocessing.ipynb` — original CSV inspection and source snapshot only; the earlier catalog-sequence proposal was superseded by waveform EEW.
2. `phase1_eew_dataset_preparation.ipynb` — identity checks, causal counterexample, candidate crops and report. Uses this project's source snapshot and read-only original waveform sources; run from this directory or the project root.
3. `phase2_eew_bilstm_training.ipynb` — records the Stage A blocker; training is deferred.
4. `phase3_eew_bilstm_evaluation.ipynb` — records unavailable metrics; inference is deferred.

All three EEW notebooks have executed top-to-bottom for the current blocked state. Their saved outputs were produced by a sequential Python cell executor with a shared namespace per notebook; no notebook magics are used. Stage B/C are guards, not completed training or evaluation implementations. They deliberately require review if the source/gate changes.

`dataset/` contains source/window manifests, raw candidate crops, feature definitions and unfitted preprocessing status. `results/` contains blocked training/evaluation status. `models/` remains empty. Original sources and previous model artifacts are unchanged.

The catalog papers are background only; see [research scope and references](reports/research_notes.md). Synthetic oracle-P windows do not establish real-world EEW performance or guaranteed warning time.
