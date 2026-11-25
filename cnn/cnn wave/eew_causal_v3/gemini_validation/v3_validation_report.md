# Gemini Flash — Causal EEW v3 Dataset Validation Report

**Review Date:** 2026-11-10 22:53:18  
**Auditor:** Gemini Flash (Validation Completion Agent)  
**Target:** `cnn/cnn wave/eew_causal_v3/`  
**Verdict:** **READY FOR INDEPENDENT VERIFICATION**

---

## 1. Current State & Deliverable Completeness
All core v3 deliverables exist and are complete:
- `01_causal_simulator_v3.ipynb`: Causal simulator definitions, analytic gamma peak scaling, impulse-response fixed filter gains, and pilot gate (PASS).
- `02_generate_dataset_v3.ipynb`: Full dataset generation loop with complete generation checkpoints (`train_generation_complete.json`, `validation_generation_complete.json`, `test_generation_complete.json`).
- `03_dataset_validation_v3.ipynb`: Implementation agent's internal validation notebook (executed).
- Datasets: `dataset/eew_dataset_train.h5` (531.5 MB), `dataset/eew_dataset_validation.h5` (114.0 MB), `dataset/eew_dataset_test.h5` (114.1 MB).
- Hashes and Manifests: All 44 baseline files from `protected_sources_before.json` verified byte-for-byte identical.

---

## 2. Exhaustive Saved-Data Checks (16,178 Records)

| Metric / Check | Required Specification | Observed Value | Verdict |
|---|---|---|:---:|
| **Train Records** | 11,324 (5,662 EQ + 5,662 Noise) | 11,324 (5,662 EQ + 5,662 Noise) | **PASS** |
| **Validation Records** | 2,426 (1,213 EQ + 1,213 Noise) | 2,426 (1,213 EQ + 1,213 Noise) | **PASS** |
| **Test Records** | 2,428 (1,214 EQ + 1,214 Noise) | 2,428 (1,214 EQ + 1,214 Noise) | **PASS** |
| **Array Finiteness** | Zero NaNs, Infs, or subnormals | 100% Finite across all arrays | **PASS** |
| **Parent Waveform Shape** | `(N, 10000)` float32 | `(N, 10000)` float32 | **PASS** |
| **Window Shapes (1s, 2s, 3s)** | `(N, 200)`, `(N, 300)`, `(N, 400)` | `(N, 200)`, `(N, 300)`, `(N, 400)` | **PASS** |
| **Pure Post-P Shapes** | `(N, 100)`, `(N, 200)`, `(N, 300)` | `(N, 100)`, `(N, 200)`, `(N, 300)` | **PASS** |
| **Parent/Window Equality** | $w[i] == parent[i, ref-100 : ref+	au]$ | Exact slice equality across 16,178 | **PASS** |
| **Normalization Match** | Recomputed pre-context norm | Max difference = $0.00000000$ | **PASS** |
| **Parent Waveform Uniqueness**| Zero cross-split duplicates | 16,178 unique SHA256 hashes | **PASS** |
| **Split Disjointness** | 0 source/pair ID overlap | 0 overlaps across train/val/test | **PASS** |
| **Model Input Purity** | No metadata in waveform arrays | Waveforms are pure 1D time-series | **PASS** |

---

## 3. Failure Mode Regression Audit (v2 Defects Repaired)
In v2, full-record noise variance scaling and full-record gamma envelope normalization leaked future energy into early windows (reproduced v2 error: **361.79** for future noise, **99.57** for future P-coda).
In v3:
- Fixed impulse-response filter gains and analytic gamma peaks ($2	au_d$) decouple early signal from future coda and noise.
- Live and saved pilot counterfactuals ($4	imes$ S-amplitude, $+50\%$ S-coda, $+50\%$ P-coda, future noise innovations, record length variations): **Maximum early-window difference = 0.00000000 (STRICT ZERO)**.

---

## 4. Boundary Conditions & S-Wave Arrival Semantics
- First-grid sample semantics: $S\_sample = \lceil t_S \cdot f_s ceil$ (first discrete sample at or after physical S arrival).
- Boundary case `NCS_EV_03038` ($S-P = 2.0018	ext{ s}$): $\Delta = 201$ samples. $2	ext{s}$ window ends at sample 200 ($200 < 201$, `pre_s_valid_2s = True`). $3	ext{s}$ window ends at sample 300 ($300 > 201$, `pre_s_valid_3s = False`).
- Boundary case `NCS_EV_03572` ($S-P = 3.0030	ext{ s}$): $\Delta = 301$ samples. $3	ext{s}$ window ends at sample 300 ($300 < 301$, `pre_s_valid_3s = True`).
- **Exact Catalog Eligibility Counts (8,089 Earthquakes):**
  - **1s Horizon:** 8,089 / 8,089 (**100.0000%**)
  - **2s Horizon:** 8,056 / 8,089 (**99.5920%**)
  - **3s Horizon:** 7,996 / 8,089 (**98.8503%**)
- Every window marked `pre_s_valid = True` ends strictly before the first S-wave sample.

---

## 5. Dataset Contracts & Training Guidance
1. **Magnitude Regression Masking:** Noise magnitude is set to $-1.0$. Models trained for magnitude regression MUST apply `magnitude_mask = (labels == 1)` or per-horizon masks (`magnitude_mask_1s`, `2s`, `3s`).
2. **Classification Masking:** `classification_mask_1s`, `2s`, `3s` are `True` for all valid noise records and for eligible earthquakes.
3. **Oracle Alignment Scope:** Waveforms are aligned around theoretical $t_P$ (at index 100) for rapid post-trigger analysis. This benchmark evaluates early-window classification and magnitude prediction, NOT autonomous continuous streaming trigger picking.

---

## 6. Final Verdict
All exhaustive data integrity, causality regression, split disjointness, and baseline protection checks have passed.

**STATUS:** **READY FOR INDEPENDENT VERIFICATION**
