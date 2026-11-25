# Phase 2 Causal EEW Dataset — Completion Report

**Project Directory:** `cnn/cnn wave/eew_causal_v2/`  
**Completion Date:** 2026-11-10 17:56:45  
**Status:** **APPROVED FOR INDEPENDENT VERIFICATION**

---

## 1. Original Simulator Components Reused
- **Wave Propagation Physics:** Elastic body wave velocities ($V_P = 6.0	ext{ km/s}, V_S = 3.5	ext{ km/s}$).
- **Crustal Attenuation:** Frequency-dependent quality factor $Q(f) = Q_0 (f/f_0)^\eta$ with $Q_{P0}=350, Q_{S0}=300, \eta=0.70$.
- **Source Scaling & Geometrical Spreading:** Magnitude scaling ($A \propto 10^{0.5(M-4)}$) and geometric decay $(R_0/R)^1$.
- **Wave Envelopes:** Smooth $C^1$ Gamma rise/decay envelopes with magnitude-dependent rupture duration.
- **Acquisition Parameters:** Sampling frequency $f_s = 100	ext{ Hz}$ ($dt = 0.01	ext{ s}$).
- **Metadata & Catalog:** 8,089 regional earthquake events joined to theoretical arrivals and original station coordinates.
- **Split Assignments:** Preserved exact pair-disjoint train (11,324), validation (2,426), and test (2,428) partitions.

---

## 2. Causality Corrections Implemented
1. **Decoupled Ambient Noise:** Ambient background noise is generated independently of earthquake energy. Future S-wave amplitude alterations produce exactly $0.00000000$ difference in pre-S and pre-P samples.
2. **Causal Preprocessing:** Waveforms are standardized strictly using trailing 1.0 s pre-P noise buffers ($[t_P - 1.0	ext{ s}, t_P]$). Zero future samples enter normalization statistics.
3. **Variable P-Wave Onsets:** P-wave arrival positions are distributed across $[3.0	ext{ s}, 7.0	ext{ s}]$ (std: 1.15 s), preventing fixed-sample shortcut learning.
4. **Multi-Horizon EEW Windows:** Extracted 1.0 s, 2.0 s, and 3.0 s post-P observation windows (shapes: 200, 300, 400 with pre-P context; 100, 200, 300 pure post-P).
5. **Strict Pre-S Cutoffs:** 100% of earthquake events satisfy $t_S > t_P + 3.0	ext{ s}$ (minimum $S-P = 3.89	ext{ s}$), guaranteeing zero S-wave contamination.
6. **Metadata Isolation:** Magnitude, distance, and arrivals are stored strictly as evaluation labels, not model inputs.

---

## 3. Pilot Validation Results
- **Test 1 (S-Wave Counterfactual):** PASS (max diff = $0.00000000$).
- **Test 2 (Coda Counterfactual):** PASS (max diff = $0.00000000$).
- **Test 3 (Causal Normalization):** PASS (max diff = $0.00000000$).
- **Test 4 (P-Onset Variability):** PASS ($t_P \in [2.63	ext{ s}, 7.60	ext{ s}]$).
- **Test 5 (Pre-S Cutoffs):** PASS (100% valid across 1s, 2s, 3s).
- **Test 6 (Split Integrity):** PASS (0 source overlap across train/val/test).

---

## 4. Final Dataset Shapes & Assets
- `eew_dataset_train.h5`: 11,324 records (5,662 EQ + 5,662 Noise) — 98.74 MB
- `eew_dataset_validation.h5`: 2,426 records (1,213 EQ + 1,213 Noise) — 21.19 MB
- `eew_dataset_test.h5`: 2,428 records (1,214 EQ + 1,214 Noise) — 21.21 MB
- `eew_windows_multitask.npz`: Complete multi-task benchmark archive — 103.02 MB
- Observation Window Arrays: `(N, 200)`, `(N, 300)`, `(N, 400)` float32.

---

## 5. Original File Hash Verification
- All 17 protected original source files, notebooks, datasets, models, and metadata files were verified.
- **Verification Result:** 17 / 17 hashes match initial SHA256 hashes byte-for-byte.
- **Zero original files were modified, moved, renamed, or deleted.**

---

## 6. Remaining Scientific Limitations
1. **Synthetic Nature:** Synthesized waveforms are physics-informed mathematical models; transfer to real continuous recorded broadband data (e.g. NCS regional stations) requires domain adaptation.
2. **Fixed Attenuation Geometry:** 1D velocity model assumption ($V_P/V_S = 1.71$) does not capture complex 3D crustal scattering and basin resonance.
