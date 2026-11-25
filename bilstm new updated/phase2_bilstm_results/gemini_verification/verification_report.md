# Independent verification of Gemini Phase 2B and 2C

Audit date: 2026-11-10. Verification only: no training, tuning, model inference, data regeneration or model selection. Original model checkpoints were hashed as bytes, never deserialized. Test targets were accessed only to reproduce Phase 2C descriptive statistics.

PASS means independently reproduced; PARTIAL means supported with limitations; FAIL means contradicted; UNKNOWN means insufficient evidence.

## Verdict

**PASS:** Phase 2B core validation metrics; recorded matching architecture, initialization and sampling; Phase 2C JSON feature correlations and descriptive distribution shift; registered artifact integrity.
**FAIL:** Several narrative numbers and interpretations in both reports.
**UNKNOWN:** Exact historical one-variable control equivalence, absolute absence of historical test access, formal Mc values and whether Phase 2C executed any unrecorded training.
**PARTIAL:** Reconstructed Phase 2C method provenance, inferential significance and coverage-adjusted event-frequency interpretation.

## Phase 2B: independently recalculated validation results

Both saved prediction files' targets exactly match the corrected-v2 validation labels in manifest order (n=1065). The original registered protocol specifies Huber delta1=1.0; the new notebook specifies delta2=2.0. Identity target scaling makes both thresholds original ML units.

| Validation metric | delta1 | delta2 |
|---|---|---|
| MAE | 0.472400 | 0.469827 |
| RMSE | 0.624177 | 0.624329 |
| R2 | 0.016240 | 0.015759 |
| Prediction SD (population) | 0.157293 | 0.124784 |
| ML>=4.5 MAE; n=75 | 1.421449 | 1.465769 |
| ML>=4.5 underprediction | 75/75 | 75/75 |
| ML<3 MAE; n=195 | 0.721114 | 0.680939 |
| Clean train MAE (history) | 0.430884 | 0.442460 |
| Val-train MAE gap (history) | 0.041516 | 0.027366 |
| Best/total epochs | 12/20 | 11/19 |
| Recorded training seconds | 10.753820 | 11.078315 |

MAE improves by -0.002573 ML (0.5447%), while prediction SD falls 20.6680% and high-magnitude MAE worsens 0.044320 ML. This is not evidence of a useful generalization breakthrough. The smaller logged train/validation gap accompanies worse training MAE and greater compression; it does not establish reduced overfitting. Paired absolute error improves on 525 and worsens on 540 validation examples. Statistical reliability remains uncertain: one seed, overlapping windows, 75 high targets, and checkpoint selection on this same validation set. No new significance experiment was introduced.

**Report corrections:** delta2 Huber loss is **0.190228**, not 0.2312. Its prediction mean moves **farther** from the actual mean (absolute offset 0.025339 to 0.077041). The 19.829% SD ratio corresponds to **3.932% of target variance**, not 19.8%. Recorded runtime is 11.078315s, not 10.3s. Objective losses with different deltas are not directly comparable quality scores. The RMSE plotting expression is fragile but uses the correct RMSE fallback for the saved baseline history, which has no val_pred_std field.

### Controls and provenance limits

The architecture AST is identical; recorded initial weight hashes match; all 19 common epochs have identical sampled-parent/order hashes and counts. The approved augmentation consumer source hash matches the original protocol. The new code reads the registered full-precision settings: hidden128, layers2, dropout 0.3144753602375439, learning rate 0.0004549047669323766, weight decay 0.00012155233771053111, batch128; AdamW over all parameters, seed42, max50, patience8 and clipping1.0. There is no warm-start load in the supplied new training cell.

The saved best-checkpoint hashes match their fit summaries. Recorded minimum-validation-MAE epochs12/11 agree with saved predictions; source saves on validation improvement and reloads best before generating those predictions. Weight-to-prediction identity was not rerun.

The delta2 source indexes training/validation tensors only and selects using validation MAE. **Exact historical control equivalence and an absolute no-test-access guarantee remain UNKNOWN:** no immutable delta2 configuration/environment snapshot or independent access/execution log establishes every runtime fact. File hashing/memory mapping of the monolithic dataset is not evidence that test labels influenced optimization. The separate huber_experiment folder preserves Phase 2 outputs, but rerunning its notebook would overwrite its own artifacts; no safe resume logic is supplied.

## Phase 2C: feature verification

All **340 numeric JSON fields** reproduce, using inverse-scaled **final historical step19**, not whole-sequence averages. Feature names/indexes match the corrected schema; labels match source IDs; targets are unique across splits; timestamps are sorted; every input precedes its target.

| Feature | Pearson H1 | Pearson H2 | Pearson val | Spearman train | Spearman val |
|---|---|---|---|---|---|
| historical_magnitude | 0.712201 | 0.173127 | 0.185696 | 0.643449 | 0.175000 |
| historical_depth_km | 0.063069 | 0.031668 | -0.050866 | 0.263053 | 0.021816 |
| historical_latitude_deg | 0.124701 | -0.016051 | 0.013412 | 0.057242 | 0.048488 |
| historical_longitude_deg | 0.219664 | 0.005774 | -0.022042 | 0.180052 | -0.040317 |
| inter_event_days | 0.298504 | 0.140053 | -0.011236 | 0.396137 | -0.051143 |
| event_rate_30d | -0.267076 | -0.281125 | 0.073794 | -0.599379 | 0.051194 |
| mean_magnitude_20 | 0.826588 | 0.290506 | 0.200656 | 0.744353 | 0.217067 |
| std_magnitude_20 | -0.256106 | -0.071988 | 0.139617 | -0.462834 | 0.091902 |
| mean_depth_20_km | 0.107170 | 0.141406 | -0.046081 | 0.340419 | -0.070057 |
| std_depth_20_km | 0.031884 | 0.081264 | -0.038605 | -0.030138 | -0.056774 |

Matching early/late reconstruction: H1 rows[0,2486), n=2486, targets 1924-08-13T23:57:42+00:00 through 2013-01-29T20:25:30+00:00; H2 rows[2486,4973), n=2487, targets 2013-01-31T13:11:02+00:00 through 2024-12-20T09:43:29+00:00. These reproduce all H1/H2 correlations. **The named phase2c_feature_temporal_audit.ipynb is absent in the scoped project search.** The boundary definition is independently reconstructed, not recovered original source.

The report's feature table has **19/40 incorrect mean/SD entries** beyond displayed rounding tolerance; corrected values are in feature_verification.csv. Examples: validation mean_magnitude_20=3.511981+/-0.232645, historical_depth_km=23.707982+/-33.102461, and training inter_event_days SD=22.550500. Depth redundancy is pair8/9, r=0.657882, not pair1/8. Reported Cohen d uses overall-split SD, not pooled within-group SD. The claim that no validation |r| exceeds0.20 is contradicted by rolling-mean r=0.200656. Individual correlations do not establish absence of nonlinear/joint/sequence predictive signal or joint held-out explained variance.

## Temporal statistics and scientific interpretation

| Split | n | Target mean | ML>=4.5 | Elapsed years | N/elapsed year | Mean 30d rate |
|---|---|---|---|---|---|---|
| train | 4973 | 4.033843 | 1553 (31.229%) | 100.353619 | 49.555 | 0.554300 |
| val | 1065 | 3.510610 | 75 (7.042%) | 1.128955 | 943.350 | 3.007293 |
| test | 1067 | 3.101218 | 38 (3.561%) | 0.663893 | 1607.187 | 4.496407 |

Rates in the table are retained-ML events/day; mean 30-day test/train ratio=8.1119. Final-input gaps and rolling rates were recomputed from explicit UTC nanoseconds with interval (time-30days,time]; target gaps independently match the manifest. Final-input gap medians are 1.556725/0.197836/0.131713 days.

Train-to-validation Wasserstein=0.524702, KS D=0.314511; train-to-test Wasserstein=0.932624. The nominal validation KS p=2.335e-77 is reproducible, but it is not a dependence-robust significance claim. Overlapping histories, time trends and tied magnitudes limit iid Pearson/Spearman/KS p-value interpretations. Saved p=0 values reflect numerical underflow.

Annual rates above use exact first/last target elapsed seconds divided by365.2425days, handling partial years; both N/span and (N-1)/span are saved. Gemini's rounded 1.13/0.67-year denominators introduce approximation error. These are retained ML target counts, **not complete regional seismicity rates**; catalog uptime, gaps, magnitude-selection/reporting policies and true observation exposure are unknown.

**Confirmed:** empirical temporal shifts, correlation weakening/sign changes, and compressed validation predictions.
**Hypotheses:** changing detection/reporting thresholds, spatial sampling or physical clustering contributing to those shifts. No validated Mc fit, uncertainty estimate, station-coverage history, learned-rule analysis or conditional-independence test was supplied. Historical Mc~4.0 and modern Mc~1.5 remain **UNKNOWN**; causal completeness and forced-collapse explanations are unsupported as established conclusions.

Mc needs a defined estimation/validation procedure, rather than an observed minimum. The Gutenberg-Richter relation describes a frequency-magnitude distribution and does not specify temporal independence. See [Mignan & Woessner (2012), sections4-7](https://www.corssa.org/export/sites/corssa/.galleries/articles-pdf/Mignan-Woessner-2012-CORSSA-Magnitude-of-completeness.pdf_2063069299.pdf). This methodological reference does not establish completeness or any cause for this catalog.

## Integrity and required corrections

All **29 registered original Phase 2/v2 hashes match**, including four original best checkpoints, corrected .pt/.npz, schema, manifest and scalers. All 52 inspected source hashes remain identical before/after this audit. Registered model preservation is verified; the missing Phase 2C execution source prevents certifying that no other training occurred.

Before another research experiment: correct the separately enumerated report numbers/labels, recover the original Phase 2C notebook and execution provenance, preserve a complete immutable Phase 2B run configuration, and downgrade completeness/independence/causation assertions to untested hypotheses. Preserve this verification and the original reports side by side; nothing was silently repaired.

Evidence: [verification_metrics.json](verification_metrics.json), [discrepancy_table.csv](discrepancy_table.csv), [feature_verification.csv](feature_verification.csv), and [executed verification notebook](../../phase2d_gemini_verification.ipynb). Gemini originals: ../huber_experiment/huber_experiment_report.md and ../error_diagnosis/phase2c_feature_informativeness_report.md. Verification stops here.
