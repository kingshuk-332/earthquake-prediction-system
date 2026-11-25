# Final results summary

**Scope:** synthetic oracle-P-aligned early-observation benchmark, not operational EEW or next-earthquake prediction. CNN and BiLSTM use different representations of the same simulated signals.

## Preserved common-test point estimates

F1 is measured at threshold 0.5; MAE includes all 1,200 eligible earthquakes. Detection includes those earthquakes plus 1,214 noise records.

| Model | 1s F1 / MAE | 2s F1 / MAE | 3s F1 / MAE |
|---|---|---|---|
| CNN | 0.692712 / 0.665020 | 0.712246 / 0.635296 | 0.757641 / 0.613005 |
| BiLSTM | 0.727438 / 0.670213 | 0.646637 / 0.634122 | 0.694690 / 0.615082 |
| Fusion | 0.679908 / 0.664240 | 0.716312 / 0.635879 | 0.747475 / 0.608215 |
| Attention | 0.672045 / 0.666359 | 0.717199 / 0.636489 | 0.747428 / 0.606601 |

Numerical leaders: BiLSTM F1 and Fusion MAE at 1s; Attention F1 and BiLSTM MAE at 2s; CNN F1 and Attention MAE at 3s. These are metric-specific point estimates, not statistically established winners. The 1s BiLSTM noise FPR is 73.64%.

## Integrated findings

- Attention has no statistically supported improvement over non-Attention fusion: all saved paired F1/MAE intervals include zero. Fusion has no consistent advantage over standalone models.
- All 12 controlled 3s tuning trials failed the predeclared replacement rule. The original Attention model was retained; no 1s/2s tuning followed.
- Validation threshold 0.41 yields F1 0.768585, recall 0.760599 and FPR 0.216818. Lowering from 0.50 recovers 95 earthquakes but adds 104 false positives. Common-test outcomes at the frozen 0.41 threshold are F1 0.755423, recall 0.740000 and FPR 0.216639. These are different cohorts and must not be conflated.
- Original 3s Attention underpredicts 85.50% of validation M≥5 events and 87.02% on common test. Small-event overestimation cancels strong-event underestimation in aggregate signed error.
- CNN–BiLSTM magnitude residual correlation is approximately 0.983 on validation. Shared error is evidence; frozen-feature causation is unproven. Noise-family shortcut effects remain UNKNOWN.

**Conclusion:** retain the approved baseline and stop tuning. Prioritize verified independent real recordings, realistic triggering and stronger-event evaluation before additional optimization. Historical test reuse makes later comparisons exploratory. No operational warning lead time or deployment readiness was demonstrated.

See [full report](final_research_report.md) and [limitations/future work](limitations_and_future_work.md). Existing figures are linked from the report without duplication. This integration performed no inference or new experiments and changed no prior results.
