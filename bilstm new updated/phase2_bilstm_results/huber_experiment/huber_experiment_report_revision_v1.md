# Phase 2B Huber experiment — corrected report, revision 1

**Phase 2E revision, 2026-11-10.** Supersedes the narrative and labels in [the original report](huber_experiment_report.md), which remains unchanged. Statistics reuse [independent Phase 2D verification](../gemini_verification/verification_report.md); additional checkpoint-metadata evidence is recorded in Phase 2E. No new training, inference or model selection.

## Verified results and corrections

Targets use original ML units and the identity transform. Baseline delta=1, new delta=2; validation n=1065.

| Metric | delta1, Variant C | delta2, Variant C |
|---|---|---|
| Validation MAE | 0.472400 | 0.469827 |
| Validation RMSE | 0.624177 | 0.624329 |
| Validation R2 | 0.016240 | 0.015759 |
| Validation Huber (own delta) | 0.175357 | 0.190228 |
| Prediction mean | 3.485271 | 3.433569 |
| Prediction SD | 0.157293 | 0.124784 |
| ML>=4.5 MAE (75 events) | 1.421449 | 1.465769 |
| ML<3 MAE (195 events) | 0.721114 | 0.680939 |
| Clean train MAE, logged | 0.430884 | 0.442460 |
| Val-train MAE gap, logged | 0.041516 | 0.027366 |
| Training seconds, logged | 10.753820 | 11.078315 |
| ML>=4.5 underprediction | 75/75 (100%) | 75/75 (100%) |
| Best/total epochs | 12/20 | 11/19 |

MAE improves by -0.002573 ML (0.5447%), but prediction SD falls 20.6680% and high-magnitude MAE worsens 0.044320 ML. All75 high targets remain underpredicted; RMSE/R2 do not improve. This is a modest validation tradeoff, not established generalization improvement.

- Delta2 Huber loss is **0.1902281602**, replacing0.2312. Different delta thresholds define different loss objectives; their raw values are not equivalent quality scores.
- Prediction mean moves **farther from** actual mean 3.510610: absolute offsets 0.025339 -> 0.077041.
- Delta2 prediction SD is **19.829% of actual SD**, while variance is **3.932% of actual variance**. The original19.8% variance description was incorrect.
- Recorded delta2 runtime is **11.078315s**, replacing10.3s; baseline is 10.753820s. Saved timings are not an independent latency benchmark or proof of identical efficiency.
- The smaller train/validation gap accompanies worse training MAE and more compression; it does not establish reduced overfitting across different distributions.
- Compression increased in this fit, but its causal mechanism was not established. Gradients use weighted training observations, not the validation majority. Huber optimization does not necessarily produce constant predictions or the arithmetic mean.

Statistical reliability remains uncertain: one seed,75 high validation targets, overlapping histories and checkpoint selection on the same validation set. The retained RMSE figure uses the correct fallback for this baseline history; its conditional plotting code is fragile, not evidence of an incorrect saved curve.

## Strengthened control evidence and remaining limits

**PASS — serialized settings:** Both best checkpoints retain matching hyperparameters, model configuration, dataset hash/version, Variant C, target definition, seed42,538881 parameters and initialization hash. Saved losses are Huber_delta1 versus Huber_delta2.0. Full-precision settings: hidden128, layers2, dropout 0.3144753602375439, learning rate 0.0004549047669323766, weight decay 0.00012155233771053111, batch128. Last-checkpoint AdamW parameter groups match, including betas(0.9,0.999), epsilon1e-8 and optional flags.

**PASS — recorded selection:** Best-checkpoint epochs12/11 and validation MAE match history minima and saved predictions. Delta2 last checkpoint records epoch19, best epoch11, eight stale epochs. Phase 2D independently verified identical initialization and all19 common sampled-parent/order hashes. Supplied notebook code selects by validation MAE and indexes training/validation tensors only.

**UNKNOWN — full historical execution:** Checkpoint metadata adds evidence beyond Phase 2D's source inspection, but no complete delta2 runtime/environment snapshot or independent access log establishes every execution fact. Its checkpoint omits the consumer-source hash. Exact full runtime equivalence and absolute historical test non-access remain UNKNOWN; intended validation-only selection is supported.

Metadata was read with restricted fake tensors, without materializing checkpoint tensor storage or constructing models. Original reports, predictions, metrics, histories, figures and checkpoints are unchanged. The original notebook has no safe resume logic and would overwrite its own delta2 artifacts if rerun.

Evidence: [correction summary](../gemini_verification/correction_report.md), [metadata/search record](../gemini_verification/correction_provenance_v1.json).
