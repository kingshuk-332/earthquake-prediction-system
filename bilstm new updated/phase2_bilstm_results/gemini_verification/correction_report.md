# Phase 2E correction report

Date: 2026-11-10. Reporting/provenance work only: no training, inference, optimization, dataset loading/regeneration or CNN work.

**Documentation: READY WITH LIMITATIONS for planning the next experiment.** Corrections are complete and limitations explicit. This does not certify complete historical reproducibility or authorize another experiment.

## Versioned corrections

- [Phase 2B revision 1](../huber_experiment/huber_experiment_report_revision_v1.md): corrected delta2 Huber loss to 0.1902281602; SD ratio to 19.829% versus variance ratio 3.932%; prediction mean moves farther from the validation mean; runtime 11.078315s versus baseline 10.753820s. Overfitting, significance and compression-mechanism claims are qualified.
- [Phase 2C revision 1](../error_diagnosis/phase2c_feature_informativeness_report_revision_v1.md): complete corrected feature table and all 19 original/replacement entries. Corrected redundancy pairing, effect-size label, correlation-threshold overstatement and annual-rate approximations.
- Confirmed distribution shifts are separated from completeness, station-network and learned-rule hypotheses. Historical/current Mc are UNKNOWN. Gutenberg-Richter statistics do not prove independent magnitudes; significance is qualified for temporal dependence.
- Original reports, numerical artifacts, figures, checkpoints and Phase 2D verification are preserved. Cite the versioned revisions for current interpretation.

## Additional provenance evidence

**PASS — saved controls:** restricted fake-tensor metadata inspection confirms matching full-precision hyperparameters, model configuration, v2 dataset hash, Variant C, identity target definition, seed, parameter count and initialization hash. Last-checkpoint AdamW parameter groups match. Best-checkpoint epochs 12/11 and validation MAE agree with history minima; delta2 last state records epoch 19 and eight stale epochs. No model was constructed or run.

**UNKNOWN — full execution guarantees:** complete delta2 runtime/environment and independent test-access logs remain unavailable; its checkpoint omits the augmentation consumer-source hash. Supplied code supports validation-only selection, but exact full historical execution equivalence and absolute test non-access cannot be certified.

**NOT FOUND within targeted scope — original Phase 2C notebook:** no matching notebook or checkpoint copy was recovered. The content-search candidate is Phase 2A. Preserve verified statistics and independent Phase 2D reproduction separately; original execution history remains missing.

## Before the next experiment

Use revised reports and explicitly retain unresolved provenance limits. A future run needs a complete configuration/environment snapshot, source hashes, split/selection evidence and resumable execution log. Any causal-completeness assertion requires a separately justified analysis; none was performed here.

Evidence: [provenance/preservation record](correction_provenance_v1.json), [independent verification](verification_report.md), [executed Phase 2E notebook](../../phase2e_report_corrections.ipynb).
