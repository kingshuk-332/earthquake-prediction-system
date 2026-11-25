# Final consistency check

**Integration status: COMPLETE. SHAP scientific verdict: CONDITIONAL.**

## Scope and preservation

The latest available academic report was `project_report/academic_final_v1/final_minor_project_report.md`. It supplied the existing model comparison, tuning findings and bibliography; fallback reconstruction was unnecessary. Only the five specified SHAP documents, the academic report and exact referenced figure paths were read for substantive integration. No datasets, checkpoints, attribution arrays or executable notebooks were opened or run. No training, inference, SHAP recomputation, tuning or cleanup occurred.

Created exactly the three requested Markdown deliverables in this new directory. All six source documents and every referenced raster figure retain their before/after SHA256 hashes. Chapters 1–7 are text-identical to the prior report, and all original table rows remain unchanged and ordered. The abstract, contents and conclusion now incorporate SHAP; the previous conclusion is Chapter 9. Inherited local manifest links were corrected for the new directory; destinations containing spaces use Markdown angle brackets.

## Scientific consistency checks

| Check | Result and resolution |
|---|---|
| Four-model comparison | Preserved: 3s CNN fixed-0.5 F1 0.757641 is highest; Attention numerical MAE 0.606601 is lowest. No universal winner is asserted. |
| Paired uncertainty | Preserved: all saved Attention-minus-fusion F1/MAE intervals include zero; SHAP supplies no evidence of improved accuracy or statistical superiority. |
| Tuning | All twelve trials and replacement criteria remain unchanged. Original hidden-32/dropout-0.2 Attention head retained; SHAP explains that model, not a tuned candidate. |
| Predictive versus attribution error | Test/validation magnitude errors and strong-event bias remain distinct from SHAP reconstruction residuals. A SHAP pass does not resolve magnitude compression or underprediction. |
| Cohorts | Common-test 2,414-record comparisons, original 12-case/9-earthquake SHAP cohorts and four-failure/two-seed diagnostic are explicitly distinguished. No test-selection claim is introduced. |
| Initial reconstruction | 8/12 detection passes, mean/max absolute residual 0.205840/0.622584 logits; 9/9 magnitude passes, 0.012144/0.024013 magnitude units. |
| Higher sampling | Diagnostic mean error 0.212092 to 0.119120 with 2,048 to 8,192 draws; 4/8 to 5/8 passing case/seed runs. This is not a before/after comparison of the original 12-case mean. |
| Diagnostic budget | 81,920 retained draws; one bookkeeping-repeat call yields 83,968 actually computed draws. No adaptive sampling configurations or report-stage computation. |
| All detection failures | NOISE_P7567, NCS_EV_03052, NCS_EV_07170 and NCS_EV_02345 appear with signed residuals and original tolerances. Noise sentinel is not represented as physical magnitude. |
| Local example | NCS_EV_03910 is correctly described as preselected and passing its original tolerance, with nonzero residual 0.201782. It is not incorrectly labelled a failed case or used to hide the four failures. |
| Feature evidence | Seven saved importance values retained to eight decimals; detection top three RMS / first-difference energy / zero-crossing rate; magnitude RMS / mean absolute amplitude / first-difference energy. No unsupported physical or precise temporal peak ranking added. |
| Background sensitivity | Detection reference shift 1.973908 logits retained alongside rank rho 0.892857; magnitude shift 0.008048 and rho 0.964286 retained. Stable ranks do not establish stable expected values. |
| Dependency | Both branches derive from the same synthetic observations. Paired endpoints do not guarantee physically consistent interpolated points; no causal or independent branch-contribution claims. |
| Attention | Attention pooling weights are explicitly distinct from SHAP values; neither visual resemblance nor numerical reconstruction establishes physical validity. |
| Figure use | All 12 originals used with target-specific warnings. They remain 2,048-draw figures; no diagnostic values are misattributed to unchanged plots. |
| Verdict | CONDITIONAL appears in abstract, chapter and conclusion; no unqualified approval or prediction-accuracy gain is claimed. |
| References | New local evidence citations X1–X5 map to supplied SHAP reports. Prior external bibliography/access dates retained, not newly verified. |

## Document and link verification

- Original table rows preserved: 116.
- Existing raster image references resolved: 17; SHAP images: 12, all nominal 300 DPI.
- Other local link destinations verified: 15 (plus the two new companion deliverables).
- Chapter numbering and contents updated; SHAP figures 8.1–8.12 and tables 8.1–8.5 are distinct from the prior figures/tables.
- No claim of operational warning readiness, noise-family causality, independent sensor contributions or SHAP-induced model improvement was added.

## Source document hashes

| Source | SHA256 |
|---|---|
| project_report/academic_final_v1/final_minor_project_report.md | `cb2a3f1724f1c2c8288f8ffadf7dca3753264c7f6286099125e0473e22d57ca2` |
| explainability/final_attention_shap_v1/shap_explainability_report.md | `63f9c4f891c7950a1fa0092c703f2640d4e38e6691b61a6ab6e83a3697f76b7f` |
| explainability/final_attention_shap_v1/shap_validation_report.md | `46d31765345084795dcf04a05c71e27e117420ef56ec189c787ea2f0f69da62d` |
| explainability/final_attention_shap_v1/academic_shap_section.md | `ba75bf48d4797edbba742fe254e87b40e8f8930dfab7f634e6a8f5513d5fa4a1` |
| explainability/shap_reliability_v1/reliability_report.md | `6da438ebdd0d31e09fd8df6157ccf823cc9406f6e9ce7dae91b09c93ea3b7d58` |
| explainability/shap_reliability_v1/report_integration_guidance.md | `76cee029969512f473168ac3dc30c11c0e2ce57ab6d7d09943776f0d0092f541` |

This is a document-level consistency check against the supplied completed studies. It is not a repeated model audit or an independent revalidation of upstream datasets, cited external literature or experimental results. Existing institutional placeholders and provenance limitations remain for the authors to complete.
