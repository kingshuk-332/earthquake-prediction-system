# SHAP figures manifest

All 12 existing figures are embedded by relative reference in Chapter 8. No images were copied, regenerated, resampled or edited. The original 2,048-draw figures are not relabelled as 8,192-draw diagnostic results. Selection and qualifications follow [report integration guidance](../../explainability/shap_reliability_v1/report_integration_guidance.md).

| Report figure | Existing source | Dimensions (pixels) | Embedded DPI | Permitted interpretation and required warning |
|---|---|---|---|---|
| 8.1 | [detection_waveform_shap.png](../../explainability/final_attention_shap_v1/plots/detection_waveform_shap.png) | 2970 × 1166 | 299.9994 × 299.9994 | Approximate aggregate; all 12 cases included; 4/12 failures and 1.974-logit reference shift; no causal timing. |
| 8.2 | [detection_feature_importance.png](../../explainability/final_attention_shap_v1/plots/detection_feature_importance.png) | 2968 × 1165 | 299.9994 × 299.9994 | Approximate normalized-input ranking; 4/12 failures; alternative-background rank rho=0.893; no independent physical effects. |
| 8.3 | [detection_temporal_importance.png](../../explainability/final_attention_shap_v1/plots/detection_temporal_importance.png) | 2970 × 1166 | 299.9994 × 299.9994 | Absolute feature sum per bin, not signed evidence or Attention; original detection numerical/reference warnings. |
| 8.4 | [detection_feature_time_heatmap.png](../../explainability/final_attention_shap_v1/plots/detection_feature_time_heatmap.png) | 3100 × 1166 | 299.9994 × 299.9994 | Absolute attribution hides sign and record variation; dependent inputs; original numerical/reference warnings. |
| 8.5 | [detection_local_explanation.png](../../explainability/final_attention_shap_v1/plots/detection_local_explanation.png) | 3268 × 2666 | 299.9994 × 299.9994 | Preselected NCS_EV_03910 passes its tolerance; residual 0.201782 is retained. All four failures are listed in Table 8.3; no success-based replacement. |
| 8.6 | [magnitude_waveform_shap.png](../../explainability/final_attention_shap_v1/plots/magnitude_waveform_shap.png) | 2970 × 1166 | 299.9994 × 299.9994 | Nine selected earthquakes; initial 9/9 numerical passes do not establish causality; mixed-class background. |
| 8.7 | [magnitude_feature_importance.png](../../explainability/final_attention_shap_v1/plots/magnitude_feature_importance.png) | 2968 × 1165 | 299.9994 × 299.9994 | RMS, mean absolute amplitude, first-difference energy lead; nine-case descriptive ranking; no population claim. |
| 8.8 | [magnitude_temporal_importance.png](../../explainability/final_attention_shap_v1/plots/magnitude_temporal_importance.png) | 2970 × 1166 | 299.9994 × 299.9994 | Absolute attribution per bin with full-window bidirectional context; no isolated physical timing claim. |
| 8.9 | [magnitude_feature_time_heatmap.png](../../explainability/final_attention_shap_v1/plots/magnitude_feature_time_heatmap.png) | 3152 × 1166 | 299.9994 × 299.9994 | Dependent normalized features; sign and between-record variation suppressed. |
| 8.10 | [magnitude_local_explanation.png](../../explainability/final_attention_shap_v1/plots/magnitude_local_explanation.png) | 3268 × 2666 | 299.9994 × 299.9994 | Original first selected M>=5 example; retain predicted/true magnitude and residual; numerical pass is not prediction accuracy. |
| 8.11 | [detection_attention_vs_shap.png](../../explainability/final_attention_shap_v1/plots/detection_attention_vs_shap.png) | 2970 × 1765 | 299.9994 × 299.9994 | Separate units/meanings; Attention is not SHAP. Original 4/12 failures and reference sensitivity apply. |
| 8.12 | [magnitude_attention_vs_shap.png](../../explainability/final_attention_shap_v1/plots/magnitude_attention_vs_shap.png) | 2945 × 1765 | 299.9994 × 299.9994 | Separate weights and input attributions; visual similarity is not agreement, validation or causality. |

PNG metadata rounds to 299.9994 DPI (nominal 300 DPI); all twelve files satisfy the 300-DPI check. Original comparison figures and architecture diagrams remain in their existing chapters. No feature or temporal ranking beyond the saved SHAP evidence was inferred.

## Source preservation hashes

| Figure | SHA256 |
|---|---|
| 8.1: detection_waveform_shap.png | `d93cbf07e98f9687e8e83acb54104e95ba4de8ad5c682c13ebfbb2eb7783b2c9` |
| 8.2: detection_feature_importance.png | `e36591cc22231c82cad3ed3f71d1e61626800fc0b531effd198d01927a7e7edd` |
| 8.3: detection_temporal_importance.png | `6f215b5e649b0236687b4c6b3d3fde4eedb937130cc66f90f232245c94e70c5b` |
| 8.4: detection_feature_time_heatmap.png | `c56842a1e704ea5fcbbe88cdc978d5d216775926bc22eb82f4c45b26723a223c` |
| 8.5: detection_local_explanation.png | `b3f333138ae4793d3ffc7f926a1992a44ed27889be8dbd078c0d63667ed3deee` |
| 8.6: magnitude_waveform_shap.png | `307c79600597fd5d8c22194e514e9fe42f9a6e6ab0f30307590f02c130bcd6a6` |
| 8.7: magnitude_feature_importance.png | `f445bd127de3334de2f4256928a4d9af951e3accdd960f478c392978069f8a59` |
| 8.8: magnitude_temporal_importance.png | `3eb69bef61d13e02210d90da2dbe658b9b64a6ded33eaf381a28f7bacf56b0f2` |
| 8.9: magnitude_feature_time_heatmap.png | `689b2e240ab698ad8233edc366fe378016d91385852aef7fe3703433b1fe5696` |
| 8.10: magnitude_local_explanation.png | `8ac29f9def582b07dd6e728798d0a7026ed1ca730f5fdaf49bc020d4468580da` |
| 8.11: detection_attention_vs_shap.png | `2b904258f0bfe07a841293b82678c0a8afc7c77092b6ffd7cb2f87ceae2866b6` |
| 8.12: magnitude_attention_vs_shap.png | `67fa513d70e61d0a88844c2ec66414f46c893b67d2cca766a1fb11798513ea1d` |
