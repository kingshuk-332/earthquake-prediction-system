# Figure and table manifest

Existing raster figures are referenced, not regenerated or copied.

| Figure | Content | Source |
|---|---|---|
| 6.1 | Four-model F1 at fixed0.5;300 DPI | [final_comparison/four_model_eew_v1/plots/four_model_f1.png](../../final_comparison/four_model_eew_v1/plots/four_model_f1.png) |
| 6.2 | Magnitude MAE on eligible earthquakes;300 DPI | [final_comparison/four_model_eew_v1/plots/four_model_mae.png](../../final_comparison/four_model_eew_v1/plots/four_model_mae.png) |
| 6.3 | Four-model FPR at fixed0.5;300 DPI | [final_comparison/four_model_eew_v1/plots/four_model_fpr.png](../../final_comparison/four_model_eew_v1/plots/four_model_fpr.png) |
| 6.4 | Magnitude RMSE;300 DPI | [final_comparison/four_model_eew_v1/plots/four_model_rmse.png](../../final_comparison/four_model_eew_v1/plots/four_model_rmse.png) |
| 6.5 | Attention confusion matrices, common cohort, fixed0.5;300 DPI | [fusion/cnn_bilstm_attention_v1/plots/confusion_matrices.png](../../fusion/cnn_bilstm_attention_v1/plots/confusion_matrices.png) |
| 4.1 | Embedded Mermaid architecture diagram | [final_cnn_eew_report.md](../../cnn/cnn wave/eew_causal_v3/cnn_eew_training/final_cnn_eew_report.md) |
| 4.2 | Embedded Mermaid architecture diagram | [final_bilstm_eew_report.md](../../bilstm_eew/bilstm_training_v1/final_bilstm_eew_report.md) |
| 4.3 | Embedded Mermaid architecture diagram | [final_fusion_report.md](../../fusion/cnn_bilstm_v1/final_fusion_report.md) |
| 4.4 | Embedded Mermaid architecture diagram | [final_attention_report.md](../../fusion/cnn_bilstm_attention_v1/final_attention_report.md) |

## Tables and numerical provenance

| Location | Content | Source / qualification |
|---|---|---|
| 3.1 | Parent splits | Fusion final report; total also supplied in task |
| 3.2 | Input shapes | BiLSTM/fusion/Attention reports |
| 5.1 | Budgets, epochs, parameter counts | Four model final reports; frozen parameters excluded from trained-head counts |
| 6.1 and leader subtable | Fixed-threshold comparison and metric-specific leaders | Integrated v2 report and v1 full-precision CSV; no new inference |
| 6.2 | Common-test results at validation-selected thresholds | Integrated report; CNN corresponding result unavailable |
| 6.3 | Fixed-.5 confusion counts | Exact integer reconstruction from full-precision recall/FPR and class totals; checked against accuracy |
| 6.4 | Paired differences/intervals | Integrated report; no new bootstrap |
| Chapter6 magnitude tables | Common-test compression and separately labeled validation magnitude-band diagnosis | Integrated report; cohort labels preserved |
| Chapter7 trial table | All12 validation trials | Integrated report; no retraining |
| Chapter7 threshold table | Validation and exploratory test outcomes | Integrated report; no threshold adjustment |

The original standalone1s/2s horizon-specific plots were not inserted as common-cohort comparisons. The reused Attention confusion plot uses the common cohort. Mermaid diagrams require a compatible renderer; preserve relative image paths when producing a typeset PDF. Experimental figure captions distinguish common-test metrics from validation analyses.
