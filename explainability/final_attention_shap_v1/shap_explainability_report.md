# SHAP explainability report

**VERDICT: CONDITIONAL**

Only the frozen full 3s CNN + BiLSTM + Temporal Attention model is explained. The original checkpoint and referenced backbones were loaded strictly. Both scalar wrappers retain both branches, temporal Attention, original scalers and final prediction heads. Detection is explained in logit units; magnitude in original target units. No training or tuning.

## Sampling and computation

Seed 20261114. Two 48-record paired training backgrounds with proportional class/magnitude strata. Validation: 12 records (3 noise, 3 EQ M<4, 3 EQ 4<=M<5, 3 EQ M>=5); magnitude uses only the 9 eligible earthquakes. Strong events are deliberately overrepresented in the small explanation cohort, so aggregate plots are descriptive, not population estimates. Selection precedes SHAP and uses no prediction quality or test observations. See selection_manifest.json. CPU evaluation mode with frozen weights avoids cuDNN evaluation-mode LSTM backward restrictions. GradientExplainer pilot: 128 draws; final and sensitivity runs: 2048 each, no smoothing.

The seven tabular features are deterministic summaries of the same waveform; they are also internally dependent (energy = 10*RMS squared before normalization). Background endpoints are paired real training records. SHAP GradientExplainer uses a shared background row and interpolation coefficient for both tensors, but simultaneous straight-line interpolation generally does not preserve nonlinear feature-extraction identities. Thus intermediate points can be physically inconsistent even with paired endpoints. These are model-input expected-gradient attributions under the selected reference distribution, not unique physical contributions of independent modalities, conditional-distribution SHAP, or evidence of causality. Attention weights are learned hidden-state pooling weights, not SHAP values; bidirectional states mix information throughout the available window.

Both targets use the same mixed-class training backgrounds. The magnitude expected value therefore includes outputs on noise records: it is a mathematical model-output reference, not an average physical earthquake magnitude. Only eligible earthquakes are explained for magnitude.

## Detection

Top three tabular inputs by mean absolute attribution: rms, first_difference_energy, zero_crossing_rate. This ordering concerns normalized model inputs only. Expected output: -0.762527. Reconstruction mean/max absolute residual: 0.205840/0.622584; 8/12 satisfy the predeclared tolerance. Alternative-background feature rank correlation: 0.893; relative L1 difference: 0.136. These small-sample sensitivity results do not establish universal ranking stability.

## Magnitude

Top three tabular inputs by mean absolute attribution: rms, mean_absolute_amplitude, first_difference_energy. This ordering concerns normalized model inputs only. Expected output: 3.853153. Reconstruction mean/max absolute residual: 0.012144/0.024013; 9/9 satisfy the predeclared tolerance. Alternative-background feature rank correlation: 0.964; relative L1 difference: 0.108. These small-sample sensitivity results do not establish universal ranking stability.

## Limits of interpretation

The dependency issue structurally prevents independent-modality conclusions; its effect size on real-world conclusions is not identifiable from this experiment. No off-manifold causal claims are made. The synthetic oracle-aligned window spans [-1,3) seconds (last waveform sample 2.99 s; last bin ends at 3 s). No future observations enter the inputs. This is completed-window inference, not an online prefix-invariant model. Existing train-fitted preprocessing is reused unchanged. No standalone model explanations or test-set inference were performed.

## Figures

- [detection_attention_vs_shap.png](plots/detection_attention_vs_shap.png) (300 DPI)
- [detection_feature_importance.png](plots/detection_feature_importance.png) (300 DPI)
- [detection_feature_time_heatmap.png](plots/detection_feature_time_heatmap.png) (300 DPI)
- [detection_local_explanation.png](plots/detection_local_explanation.png) (300 DPI)
- [detection_temporal_importance.png](plots/detection_temporal_importance.png) (300 DPI)
- [detection_waveform_shap.png](plots/detection_waveform_shap.png) (300 DPI)
- [magnitude_attention_vs_shap.png](plots/magnitude_attention_vs_shap.png) (300 DPI)
- [magnitude_feature_importance.png](plots/magnitude_feature_importance.png) (300 DPI)
- [magnitude_feature_time_heatmap.png](plots/magnitude_feature_time_heatmap.png) (300 DPI)
- [magnitude_local_explanation.png](plots/magnitude_local_explanation.png) (300 DPI)
- [magnitude_temporal_importance.png](plots/magnitude_temporal_importance.png) (300 DPI)
- [magnitude_waveform_shap.png](plots/magnitude_waveform_shap.png) (300 DPI)

## Numerical interpretation restrictions

The detection reconstruction check is **not fully passed**: 4 of 12 records exceed the predeclared tolerance of 0.1 + 0.1*abs(output minus expected output). Its maximum absolute residual is 0.622584 logits. Detection plots are therefore approximate descriptive attributions; they must not be used as precise local additive decompositions or exact probability explanations. Residuals are displayed explicitly in the local plots and retained for every record in the NPZ files. No extra sampling or explainer tuning was performed after these results.

Changing the 48-record training background shifts the detection expected logit from -0.762527 to 1.211380 (a 1.973908-logit shift), despite the same class/magnitude stratum counts. This is material reference-distribution sensitivity, even though aggregate feature rank correlation is 0.893 and the minimum local signed-attribution cosine is 0.930. Stable feature rankings do not establish a stable reference value or exact local explanation. The independent Monte Carlo repeat has feature rank correlation 1.000 and minimum local cosine 0.991.

Magnitude passes the reconstruction tolerance for all 9 records (maximum absolute residual 0.024013 magnitude units). Its expected output changes from 3.853153 to 3.861201 across backgrounds; feature rank correlation is 0.964, with minimum local cosine 0.887. The independent repeat has feature rank correlation 1.000 and minimum local cosine 0.995. These results support approximate descriptions of these selected model outputs, subject to the paired-input dependency limitations. The magnitude reference includes noise model outputs and has no interpretation as an average physical earthquake magnitude.

**CONDITIONAL** applies to the experiment as a whole: genuine SHAP expected gradients were obtained, but detection numerical reconstruction, background sensitivity, dependent inputs, and the small stratified cohort prevent an unqualified approval. Attention weights remain separate diagnostics, never SHAP values.
