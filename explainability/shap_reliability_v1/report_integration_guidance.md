# Academic report integration guidance

**FINAL VERDICT: CONDITIONAL.** Use existing figures only as descriptive explanations of the frozen model on the preselected validation cohort. No existing plot has been replaced by this diagnostic.

## Figure-by-figure disposition

| Existing figure | Academic use | Required qualification |
|---|---|---|
| detection_feature_importance.png | Qualified aggregate | 12 deliberately stratified cases; approximate mean absolute model-input SHAP; background feature rank rho=0.893; original 4/12 reconstruction failures. |
| detection_waveform_shap.png | Qualified aggregate | Not a physical time-local causal signal; original 4/12 reconstruction failures and 1.974-logit reference shift. |
| detection_temporal_importance.png | Qualified aggregate | Sum of absolute feature attributions per bin, not signed evidence or Attention; original numerical limitations apply. |
| detection_feature_time_heatmap.png | Qualified aggregate | Mean absolute SHAP hides sign/record variation; dependent inputs; original numerical and reference-sensitivity warning required. |
| detection_local_explanation.png | Usable as an explicitly approximate local explanation | NCS_EV_03910 passes the original tolerance (residual 0.201782 < tolerance about 0.25271). Retain its residual and reference-sensitivity warning; do not present an exact logit or probability decomposition. |
| detection_attention_vs_shap.png | Methodological comparison with explicit warning | Attention and SHAP differ in units/meaning; resemblance is not validation or causality; original detection reconstruction limits apply. |
| magnitude_feature_importance.png | Usable with study limitations | Nine eligible earthquakes; strong events deliberately included; mixed-class mathematical background; not population importance. |
| magnitude_waveform_shap.png | Usable with study limitations | Model-input attributions only; no physical causality despite 9/9 original reconstruction passes. |
| magnitude_temporal_importance.png | Usable with study limitations | Absolute attribution across seven inputs; bidirectional context spans the full observed window. |
| magnitude_feature_time_heatmap.png | Usable with study limitations | Dependent normalized features; mean absolute values do not show direction. |
| magnitude_local_explanation.png | Approximate local model explanation | Retain residual and predicted/true magnitude; numerical pass does not establish prediction accuracy or physical validity. |
| magnitude_attention_vs_shap.png | Methodological comparison with explicit warning | Attention is not SHAP; weights cannot validate SHAP and axes are not directly comparable. |

All filenames refer to ../final_attention_shap_v1/plots/. Existing 300-DPI images remain unchanged. Magnitude plots have stronger numerical reconstruction support, but still require dependency, sampling and synthetic oracle-alignment limitations. Detection aggregate plots require explicit numerical and reference-sensitivity warnings, not merely a generic SHAP caveat.

## Suggested academic text

A targeted fixed-budget check examined the four detection records that failed the original SHAP reconstruction tolerance. Mean absolute reconstruction error decreased from 0.212092 to 0.119120 logits; 6/8 matched case-seed comparisons improved. These are four deliberately selected failures and two seeds, not a convergence guarantee. No post-result budget extension or tolerance adjustment was made. The frozen predictions, paired inputs, logit handling, empirical expected value and signed multi-input aggregation were verified. The detection reference logit differed by 1.974 between two existing training backgrounds despite relatively stable feature rankings. Detection explanations therefore remain descriptive and approximate; the original local detection figure is presented only with its reconstruction residual. Magnitude reconstruction passed for all nine original earthquake cases, without establishing causal or independent-modality interpretations.

## Required shared limitations

Passing reconstruction establishes only approximate numerical accounting for the chosen model function and reference distribution. It does not validate the counterfactual inputs. Waveform and seven-feature tensors share a source, and paired real endpoints do not make straight-line interpolated points obey nonlinear feature extraction (including energy/RMS dependence). Consequently these are expected-gradient model-input attributions, not uniquely identifiable physical branch contributions, conditional-distribution SHAP, independent sensor effects, or causal effects. Attention pools bidirectional hidden states and remains distinct from SHAP; a bin cannot be interpreted as isolated physical evidence.

Do not caption any figure as a physical causal contribution, independent modality importance, exact probability decomposition, real-world warning validation, or proof that Attention agrees with SHAP. Do not attach new high-sampling diagnostic values to unchanged old figures as if those figures were regenerated.
