# Targeted SHAP reliability report

**FINAL VERDICT: CONDITIONAL**

## Four original detection failures

| Validation source ID | Label | Magnitude | Logit | Wave sum | Tabular sum | Residual | Tolerance |
|---|---|---|---|---|---|---|---|
| NOISE_P7567 | 0 | -1.00 | -0.527897 | -1.011896 | 1.112619 | 0.133907 | 0.123463 |
| NCS_EV_03052 | 1 | 3.80 | -0.761951 | -0.874232 | 1.004737 | -0.129928 | 0.100058 |
| NCS_EV_07170 | 1 | 5.00 | -0.003779 | -0.683296 | 1.817967 | -0.375923 | 0.175875 |
| NCS_EV_02345 | 1 | 5.70 | 0.725666 | 0.527812 | 1.582965 | -0.622584 | 0.248819 |

## Implementation checks

The original residual is logit minus the empirical mean of 48 training-background logits minus the signed sum over all 400 waveform and 40x7 tabular SHAP values. No sigmoid, probability/logit mixing, absolute-value summation, or branch averaging was found. The original scalar wrapper returns shape (N,1) from the full fused model. SHAP singleton output axes are removed before aggregation. Recomputed saved-array residuals agree to 1e-10; reloaded frozen predictions agree to 1e-5 and background means to 1e-6. Expected values are mean(model(background)), not model(mean(background)) or logit(mean probability).

The unchanged rule abs(residual) <= 0.1 + 0.1*abs(logit - expected logit) is a prespecified heuristic screen in logit units. It is neither SHAP exact local accuracy nor a statistical confidence bound. It is reference-dependent and has an absolute floor, so passing fractions alone are not adequate reliability evidence. Signed residuals and tolerances are shown rather than concealing failures.

## Fixed-budget sampling diagnostic

Execution note: the retained diagnostic contains 81,920 path draws. One 2,048-draw call was repeated after a float32 bookkeeping assertion interrupted the first attempt (83,968 draws computed including that repeat). The repair promoted scalar arithmetic to Python float; it did not change SHAP settings, inputs, weights, tolerances, seeds, or the planned comparisons. No sampling configurations were added in response to results.

Two nested sampling budgets, 2048 and 8192, two seeds per failed case (730+j, 830+j), four cases, 81,920 total path draws. Primary training background unchanged; every case is called separately so matching seeds use the same random-sample prefix. These newly seeded runs are not exact continuations of the original batched seed-720 run. Only GradientExplainer attributions are used.

| Draws | Mean absolute residual | RMS residual | Maximum residual | Passes / runs | Mean absolute baseline sampling error | Mean absolute remaining path error |
|---|---|---|---|---|---|---|
| 2048 | 0.212092 | 0.245736 | 0.381776 | 4/8 | 0.097903 | 0.137453 |
| 8192 | 0.119120 | 0.144583 | 0.226678 | 5/8 | 0.041866 | 0.081079 |

Mean absolute reconstruction error decreased from 0.212092 to 0.119120 logits; 6/8 matched case-seed comparisons improved. These are four deliberately selected failures and two seeds, not a convergence guarantee. No post-result budget extension or tolerance adjustment was made.

| Case | Seed | Draws | Signed residual | Tolerance | Pass |
|---|---|---|---|---|---|
| NOISE_P7567 | 730 | 2048 | 0.370391 | 0.123463 | False |
| NOISE_P7567 | 730 | 8192 | 0.208425 | 0.123463 | False |
| NOISE_P7567 | 830 | 2048 | 0.017828 | 0.123463 | True |
| NOISE_P7567 | 830 | 8192 | -0.111802 | 0.123463 | True |
| NCS_EV_03052 | 731 | 2048 | -0.172158 | 0.100058 | False |
| NCS_EV_03052 | 731 | 8192 | 0.186890 | 0.100058 | False |
| NCS_EV_03052 | 831 | 2048 | 0.288194 | 0.100058 | False |
| NCS_EV_03052 | 831 | 8192 | 0.226678 | 0.100058 | False |
| NCS_EV_07170 | 732 | 2048 | -0.381776 | 0.175875 | False |
| NCS_EV_07170 | 732 | 8192 | 0.018961 | 0.175875 | True |
| NCS_EV_07170 | 832 | 2048 | -0.162913 | 0.175875 | True |
| NCS_EV_07170 | 832 | 8192 | 0.037224 | 0.175875 | True |
| NCS_EV_02345 | 733 | 2048 | 0.237014 | 0.248819 | True |
| NCS_EV_02345 | 733 | 8192 | 0.152043 | 0.248819 | True |
| NCS_EV_02345 | 833 | 2048 | -0.066466 | 0.248819 | True |
| NCS_EV_02345 | 833 | 8192 | 0.010940 | 0.248819 | True |

The residual was partitioned by replaying the installed SHAP sampler: sampled-background mean minus full-background mean, plus logit minus sampled-background mean minus summed attribution. The second term includes finite path-integration/gradient-sampling and floating-point error; the partition is not a formal variance decomposition or proof of the sole cause. SHAP draws the same background row and interpolation coefficient for both branches.

## Existing background and repeat stability

Detection expected logit changes from -0.762527 to 1.211380 (+1.973908). Magnitude expected output changes from 3.853153 to 3.861201 (+0.008048). Same coarse stratum counts do not guarantee similar background output distributions. Feature rankings alone therefore cannot justify reference-invariant local interpretations. No new backgrounds were selected.

### detection

| Measure | Same-background repeat | Alternative background |
|---|---|---|
| feature_rank_spearman | 1.000000 | 0.892857 |
| feature_relative_L1 | 0.007546 | 0.136493 |
| local_cosine_min | 0.991375 | 0.930443 |
| local_cosine_mean | 0.997065 | 0.948961 |
| waveform_temporal_spearman | 0.997983 | 0.954385 |
| tabular_temporal_spearman | 0.995310 | 0.871482 |

Reconstruction passes: original 8/12, repeat 10/12, alternative 9/12.

### magnitude

| Measure | Same-background repeat | Alternative background |
|---|---|---|
| feature_rank_spearman | 1.000000 | 0.964286 |
| feature_relative_L1 | 0.005631 | 0.108198 |
| local_cosine_min | 0.995451 | 0.887271 |
| local_cosine_mean | 0.997294 | 0.930458 |
| waveform_temporal_spearman | 0.995298 | 0.905420 |
| tabular_temporal_spearman | 0.998874 | 0.975985 |

Reconstruction passes: original 9/9, repeat 9/9, alternative 9/9.

## Paired-input and preservation checks

Both fixed 48-record training backgrounds and the 12 existing validation records were reloaded by source ID and joined by unique parent ID. Source ID, parent ID, label, magnitude and 3s eligibility masks match across branches; ordered source IDs match the saved attribution arrays. Input shapes are (N,400)/(N,40,7); 100 Hz, one second pre-context, and final bin endpoint 3s are preserved. Only selected waveform/feature rows were loaded. No test group, new validation sample, whole-model audit, training or model selection was used. Existing SHAP sources and checkpoint hashes remain unchanged; model state tensors and evaluation mode are unchanged. No assertion here independently re-audits original feature generation.

## Remaining scientific limits

Passing reconstruction establishes only approximate numerical accounting for the chosen model function and reference distribution. It does not validate the counterfactual inputs. Waveform and seven-feature tensors share a source, and paired real endpoints do not make straight-line interpolated points obey nonlinear feature extraction (including energy/RMS dependence). Consequently these are expected-gradient model-input attributions, not uniquely identifiable physical branch contributions, conditional-distribution SHAP, independent sensor effects, or causal effects. Attention pools bidirectional hidden states and remains distinct from SHAP; a bin cannot be interpreted as isolated physical evidence.

The verdict remains CONDITIONAL even if the larger-budget numerical checks pass: reference sensitivity and dependent-input interpretation remain, while the original figures still contain their original 2048-draw values. This diagnostic does not repair or overwrite them. All results and code are retained in the executed notebook; diagnostic arrays and machine-readable results are new artifacts in this directory.
