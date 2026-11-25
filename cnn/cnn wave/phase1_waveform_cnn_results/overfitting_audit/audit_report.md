# Frozen CNN overfitting audit — 2026-11-06
Experiment wavecnn_ccb3b06b3d37; best epoch 22/25. No new trained experiment; registry unchanged.

## Comparable best-checkpoint metrics
            samples     loss  accuracy  precision   recall       f1  roc_auc   pr_auc  average_precision    tn  fp  fn    tp
split                                                                                                                       
train         11324 0.007772  0.997351   0.999114 0.995585 0.997346 0.999978 0.999978           0.999978  5657   5  25  5637
validation     2426 0.010031  0.997115   0.997525 0.996702 0.997113 0.999779 0.999748           0.999748  1210   3   4  1209
test           2428 0.005562  0.998353   0.999175 0.997529 0.998351 0.999995 0.999995           0.999995  1213   1   3  1211

Accuracy gaps (train minus held out), percentage points: validation 0.023617; test -0.100179.
Loss gaps (held out minus train): validation 0.002260; test -0.002209.
Best validation loss 0.010031; final 0.010481; peak after best 0.017998. Late fluctuations are not sustained divergence; best weights were restored. History train metrics include dropout and changing weights; use frozen metrics above for gaps. No clear underfitting on these synthetic splits.

## Errors and subgroups
  group  samples  errors  accuracy
    M<3      195       3  0.984615
 3<=M<5      888       0  1.000000
   M>=5      131       0  1.000000
 SNR<10      699       3  0.995708
SNR>=10      515       0  1.000000
Test false positives: 1; false negatives: 3. Magnitude/SNR bins are earthquake-only; accuracy equals recall. Noise has no physical earthquake magnitude. The cached noise subgroup report assigns the false positive to transient disturbances. Individual errors are in test_errors.csv; confidence summaries include confidently wrong cases.

## Robustness
Reused circular shift accuracy 0.997529, reversal 0.997117, phase randomization 0.941928 (ROC-AUC 0.973689); repeated pre-onset prefix FPR 0.067957. Prefix original-label AUC 0.563610 is source-class discrimination, not earthquake detection, because all transformed examples are noise. Full-trace detrending and repetition can introduce artifacts.
              family  samples  false_positives  false_positive_rate  specificity  fpr_ci95_low  fpr_ci95_high  mean_earthquake_probability
      white_gaussian      256                0                  0.0          1.0           0.0       0.014784                     0.000212
lowpass_gaussian_5hz      256                0                  0.0          1.0           0.0       0.014784                     0.000864
  impulsive_gaussian      256                0                  0.0          1.0           0.0       0.014784                     0.000003
Independent draws above use simple, prespecified noise recipes, not the original generator. They test false alarms only; no earthquake sensitivity or real-field generalization can be inferred. Wilson intervals assume independent generated draws. No threshold, model or recipe was tuned on test outcomes.

## Verdict
Overfitting: No clear evidence of ordinary sample overfitting within the supplied synthetic distribution; this does not establish external generalization. Small frozen-checkpoint gaps and late validation fluctuations must be distinguished from synthetic shortcuts.
Leakage: not detected in completed exact-duplicate, source/pair/event disjointness and label/alignment checks. Exhaustive near-duplicate correlation remains unresolved; shared simulator mechanisms are a distribution limitation, not proven row leakage. Existing complete checks reused, not repeated.
Synthetic shortcuts: evidence consistent with spectral/background discrimination: phase-randomized traces retain high accuracy and time reversal/shift have little effect. This is not causal proof of specific learned features. Noise stress results provide additional negative-only evidence. Full-waveform preprocessing is offline detection, not forecasting.
Ready for BiLSTM comparison: Yes for a controlled synthetic baseline using identical splits, preprocessing and validation selection; no claim of field readiness. Freeze this CNN and threshold. Official test has already been inspected for diagnostics: do not use it to tune architectures or hyperparameters; reserve a new untouched distribution/holdout for claims beyond this benchmark.
