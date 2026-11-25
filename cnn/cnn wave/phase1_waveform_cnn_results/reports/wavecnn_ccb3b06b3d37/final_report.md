# Waveform CNN detection results — wavecnn_ccb3b06b3d37
Date: 2026-11-05. This is synthetic waveform detection, not future forecasting.
Input: Phase 11 waveforms_standardized, (10000,1), 100 Hz.
Splits: train 11324, validation 2426, test 2428; balanced earthquake/noise.
Parameters: 4177; epochs: 25; restored epoch: 22; GPU: NVIDIA GeForce RTX 3050 Laptop GPU.

## Detection metrics
```
split            model  samples  accuracy  precision   recall       f1  roc_auc  average_precision  pr_auc_trapezoidal  log_loss   tn   fp  fn   tp  threshold
 test              CNN     2428  0.998353   0.999175 0.997529 0.998351 0.999995           0.999995            0.999995  0.005562 1213    1   3 1211   0.500000
 test  signal_logistic     2428  0.915568   0.932305 0.896211 0.913902 0.973751           0.976230            0.976221  0.211347 1135   79 126 1088   0.500000
 test raw_RMS_logistic     2428  0.578254   0.579299 0.571664 0.575456 0.593738           0.623917            0.623650  0.680251  710  504 520  694   0.500000
 test      max_STA_LTA     2428  0.512768   0.506766 0.956343 0.662482 0.451261           0.430744            0.430393       NaN   84 1130  53 1161   1.554137
```

## Frozen-model shortcut probes
```
               probe  samples  accuracy  precision   recall       f1  roc_auc  average_precision  pr_auc_trapezoidal  log_loss   tn  fp   fn   tp  threshold  all_noise_false_positive_rate  original_EQ_noise_false_positive_rate  original_noise_false_positive_rate
           shift_25s     2428  0.997529   1.000000 0.995058 0.997523 0.999996           0.999996            0.999996  0.007271 1214   0    6 1208        0.5                            NaN                                    NaN                                 NaN
       time_reversal     2428  0.997117   0.999173 0.995058 0.997111 0.999988           0.999988            0.999988  0.008304 1213   1    6 1208        0.5                            NaN                                    NaN                                 NaN
    phase_randomized     2428  0.941928   0.897113 0.998353 0.945029 0.973689           0.962489            0.969394  0.525571 1075 139    2 1212        0.5                            NaN                                    NaN                                 NaN
pre_onset_noise_only     2428  0.503707   0.527273 0.071664 0.126178 0.563610           0.529506            0.528661  4.273179 1136  78 1127   87        0.5                       0.067957                               0.071664                             0.06425
```
Pre-onset-noise probe labels for physical detection are all noise. Its original-label accuracy/AUC only test residual source-class signatures.

## Interpretation constraints
No metadata entered the CNN. Performance on transformed traces and signal baselines indicates possible shortcut reliance but cannot prove physical mechanisms. Synthetic in-distribution test performance is not real-field performance.
Experiment B was audited, not trained: magnitude labels exist but amplitude normalization and simulator inversion confound interpretation; absolute UTC origin time is absent from inputs; single-component traces cannot uniquely identify geographic epicenter. P/S indices are available but P is artificially anchored near six seconds.
Catalog forecasting notebook/results remain byte-for-byte unchanged.

## Verified waveform findings — 2026-11-05
The CNN has 4,177 trainable parameters. It completed 25 epochs and restored epoch 22 (validation BCE 0.010031).
Held-out test: accuracy 0.998353, precision 0.999175, recall 0.997529, F1 0.998351, ROC-AUC 0.999995, average precision 0.999995, trapezoidal PR-AUC 0.999995.
Confusion matrix (rows true noise/EQ, columns predicted noise/EQ): [[1213, 1], [3, 1211]].
The five-signal-feature logistic baseline has accuracy 0.915568 and F1 0.913902. This is an in-simulator comparison, not an external field benchmark.

### What the shortcut probes show
- Circular 25 s shift: accuracy 0.997529, AUC 0.999996.
- Time reversal: accuracy 0.997117, AUC 0.999988.
- Spectrum-preserving phase randomization: accuracy 0.941928, AUC 0.973689.
- Pre-onset-noise-only: overall false-positive rate 0.067957; noise taken from earthquake traces 0.071664; noise taken from synthetic-noise traces 0.064250. Its AUC against original source labels is 0.563610; those original labels are diagnostic group labels, not correct event labels after signal removal.

These probes do not establish exactly which physical features are learned. The CNN separates the supplied generated classes; disruption of phase/ordering tests sensitivity to waveform structure, while remaining discrimination can expose generator-specific spectral/background differences. Noise-only probes are derived after full-trace detrending, which itself can leave a class-associated linear-trend imprint; this is another preprocessing shortcut risk. Repetition introduces artificial boundaries. Treat all probes as diagnostics, not causal proof or realistic out-of-distribution performance estimates.

Experiment B is **not trained**. Earthquake magnitude/P/S labels are verified; origin times and coordinates require source-ID joins. Standardization removes absolute amplitude, and waveform shape is generated from known metadata formulas. Absolute origin time and unique epicentral coordinates cannot be recovered from this single clock-free vertical trace. Separate magnitude-inversion/picking studies need explicit baselines and independent data.

Paired-bootstrap accuracy difference (CNN minus signal logistic): 0.082784, 95% interval [0.071664, 0.093904]. This interval covers sampled pairs, not uncertainty across generator families or seeds.
