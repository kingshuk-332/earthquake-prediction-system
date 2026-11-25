# Phase 1 results — cnn_b86050c74334
Date: 2026-11-05. Dataset SHA256: 5cfebe7268a87940ec9489d4f72b45381e2c940cc3f38d96d60b4575355d8bf3.
Raw rows: 8089; selected complete-day records: 5603.
Samples: {'train': 562, 'validation': 110, 'test': 110}. Input: (782, 12, 90); output: (782, 7).
Trainable parameters: 3223. Epochs completed: 21; restored epoch: 6.
Device: NVIDIA GeForce RTX 3050 Laptop GPU.

CNN does not beat all simple comparators on primary test Brier score; no general forecasting advantage is established.

```
              model  primary_brier  primary_log_loss  primary_AP  time_mean_brier  spatial_mean_brier  all_output_brier
                CNN       0.281609          0.758084    0.567793         0.234552            0.131242          0.182240
   training_Poisson       0.269457          0.732576    0.563636         0.232346            0.129448          0.178849
     recent_Poisson       0.242390          0.680387    0.649873         0.225823            0.125200          0.170691
training_prevalence       0.280619          0.755973    0.563636         0.236064            0.131875          0.182892
```

```
                   comparison  mean_brier_difference  ci95_low  ci95_high  block_weeks
   CNN minus training_Poisson               0.012152  0.005557   0.019889            4
     CNN minus recent_Poisson               0.039219 -0.011912   0.089982            4
CNN minus training_prevalence               0.000990  0.000442   0.001602            4
```

See split_summary.json for exact dates/event counts, scores.csv for each target, and calibration_bins.csv for reliability support.
Notebook limitations apply: mixed magnitude scales, unknown completeness/delays, exploratory single holdout, coarse marginal outputs and correlated histories.
This run does not establish deterministic or operational earthquake prediction.
