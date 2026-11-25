# Standalone tabular BiLSTM synthetic EEW results

**Date:** 2026-11-13. Three horizon-specific models trained, validation-selected and evaluated once on frozen test inputs.

## Protocol

Input: existing standardized feature-only HDF5, 20/30/40 completed 0.1-second bins Ã— 7 features. First ten bins provide pre-reference context. No raw waveforms, catalog history, IDs, labels, magnitude targets or arrival metadata are model features. Existing feature scalers are reused without refitting. BiLSTM processes only the selected observed prefix bidirectionally.

Architecture: two-layer bidirectional LSTM, hidden32/direction, interlayer dropout .2, temporal mean pool, dropout .2, 64â†’32 SiLU shared representation, binary-logit and magnitude heads; 37,730 parameters. Settings: `{"seed": 42, "hidden": 32, "layers": 2, "dropout": 0.2, "batch_size": 256, "max_epochs": 35, "patience": 7, "min_delta": 0.0001, "lr": 0.001, "weight_decay": 0.0001, "clip": 5.0, "magnitude_weight": 1.0, "huber_delta": 1.0}`.

Loss: equal-weight BCE and earthquake-masked Huber(delta=1) on magnitudes standardized by eligible training-earthquake mean/population SD. Sentinel -1 never contributes. Poisoned-sentinel and empty-regression guards passed. Validation selection minimizes full-set mean BCE plus eligible-earthquake mean Huber. Epoch-end training/validation curves disable dropout. All three checkpoints and thresholds were selected before test evaluation.

Threshold rule: maximize validation F1 over .10 to .90 in steps .01; ties prefer closest .5 then lower threshold. No test tuning. Fixed-.5 results are also reported to match the CNN. Ineligible rows are excluded before batching; magnitude metrics include every eligible earthquake regardless of detection outcome.

## Checkpoints

| Horizon | Best / run epochs | Val loss | Threshold | Fit seconds | Checkpoint |
|---|---:|---:|---:|---:|---|
| 1s | 18 / 25 | 0.9557138 | 0.47 | 21.98 | models/bilstm_1s_best.pt |
| 2s | 27 / 34 | 0.8685476 | 0.36 | 28.46 | models/bilstm_2s_best.pt |
| 3s | 26 / 33 | 0.8157979 | 0.36 | 27.86 | models/bilstm_3s_best.pt |

## Horizon-eligible test results

PR-AUC uses average precision. Confusion matrices are actual rows/predicted columns [noise, earthquake].

| Horizon | N / EQ / excluded | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1s | 2428 / 1214 / 0 | 0.629325 | 0.574337 | 0.999176 | 0.729405 | 0.677093 | 0.658111 | [[315, 899], [1, 1213]] |
| 2s | 2421 / 1207 / 7 | 0.628666 | 0.573194 | 0.999171 | 0.728481 | 0.764507 | 0.780146 | [[316, 898], [1, 1206]] |
| 3s | 2414 / 1200 / 14 | 0.648302 | 0.589404 | 0.964167 | 0.731584 | 0.813242 | 0.834604 | [[408, 806], [43, 1157]] |

| Horizon | MAE | RMSE | RÂ² | Signed error | Actual / predicted variance | Pred/actual SD | Slope |
|---|---:|---:|---:|---:|---|---:|---:|
| 1s | 0.678042 | 0.833247 | 0.143602 | -0.060272 | 0.810722 / 0.102291 | 0.3552 | 0.1371 |
| 2s | 0.638256 | 0.785525 | 0.237753 | -0.046810 | 0.809514 / 0.192142 | 0.4872 | 0.2389 |
| 3s | 0.615082 | 0.758421 | 0.288455 | -0.082112 | 0.808385 / 0.215269 | 0.5160 | 0.2815 |

| Horizon | Magnitude band | N | MAE | Signed error | Underprediction % |
|---|---|---:|---:|---:|---:|
| 1s | M<3 | 195 | 1.090304 | 1.090304 | 0.00 |
| 1s | 3<=M<4 | 495 | 0.333930 | 0.290053 | 18.99 |
| 1s | 4<=M<5 | 393 | 0.660058 | -0.622653 | 93.89 |
| 1s | M>=5 | 131 | 1.418591 | -1.409562 | 97.71 |
| 2s | M<3 | 192 | 1.005128 | 1.005128 | 0.00 |
| 2s | 3<=M<4 | 491 | 0.325864 | 0.250926 | 24.85 |
| 2s | 4<=M<5 | 393 | 0.651660 | -0.547442 | 86.51 |
| 2s | M>=5 | 131 | 1.231207 | -1.202630 | 93.13 |
| 3s | M<3 | 190 | 0.933111 | 0.933111 | 0.00 |
| 3s | 3<=M<4 | 487 | 0.316378 | 0.199704 | 29.77 |
| 3s | 4<=M<5 | 392 | 0.652361 | -0.572566 | 87.50 |
| 3s | M>=5 | 131 | 1.152716 | -1.134616 | 96.18 |

## Identical common-cohort comparison

Common cohort identity hash matches the audited CNN exactly: `862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601`. Detection: 1,200 earthquakes and 1,214 noise; magnitude: those 1,200 earthquakes. Common metrics reuse the single test prediction pass. The two representations differ (CNN signed-log waveform versus seven standardized summaries), model sizes, budgets and threshold rules differ; these are benchmark comparisons, not isolated architecture effects.

| Horizon | CNN acc / F1 (.5) | BiLSTM acc / F1 (.5) | BiLSTM acc / F1 (val threshold) | CNN / BiLSTM MAE | CNN / BiLSTM RMSE | CNN / BiLSTM RÂ² |
|---|---|---|---|---|---|---|
| 1s | 0.661143 / 0.692712 | 0.628418 / 0.727438 | 0.627175 / 0.727107 | 0.665020 / 0.670213 | 0.821894 / 0.824230 | 0.164370 / 0.159615 |
| 2s | 0.724524 / 0.712246 | 0.673571 / 0.646637 | 0.627589 / 0.727328 | 0.635296 / 0.634122 | 0.783164 / 0.780973 | 0.241270 / 0.245509 |
| 3s | 0.766777 / 0.757641 | 0.714167 / 0.694690 | 0.648302 / 0.731584 | 0.613005 / 0.615082 | 0.757702 / 0.758421 | 0.289803 / 0.288455 |

## Timing, generalization and scientific limitations

- 1s: model-only median/p95 latency 0.7122/1.1162 ms. Prediction SD ratio 0.355, slope 0.137; Mâ‰¥5 underprediction 97.71% (n=131), bias -1.410. Test-minus-validation gaps: F1 -0.0010, magnitude MAE -0.0028.
- 2s: model-only median/p95 latency 0.7835/1.2780 ms. Prediction SD ratio 0.487, slope 0.239; Mâ‰¥5 underprediction 93.13% (n=131), bias -1.203. Test-minus-validation gaps: F1 -0.0024, magnitude MAE -0.0101.
- 3s: model-only median/p95 latency 0.8030/1.3195 ms. Prediction SD ratio 0.516, slope 0.282; Mâ‰¥5 underprediction 96.18% (n=131), bias -1.135. Test-minus-validation gaps: F1 -0.0096, magnitude MAE -0.0062.

Timing hardware: {'torch': '2.6.0+cu124', 'device': 'cuda', 'hardware': 'NVIDIA GeForce RTX 3050 Laptop GPU'}. All horizons use batch1,25 warmups,200 synchronized forwards on preloaded tensors. Excludes feature extraction, disk IO, device transfer, trigger and acquisition; this is network-only latency, not end-to-end compute. Oracle-aligned 1/2/3-second observation duration is not measured autonomous detection delay or operational warning lead time.

Prediction SD/slope below one indicates compression. Strong-event error and low-range bias must accompany aggregate MAE; good global metrics do not establish reliable strong-event estimates. Epoch selection may stop at the cap without convergence; no extra tuning is performed. One seed, no confidence intervals or significance testing. The held-out cohort has been inspected in prior research and is not fresh confirmatory evidence.

Noise-family shortcut check: UNKNOWN. Feature HDF5 contains no noise-profile labels; no families were invented or inferred from IDs, and upstream waveforms/metadata were not loaded. Previous CNN shortcut concerns can also affect this representation; absence of an audit here is not absence of shortcuts. Seven features include redundant energy/RMS information. No metadata input leakage or masking inconsistency was found by loader guards; original 94 feature checks were reused rather than rerun.

Selected checkpoints reload and reproduce validation loss within 1e-6. Validation/test differences are reported above. Original per-horizon populations differ; only the identical common cohort supports direct cross-horizon/model comparisons. No operational or real-data generalization claim.

Observed results: validation-F1-selected thresholds trade specificity for recall; false-positive rates are 1s 74.05%, 2s 73.97%, 3s 66.39%. These are weak detection operating points despite F1 around 0.73.

All horizons show magnitude compression (SD ratios 0.355–0.516), overprediction below magnitude 3, and 93.13–97.71% underprediction among magnitude≥5 events. Small validation/test MAE gaps do not establish real-world validity.

On identical common samples at fixed threshold .5, BiLSTM F1 exceeds CNN at 1s but is lower at 2s and 3s; accuracy is lower at all three horizons. Common-cohort magnitude MAE differs from CNN by only about +0.0052, −0.0012 and +0.0021 at 1s/2s/3s. These small untested differences do not justify a general superiority claim. No settings were changed after observing them.

## Outputs and preservation

- [plots/training_validation_loss.png](plots/training_validation_loss.png) â€” 300 DPI PNG
- [plots/actual_predicted_magnitude.png](plots/actual_predicted_magnitude.png) â€” 300 DPI PNG
- [plots/residual_distribution.png](plots/residual_distribution.png) â€” 300 DPI PNG
- [plots/confusion_matrices.png](plots/confusion_matrices.png) â€” 300 DPI PNG
- [plots/detection_f1_comparison.png](plots/detection_f1_comparison.png) â€” 300 DPI PNG
- [plots/magnitude_mae_comparison.png](plots/magnitude_mae_comparison.png) â€” 300 DPI PNG

Two executed notebooks, three best checkpoints, six plots and this report only. Checkpoints retain configuration, histories, validation-selected threshold, train-only target transforms, feature scaler metadata, input hashes and environment. No duplicate dataset or per-epoch checkpoints. Original feature HDF5, validation report and CNN reference JSON hashes match before/after; other existing files were not opened for writing.

Full numerical results follow for reproducibility without another results file.

```json
{
  "1": {
    "threshold": 0.47,
    "eligible": 2428,
    "earthquakes": 1214,
    "excluded": 0,
    "detection": {
      "accuracy": 0.6293245469522241,
      "precision": 0.5743371212121212,
      "recall": 0.999176276771005,
      "f1": 0.7294046903187011,
      "roc_auc": 0.6770930305143996,
      "pr_auc_ap": 0.658110894314985,
      "confusion_matrix": [
        [
          315,
          899
        ],
        [
          1,
          1213
        ]
      ]
    },
    "fixed_05_detection": {
      "accuracy": 0.6305601317957167,
      "precision": 0.5752969121140142,
      "recall": 0.9975288303130149,
      "f1": 0.7297378728532691,
      "roc_auc": 0.6770930305143996,
      "pr_auc_ap": 0.658110894314985,
      "confusion_matrix": [
        [
          320,
          894
        ],
        [
          3,
          1211
        ]
      ]
    },
    "magnitude": {
      "n": 1214,
      "mae": 0.678041924754713,
      "rmse": 0.8332470473886122,
      "r2": 0.14360165970224226,
      "bias": -0.0602715537811229,
      "actual_variance": 0.8107216108574445,
      "predicted_variance": 0.10229093535408357,
      "std_ratio": 0.3552079685792337,
      "slope": 0.13712756724821926,
      "actual_range": [
        2.0,
        7.199999809265137
      ],
      "predicted_range": [
        3.6250758171081543,
        6.399547100067139
      ]
    },
    "bands": {
      "M<3": {
        "n": 195,
        "mae": 1.090303897857666,
        "bias": 1.090303897857666,
        "underprediction_percent": 0.0
      },
      "3<=M<4": {
        "n": 495,
        "mae": 0.3339303731918335,
        "bias": 0.29005321860313416,
        "underprediction_percent": 18.98989898989899
      },
      "4<=M<5": {
        "n": 393,
        "mae": 0.660057783126831,
        "bias": -0.6226530075073242,
        "underprediction_percent": 93.89312977099237
      },
      "M>=5": {
        "n": 131,
        "mae": 1.4185909032821655,
        "bias": -1.4095618724822998,
        "underprediction_percent": 97.70992366412213
      }
    },
    "common": {
      "detection": {
        "accuracy": 0.6271748135874068,
        "precision": 0.5714966634890372,
        "recall": 0.9991666666666666,
        "f1": 0.727107337780473,
        "roc_auc": 0.6759157056562327,
        "pr_auc_ap": 0.6518563098718889,
        "confusion_matrix": [
          [
            315,
            899
          ],
          [
            1,
            1199
          ]
        ]
      },
      "fixed_05_detection": {
        "accuracy": 0.6284175642087821,
        "precision": 0.5724533715925395,
        "recall": 0.9975,
        "f1": 0.7274384685505926,
        "roc_auc": 0.6759157056562327,
        "pr_auc_ap": 0.6518563098718889,
        "confusion_matrix": [
          [
            320,
            894
          ],
          [
            3,
            1197
          ]
        ]
      },
      "magnitude": {
        "n": 1200,
        "mae": 0.6702129632234574,
        "rmse": 0.8242297777032527,
        "r2": 0.15961480519523186,
        "bias": -0.0767141725619634,
        "actual_variance": 0.8083849294972121,
        "predicted_variance": 0.09465369453389881,
        "std_ratio": 0.3421839858609602,
        "slope": 0.1419923562856427,
        "actual_range": [
          2.0,
          7.199999809265137
        ],
        "predicted_range": [
          3.6250758171081543,
          6.399547100067139
        ]
      },
      "identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601"
    },
    "cnn_common": {
      "n": 2414,
      "earthquakes": 1200,
      "accuracy": 0.6611433305716653,
      "precision": 0.6306429548563611,
      "recall": 0.7683333333333333,
      "f1": 0.6927122464312547,
      "roc_auc": 0.7487239154310819,
      "pr_auc_ap": 0.7472576176009302,
      "confusion_matrix": [
        [
          674,
          540
        ],
        [
          278,
          922
        ]
      ],
      "mae": 0.6650200092792511,
      "rmse": 0.8218944400551338,
      "r2": 0.1643702820960723
    },
    "common_bilstm_minus_cnn": {
      "fixed_05_accuracy": -0.03272576636288316,
      "fixed_05_f1": 0.034726222119337846,
      "mae": 0.005192953944206269,
      "rmse": 0.0023353376481188848,
      "r2": -0.0047554769008404385
    },
    "latency_ms": {
      "median": 0.7122000097297132,
      "p95": 1.1162050010170788,
      "warmups": 25,
      "repeats": 200,
      "batch_size": 1
    },
    "validation_test_gaps": {
      "f1_test_minus_val": -0.0009933796451107035,
      "mae_test_minus_val": -0.002833572471263035
    },
    "noise_family_shortcuts": "UNKNOWN: no noise_profile field in validated feature file. No invented family labels or expanded source access."
  },
  "2": {
    "threshold": 0.36,
    "eligible": 2421,
    "earthquakes": 1207,
    "excluded": 7,
    "detection": {
      "accuracy": 0.6286658405617513,
      "precision": 0.5731939163498099,
      "recall": 0.9991714995857498,
      "f1": 0.7284808215040773,
      "roc_auc": 0.7645066054822978,
      "pr_auc_ap": 0.780146193049818,
      "confusion_matrix": [
        [
          316,
          898
        ],
        [
          1,
          1206
        ]
      ]
    },
    "fixed_05_detection": {
      "accuracy": 0.674514663362247,
      "precision": 0.7020250723240116,
      "recall": 0.6031483015741508,
      "f1": 0.6488413547237076,
      "roc_auc": 0.7645066054822978,
      "pr_auc_ap": 0.780146193049818,
      "confusion_matrix": [
        [
          905,
          309
        ],
        [
          479,
          728
        ]
      ]
    },
    "magnitude": {
      "n": 1207,
      "mae": 0.638255535519212,
      "rmse": 0.7855251941142933,
      "r2": 0.23775259950675465,
      "bias": -0.0468101238750088,
      "actual_variance": 0.8095138536241766,
      "predicted_variance": 0.19214236774978052,
      "std_ratio": 0.487191190126843,
      "slope": 0.23890732490319158,
      "actual_range": [
        2.0,
        7.199999809265137
      ],
      "predicted_range": [
        3.543304681777954,
        6.364565849304199
      ]
    },
    "bands": {
      "M<3": {
        "n": 192,
        "mae": 1.005128264427185,
        "bias": 1.005128264427185,
        "underprediction_percent": 0.0
      },
      "3<=M<4": {
        "n": 491,
        "mae": 0.32586392760276794,
        "bias": 0.25092604756355286,
        "underprediction_percent": 24.84725050916497
      },
      "4<=M<5": {
        "n": 393,
        "mae": 0.6516602635383606,
        "bias": -0.5474416613578796,
        "underprediction_percent": 86.51399491094148
      },
      "M>=5": {
        "n": 131,
        "mae": 1.2312073707580566,
        "bias": -1.2026304006576538,
        "underprediction_percent": 93.12977099236642
      }
    },
    "common": {
      "detection": {
        "accuracy": 0.6275890637945319,
        "precision": 0.5717691940867906,
        "recall": 0.9991666666666666,
        "f1": 0.7273278738246891,
        "roc_auc": 0.7631727073036794,
        "pr_auc_ap": 0.7775338100583221,
        "confusion_matrix": [
          [
            316,
            898
          ],
          [
            1,
            1199
          ]
        ]
      },
      "fixed_05_detection": {
        "accuracy": 0.6735708367854184,
        "precision": 0.7,
        "recall": 0.6008333333333333,
        "f1": 0.6466367713004484,
        "roc_auc": 0.7631727073036794,
        "pr_auc_ap": 0.7775338100583221,
        "confusion_matrix": [
          [
            905,
            309
          ],
          [
            479,
            721
          ]
        ]
      },
      "magnitude": {
        "n": 1200,
        "mae": 0.6341219729185105,
        "rmse": 0.7809729622197167,
        "r2": 0.24550947764750186,
        "bias": -0.05493990282217662,
        "actual_variance": 0.8083849294972121,
        "predicted_variance": 0.18797898592663384,
        "std_ratio": 0.48222036448205485,
        "slope": 0.24088990678608346,
        "actual_range": [
          2.0,
          7.199999809265137
        ],
        "predicted_range": [
          3.543304681777954,
          6.364565849304199
        ]
      },
      "identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601"
    },
    "cnn_common": {
      "n": 2414,
      "earthquakes": 1200,
      "accuracy": 0.7245236122618062,
      "precision": 0.7407740774077408,
      "recall": 0.6858333333333333,
      "f1": 0.7122457810471657,
      "roc_auc": 0.8255271828665569,
      "pr_auc_ap": 0.8455008164565294,
      "confusion_matrix": [
        [
          926,
          288
        ],
        [
          377,
          823
        ]
      ],
      "mae": 0.6352955295642218,
      "rmse": 0.783163981207754,
      "r2": 0.2412700941336703
    },
    "common_bilstm_minus_cnn": {
      "fixed_05_accuracy": -0.05095277547638777,
      "fixed_05_f1": -0.06560900974671735,
      "mae": -0.0011735566457112823,
      "rmse": -0.0021910189880373077,
      "r2": 0.004239383513831574
    },
    "latency_ms": {
      "median": 0.7834500138415024,
      "p95": 1.278040003671776,
      "warmups": 25,
      "repeats": 200,
      "batch_size": 1
    },
    "validation_test_gaps": {
      "f1_test_minus_val": -0.0023699202458469237,
      "mae_test_minus_val": -0.010085808138042607
    },
    "noise_family_shortcuts": "UNKNOWN: no noise_profile field in validated feature file. No invented family labels or expanded source access."
  },
  "3": {
    "threshold": 0.36,
    "eligible": 2414,
    "earthquakes": 1200,
    "excluded": 14,
    "detection": {
      "accuracy": 0.648301574150787,
      "precision": 0.5894039735099338,
      "recall": 0.9641666666666666,
      "f1": 0.7315839392981347,
      "roc_auc": 0.8132420373421198,
      "pr_auc_ap": 0.8346043836399112,
      "confusion_matrix": [
        [
          408,
          806
        ],
        [
          43,
          1157
        ]
      ]
    },
    "fixed_05_detection": {
      "accuracy": 0.7141673570836785,
      "precision": 0.7405660377358491,
      "recall": 0.6541666666666667,
      "f1": 0.6946902654867256,
      "roc_auc": 0.8132420373421198,
      "pr_auc_ap": 0.8346043836399112,
      "confusion_matrix": [
        [
          939,
          275
        ],
        [
          415,
          785
        ]
      ]
    },
    "magnitude": {
      "n": 1200,
      "mae": 0.6150817563136418,
      "rmse": 0.7584208797158002,
      "r2": 0.2884550295282664,
      "bias": -0.08211152732372284,
      "actual_variance": 0.8083849294972121,
      "predicted_variance": 0.2152688049281349,
      "std_ratio": 0.5160377175516065,
      "slope": 0.28154520819618056,
      "actual_range": [
        2.0,
        7.199999809265137
      ],
      "predicted_range": [
        3.370150327682495,
        6.515814781188965
      ]
    },
    "bands": {
      "M<3": {
        "n": 190,
        "mae": 0.9331105947494507,
        "bias": 0.9331105947494507,
        "underprediction_percent": 0.0
      },
      "3<=M<4": {
        "n": 487,
        "mae": 0.31637778878211975,
        "bias": 0.19970397651195526,
        "underprediction_percent": 29.774127310061605
      },
      "4<=M<5": {
        "n": 392,
        "mae": 0.6523606181144714,
        "bias": -0.5725661516189575,
        "underprediction_percent": 87.5
      },
      "M>=5": {
        "n": 131,
        "mae": 1.152715802192688,
        "bias": -1.134616494178772,
        "underprediction_percent": 96.18320610687023
      }
    },
    "common": {
      "detection": {
        "accuracy": 0.648301574150787,
        "precision": 0.5894039735099338,
        "recall": 0.9641666666666666,
        "f1": 0.7315839392981347,
        "roc_auc": 0.8132420373421198,
        "pr_auc_ap": 0.8346043836399112,
        "confusion_matrix": [
          [
            408,
            806
          ],
          [
            43,
            1157
          ]
        ]
      },
      "fixed_05_detection": {
        "accuracy": 0.7141673570836785,
        "precision": 0.7405660377358491,
        "recall": 0.6541666666666667,
        "f1": 0.6946902654867256,
        "roc_auc": 0.8132420373421198,
        "pr_auc_ap": 0.8346043836399112,
        "confusion_matrix": [
          [
            939,
            275
          ],
          [
            415,
            785
          ]
        ]
      },
      "magnitude": {
        "n": 1200,
        "mae": 0.6150817563136418,
        "rmse": 0.7584208797158002,
        "r2": 0.2884550295282664,
        "bias": -0.08211152732372284,
        "actual_variance": 0.8083849294972121,
        "predicted_variance": 0.2152688049281349,
        "std_ratio": 0.5160377175516065,
        "slope": 0.28154520819618056,
        "actual_range": [
          2.0,
          7.199999809265137
        ],
        "predicted_range": [
          3.370150327682495,
          6.515814781188965
        ]
      },
      "identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601"
    },
    "cnn_common": {
      "n": 2414,
      "earthquakes": 1200,
      "accuracy": 0.7667771333885667,
      "precision": 0.7836153161175423,
      "recall": 0.7333333333333333,
      "f1": 0.7576409814894532,
      "roc_auc": 0.8584740527182867,
      "pr_auc_ap": 0.882497776373032,
      "confusion_matrix": [
        [
          971,
          243
        ],
        [
          320,
          880
        ]
      ],
      "mae": 0.6130052047967911,
      "rmse": 0.7577022528451374,
      "r2": 0.2898028117326791
    },
    "common_bilstm_minus_cnn": {
      "fixed_05_accuracy": -0.05260977630488817,
      "fixed_05_f1": -0.06295071600272761,
      "mae": 0.002076551516850711,
      "rmse": 0.0007186268706628196,
      "r2": -0.001347782204412673
    },
    "latency_ms": {
      "median": 0.8029500022530556,
      "p95": 1.3194650033256037,
      "warmups": 25,
      "repeats": 200,
      "batch_size": 1
    },
    "validation_test_gaps": {
      "f1_test_minus_val": -0.00963011820985249,
      "mae_test_minus_val": -0.006180788590228303
    },
    "noise_family_shortcuts": "UNKNOWN: no noise_profile field in validated feature file. No invented family labels or expanded source access."
  }
}
```

**STOP: no fusion, Attention, further tuning or cleanup.**
