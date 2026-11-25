# CNN + BiLSTM fusion v1

**Date:** 2026-11-13. Three heads trained and evaluated; no Attention.

## Architecture and fixed protocol

Exact standalone architecture classes imported from their notebook ASTs without executing original training code. Each horizon loads its own best CNN/BiLSTM weights, frozen with requires_grad=False and eval mode (including backbone dropout). Forward hooks capture each 32-dimensional shared/penultimate representation. Only these learned embeddings are concatenated, not raw inputs or standalone predictions. Embeddings are cached in memory, never saved as a duplicate dataset.

The 64-dimensional concatenation is standardized by eligible training-embedding means/SD (SD<=1e-8 uses 1); frozen transforms apply to validation/test. Fusion MLP: Linear(64,32), SiLU, dropout .2, then separate binary-logit and magnitude heads. AdamW .001, weight_decay .0001; BCE plus masked Huber(delta1) with train-only eligible magnitude mean/SD; gradient norm5; seed42; maximum35 epochs, patience7, minimum validation improvement .0001. Validation criterion is full-set mean BCE plus eligible-EQ mean Huber. No noise magnitude sentinel enters regression.

Fusion threshold policy: validation-only F1 grid .10:.01:.90, tie closest .5 then lower. Low-FPR operating point prespecified as <=5% empirical validation noise FPR, using the negative-score order-statistic boundary; ties can make it conservative. Test FPR is measured, not guaranteed <=5%. Both policies frozen before test. Fixed .5 is the only direct baseline threshold comparison.

Trainable parameters per head: 2146. Environment: {'torch': '2.6.0+cu124', 'device': 'cuda', 'hardware': 'NVIDIA GeForce RTX 3050 Laptop GPU'}.

## Mandatory alignment

Parent-ID keyed join with exact source/pair/combined ID, labels, magnitudes and all horizon eligibility/regression-mask equality: `{"train": {"records": 11324, "reordered_tabular_rows": 0, "parent_order_sha256": "102a89512fab6999895e4f709c07a88ff339b20989845845f947abbf5ae39df8"}, "validation": {"records": 2426, "reordered_tabular_rows": 0, "parent_order_sha256": "24a43418df3eaec8567d0bf709cff92984261fbd2df17796f11c741e260e8f64"}, "test": {"records": 2428, "reordered_tabular_rows": 0, "parent_order_sha256": "07c764593ab688f9e64026992ced3ec3161dfe5459482e064fd86106bd32e134"}}`. Cross-split parent intersections zero. Sampling100Hz, shared100-sample pre-context, waveform200/300/400 samples, tabular20/30/40 bins verified. Tabular records are explicitly reordered by parent lookup, even where resulting order is identical. Alignment failure raises before training. No arrival values, true magnitude, identity or masks are network input features.

## Selected fusion checkpoints

| Horizon | Best / run epochs | Validation loss | F1 threshold | Low-FPR threshold | Fit seconds | Checkpoint |
|---|---:|---:|---:|---:|---:|---|
| 1s | 2 / 9 | 0.9279768 | 0.42 | 0.674369335 | 5.19 | models/fusion_1s_best.pt |
| 2s | 15 / 22 | 0.8038182 | 0.41 | 0.712020636 | 13.40 | models/fusion_2s_best.pt |
| 3s | 27 / 34 | 0.7354916 | 0.36 | 0.692197263 | 19.98 | models/fusion_3s_best.pt |

Best head weights only are saved, with frozen-backbone paths/hashes, embedding scalers, target transforms, thresholds, histories and configuration. Standalone weights are not duplicated in these small checkpoints; reconstruct using the exact referenced source checkpoints. Weights checked tensor-exact after training and all six original checkpoint file hashes checked before/after.

## Identical common-cohort results

All models evaluated on the same 1,200 earthquakes and 1,214 noise records, with ordered source-ID hash matching the audited CNN. Regression covers all 1,200 eligible earthquakes, independent of detection success. Baseline fixed-.5 accuracy/precision/recall/F1 and magnitude MAE reproduced within1e-6. No standalone retraining or full audit repeated.

| Horizon | Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC (AP) | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | CNN | 0.661143 | 0.630643 | 0.768333 | 0.692712 | 0.444811 | 0.748724 | 0.747258 | 0.665020 | 0.821894 | 0.164370 |
| 1s | BiLSTM | 0.628418 | 0.572453 | 0.997500 | 0.727438 | 0.736409 | 0.675916 | 0.651856 | 0.670213 | 0.824230 | 0.159615 |
| 1s | Fusion | 0.652858 | 0.627645 | 0.741667 | 0.679908 | 0.434926 | 0.746584 | 0.744865 | 0.664240 | 0.817358 | 0.173570 |
| 2s | CNN | 0.724524 | 0.740774 | 0.685833 | 0.712246 | 0.237232 | 0.825527 | 0.845501 | 0.635296 | 0.783164 | 0.241270 |
| 2s | BiLSTM | 0.673571 | 0.700000 | 0.600833 | 0.646637 | 0.254530 | 0.763173 | 0.777534 | 0.634122 | 0.780973 | 0.245509 |
| 2s | Fusion | 0.734880 | 0.765152 | 0.673333 | 0.716312 | 0.204283 | 0.831996 | 0.849300 | 0.635879 | 0.783324 | 0.240961 |
| 3s | CNN | 0.766777 | 0.783615 | 0.733333 | 0.757641 | 0.200165 | 0.858474 | 0.882498 | 0.613005 | 0.757702 | 0.289803 |
| 3s | BiLSTM | 0.714167 | 0.740566 | 0.654167 | 0.694690 | 0.226524 | 0.813242 | 0.834604 | 0.615082 | 0.758421 | 0.288455 |
| 3s | Fusion | 0.772162 | 0.832311 | 0.678333 | 0.747475 | 0.135091 | 0.863415 | 0.886098 | 0.608215 | 0.753055 | 0.298488 |

## Fusion operating points

| Horizon | Policy | Threshold | Precision | Recall | F1 | Test FPR | Confusion matrix [noise,EQ] |
|---|---|---:|---:|---:|---:|---:|---|
| 1s | Fixed | 0.500000000 | 0.627645 | 0.741667 | 0.679908 | 0.434926 | [[686, 528], [310, 890]] |
| 1s | Validation F1 | 0.420000000 | 0.576453 | 0.983333 | 0.726825 | 0.714168 | [[347, 867], [20, 1180]] |
| 1s | Validation FPR<=.05 | 0.674369335 | 0.845921 | 0.233333 | 0.365774 | 0.042010 | [[1163, 51], [920, 280]] |

1s low-FPR validation precision/recall/FPR: 0.827586 / 0.237428 / 0.049464.

| 2s | Fixed | 0.500000000 | 0.765152 | 0.673333 | 0.716312 | 0.204283 | [[966, 248], [392, 808]] |
| 2s | Validation F1 | 0.410000000 | 0.691860 | 0.793333 | 0.739130 | 0.349259 | [[790, 424], [248, 952]] |
| 2s | Validation FPR<=.05 | 0.712020636 | 0.910277 | 0.465000 | 0.615554 | 0.045305 | [[1159, 55], [642, 558]] |

2s low-FPR validation precision/recall/FPR: 0.906687 / 0.482217 / 0.049464.

| 3s | Fixed | 0.500000000 | 0.832311 | 0.678333 | 0.747475 | 0.135091 | [[1050, 164], [386, 814]] |
| 3s | Validation F1 | 0.360000000 | 0.721033 | 0.814167 | 0.764775 | 0.311367 | [[836, 378], [223, 977]] |
| 3s | Validation FPR<=.05 | 0.692197263 | 0.923592 | 0.574167 | 0.708119 | 0.046952 | [[1157, 57], [511, 689]] |

3s low-FPR validation precision/recall/FPR: 0.920213 / 0.575229 / 0.049464.


## Paired evidence and bias

Differences below are fusion minus baseline, at fixed .5 for F1, same EQ targets for MAE. 500 paired stratified bootstrap resamples, seed2026, provide descriptive percentile95% intervals. Negative MAE difference and positive F1 difference favor fusion; intervals crossing zero do not establish a robust gain. One training seed, reused exploratory test cohort, and no multiple-comparison adjustment: no general or operational superiority claim.

| Horizon | Baseline | F1 difference [CI95] | MAE difference [CI95] |
|---|---|---|---|
| 1s | CNN | -0.012804 [-0.023792306807676085, -3.167606590949191e-05] | -0.000780 [-0.0034688999352511015, 0.0017248650634428485] |
| 1s | BiLSTM | -0.047530 [-0.06428662499380051, -0.03231501238265269] | -0.005973 [-0.015197540121152996, 0.0032984839810524117] |
| 2s | CNN | 0.004066 [-0.0051605001703711765, 0.014231114089397819] | 0.000583 [-0.004587645235005766, 0.005602469923906027] |
| 2s | BiLSTM | 0.069675 [0.04497768064970756, 0.09491167683694436] | 0.001757 [-0.0046189094544388356, 0.008094231225550173] |
| 3s | CNN | -0.010166 [-0.02269079653232397, -0.000400259480366555] | -0.004790 [-0.011949946149252355, 0.0021605384710710472] |
| 3s | BiLSTM | 0.052784 [0.031601839949213034, 0.0756103546455666] | -0.006867 [-0.014054152299650013, 0.001418869048939083] |

| Horizon | Fusion bias | Pred/actual SD | Slope | Actual/predicted variance | M>=5 N | Underprediction % | Strong bias |
|---|---:|---:|---:|---|---:|---:|---:|
| 1s | -0.074875 | 0.39573 | 0.16855 | 0.808385/0.126593 | 131 | 98.473 | -1.364506 |
| 2s | -0.067043 | 0.48983 | 0.24323 | 0.808385/0.193961 | 131 | 93.893 | -1.218227 |
| 3s | -0.061756 | 0.56992 | 0.31401 | 0.808385/0.262567 | 131 | 92.366 | -1.047271 |

## Observed outcome

Fusion does not show a consistent advantage over the frozen CNN. Fixed-.5 F1 is lower at 1s and 3s; the small 2s F1 increase has a paired interval crossing zero. All fusion-versus-CNN MAE intervals cross zero, including the 3s point estimate improvement. Compared with BiLSTM, fixed-.5 F1 is worse at 1s and better at 2s/3s in the descriptive paired bootstrap; magnitude gains remain uncertain. These are exploratory, single-seed comparisons, not confirmatory superiority findings.

Validation-defined low-FPR thresholds yield test FPR 4.20%, 4.53%, 4.70% with recall 23.33%, 46.50%, 57.42% at 1s/2s/3s. This operating-point tradeoff should not be conflated with fixed-.5 comparisons. Magnitude predictions remain compressed (SD ratios 0.396–0.570) and underpredict 92.37–98.47% of magnitude>=5 events. No settings or thresholds were changed after these findings.

## Scientific limitations

Both branches derive from the same synthetic signal source; tabular summaries are not independent sensor observations. Oracle P/reference alignment is assumed. Eligible S masks define offline research populations, not a deployed decision rule. Operational EEW latency, autonomous picking and warning lead time remain unverified; no runtime claim is inferred from head-training speed. Noise-family shortcut status is UNKNOWN, and fusion does not prove removal of known background risks. Magnitude compression, strong-event underprediction and high false-positive rates remain risks quantified above. Differences include representation/capacity/budget effects and cannot isolate architecture alone.

The test set has already been inspected in earlier research. Architecture, budgets, thresholds and checkpoints here were chosen using only training/validation; test paired comparisons are descriptive and did not trigger tuning. All fusion checkpoint validation losses reproduced after reload within1e-6. No Attention or further model change follows this evaluation.

## Outputs and preservation

- [plots/training_validation_loss.png](plots/training_validation_loss.png) — PNG at300 DPI
- [plots/confusion_matrices.png](plots/confusion_matrices.png) — PNG at300 DPI
- [plots/actual_predicted_magnitude.png](plots/actual_predicted_magnitude.png) — PNG at300 DPI
- [plots/magnitude_residuals.png](plots/magnitude_residuals.png) — PNG at300 DPI
- [plots/three_model_f1.png](plots/three_model_f1.png) — PNG at300 DPI
- [plots/three_model_mae.png](plots/three_model_mae.png) — PNG at300 DPI

All 17 original dependencies hashed unchanged, including all six standalone checkpoints, both datasets and original reports/notebooks. Writes confined to fusion/cnn_bilstm_v1. No waveforms/features regenerated, no full datasets copied, no existing file modified/deleted/moved. Complete hashes are embedded in each fusion checkpoint.

Complete evaluation values, operating points and paired intervals:

```json
{
  "1": {
    "common_identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601",
    "n": 2414,
    "earthquakes": 1200,
    "noise": 1214,
    "models": {
      "CNN": {
        "fixed_05": {
          "accuracy": 0.6611433305716653,
          "precision": 0.6306429548563611,
          "recall": 0.7683333333333333,
          "f1": 0.6927122464312547,
          "false_positive_rate": 0.44481054365733114,
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
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6650200092792511,
          "rmse": 0.8218944400551338,
          "r2": 0.1643702820960723,
          "bias": -0.10166003902753194,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.13666959768247441,
          "std_ratio": 0.4111751445269636,
          "slope": 0.17310987000666828,
          "strong_n": 131,
          "strong_underprediction_percent": 97.70992366412213,
          "strong_bias": -1.3813914979687174
        }
      },
      "BiLSTM": {
        "fixed_05": {
          "accuracy": 0.6284175642087821,
          "precision": 0.5724533715925395,
          "recall": 0.9975,
          "f1": 0.7274384685505926,
          "false_positive_rate": 0.7364085667215815,
          "roc_auc": 0.6759157056562328,
          "pr_auc_ap": 0.651856309871889,
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
          "strong_n": 131,
          "strong_underprediction_percent": 97.70992366412213,
          "strong_bias": -1.4095619434618767
        }
      },
      "Fusion": {
        "fixed_05": {
          "accuracy": 0.6528583264291632,
          "precision": 0.6276445698166432,
          "recall": 0.7416666666666667,
          "f1": 0.6799083269671505,
          "false_positive_rate": 0.43492586490939045,
          "roc_auc": 0.746583607907743,
          "pr_auc_ap": 0.7448651905693098,
          "confusion_matrix": [
            [
              686,
              528
            ],
            [
              310,
              890
            ]
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6642395887772242,
          "rmse": 0.8173578412873941,
          "r2": 0.17356964938781316,
          "bias": -0.07487517058849334,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.1265931723180357,
          "std_ratio": 0.3957273231039049,
          "slope": 0.16855246945375862,
          "strong_n": 131,
          "strong_underprediction_percent": 98.47328244274809,
          "strong_bias": -1.3645059771210182
        }
      }
    },
    "fusion_validation_threshold": 0.42,
    "fusion_selected": {
      "accuracy": 0.6325600662800331,
      "precision": 0.5764533463605276,
      "recall": 0.9833333333333333,
      "f1": 0.7268247613181398,
      "false_positive_rate": 0.714168039538715,
      "roc_auc": 0.746583607907743,
      "pr_auc_ap": 0.7448651905693098,
      "confusion_matrix": [
        [
          347,
          867
        ],
        [
          20,
          1180
        ]
      ]
    },
    "fusion_low_fpr_threshold": 0.6743693351745605,
    "fusion_low_fpr_validation": {
      "accuracy": 0.5939818631492169,
      "precision": 0.8275862068965517,
      "recall": 0.23742786479802144,
      "f1": 0.3689942344650865,
      "false_positive_rate": 0.0494641384995878,
      "roc_auc": 0.7403424973613009,
      "pr_auc_ap": 0.7385615443321984,
      "confusion_matrix": [
        [
          1153,
          60
        ],
        [
          925,
          288
        ]
      ]
    },
    "fusion_low_fpr_test": {
      "accuracy": 0.5977630488815244,
      "precision": 0.8459214501510574,
      "recall": 0.23333333333333334,
      "f1": 0.36577400391900716,
      "false_positive_rate": 0.042009884678747944,
      "roc_auc": 0.746583607907743,
      "pr_auc_ap": 0.7448651905693098,
      "confusion_matrix": [
        [
          1163,
          51
        ],
        [
          920,
          280
        ]
      ]
    },
    "paired_comparisons": {
      "CNN": {
        "fusion_minus_baseline_f1": -0.012803919464104196,
        "fusion_minus_baseline_mae": -0.0007804205020268684,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.023792306807676085,
          -3.167606590949191e-05
        ],
        "mae_difference_ci95": [
          -0.0034688999352511015,
          0.0017248650634428485
        ]
      },
      "BiLSTM": {
        "fusion_minus_baseline_f1": -0.04753014158344204,
        "fusion_minus_baseline_mae": -0.005973374446233137,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.06428662499380051,
          -0.03231501238265269
        ],
        "mae_difference_ci95": [
          -0.015197540121152996,
          0.0032984839810524117
        ]
      }
    },
    "baseline_threshold_policy": "fixed .5; selected fusion and low-FPR metrics reported separately",
    "noise_family_shortcuts": "UNKNOWN; not established by this experiment"
  },
  "2": {
    "common_identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601",
    "n": 2414,
    "earthquakes": 1200,
    "noise": 1214,
    "models": {
      "CNN": {
        "fixed_05": {
          "accuracy": 0.7245236122618062,
          "precision": 0.7407740774077408,
          "recall": 0.6858333333333333,
          "f1": 0.7122457810471657,
          "false_positive_rate": 0.2372322899505766,
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
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6352955295642218,
          "rmse": 0.783163981207754,
          "r2": 0.2412700941336703,
          "bias": -0.06579678634802501,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.23993120480157107,
          "std_ratio": 0.5447964427161235,
          "slope": 0.27171432439033805,
          "strong_n": 131,
          "strong_underprediction_percent": 94.65648854961832,
          "strong_bias": -1.162960511127501
        }
      },
      "BiLSTM": {
        "fixed_05": {
          "accuracy": 0.6735708367854184,
          "precision": 0.7,
          "recall": 0.6008333333333333,
          "f1": 0.6466367713004484,
          "false_positive_rate": 0.2545304777594728,
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
          "strong_n": 131,
          "strong_underprediction_percent": 93.12977099236642,
          "strong_bias": -1.2026304106675942
        }
      },
      "Fusion": {
        "fixed_05": {
          "accuracy": 0.7348798674399337,
          "precision": 0.7651515151515151,
          "recall": 0.6733333333333333,
          "f1": 0.7163120567375887,
          "false_positive_rate": 0.2042833607907743,
          "roc_auc": 0.8319961559582647,
          "pr_auc_ap": 0.8492995205727536,
          "confusion_matrix": [
            [
              966,
              248
            ],
            [
              392,
              808
            ]
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6358787045876185,
          "rmse": 0.78332360405291,
          "r2": 0.24096077712869313,
          "bias": -0.06704320053259531,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.19396078441340306,
          "std_ratio": 0.4898327986906324,
          "slope": 0.2432285793764195,
          "strong_n": 131,
          "strong_underprediction_percent": 93.89312977099237,
          "strong_bias": -1.2182269296573318
        }
      }
    },
    "fusion_validation_threshold": 0.41,
    "fusion_selected": {
      "accuracy": 0.7216238608119304,
      "precision": 0.6918604651162791,
      "recall": 0.7933333333333333,
      "f1": 0.7391304347826086,
      "false_positive_rate": 0.34925864909390447,
      "roc_auc": 0.8319961559582647,
      "pr_auc_ap": 0.8492995205727536,
      "confusion_matrix": [
        [
          790,
          424
        ],
        [
          248,
          952
        ]
      ]
    },
    "fusion_low_fpr_threshold": 0.7120206356048584,
    "fusion_low_fpr_validation": {
      "accuracy": 0.7167630057803468,
      "precision": 0.9066874027993779,
      "recall": 0.48221670802315963,
      "f1": 0.6295896328293736,
      "false_positive_rate": 0.0494641384995878,
      "roc_auc": 0.8411317427619318,
      "pr_auc_ap": 0.8594032697929179,
      "confusion_matrix": [
        [
          1153,
          60
        ],
        [
          626,
          583
        ]
      ]
    },
    "fusion_low_fpr_test": {
      "accuracy": 0.7112676056338029,
      "precision": 0.9102773246329527,
      "recall": 0.465,
      "f1": 0.6155543298400441,
      "false_positive_rate": 0.045304777594728174,
      "roc_auc": 0.8319961559582647,
      "pr_auc_ap": 0.8492995205727536,
      "confusion_matrix": [
        [
          1159,
          55
        ],
        [
          642,
          558
        ]
      ]
    },
    "paired_comparisons": {
      "CNN": {
        "fusion_minus_baseline_f1": 0.004066275690422949,
        "fusion_minus_baseline_mae": 0.0005831750233967758,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.0051605001703711765,
          0.014231114089397819
        ],
        "mae_difference_ci95": [
          -0.004587645235005766,
          0.005602469923906027
        ]
      },
      "BiLSTM": {
        "fusion_minus_baseline_f1": 0.0696752854371403,
        "fusion_minus_baseline_mae": 0.0017567316691080581,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          0.04497768064970756,
          0.09491167683694436
        ],
        "mae_difference_ci95": [
          -0.0046189094544388356,
          0.008094231225550173
        ]
      }
    },
    "baseline_threshold_policy": "fixed .5; selected fusion and low-FPR metrics reported separately",
    "noise_family_shortcuts": "UNKNOWN; not established by this experiment"
  },
  "3": {
    "common_identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601",
    "n": 2414,
    "earthquakes": 1200,
    "noise": 1214,
    "models": {
      "CNN": {
        "fixed_05": {
          "accuracy": 0.7667771333885667,
          "precision": 0.7836153161175423,
          "recall": 0.7333333333333333,
          "f1": 0.7576409814894532,
          "false_positive_rate": 0.200164744645799,
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
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6130052047967911,
          "rmse": 0.7577022528451374,
          "r2": 0.2898028117326791,
          "bias": -0.09076991856098175,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.21333789166312078,
          "std_ratio": 0.5137181325619806,
          "slope": 0.2819506392782616,
          "strong_n": 131,
          "strong_underprediction_percent": 96.94656488549617,
          "strong_bias": -1.1642182965315024
        }
      },
      "BiLSTM": {
        "fixed_05": {
          "accuracy": 0.7141673570836785,
          "precision": 0.7405660377358491,
          "recall": 0.6541666666666667,
          "f1": 0.6946902654867256,
          "false_positive_rate": 0.22652388797364087,
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
          "strong_n": 131,
          "strong_underprediction_percent": 96.18320610687023,
          "strong_bias": -1.1346164987287448
        }
      },
      "Fusion": {
        "fixed_05": {
          "accuracy": 0.772162386081193,
          "precision": 0.8323108384458078,
          "recall": 0.6783333333333333,
          "f1": 0.7474747474747475,
          "false_positive_rate": 0.13509060955518945,
          "roc_auc": 0.8634150192202087,
          "pr_auc_ap": 0.8860976972692773,
          "confusion_matrix": [
            [
              1050,
              164
            ],
            [
              386,
              814
            ]
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6082151192426681,
          "rmse": 0.753054914220647,
          "r2": 0.2984880307150485,
          "bias": -0.061756018996238705,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.2625674167193923,
          "std_ratio": 0.5699166059782849,
          "slope": 0.3140053888577295,
          "strong_n": 131,
          "strong_underprediction_percent": 92.36641221374046,
          "strong_bias": -1.0472708840406577
        }
      }
    },
    "fusion_validation_threshold": 0.36,
    "fusion_selected": {
      "accuracy": 0.7510356255178128,
      "precision": 0.7210332103321033,
      "recall": 0.8141666666666667,
      "f1": 0.764774951076321,
      "false_positive_rate": 0.3113673805601318,
      "roc_auc": 0.8634150192202087,
      "pr_auc_ap": 0.8860976972692773,
      "confusion_matrix": [
        [
          836,
          378
        ],
        [
          223,
          977
        ]
      ]
    },
    "fusion_low_fpr_threshold": 0.6921972632408142,
    "fusion_low_fpr_validation": {
      "accuracy": 0.7636589403973509,
      "precision": 0.9202127659574468,
      "recall": 0.5752285951787198,
      "f1": 0.7079283887468031,
      "false_positive_rate": 0.0494641384995878,
      "roc_auc": 0.8636830567165488,
      "pr_auc_ap": 0.8861262646475905,
      "confusion_matrix": [
        [
          1153,
          60
        ],
        [
          511,
          692
        ]
      ]
    },
    "fusion_low_fpr_test": {
      "accuracy": 0.7647058823529411,
      "precision": 0.9235924932975871,
      "recall": 0.5741666666666667,
      "f1": 0.7081192189105858,
      "false_positive_rate": 0.04695222405271829,
      "roc_auc": 0.8634150192202087,
      "pr_auc_ap": 0.8860976972692773,
      "confusion_matrix": [
        [
          1157,
          57
        ],
        [
          511,
          689
        ]
      ]
    },
    "paired_comparisons": {
      "CNN": {
        "fusion_minus_baseline_f1": -0.010166234014705733,
        "fusion_minus_baseline_mae": -0.004790085554122969,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.02269079653232397,
          -0.000400259480366555
        ],
        "mae_difference_ci95": [
          -0.011949946149252355,
          0.0021605384710710472
        ]
      },
      "BiLSTM": {
        "fusion_minus_baseline_f1": 0.05278448198802188,
        "fusion_minus_baseline_mae": -0.00686663707097368,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          0.031601839949213034,
          0.0756103546455666
        ],
        "mae_difference_ci95": [
          -0.014054152299650013,
          0.001418869048939083
        ]
      }
    },
    "baseline_threshold_policy": "fixed .5; selected fusion and low-FPR metrics reported separately",
    "noise_family_shortcuts": "UNKNOWN; not established by this experiment"
  }
}
```

**STOP after fusion v1. Independent audit required before Attention.**
