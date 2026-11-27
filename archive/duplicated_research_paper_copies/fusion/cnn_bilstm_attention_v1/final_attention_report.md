# CNN + BiLSTM + temporal attention v1

**Scope:** synthetic oracle-P-aligned research; three models trained and evaluated.

## Architecture and prespecified protocol

Frozen horizon-specific CNN learned 32D shared embedding plus ALL bidirectional LSTM hidden states (20/30/40 steps, 64D each). Additive attention Linear(64,32), tanh, Linear(32,1), temporal softmax, weighted sum of states; concatenate with CNN to 96D; Linear(96,32), SiLU, dropout .2 and separate binary-logit/magnitude heads. The original BiLSTM shared pooling/head is bypassed, not changed. Bidirectionality uses only the fully observed window; attention is not an online prefix-invariant representation before that cutoff.

Per-channel CNN and hidden-state scaling fitted exclusively on eligible training records (hidden-state moments across records/time); fixed on validation/test. Existing seven-feature train-only normalization reused. BCE plus earthquake-only standardized Huber, AdamW lr .001/weight decay .0001, gradient clip5, seed42, maximum35 epochs, patience7, min_delta .0001. Selection minimizes full validation BCE plus earthquake mean Huber. Validation F1 threshold grid .10:.01:.90, ties closest .5 then lower. All three checkpoints saved before test inference. No test-driven tuning.

Exact parent-key joins, all source/label/magnitude/mask equality and split-parent isolation checked before training. No target/arrival/ID metadata enters either network. Per-horizon pre-S masks filter train/validation; test uses shared 1,200 earthquakes and 1,214 noise. Regression includes every eligible earthquake independent of detection. All baseline fixed-.5 F1 and MAE reproduce within1e-6 on the identical ordered source-ID cohort.

## Selected checkpoints

| Horizon | Best/run epochs | Validation loss | Threshold | Parameters | Training seconds |
|---|---:|---:|---:|---:|---:|
| 1s | 3/10 | 0.93035614 | 0.28 | 5282 | 9.02 |
| 2s | 17/24 | 0.80256573 | 0.41 | 5282 | 21.37 |
| 3s | 32/35 | 0.73184561 | 0.41 | 5282 | 33.57 |

Checkpoints: `models/attention_1s_best.pt`, `models/attention_2s_best.pt`, `models/attention_3s_best.pt`. These contain attention/head weights, train-fitted scalers, histories, thresholds and original dependency hashes; original backbone weights are reused without duplication. Reloaded validation losses match within1e-6.

## Four-model comparison (fixed 0.5 throughout)

| Horizon | Model | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC(AP) | MAE | RMSE | R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | CNN | 0.661143 | 0.630643 | 0.768333 | 0.692712 | 0.444811 | 0.748724 | 0.747258 | 0.665020 | 0.821894 | 0.164370 |
| 1s | BiLSTM | 0.628418 | 0.572453 | 0.997500 | 0.727438 | 0.736409 | 0.675916 | 0.651856 | 0.670213 | 0.824230 | 0.159615 |
| 1s | Fusion | 0.652858 | 0.627645 | 0.741667 | 0.679908 | 0.434926 | 0.746584 | 0.744865 | 0.664240 | 0.817358 | 0.173570 |
| 1s | Attention | 0.663215 | 0.651290 | 0.694167 | 0.672045 | 0.367381 | 0.751284 | 0.748852 | 0.666359 | 0.818941 | 0.170365 |
| 2s | CNN | 0.724524 | 0.740774 | 0.685833 | 0.712246 | 0.237232 | 0.825527 | 0.845501 | 0.635296 | 0.783164 | 0.241270 |
| 2s | BiLSTM | 0.673571 | 0.700000 | 0.600833 | 0.646637 | 0.254530 | 0.763173 | 0.777534 | 0.634122 | 0.780973 | 0.245509 |
| 2s | Fusion | 0.734880 | 0.765152 | 0.673333 | 0.716312 | 0.204283 | 0.831996 | 0.849300 | 0.635879 | 0.783324 | 0.240961 |
| 2s | Attention | 0.735708 | 0.766098 | 0.674167 | 0.717199 | 0.203460 | 0.832168 | 0.849352 | 0.636489 | 0.782980 | 0.241627 |
| 3s | CNN | 0.766777 | 0.783615 | 0.733333 | 0.757641 | 0.200165 | 0.858474 | 0.882498 | 0.613005 | 0.757702 | 0.289803 |
| 3s | BiLSTM | 0.714167 | 0.740566 | 0.654167 | 0.694690 | 0.226524 | 0.813242 | 0.834604 | 0.615082 | 0.758421 | 0.288455 |
| 3s | Fusion | 0.772162 | 0.832311 | 0.678333 | 0.747475 | 0.135091 | 0.863415 | 0.886098 | 0.608215 | 0.753055 | 0.298488 |
| 3s | Attention | 0.776305 | 0.851812 | 0.665833 | 0.747428 | 0.114498 | 0.865393 | 0.887532 | 0.606601 | 0.748621 | 0.306725 |

## Attention operating points

| Horizon | Threshold | Precision | Recall | F1 | FPR | Confusion matrix [noise,EQ] |
|---|---:|---:|---:|---:|---:|---|
| 1s | 0.50 | 0.651290 | 0.694167 | 0.672045 | 0.367381 | [[768, 446], [367, 833]] |
| 1s | 0.28 | 0.571224 | 0.999167 | 0.726887 | 0.741351 | [[314, 900], [1, 1199]] |
| 2s | 0.50 | 0.766098 | 0.674167 | 0.717199 | 0.203460 | [[967, 247], [391, 809]] |
| 2s | 0.41 | 0.689130 | 0.792500 | 0.737209 | 0.353377 | [[785, 429], [249, 951]] |
| 3s | 0.50 | 0.851812 | 0.665833 | 0.747428 | 0.114498 | [[1075, 139], [401, 799]] |
| 3s | 0.41 | 0.771503 | 0.740000 | 0.755423 | 0.216639 | [[951, 263], [312, 888]] |

Validation-selected thresholds trade precision/FPR against recall; do not compare these to fixed-.5 baseline values as equivalent policies. Complete accuracy/AUC and confusion matrices for both operating points are in the embedded results.

## Magnitude compression and strong-event bias

| Horizon | Mean signed error | Pred/actual SD | Slope | M>=5 N | Underprediction % | Strong bias |
|---|---:|---:|---:|---:|---:|---:|
| 1s | -0.076292 | 0.387569 | 0.163888 | 131 | 98.473 | -1.373537 |
| 2s | -0.056138 | 0.512419 | 0.254049 | 131 | 95.420 | -1.182187 |
| 3s | -0.004881 | 0.600623 | 0.333751 | 131 | 87.023 | -0.930820 |

## Paired uncertainty

Attention-minus-baseline differences; 500 paired, class-stratified bootstrap resamples, seed2026; percentile95% intervals. Same earthquake draws for magnitude. Single-seed, exploratory, no multiplicity correction; tiny changes or intervals crossing zero do not establish improvement.

| Horizon | Baseline | F1 difference [95% CI] | MAE difference [95% CI] |
|---|---|---|---|
| 1s | CNN | -0.020667 [-0.032711352671601135, -0.009071786052482529] | 0.001339 [-0.0028259355633053927, 0.00537417507730424] |
| 1s | BiLSTM | -0.055393 [-0.07388907972236253, -0.03813968519009021] | -0.003854 [-0.013493929337710143, 0.005021539563313125] |
| 1s | Fusion | -0.007863 [-0.018330848456274, 0.001981389353165757] | 0.002119 [-0.0012190080509753896, 0.0054234042880125346] |
| 2s | CNN | 0.004953 [-0.005296183689007172, 0.01593688664743822] | 0.001194 [-0.003874334110878408, 0.005953477998264133] |
| 2s | BiLSTM | 0.070562 [0.04625659904012046, 0.09408423209782517] | 0.002367 [-0.004293993522878736, 0.009471768513321871] |
| 2s | Fusion | 0.000887 [-0.003969822205396578, 0.005982555642963445] | 0.000610 [-0.0018039557384327053, 0.002919959701830521] |
| 3s | CNN | -0.010213 [-0.023373921465093036, 0.0013025063222041066] | -0.006405 [-0.016216625086963175, 0.0034196724591311037] |
| 3s | BiLSTM | 0.052737 [0.03329790459158589, 0.0738269253297935] | -0.008481 [-0.017232834408059717, 0.0020405739516718238] |
| 3s | Fusion | -0.000047 [-0.006012117100356584, 0.0065387337626778815] | -0.001614 [-0.0063754469039849935, 0.003244647354586046] |

**Consistent improvement over non-Attention fusion:** Not established across horizons and both metrics. Report mixed/worse outcomes as measured; no settings were changed after viewing test results.

## Inference timing and attention diagnostics

Environment: {'torch': '2.6.0+cu124', 'device': 'cuda', 'hardware': 'NVIDIA GeForce RTX 3050 Laptop GPU'}

| Horizon | Warm batch-1 median ms | p95 ms |
|---|---:|---:|
| 1s | 1.4483 | 1.9144 |
| 2s | 1.4383 | 2.8161 |
| 3s | 1.4215 | 2.8192 |

Timing includes both backbones and attention/head on device-resident inputs, 10 warmups/100 repeats, synchronized GPU timing. Excludes I/O, feature extraction and transfer; not operational latency or streaming detection delay. Observation horizons are 1/2/3s post-oracle-reference with 1s preceding context.

Attention figure selects the first noise, first earthquake and first M>=5 earthquake in cohort order, independently of prediction quality. Bin centers are plotted relative to the reference; weights sum to1. These distributions are diagnostics, not proof of physical feature importance. Bidirectional states contain contextual information from throughout the available window, so a high weight cannot be assigned uniquely to an isolated physical time sample.

## Scientific limitations

Synthetic oracle-P-aligned benchmark; both branches share the same synthetic observations. Operational EEW performance, autonomous triggering and real-world generalization are unverified. High false-positive rates, magnitude compression and strong-event underprediction remain risks. Existing test cohort has been inspected in prior experiments; present comparisons are exploratory rather than fresh confirmatory evidence. Architecture/capacity differences prevent attributing all changes uniquely to attention. No further tuning, independent audit or cleanup performed.

## Outputs and preservation

- [plots/actual_predicted_magnitude.png](plots/actual_predicted_magnitude.png) (300 DPI)
- [plots/confusion_matrices.png](plots/confusion_matrices.png) (300 DPI)
- [plots/four_model_f1.png](plots/four_model_f1.png) (300 DPI)
- [plots/four_model_mae.png](plots/four_model_mae.png) (300 DPI)
- [plots/magnitude_residuals.png](plots/magnitude_residuals.png) (300 DPI)
- [plots/temporal_attention_weights.png](plots/temporal_attention_weights.png) (300 DPI)
- [plots/training_validation_loss.png](plots/training_validation_loss.png) (300 DPI)

All 17 original dependencies retain exact SHA256 hashes. Both backbones remain requires_grad=False, eval mode and tensor-identical to source checkpoints after training. Writes confined to this new output directory. No datasets regenerated or copied. Hashes and environment are in each checkpoint.

Complete results:

```json
{
  "1": {
    "best_epoch": 3,
    "common_identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601",
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
      },
      "Attention": {
        "fixed_05": {
          "accuracy": 0.6632145816072909,
          "precision": 0.6512900703674745,
          "recall": 0.6941666666666667,
          "f1": 0.672045179507866,
          "false_positive_rate": 0.3673805601317957,
          "roc_auc": 0.7512836353651839,
          "pr_auc_ap": 0.748852252157705,
          "confusion_matrix": [
            [
              768,
              446
            ],
            [
              367,
              833
            ]
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6663588509956996,
          "rmse": 0.818940840992192,
          "r2": 0.1703654081451904,
          "bias": -0.07629210571448009,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.12142718435358073,
          "std_ratio": 0.3875688444820556,
          "slope": 0.16388757912950222,
          "strong_n": 131,
          "strong_underprediction_percent": 98.47328244274809,
          "strong_bias": -1.3735367378205743
        }
      }
    },
    "validation_threshold": 0.28,
    "validation_selected_test": {
      "accuracy": 0.6267605633802817,
      "precision": 0.5712243925678895,
      "recall": 0.9991666666666666,
      "f1": 0.7268869354349803,
      "false_positive_rate": 0.7413509060955519,
      "roc_auc": 0.7512836353651839,
      "pr_auc_ap": 0.748852252157705,
      "confusion_matrix": [
        [
          314,
          900
        ],
        [
          1,
          1199
        ]
      ]
    },
    "paired_comparisons": {
      "CNN": {
        "f1_difference": -0.02066706692338871,
        "mae_difference": 0.0013388417164484911,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.032711352671601135,
          -0.009071786052482529
        ],
        "mae_difference_ci95": [
          -0.0028259355633053927,
          0.00537417507730424
        ]
      },
      "BiLSTM": {
        "f1_difference": -0.05539328904272656,
        "mae_difference": -0.0038541122277577777,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.07388907972236253,
          -0.03813968519009021
        ],
        "mae_difference_ci95": [
          -0.013493929337710143,
          0.005021539563313125
        ]
      },
      "Fusion": {
        "f1_difference": -0.007863147459284514,
        "mae_difference": 0.0021192622184753596,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.018330848456274,
          0.001981389353165757
        ],
        "mae_difference_ci95": [
          -0.0012190080509753896,
          0.0054234042880125346
        ]
      }
    },
    "latency": {
      "median_ms": 1.4483499980997294,
      "p95_ms": 1.9143900033668613,
      "repeats": 100,
      "batch_size": 1,
      "scope": "warm device-resident two-backbone plus attention inference; excludes I/O, tabular extraction and transfer; not streaming detection delay"
    },
    "attention_mean_entropy": 2.8978333473205566,
    "mean_attention_by_bin": [
      0.06772282719612122,
      0.06350068747997284,
      0.06162021681666374,
      0.06125397980213165,
      0.061122968792915344,
      0.06088811904191971,
      0.06020008772611618,
      0.05927509069442749,
      0.05784759297966957,
      0.056113362312316895,
      0.05372684821486473,
      0.05099615082144737,
      0.04782627522945404,
      0.0440465472638607,
      0.039983149617910385,
      0.03587128221988678,
      0.03206879273056984,
      0.029146980494260788,
      0.027851495891809464,
      0.02893754094839096
    ]
  },
  "2": {
    "best_epoch": 17,
    "common_identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601",
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
      },
      "Attention": {
        "fixed_05": {
          "accuracy": 0.7357083678541839,
          "precision": 0.7660984848484849,
          "recall": 0.6741666666666667,
          "f1": 0.7171985815602837,
          "false_positive_rate": 0.20345963756177923,
          "roc_auc": 0.8321677649643053,
          "pr_auc_ap": 0.8493518607704592,
          "confusion_matrix": [
            [
              967,
              247
            ],
            [
              391,
              809
            ]
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6364890929063162,
          "rmse": 0.7829799640130151,
          "r2": 0.24162660426249727,
          "bias": -0.05613822380701701,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.2122605397526771,
          "std_ratio": 0.5124193578543239,
          "slope": 0.25404935841130716,
          "strong_n": 131,
          "strong_underprediction_percent": 95.41984732824427,
          "strong_bias": -1.1821874917008495
        }
      }
    },
    "validation_threshold": 0.41,
    "validation_selected_test": {
      "accuracy": 0.7191383595691798,
      "precision": 0.6891304347826087,
      "recall": 0.7925,
      "f1": 0.7372093023255814,
      "false_positive_rate": 0.35337726523887975,
      "roc_auc": 0.8321677649643053,
      "pr_auc_ap": 0.8493518607704592,
      "confusion_matrix": [
        [
          785,
          429
        ],
        [
          249,
          951
        ]
      ]
    },
    "paired_comparisons": {
      "CNN": {
        "f1_difference": 0.004952800513117972,
        "mae_difference": 0.001193563342094417,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.005296183689007172,
          0.01593688664743822
        ],
        "mae_difference_ci95": [
          -0.003874334110878408,
          0.005953477998264133
        ]
      },
      "BiLSTM": {
        "f1_difference": 0.07056181025983532,
        "mae_difference": 0.002367119987805699,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          0.04625659904012046,
          0.09408423209782517
        ],
        "mae_difference_ci95": [
          -0.004293993522878736,
          0.009471768513321871
        ]
      },
      "Fusion": {
        "f1_difference": 0.0008865248226950229,
        "mae_difference": 0.0006103883186976411,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.003969822205396578,
          0.005982555642963445
        ],
        "mae_difference_ci95": [
          -0.0018039557384327053,
          0.002919959701830521
        ]
      }
    },
    "latency": {
      "median_ms": 1.4383000088855624,
      "p95_ms": 2.8161299997009337,
      "repeats": 100,
      "batch_size": 1,
      "scope": "warm device-resident two-backbone plus attention inference; excludes I/O, tabular extraction and transfer; not streaming detection delay"
    },
    "attention_mean_entropy": 3.3388986587524414,
    "mean_attention_by_bin": [
      0.034181032329797745,
      0.036955464631319046,
      0.038333818316459656,
      0.039078038185834885,
      0.039408065378665924,
      0.03944091126322746,
      0.03927401453256607,
      0.039034660905599594,
      0.0387437641620636,
      0.0383807010948658,
      0.03802083805203438,
      0.03765295445919037,
      0.037233226001262665,
      0.03675151243805885,
      0.03617092967033386,
      0.03545841574668884,
      0.03462877869606018,
      0.03366730734705925,
      0.03257378563284874,
      0.03135201334953308,
      0.03001449629664421,
      0.02865603379905224,
      0.027333032339811325,
      0.026104198768734932,
      0.02509211003780365,
      0.024436283856630325,
      0.024193547666072845,
      0.02452157996594906,
      0.025671884417533875,
      0.027636433020234108
    ]
  },
  "3": {
    "best_epoch": 32,
    "common_identity_sha256": "862b467ff6219c5193f4c99257a5971eea395581669302748d006faa78b78601",
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
      },
      "Attention": {
        "fixed_05": {
          "accuracy": 0.776304888152444,
          "precision": 0.8518123667377399,
          "recall": 0.6658333333333334,
          "f1": 0.7474275023386342,
          "false_positive_rate": 0.11449752883031301,
          "roc_auc": 0.865392641405821,
          "pr_auc_ap": 0.887531712083913,
          "confusion_matrix": [
            [
              1075,
              139
            ],
            [
              401,
              799
            ]
          ]
        },
        "magnitude": {
          "n": 1200,
          "mae": 0.6066006402174632,
          "rmse": 0.7486207347412538,
          "r2": 0.30672507114512293,
          "bias": -0.004880607922871908,
          "actual_variance": 0.8083849294972121,
          "predicted_variance": 0.29162310351765447,
          "std_ratio": 0.6006228625923995,
          "slope": 0.3337511803935037,
          "strong_n": 131,
          "strong_underprediction_percent": 87.02290076335878,
          "strong_bias": -0.9308202175693657
        }
      }
    },
    "validation_threshold": 0.41,
    "validation_selected_test": {
      "accuracy": 0.7618061309030655,
      "precision": 0.7715030408340573,
      "recall": 0.74,
      "f1": 0.7554232241599319,
      "false_positive_rate": 0.21663920922570015,
      "roc_auc": 0.865392641405821,
      "pr_auc_ap": 0.887531712083913,
      "confusion_matrix": [
        [
          951,
          263
        ],
        [
          312,
          888
        ]
      ]
    },
    "paired_comparisons": {
      "CNN": {
        "f1_difference": -0.01021347915081905,
        "mae_difference": -0.0064045645793279204,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.023373921465093036,
          0.0013025063222041066
        ],
        "mae_difference_ci95": [
          -0.016216625086963175,
          0.0034196724591311037
        ]
      },
      "BiLSTM": {
        "f1_difference": 0.052737236851908564,
        "mae_difference": -0.008481116096178631,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          0.03329790459158589,
          0.0738269253297935
        ],
        "mae_difference_ci95": [
          -0.017232834408059717,
          0.0020405739516718238
        ]
      },
      "Fusion": {
        "f1_difference": -4.724513611331638e-05,
        "mae_difference": -0.0016144790252049512,
        "replicates": 500,
        "seed": 2026,
        "method": "paired stratified bootstrap; same sampled EQ/noise indices for both models",
        "f1_difference_ci95": [
          -0.006012117100356584,
          0.0065387337626778815
        ],
        "mae_difference_ci95": [
          -0.0063754469039849935,
          0.003244647354586046
        ]
      }
    },
    "latency": {
      "median_ms": 1.4215499977581203,
      "p95_ms": 2.8191949866595674,
      "repeats": 100,
      "batch_size": 1,
      "scope": "warm device-resident two-backbone plus attention inference; excludes I/O, tabular extraction and transfer; not streaming detection delay"
    },
    "attention_mean_entropy": 3.638434886932373,
    "mean_attention_by_bin": [
      0.020057033747434616,
      0.022769922390580177,
      0.024680867791175842,
      0.026150725781917572,
      0.027299221605062485,
      0.028172247111797333,
      0.028807613998651505,
      0.029232796281576157,
      0.029489221051335335,
      0.02961382083594799,
      0.029546022415161133,
      0.029399722814559937,
      0.029205722734332085,
      0.028917105868458748,
      0.028563745319843292,
      0.028167329728603363,
      0.027726173400878906,
      0.02721422351896763,
      0.026705356314778328,
      0.0261541735380888,
      0.025545351207256317,
      0.024926362559199333,
      0.024322593584656715,
      0.023739799857139587,
      0.023172900080680847,
      0.022654255852103233,
      0.02218114398419857,
      0.021767325699329376,
      0.021407583728432655,
      0.02111164480447769,
      0.02091832645237446,
      0.020836181938648224,
      0.02088187262415886,
      0.021045733243227005,
      0.021323222666978836,
      0.021751387044787407,
      0.022387977689504623,
      0.023247621953487396,
      0.0241392869502306,
      0.02476630173623562
    ]
  }
}
```
