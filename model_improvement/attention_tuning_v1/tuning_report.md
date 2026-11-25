# Controlled3s Attention tuning study

## Scope and preregistered protocol

Only3s studied. Existing backbones frozen and unchanged. All trial fitting and selection used training/validation only; no test prediction used until final checkpoint selection. Existing reports/checkpoints contain no stored validation probability archive, so Phase A probabilities were reproduced from the frozen original checkpoint. Existing synthetic oracle-aligned data reused without regeneration.

Protocol recorded in the executed tuning notebook before trial execution:

```json
{
  "max_trials": 12,
  "seed": 42,
  "epochs": 35,
  "patience": 7,
  "min_delta": 0.0001,
  "threshold_grid": "0.01 through0.99 step0.01 plus0,1 and nextafter(max probability,+infinity)",
  "operating_point": "Among FPR<=0.30 and recall>=0.70 maximize F1; ties lower FPR then closer0.5. If infeasible maximize recall under FPR<=0.30; label infeasible. Also unconstrained maximum-F1 threshold reported.",
  "selection": "Gate feasible operating point, overall validation MAE <= original+0.02, strong MAE <=original+0.10. Useful improvement: FPR decreases >=0.02 with F1 >=original-0.01, OR F1 increases >=0.01 with FPR<=original+0.01, OR MAE decreases>=0.02 with FPR<=original+0.01 and F1>=original-0.01. Within useful candidates lexicographic lowest FPR, highest F1, lowest overall MAE, lowest strong MAE, lowest strong-underprediction percentage, trial ID. Otherwise retain original.",
  "variants": "If selected phase-B configuration validation SD ratio<0.8, train exactly two from scratch with same configuration: strong_bin (Huber weight2 for training magnitude>=5), asymmetric (Huber weight2 when training magnitude>=5 and predicted<target). Phase B reference uses same feasible/error gates and lexicographic ranking, falling back to lowest overall validation MAE if none qualify. No validation examples in loss.",
  "checkpoint_selection": "Unweighted validation BCE + standardized masked Huber for every trial; optimizer uses predefined regression weight/variant; identical fixed checkpoint criterion."
}
```

Ten fixed Phase-B configurations cover learning rates1e-4/3e-4/1e-3, dropout0.1–0.4, hidden64/128, regression weights0.5/1/2. Up to two Phase-C variants are included in the12-trial cap, not extra trials. Seed42, batch256, AdamW weight decay1e-4, clipping5, maximum35 epochs/patience7. Original train-fitted hidden-state/CNN and target scalers exactly match moments recomputed from training records. Selection is gated lexicographic, not a hidden aggregate score. Epoch checkpoint selection uses the declared unweighted validation BCE+Huber criterion.

## Phase A: original validation operating points

Validation cohort:1,203 eligible earthquakes +1,213 noise; distinct from test. FPR<=30% alone can be achieved by rejecting all examples, so useful recall>=70% is explicitly required.

| Policy | Threshold | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|
| Fixed | 0.5000 | 0.837589 | 0.681629 | 0.751604 | 0.131080 |
| Unconstrained validation F1 | 0.4100 | 0.776740 | 0.760599 | 0.768585 | 0.216818 |
| Constrained validation operating point | 0.4100 | 0.776740 | 0.760599 | 0.768585 | 0.216818 |

Useful target feasible on validation: True. A validation constraint is not a test or deployment FPR guarantee.

## Phase B/C trials and final selection

Trials executed: 12. Phase-B reference for loss variants: trial10. Selected trial: None; None means original retained. Configuration: `{"hidden": 32, "dropout": 0.2, "source": "original"}`.

| Trial | LR | Dropout | Hidden | Reg weight | Variant | Best epoch | F1 | FPR | Recall | MAE | Strong MAE | Underprediction % |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0001 | 0.1 | 64 | 1.0 | standard | 33 | 0.765403 | 0.296785 | 0.805486 | 0.613366 | 1.094090 | 89.313 |
| 2 | 0.0003 | 0.2 | 64 | 1.0 | standard | 34 | 0.766929 | 0.299258 | 0.809643 | 0.614142 | 1.070109 | 88.550 |
| 3 | 0.001 | 0.3 | 64 | 1.0 | standard | 32 | 0.767812 | 0.220115 | 0.761430 | 0.613804 | 1.084246 | 88.550 |
| 4 | 0.0001 | 0.2 | 128 | 1.0 | standard | 35 | 0.765873 | 0.290190 | 0.802161 | 0.614128 | 1.075411 | 87.786 |
| 5 | 0.0003 | 0.3 | 128 | 1.0 | standard | 34 | 0.767645 | 0.289365 | 0.804655 | 0.613727 | 1.061949 | 87.786 |
| 6 | 0.001 | 0.4 | 128 | 1.0 | standard | 27 | 0.767031 | 0.216818 | 0.758105 | 0.613029 | 1.085261 | 88.550 |
| 7 | 0.0003 | 0.1 | 64 | 0.5 | standard | 34 | 0.766733 | 0.291838 | 0.804655 | 0.613956 | 1.077375 | 89.313 |
| 8 | 0.0003 | 0.3 | 128 | 0.5 | standard | 32 | 0.767311 | 0.270404 | 0.792186 | 0.614046 | 1.068495 | 87.023 |
| 9 | 0.0003 | 0.2 | 64 | 2.0 | standard | 34 | 0.767836 | 0.296785 | 0.809643 | 0.613902 | 1.067140 | 88.550 |
| 10 | 0.001 | 0.2 | 128 | 2.0 | standard | 27 | 0.766962 | 0.214345 | 0.756442 | 0.613496 | 1.086542 | 88.550 |
| 11 | 0.001 | 0.2 | 128 | 2.0 | strong_bin | 21 | 0.767221 | 0.291838 | 0.805486 | 0.616033 | 1.098798 | 88.550 |
| 12 | 0.001 | 0.2 | 128 | 2.0 | asymmetric | 21 | 0.766745 | 0.298434 | 0.808811 | 0.615486 | 1.077363 | 87.786 |

Full trial MAE/RMSE/R2/bias/compression and fixed-.5 detection values are in trial_results.csv. Trial histories and complete validation threshold curves are retained in the single necessary selected study checkpoint; discarded trial weights were not saved. Magnitude variants use true magnitude only to weight training losses, never as an input feature.

## Validation magnitude comparison

| Model | MAE | RMSE | R2 | Bias | SD ratio | Strong MAE | Underprediction % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original Attention | 0.615548 | 0.752131 | 0.324633 | 0.003205 | 0.605680 | 1.043461 | 85.496183 |
| Selected | 0.615548 | 0.752131 | 0.324633 | 0.003205 | 0.605680 | 1.043461 | 85.496183 |

CNN validation fixed-.5 metrics and magnitude (same eligible validation cohort): `{"fixed_05": {"accuracy": 0.7723509933774835, "precision": 0.7902222222222223, "recall": 0.7389858686616791, "f1": 0.7637457044673539, "false_positive_rate": 0.19455894476504534, "roc_auc": 0.8614497008372172, "pr_auc_ap": 0.8841166764520607, "confusion_matrix": [[977, 236], [314, 889]]}, "magnitude": {"n": 1203, "mae": 0.6198669128980819, "rmse": 0.7643337308820294, "r2": 0.30254168884610566, "bias": -0.07902118878273397, "actual_variance": 0.8376214647116554, "predicted_variance": 0.22088591698102317, "std_ratio": 0.5135232541252361, "slope": 0.28685133920887446, "strong_n": 131, "strong_underprediction_percent": 92.36641221374046, "strong_bias": -1.1779837025940874}}`.

## One final exploratory test comparison

No authorized untouched evaluation set was supplied; no independent data collection or project scan was performed. The existing test cohort has been used in previous studies. This is therefore exploratory, not an unbiased confirmation. Exactly1,200 earthquakes +1,214 noise with the audited ordered identity hash. Both selected/original predictions computed only after selection; CNN frozen predictions provide the requested reference.

| Model | Policy | Threshold | Precision | Recall | F1 | FPR | MAE | RMSE | R2 | M>=5 underprediction % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected | Fixed | 0.5000 | 0.851812 | 0.665833 | 0.747428 | 0.114498 | 0.606601 | 0.748621 | 0.306725 | 87.023 |
| Selected | Validation operating point | 0.4100 | 0.771503 | 0.740000 | 0.755423 | 0.216639 | 0.606601 | 0.748621 | 0.306725 | 87.023 |
| Original Attention | Fixed | 0.5000 | 0.851812 | 0.665833 | 0.747428 | 0.114498 | 0.606601 | 0.748621 | 0.306725 | 87.023 |
| Original Attention | Validation operating point | 0.4100 | 0.771503 | 0.740000 | 0.755423 | 0.216639 | 0.606601 | 0.748621 | 0.306725 | 87.023 |
| CNN | Fixed | 0.5000 | 0.783615 | 0.733333 | 0.757641 | 0.200165 | 0.613005 | 0.757702 | 0.289803 | 96.947 |

Same-cohort magnitude covers every eligible earthquake independent of threshold; noise excluded. Fixed-.5 CNN comparisons are separate from validation-selected operating points; no CNN threshold was tuned.

Paired500-resample stratified bootstrap, seed2026, selected minus baseline; same event draws per model; descriptive percentile95% intervals (not corrected for search or historical test reuse):

```json
{
  "original_fixed": {
    "f1_difference_ci95": [
      0.0,
      0.0
    ],
    "mae_difference_ci95": [
      0.0,
      0.0
    ]
  },
  "cnn_fixed": {
    "f1_difference_ci95": [
      -0.023373921465093036,
      0.0013025063222041066
    ],
    "mae_difference_ci95": [
      -0.016216625086963175,
      0.0034196724591311037
    ]
  },
  "original_validation_operating": {
    "f1_difference_ci95": [
      0.0,
      0.0
    ],
    "mae_difference_ci95": [
      0.0,
      0.0
    ]
  }
}
```

## Interpretation

No trial met the predeclared useful-improvement gates; original Attention retained. Tuning did not establish useful validation improvement.

Generalizable improvement remains uncertain: single seed, small validation cohort, up to12 comparisons, known synthetic/test limitations. Strong-event loss weighting can reduce underprediction while increasing overall bias/error; guards prevent selection solely on strong-event scores. Check numerical tables rather than assuming that larger heads or weighted losses improve all objectives.

## Limitations and preservation

Synthetic oracle-P-aligned benchmark; both branches derive from the same simulated waveform source. No real-world triggering, warning lead time or operational EEW readiness established. High false positives, magnitude compression and strong-event underprediction remain relevant. Bidirectional temporal states use only the observed400-sample/40-bin horizon. Validation-derived thresholds do not guarantee deployment false-alarm rates.

Only 01_tuning.ipynb, 02_tuning_evaluation.ipynb, trial_results.csv, models/, plots/ and tuning_report.md created. No1s/2s training, cleanup or changes to approved experiments. Checkpoint includes source SHA256 hashes.

All 13 original dependencies unchanged; pretrained tensors remained exactly equal after training. Changed files: [].

Complete final test values:

```json
{
  "Selected": {
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
    "validation_operating_test": {
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
    "validation_threshold": 0.41,
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
      "strong_bias": -0.9308202175693657,
      "strong_mae": 0.9943339824676514
    }
  },
  "Original Attention": {
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
    "validation_operating_test": {
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
    "validation_threshold": 0.41,
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
      "strong_bias": -0.9308202175693657,
      "strong_mae": 0.9943339824676514
    }
  },
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
    "validation_operating_test": null,
    "validation_threshold": null,
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
      "strong_bias": -1.1642182965315024,
      "strong_mae": 1.1726391315460205
    }
  }
}
```
