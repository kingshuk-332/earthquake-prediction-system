# 3s EEW error diagnosis

## Scope and evidence

Validation-only diagnosis:1,203 eligible earthquakes and1,213 noise. Saved12-trial summaries and relevant evaluation code were read. No per-record validation prediction archive or discarded trial weights was available, so only original frozen Attention/CNN/BiLSTM validation predictions were reproduced. No test inputs or predictions were loaded; no training, fitting, threshold selection, model alteration or repeated preprocessing audit occurred. Train metadata was read for magnitude distributions.

## Dominant bottlenecks

Both detection discrimination and magnitude compression constrain improvement; no single causal mechanism is established. Detection F1 failed to improve across all12 trial operating points, while overall MAE moved by only a few thousandths and strong-event MAE worsened for every trial. Therefore this is not simply a successful magnitude improvement rejected by the detection objective. Loss-weight conflict remains a hypothesis, not a measured gradient result.

| Trial | Reg weight | Variant | F1 | FPR | MAE | Strong MAE | Strong underprediction % | SD ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | standard | 0.765403 | 0.296785 | 0.613366 | 1.094090 | 89.312977 | 0.590657 |
| 2 | 1.0 | standard | 0.766929 | 0.299258 | 0.614142 | 1.070109 | 88.549618 | 0.592492 |
| 3 | 1.0 | standard | 0.767812 | 0.220115 | 0.613804 | 1.084246 | 88.549618 | 0.571705 |
| 4 | 1.0 | standard | 0.765873 | 0.290190 | 0.614128 | 1.075411 | 87.786260 | 0.605284 |
| 5 | 1.0 | standard | 0.767645 | 0.289365 | 0.613727 | 1.061949 | 87.786260 | 0.604142 |
| 6 | 1.0 | standard | 0.767031 | 0.216818 | 0.613029 | 1.085261 | 88.549618 | 0.589283 |
| 7 | 0.5 | standard | 0.766733 | 0.291838 | 0.613956 | 1.077375 | 89.312977 | 0.584388 |
| 8 | 0.5 | standard | 0.767311 | 0.270404 | 0.614046 | 1.068495 | 87.022901 | 0.605197 |
| 9 | 2.0 | standard | 0.767836 | 0.296785 | 0.613902 | 1.067140 | 88.549618 | 0.597679 |
| 10 | 2.0 | standard | 0.766962 | 0.214345 | 0.613496 | 1.086542 | 88.549618 | 0.589390 |
| 11 | 2.0 | strong_bin | 0.767221 | 0.291838 | 0.616033 | 1.098798 | 88.549618 | 0.562904 |
| 12 | 2.0 | asymmetric | 0.766745 | 0.298434 | 0.615486 | 1.077363 | 87.786260 | 0.590806 |

Original validation operating point: F1=0.768585, FPR=0.216818, overall MAE=0.615548, strong MAE=1.043461, strong underprediction85.50%, predicted/actual SD=0.605680. Trial F1 spans0.765403–0.767836. Best MAE0.613029 improves by only0.002518; strong MAE spans1.061949–1.098798, uniformly worse. Minimum FPR0.214345 improves by only0.002473, far short of the declared0.02 reduction. None met the original selection criteria; those criteria were not changed here.

Trials10/11/12 provide the closest controlled loss comparison (same LR, width, dropout, regression weight, seed; checkpoint epochs differ): standard strong MAE1.086542; strong-bin weighting1.098798; asymmetric penalty1.077363. The asymmetric variant improves relative to trial10 but remains worse than the original1.043461. Trial11 also increases overall MAE to0.616033. This supports a local bias/error tradeoff, not evidence that weighting solved compression. Other cross-trial effects are confounded by simultaneous settings changes. Unweighted BCE+Huber checkpoint selection can fail to favor tail-specific improvements; epochwise strong-event predictions were not saved, so this is a plausible selection-objective limitation, not a demonstrated cause.

## False positives and false negatives

| Threshold | TN | FP | FN | TP | Precision | Recall | F1 | FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1054 | 159 | 383 | 820 | 0.837589 | 0.681629 | 0.751604 | 0.131080 |
| 0.41 | 950 | 263 | 288 | 915 | 0.776740 | 0.760599 | 0.768585 | 0.216818 |

Changing0.50 to0.41 adds95 true positives and104 false positives. Recall rises7.90 percentage points, but FPR rises8.57 points. Threshold adjustment trades errors rather than creating new discriminative information. The0.41 point meets the study’s FPR<=30%/recall>=70% research constraint; it still misclassifies263/1,213 noise windows and misses288/1,203 earthquakes. Operational acceptability is UNKNOWN without alarm-cost, prevalence and continuous-stream false-alarm requirements. Window FPR cannot be converted to false alarms per hour from these records.

## Error by true magnitude (eligible validation earthquakes)

| Range | N | MAE | RMSE | Signed error | Underprediction % | FN at0.50 | FN at0.41 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M<3 | 191 | 0.965733 | 1.009972 | 0.965733 | 0.000 | 96 | 74 |
| 3<=M<4 | 487 | 0.371700 | 0.457559 | 0.271721 | 23.614 | 180 | 136 |
| 4<=M<5 | 394 | 0.604918 | 0.690173 | -0.479795 | 84.518 | 95 | 71 |
| M>=5 | 131 | 1.043461 | 1.223419 | -0.945713 | 85.496 | 12 | 7 |

All191 M<3 predictions overestimate magnitude; mean error+0.9657. M>=5 has mean error−0.9457 and85.50% underprediction, despite only7/131 false negatives at0.41. Strong-earthquake detection therefore does not imply accurate strong-event magnitude estimation. Global mean error+0.0032 hides cancellation between opposite tail biases. Predicted variance0.30728 versus true0.83762, SD ratio0.60568 and regression slope0.34575 quantify compression. Overall RMSE0.75213 and R2=0.32463 show substantial unexplained variation. This pattern is consistent with central-value shrinkage, but does not prove whether loss, frozen representation or limited synthetic observability causes it.

## Magnitude distributions

| Split | Eligible EQ | Mean | SD | M<3 | 3–4 | 4–5 | M>=5 | Strong fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5593 | 3.865957 | 0.904796 | 886 | 2269 | 1829 | 609 | 10.8886% |
| validation | 1203 | 3.869742 | 0.915217 | 191 | 487 | 394 | 131 | 10.8894% |

Train/validation means, spreads and strong-event proportions closely match; coarse magnitude-distribution shift is not supported as the main cause. Strong events constitute only about10.9% in each split, making average loss dominated by other events. Minority representation is evidence; a causal claim that imbalance alone explains underprediction is unsupported. The tested weighting variants did not improve strong errors over the original. Coarse counts do not rule out shifts in distance, station, signal-to-noise or other unmeasured conditions.

## CNN–BiLSTM complementarity

Metrics below use both standalone models at0.50 on the same validation records. Different standalone calibration can affect classification overlap.

```json
{
  "probability_correlation": 0.7995704460129246,
  "magnitude_residual_correlation": 0.9826663285474849,
  "cnn_wrong_bilstm_right": 244,
  "bilstm_wrong_cnn_right": 413,
  "both_wrong": 306,
  "both_correct": 1453,
  "cnn_errors": 550,
  "bilstm_errors": 719,
  "both_false_positive": 74,
  "both_false_negative": 232,
  "oracle_best_absolute_error_mae": 0.568784773349762,
  "cnn_mae": 0.6198669075965881,
  "bilstm_mae": 0.6212625503540039
}
```

There is some detection complementarity:244 CNN errors are correctly classified by BiLSTM, while413 BiLSTM errors are correctly classified by CNN;306 errors are shared, including232 shared misses. This does not prove a learned fusion can identify which branch is correct. Probability correlation0.800 is substantial, while magnitude residual correlation0.983 indicates strongly shared magnitude errors. Oracle minimum-absolute-error MAE0.568785 is an unattainable target-aware selector diagnostic, not an achievable model or validated improvement. No blending coefficients or gates were fitted. These results do not justify more blind fusion-head search; they also do not prove that all fusion strategies are impossible.

## Frozen representations and unavailable causes

All trials adapted attention/heads atop the same frozen extractors. Near-flat validation performance across widths64/128, learning rates and loss weights is compatible with a representation bottleneck. Both branches derive from the same synthetic observations, consistent with correlated residuals. However, no controlled unfreezing comparison, representation sufficiency test, gradient-conflict measurement or irreducible-error analysis exists. Thus “frozen backbones caused failure” is SPECULATION, not a finding. Several selected epochs are near the35-epoch cap; convergence beyond the cap was not measured, so optimal convergence cannot be claimed.

Noise-family shortcuts: UNKNOWN. No explicit family/noise-type keys or attributes were found in the inspected train/validation waveform schema or validation feature group. No labels were inferred from IDs, amplitudes or waveform appearance; missing trustworthy metadata prevents family-specific false-positive rates. This does not prove such metadata is absent elsewhere.

Unavailable: per-record predictions for discarded trials, trial-specific magnitude-band/FN/FP overlap, epochwise strong-event selection diagnostics, trustworthy noise-family labels, multiple-seed variability and operational false-alarm rates. Trial CSV gives aggregate fixed-.5 and selected-threshold metrics, not arbitrary0.41 per-trial confusion matrices. No missing measurements were invented.

## Recommendation

**NO FURTHER TUNING at this stage.** The available evidence does not identify one sufficiently supported intervention:12 head/attention trials failed, both magnitude-loss variants underperformed the original on strong events, and shared regression errors are very high. Keep the original model and its documented operating tradeoff. Do not infer that another loss, more fusion capacity or backbone unfreezing will help without a causal diagnostic. No new experiment is recommended or initiated here.

Synthetic oracle-P alignment, shared signal source, single seed and reused validation set limit this diagnosis. These descriptive validation findings are not an independent confirmation or real-world EEW claim.

## Source preservation

All 11 hashed source dependencies unchanged. Only diagnostic_analysis.ipynb and diagnostic_report.md created; no checkpoints or data written.
- `C:\Users\choud\Desktop\Minor project\model_improvement\attention_tuning_v1\tuning_report.md`: `63a64f4902c0b066e5a362af0220198f3e810e483f8804c6ee4e387851ec5747`
- `C:\Users\choud\Desktop\Minor project\model_improvement\attention_tuning_v1\trial_results.csv`: `d82357b592d49be4a6d9e23940eadc129a615269f44ab1914d142c00426b6380`
- `C:\Users\choud\Desktop\Minor project\model_improvement\attention_tuning_v1\01_tuning.ipynb`: `932a9083e9b481e594e185d65268f5d5884bc7f685560a9557c881e0facfeb55`
- `C:\Users\choud\Desktop\Minor project\model_improvement\attention_tuning_v1\02_tuning_evaluation.ipynb`: `8db365c33949f855a8a4cacd3dfcf1d734eb7263d91313b105759dfe9899846c`
- `C:\Users\choud\Desktop\Minor project\fusion\cnn_bilstm_attention_v1\final_attention_report.md`: `d980e25c9a1eac847ddbdbade33635bf030856dab198699963a411dc6c550e41`
- `C:\Users\choud\Desktop\Minor project\fusion\cnn_bilstm_attention_v1\models\attention_3s_best.pt`: `854608e8787d677d5afbfcc59dccae99f669722fcd7e04b1849673830827fa1a`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\cnn_eew_training\models\cnn_3s_best.pt`: `f7e7617552bc35cf5620f12b32e53bf2232efaf0d1a14e345cf220eb64ce287a`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\bilstm_training_v1\models\bilstm_3s_best.pt`: `28a40646277f114e9a9bbb25763f765dadb47b6cfff602e250d4d8b5b743c1d0`
- `C:\Users\choud\Desktop\Minor project\bilstm_eew\causal_tabular_v1\tabular_features.h5`: `8905cfa29770a3a59d52fe26b766c5060be8dfcccc99c7c13caeef14525f08fe`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\dataset\eew_dataset_train.h5`: `2f88d8c1b50c1d52d0814360d7a32beabb5c90a822e121940a0db875ad49be15`
- `C:\Users\choud\Desktop\Minor project\cnn\cnn wave\eew_causal_v3\dataset\eew_dataset_validation.h5`: `c983d0fa42aabf107204b1e7261e8f0b4ee93a86aca64fdf7ce0a270877ec1d1`
