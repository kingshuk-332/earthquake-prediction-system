# SHAP validation report

**VERDICT: CONDITIONAL**

Frozen full-validation loss reproduces the approved checkpoint within 1e-6. Scalar outputs agree with the original two-stage computation. Both input shapes, finite attributions, Attention normalization, paired identities, 3s endpoint, unchanged state tensors and input-file hashes were checked. Pilot and final arrays are genuine SHAP GradientExplainer expected gradients. Residuals are Monte Carlo approximation diagnostics, not exact local accuracy. Sensitivity includes an independent sampling-seed repeat and a distinct paired background; neither changes model parameters.

The seven tabular features are deterministic summaries of the same waveform; they are also internally dependent (energy = 10*RMS squared before normalization). Background endpoints are paired real training records. SHAP GradientExplainer uses a shared background row and interpolation coefficient for both tensors, but simultaneous straight-line interpolation generally does not preserve nonlinear feature-extraction identities. Thus intermediate points can be physically inconsistent even with paired endpoints. These are model-input expected-gradient attributions under the selected reference distribution, not unique physical contributions of independent modalities, conditional-distribution SHAP, or evidence of causality. Attention weights are learned hidden-state pooling weights, not SHAP values; bidirectional states mix information throughout the available window.

```json
{
  "detection": {
    "n": 12,
    "shapes": [
      [
        12,
        400
      ],
      [
        12,
        40,
        7
      ]
    ],
    "expected_output": -0.7625274658203125,
    "max_abs_reconstruction_residual": 0.6225839405960079,
    "mean_abs_reconstruction_residual": 0.20584006676941277,
    "reconstruction_passes": 8,
    "repeat_stability": {
      "feature_spearman": 1.0,
      "feature_relative_L1": 0.007546031055485575,
      "local_signed_cosine_min": 0.9913747632126321,
      "local_signed_cosine_mean": 0.9970651589377323,
      "waveform_temporal_spearman": 0.9979828623928899,
      "tabular_temporal_spearman": 0.9953095684803004
    },
    "background_stability": {
      "feature_spearman": 0.8928571428571429,
      "feature_relative_L1": 0.13649307648556994,
      "local_signed_cosine_min": 0.9304429567599237,
      "local_signed_cosine_mean": 0.9489614995645762,
      "waveform_temporal_spearman": 0.9543852774079836,
      "tabular_temporal_spearman": 0.8714821763602253
    },
    "alternative_expected_output": 1.2113803625106812,
    "alternative_max_abs_residual": 0.5802734082810916,
    "feature_importance": {
      "rms": 0.030850358939664056,
      "mean_absolute_amplitude": 0.024084690510477867,
      "peak_absolute_amplitude": 0.011280034469800509,
      "variance": 0.016997999264868516,
      "signal_energy": 0.0009306950913206819,
      "zero_crossing_rate": 0.025649200386955482,
      "first_difference_energy": 0.02688709087301283
    }
  },
  "magnitude": {
    "n": 9,
    "shapes": [
      [
        9,
        400
      ],
      [
        9,
        40,
        7
      ]
    ],
    "expected_output": 3.8531529903411865,
    "max_abs_reconstruction_residual": 0.0240126190558263,
    "mean_abs_reconstruction_residual": 0.012143724826828503,
    "reconstruction_passes": 9,
    "repeat_stability": {
      "feature_spearman": 1.0,
      "feature_relative_L1": 0.005630774521452867,
      "local_signed_cosine_min": 0.9954507529060268,
      "local_signed_cosine_mean": 0.9972944978109061,
      "waveform_temporal_spearman": 0.9952980331127068,
      "tabular_temporal_spearman": 0.998874296435272
    },
    "background_stability": {
      "feature_spearman": 0.9642857142857145,
      "feature_relative_L1": 0.10819769391846144,
      "local_signed_cosine_min": 0.8872712306971544,
      "local_signed_cosine_mean": 0.9304577085214443,
      "waveform_temporal_spearman": 0.9054197213732585,
      "tabular_temporal_spearman": 0.9759849906191371
    },
    "alternative_expected_output": 3.8612005710601807,
    "alternative_max_abs_residual": 0.030466511154365827,
    "feature_importance": {
      "rms": 0.0050789884298891004,
      "mean_absolute_amplitude": 0.0039057885928444436,
      "peak_absolute_amplitude": 0.0026914598146841667,
      "variance": 0.002088920227876535,
      "signal_energy": 0.00012929427733196407,
      "zero_crossing_rate": 0.001666478810937426,
      "first_difference_energy": 0.0029820992372683146
    }
  },
  "verdict": "CONDITIONAL",
  "validation_loss": 0.7318453788757324,
  "approved_validation_loss": 0.7318456104260274,
  "validation_loss_error": 2.3155029493793222e-07,
  "reference_prediction_max_error": 3.814697265625e-06,
  "weights_unchanged": true,
  "source_files_unchanged": true,
  "test_data_opened": false,
  "device": "CPU; cuDNN/MKLDNN disabled",
  "torch": "2.6.0+cu124",
  "shap": "0.54.1",
  "input_sha256": {
    "C:\\Users\\choud\\Desktop\\Minor project\\fusion\\cnn_bilstm_attention_v1\\models\\attention_3s_best.pt": "854608e8787d677d5afbfcc59dccae99f669722fcd7e04b1849673830827fa1a",
    "C:\\Users\\choud\\Desktop\\Minor project\\cnn\\cnn wave\\eew_causal_v3\\cnn_eew_training\\models\\cnn_3s_best.pt": "f7e7617552bc35cf5620f12b32e53bf2232efaf0d1a14e345cf220eb64ce287a",
    "C:\\Users\\choud\\Desktop\\Minor project\\bilstm_eew\\bilstm_training_v1\\models\\bilstm_3s_best.pt": "28a40646277f114e9a9bbb25763f765dadb47b6cfff602e250d4d8b5b743c1d0",
    "C:\\Users\\choud\\Desktop\\Minor project\\bilstm_eew\\causal_tabular_v1\\tabular_features.h5": "8905cfa29770a3a59d52fe26b766c5060be8dfcccc99c7c13caeef14525f08fe",
    "C:\\Users\\choud\\Desktop\\Minor project\\cnn\\cnn wave\\eew_causal_v3\\cnn_eew_training\\01_cnn_training.ipynb": "20003e98aa1718dc2d6f545b6353b203bb9e725114cf630667f4048bf6635e54",
    "C:\\Users\\choud\\Desktop\\Minor project\\bilstm_eew\\bilstm_training_v1\\01_bilstm_training.ipynb": "2df085de20bd88b75ab8dfcdd21de7fc5be8680edaaad4c54a901f74617b7f18",
    "C:\\Users\\choud\\Desktop\\Minor project\\fusion\\cnn_bilstm_attention_v1\\01_attention_training.ipynb": "0289b1211e790ea998ac0b9f6c562bbc407d68a8f03f95e4d0526906705b0da9",
    "C:\\Users\\choud\\Desktop\\Minor project\\fusion\\cnn_bilstm_attention_v1\\02_attention_evaluation.ipynb": "a652f345563cba1539d215caba34cc24a59410c71b6dafe2b77da63137b21466",
    "C:\\Users\\choud\\Desktop\\Minor project\\fusion\\cnn_bilstm_attention_v1\\final_attention_report.md": "d980e25c9a1eac847ddbdbade33635bf030856dab198699963a411dc6c550e41",
    "C:\\Users\\choud\\Desktop\\Minor project\\cnn\\cnn wave\\eew_causal_v3\\dataset\\eew_dataset_train.h5": "2f88d8c1b50c1d52d0814360d7a32beabb5c90a822e121940a0db875ad49be15",
    "C:\\Users\\choud\\Desktop\\Minor project\\cnn\\cnn wave\\eew_causal_v3\\dataset\\eew_dataset_validation.h5": "c983d0fa42aabf107204b1e7261e8f0b4ee93a86aca64fdf7ce0a270877ec1d1"
  }
}
```


## Numerical interpretation restrictions

The detection reconstruction check is **not fully passed**: 4 of 12 records exceed the predeclared tolerance of 0.1 + 0.1*abs(output minus expected output). Its maximum absolute residual is 0.622584 logits. Detection plots are therefore approximate descriptive attributions; they must not be used as precise local additive decompositions or exact probability explanations. Residuals are displayed explicitly in the local plots and retained for every record in the NPZ files. No extra sampling or explainer tuning was performed after these results.

Changing the 48-record training background shifts the detection expected logit from -0.762527 to 1.211380 (a 1.973908-logit shift), despite the same class/magnitude stratum counts. This is material reference-distribution sensitivity, even though aggregate feature rank correlation is 0.893 and the minimum local signed-attribution cosine is 0.930. Stable feature rankings do not establish a stable reference value or exact local explanation. The independent Monte Carlo repeat has feature rank correlation 1.000 and minimum local cosine 0.991.

Magnitude passes the reconstruction tolerance for all 9 records (maximum absolute residual 0.024013 magnitude units). Its expected output changes from 3.853153 to 3.861201 across backgrounds; feature rank correlation is 0.964, with minimum local cosine 0.887. The independent repeat has feature rank correlation 1.000 and minimum local cosine 0.995. These results support approximate descriptions of these selected model outputs, subject to the paired-input dependency limitations. The magnitude reference includes noise model outputs and has no interpretation as an average physical earthquake magnitude.

**CONDITIONAL** applies to the experiment as a whole: genuine SHAP expected gradients were obtained, but detection numerical reconstruction, background sensitivity, dependent inputs, and the small stratified cohort prevent an unqualified approval. Attention weights remain separate diagnostics, never SHAP values.
