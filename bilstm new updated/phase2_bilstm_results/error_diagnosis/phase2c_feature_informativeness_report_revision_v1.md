# Phase 2C feature and temporal audit — corrected report, revision 1

**Phase 2E revision, 2026-11-10.** Supersedes the summaries and interpretations in [the original report](phase2c_feature_informativeness_report.md). Original JSON, figures and report remain unchanged. All numbers reuse independently verified Phase 2D statistics; no data reprocessing, training or inference.

## Corrected feature table

Final historical step19, inverse-scaled to documented units; population SD. Train4973, validation1065, test1067. This is a retrospective next-retained-ML-event catalog diagnostic, not operational EEW or demonstrated earthquake forecasting.

| Index | Feature | Units | Train mean +/- SD | Validation mean +/- SD |
|---|---|---|---|---|
| 0 | historical_magnitude | ML reported units | 4.034285 +/- 0.898152 | 3.511455 +/- 0.628419 |
| 1 | historical_depth_km | km | 37.014478 +/- 35.582788 | 23.707982 +/- 33.102461 |
| 2 | historical_latitude_deg | degrees north | 25.774307 +/- 2.266943 | 26.231561 +/- 2.433653 |
| 3 | historical_longitude_deg | degrees east | 93.317730 +/- 2.505864 | 91.314345 +/- 3.298770 |
| 4 | inter_event_days | days | 7.406704 +/- 22.550500 | 0.388316 +/- 0.503241 |
| 5 | event_rate_30d | retained ML events/day | 0.554300 +/- 0.453468 | 3.007293 +/- 1.877520 |
| 6 | mean_magnitude_20 | ML reported units | 4.039364 +/- 0.743255 | 3.511981 +/- 0.232645 |
| 7 | std_magnitude_20 | ML reported units | 0.492074 +/- 0.143419 | 0.557127 +/- 0.157374 |
| 8 | mean_depth_20_km | km | 37.018817 +/- 14.862821 | 23.774085 +/- 11.492820 |
| 9 | std_depth_20_km | km | 29.444829 +/- 13.295776 | 27.890013 +/- 13.690996 |

All40 mean/SD entries are shown consistently above. The19 erroneous original entries are corrected explicitly below; fields tr/val denote training/validation.

| Index | Feature | Field | Original | Verified replacement |
|---|---|---|---|---|
| 0 | historical_magnitude | tr_mean | 4.04 | 4.03428514 |
| 0 | historical_magnitude | tr_std | 0.91 | 0.89815196 |
| 1 | historical_depth_km | tr_std | 35.5 | 35.58278792 |
| 1 | historical_depth_km | val_mean | 31.8 | 23.70798166 |
| 1 | historical_depth_km | val_std | 27.5 | 33.10246052 |
| 4 | inter_event_days | tr_std | 32.7 | 22.55050028 |
| 4 | inter_event_days | val_std | 0.91 | 0.50324108 |
| 5 | event_rate_30d | val_std | 2.05 | 1.87751987 |
| 6 | mean_magnitude_20 | tr_mean | 4.05 | 4.03936356 |
| 6 | mean_magnitude_20 | tr_std | 0.76 | 0.74325533 |
| 6 | mean_magnitude_20 | val_mean | 3.47 | 3.51198122 |
| 6 | mean_magnitude_20 | val_std | 0.38 | 0.23264547 |
| 7 | std_magnitude_20 | tr_std | 0.15 | 0.14341897 |
| 7 | std_magnitude_20 | val_mean | 0.44 | 0.55712680 |
| 7 | std_magnitude_20 | val_std | 0.11 | 0.15737431 |
| 8 | mean_depth_20_km | val_mean | 31.8 | 23.77408454 |
| 8 | mean_depth_20_km | val_std | 9.6 | 11.49281951 |
| 9 | std_depth_20_km | val_mean | 24.7 | 27.89001274 |
| 9 | std_depth_20_km | val_std | 9.1 | 13.69099568 |

## Correlations and definitions

| Index | Pearson H1 | Pearson H2 | Pearson validation | Spearman train | Spearman validation |
|---|---|---|---|---|---|
| 0 | 0.712201 | 0.173127 | 0.185696 | 0.643449 | 0.175000 |
| 1 | 0.063069 | 0.031668 | -0.050866 | 0.263053 | 0.021816 |
| 2 | 0.124701 | -0.016051 | 0.013412 | 0.057242 | 0.048488 |
| 3 | 0.219664 | 0.005774 | -0.022042 | 0.180052 | -0.040317 |
| 4 | 0.298504 | 0.140053 | -0.011236 | 0.396137 | -0.051143 |
| 5 | -0.267076 | -0.281125 | 0.073794 | -0.599379 | 0.051194 |
| 6 | 0.826588 | 0.290506 | 0.200656 | 0.744353 | 0.217067 |
| 7 | -0.256106 | -0.071988 | 0.139617 | -0.462834 | 0.091902 |
| 8 | 0.107170 | 0.141406 | -0.046081 | 0.340419 | -0.070057 |
| 9 | 0.031884 | 0.081264 | -0.038605 | -0.030138 | -0.056774 |

Phase 2D reproduced all340 original JSON numeric fields. Matching reconstructed periods: H1 rows[0,2486), n2486, target origins 1924-08-13T23:57:42+00:00 through 2013-01-29T20:25:30+00:00; H2 rows[2486,4973), n2487, 2013-01-31T13:11:02+00:00 through 2024-12-20T09:43:29+00:00. These are **reconstructed boundaries, not recovered original execution provenance**.

Rolling-mean r decreases0.826588 ->0.290506 ->0.200656. The last value exceeds0.20, so the categorical original threshold claim is withdrawn. Changes in individual correlations describe instability but do not prove absence of nonlinear, joint or sequence predictive signal. Squared univariate r is not joint or held-out model explained variance.

Training redundancy pairs are features0/6 (r0.818387) and8/9 (r0.657882). The original1/8 pairing is incorrect (actual r0.410652). Training gap Spearman is +0.396137, while event-rate Spearman is -0.599379; both are not negative.

Original d_train/d_val fields are **group mean differences divided by overall-split SD**, not pooled-within-group Cohen d. Keep their stored values under this corrected interpretation. Example: feature0 training value 1.218830, versus independently verified pooled Cohen d 1.476711. The verified feature CSV retains both definitions.

## Distribution shifts and frequency limits

| Split | n | Mean target ML | ML>=4.5 count (%) | N/elapsed year | Mean30-day rate |
|---|---|---|---|---|---|
| train | 4973 | 4.033843 | 1553 (31.229%) | 49.555 | 0.554300 |
| val | 1065 | 3.510610 | 75 (7.042%) | 943.350 | 3.007293 |
| test | 1067 | 3.101218 | 38 (3.561%) | 1607.187 | 4.496407 |

Annual rates use exact first-to-last target elapsed seconds and365.2425days/year: spans100.353619/1.128955/0.663893years. Validation/test rates943.350/1607.187 replace942/1592 calculated from rounded durations. Alternative(N-1)/span interarrival rates are saved in the verified JSON. Mean30-day test/train ratio is8.1119, replacing8.2 from rounded inputs.

Counts and rates concern **retained ML events**, not all source events or complete regional seismicity. Catalog downtime, missing periods, reporting policies and actual exposure are unverified. Exact endpoint durations account for partial years but do not establish coverage. The30-day feature counts retained events in(time-30days,time] divided by30. Final-input gap medians are1.556725/0.197836/0.131713days; these differ in definition from separately verified target-event gaps.

Train-to-validation Wasserstein=0.524702, KS D=0.314511, nominal p=2.335e-77; train-to-test Wasserstein=0.932624. These descriptive distances reproduce. Conventional KS/Pearson/Spearman p values are not dependence-robust here: overlapping histories, temporal structure and tied ML observations limit significance claims. Saved zero p values reflect numerical underflow.

## Confirmed findings versus hypotheses

**Confirmed:** temporal target/feature distribution changes; correlation weakening/sign changes; compressed model predictions and high-magnitude underprediction. Input-before-target ordering was verified; origin times do not establish historical reporting availability.

**Hypotheses only:** changing detection/reporting, spatial sampling, magnitude-type selection and physical clustering may contribute. No causal completeness analysis, validated station-network deployment history or learned-rule analysis establishes the proposed causes.

**Historical Mc UNKNOWN; current Mc UNKNOWN.** No formal validated estimator, fit diagnostics or uncertainty intervals were supplied. Claims Mc~4.0/~1.5 or that only ML>=4.0 events were historically detected are withdrawn as established findings. Observed minimum magnitudes are not Mc estimates. Modern catalog stationarity is unverified.

The Gutenberg-Richter frequency-magnitude relationship does **not** prove independent successive magnitudes or conditional independence. Small individual correlations do not prove no predictive signal, nor that Huber regression must predict the mean. The [Phase 2D report](../gemini_verification/verification_report.md) retains the methodological reference and scientific qualifications; no new completeness or independence study was introduced.

## Original Phase 2C execution provenance: missing

Targeted filename/title/content searches within this project, including hidden notebook locations and excluding dependency runtime, found no original Phase 2C notebook. Immediate workspace and named notebook-checkpoint locations yielded no matching recovery copy. The alternate content hit is the existing Phase 2A diagnosis, not Phase 2C. No machine-wide search is claimed.

Original statistics remain preserved, with independent reproducibility supported by the executed Phase 2D notebook. **Neither that verification nor this correction notebook is a replacement execution history.** Original figures are preserved; their generation provenance is missing. Registered checkpoints are unchanged; unrecorded Phase 2C training cannot be ruled out.

Evidence: [correction summary](../gemini_verification/correction_report.md), [provenance record](../gemini_verification/correction_provenance_v1.json), [verified feature statistics](../gemini_verification/feature_verification.csv).
