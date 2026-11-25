# Frozen baseline design (phase1-v1)
- Raw catalog, start 2011-01-01; discard last partial day; predeclared [20,32) N x [86,100) E, 2x2 cells split at 26 N / 93 E.
- 90 daily bins x 12 features: count, M4 count, M4.5 count, max/mean magnitude, mean depth/latitude/longitude, four cell counts. Empty bins zero; log1p counts; StandardScaler fit on unique training-history days.
- Seven marginal Bernoulli outputs: M>=4.5 any in 7 days, first 3 days, last 4 days, four cells in 7 days. Not a joint coherent distribution.
- Conv1d 12->16 k5/pad2; BN/ReLU; Conv1d 16->32 k3/pad1; ReLU/global-average; Dense 32->16/ReLU/dropout .25; Dense 16->7; sigmoid at inference.
- Unweighted BCE; AdamW lr .001/decay .001; batch32; max100 epochs; patience15; ReduceLROnPlateau patience5/factor.5; seed42; CUDA when available.
- Calendar 70/15/15; entire history+horizon inside partition; weekly nonoverlapping targets. Validation mean BCE chooses checkpoint.
- Baselines: training Poisson, smoothed trailing90-day Poisson (30-day training-rate prior), smoothed training prevalence. Brier/log loss/AP/precision/recall, calibration and paired 4-week block bootstrap.
- Version must change for architectural/protocol changes; identical completed configurations restore saved weights. last.pt contains full optimizer/scheduler/RNG state.

# Completed waveform detection architecture
Experiment wavecnn_ccb3b06b3d37: (1,10000) -> Conv8 k9/s2/BN/ReLU/pool4 -> Conv16 k7/s2/BN/ReLU/pool4 -> Conv32 k5/ReLU/global average -> dense16/ReLU/dropout.3 -> dense1/sigmoid; 4177 parameters. Full config/checkpoints under phase1_waveform_cnn_results; catalog design above preserved.
