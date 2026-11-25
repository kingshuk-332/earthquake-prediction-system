# Project conclusion

Four independently approved models were compared on the same synthetic oracle-P-aligned cohort:1,200 earthquakes and1,214 noise examples at1s,2s and3s post-P horizons. CNN used waveform windows; BiLSTM used time-ordered tabular features from those same simulated observations. Fusion and temporal attention reused frozen backbones.

No architecture consistently dominated detection and magnitude estimation. BiLSTM achieved the highest1s F1, with a73.64% noise false-positive rate. Attention had the highest2s F1 point estimate, while BiLSTM had the lowest2s magnitude MAE. At3s, CNN had the highest F1 (0.757641); Attention had the lowest MAE (0.606601). These rankings do not establish statistically superior architectures.

Fusion did not consistently improve on the standalone models. All saved Attention-versus-fusion F1 and MAE confidence intervals included zero: adding temporal attention provided no statistically supported advantage. Small numerical gains should not be described as meaningful improvements.

The contribution is a reproducible controlled comparison of four representations on verified causal synthetic observations, with matched identities, eligibility masks and evaluation policies. It is not evidence of real-world earthquake prediction or an operational warning system. Catalog-informed simulation, shared signal sources, oracle alignment, high false positives, compressed magnitude predictions and strong-earthquake underprediction constrain the conclusions.

Future research should use fresh held-out and real station data, multiple training seeds, validation-only calibration and threshold selection, realistic noise, and causal P triggering. Operational claims require end-to-end testing of false alarms, latency and physical warning intervals. No further training or experiment was performed for this synthesis.
