# Research-method adaptation

[Chawla et al. (2002), SMOTE](https://doi.org/10.1613/jair.953), Section 4.2 [accessible method](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume16/chawla02a-html/node6.html), was read. It generates minority-class feature vectors by interpolating selected nearest neighbors. Classification SMOTE does not establish a valid regression target or coherent earthquake-history interpolation here.

[Branco et al. (2017), SMOGN](https://proceedings.mlr.press/v74/branco17a/branco17a.pdf), Sections 2–4 and Algorithm 1, were read. It uses regression relevance to identify important target ranges and combines under-sampling with interpolation for sufficiently close neighbors and Gaussian perturbation otherwise. Its distance-based rule and target construction are not implemented here.

Adopted concept: emphasize relevant high regression targets conservatively. Our weight is 1+0.5×clip((ML−4)/1.5,0,1), using training labels only. Weighted replacement draws retain complete original windows and targets. Sorted parent indices preserve their temporal order. This is the user-authorized fallback, not SMOGN itself. No neighbor graph, flattened sequence interpolation, under-sampling stage or synthetic target magnitude is used.

Our coordinate-only Gaussian noise (standardized sigma 0.01, seed 42) is a separate adaptation. It leaves temporal/rolling relationships and labels unchanged and applies identical event-level perturbations across overlapping windows within an epoch. Noise is not calibrated geophysical uncertainty. Synthetic feature perturbations never become real catalog events.

All four variants keep the same epoch size and original validation/test arrays. B/D share the noise policy; C/D share draws. Future comparison must fix architecture, AdamW optimizer, compute budget and validation-based selection; use test once after selection. Report regression MAE/RMSE/R², high-magnitude MAE and underprediction bias, plus separate classification PR-AUC/precision/recall/F1. No benefit is assumed. Weighted sampling changes the training objective and may worsen overall calibration or MAE.

Independent source consistency checks are mandatory. An inherited feature error is not cured by repetition or noise; previews remain diagnostic until a separately authorized source correction passes validation.
