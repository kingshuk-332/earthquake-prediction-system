# Methodological references and scope

This project now concerns waveform-based earthquake early warning (EEW). No historical catalog sequence dataset, catalog forecasting model or earthquake forecasting capability is claimed. Catalog fields serve trace identity, supervised labels and retrospective evaluation only. Origin-time ordering does not establish historical reporting availability; publication and revision timestamps are unavailable.

## Chen & Fang (2025): background only

*Development of Attention-Based Bidirectional Long Short-Term Memory Networks for Earthquake Prediction in Taiwan*, [DOI](https://doi.org/10.1061/NHREFO.NHENG-2244). The [authors' university record and abstract](https://scholars.ncu.edu.tw/en/publications/development-of-attention-based-bidirectional-long-short-term-memo/) were accessible; the publisher full text was not available for methodological verification.

The abstract describes multivariate earthquake sequences and sliding windows, with earthquake magnitude/location classification and occurrence-time regression. This is background for sequence modeling, not evidence for initial-P-wave detection or early-warning performance. Its exact preprocessing and architecture were not reproduced. Catalog windows, attention and future-event prediction are excluded from this EEW experiment.

## Al Banna et al. (2021), including Andersson: background only

*Attention-Based Bi-Directional Long-Short Term Memory Network for Earthquake Prediction*, [DOI](https://doi.org/10.1109/ACCESS.2021.3071400), [repository full text](https://irep.ntu.ac.uk/id/eprint/42763/1/1432454_Mahmud.pdf). The requested reference lists Andersson as a coauthor; Al Banna is the first author. Methods Sections III-B/C, journal pages 56592–56594, were read.

The paper constructs catalog seismicity indicators from past events, including elapsed days and mean magnitude, and uses catalog preprocessing and attention-based sequence modeling. Its magnitude standard deviation indicator is a deviation from the Gutenberg–Richter relationship, not simply the sample standard deviation of magnitudes. No indicator formula is invented or transferred into waveform inputs. Its declustering, magnitude-scale conversion and catalog prediction protocol are not adopted. These methods do not validate P-wave EEW.

## Graves & Schmidhuber (2005): architectural principle

*Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures*, [original paper](https://www.cs.toronto.edu/~graves/nn_2005.pdf). The architecture discussion in Section II was read. Forward and backward recurrent processing supports using both directions within a finite, already observed sequence. The paper's speech results do not establish seismic performance.

For a later approved experiment, a PyTorch BiLSTM would concatenate `h_n[-2]` and `h_n[-1]`, the final layer's forward and backward hidden states, before dropout and the prediction head. Concatenating the two components of the last time-step output would not provide the same backward summary. Both directions must be restricted to samples at or before the observation cutoff. Standard PyTorch LSTM cells are an implementation choice, not an exact reproduction of the paper's cell details. No CNN or attention is included.

## Project-specific design and current limit

The 1/3/5-second windows, explicit event/noise pair partitions, original magnitude labels and source-code prefix-invariance check are this project's design decisions, not claimed reproductions of the catalog papers. Current windows use oracle P annotations and paired noise anchors. They cannot establish continuous detection false-alarm rates or guaranteed warning times.

Stage A found verified dependence of early synthetic noise on a later signal peak. The candidate crops are quarantined. Model implementation, train-only scaling, augmentation, optimization and performance evaluation are deferred until a valid source passes the gate. A separately versioned independent-noise source or aligned real recordings is required; none of the original source files were changed.
