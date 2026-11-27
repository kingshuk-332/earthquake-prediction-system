# Mathematical Formulations and Procedures for Dataset Synthesis

**Document:** `paper/01_synthesis_formulas.md`  
**Derivation Source:** Directly verified against Python source implementations in:
- `cnn/cnn wave/data wave/synthetic_waveform_simulator/improved_waveform_generator.py`
- `cnn/cnn wave/data wave/synthetic_waveform_simulator/phase10_generate_classification_dataset.py`
- `cnn/cnn wave/data wave/synthetic_waveform_simulator/output/phase8_improved_simulation/PHASE8_STEPS_AND_FORMULAS.md`
- `bilstm_eew/causal_tabular_v1/01_tabular_feature_extraction.ipynb`
- `cnn/cnn wave/eew_causal_v3/01_causal_simulator_v3.ipynb`

---

## 1. Kinematic Travel-Time Constraints & Phase Arrival Modeling

For each event in the earthquake metadata catalog ($N = 8,089$), 3D spherical chord hypocentral distance $R$ is derived from epicentral arc distance $D$, focal depth $h$, and Earth radius $R_\oplus = 6371.0\text{ km}$ (eliminating up to $9.013\text{ km}$ of flat-earth $\sqrt{D^2 + h^2}$ curvature error):

\begin{equation}
R = \sqrt{R_\oplus^2 + (R_\oplus - h)^2 - 2 R_\oplus (R_\oplus - h)\cos(D / R_\oplus)}
\label{eq:hypocentral_distance}
\end{equation}

where:
- $D$: Epicentral distance in kilometers ($\text{km}$), empirical catalog range: $[1.6, 683.9]\text{ km}$.
- $h$: Focal depth in kilometers ($\text{km}$), empirical catalog range: $[1.0, 430.0]\text{ km}$.
- $R$: Hypocentral distance ($\text{km}$), empirical range: $[8.93, 684.70]\text{ km}$.

Theoretical compressional (P) and shear (S) wave arrival times ($t_P, t_S$) and their discrete sample indices ($P_{\text{sample}}, S_{\text{sample}}$) are computed under a homogeneous 1D crustal velocity model:

\begin{equation}
t_P = \frac{R}{V_P}, \quad t_S = \frac{R}{V_S}
\label{eq:travel_times}
\end{equation}

\begin{equation}
P_{\text{sample}} = \left\lfloor t_P \cdot f_s + 0.5 \right\rfloor, \quad S_{\text{sample}} = \left\lfloor t_S \cdot f_s + 0.5 \right\rfloor
\label{eq:arrival_samples}
\end{equation}

where:
- $V_P = 6.0\text{ km/s}$: Mean upper-crustal P-wave velocity.
- $V_S = 3.5\text{ km/s}$: Mean upper-crustal S-wave velocity ($V_P / V_S \approx 1.714$, Poisson solid).
- $f_s = 100.0\text{ Hz}$: Sampling frequency ($\Delta t = 0.01\text{ s}$).
- Total waveform duration $T = 100.0\text{ s}$ ($N_{\text{samples}} = 10,000$).

*Seismological Note on Code Parity:* As established in the project audits, $P_{\text{sample}}$ and $S_{\text{sample}}$ represent theoretical 1D kinematic constraints, not empirical picks from seismic station analysts.

---

## 2. Source Amplitude Scaling and Frequency-Dependent Crustal Attenuation

The peak clean wave amplitude $A(M, R, f)$ accounts for earthquake magnitude source scaling, geometric wave-front spreading, and frequency-dependent anelastic absorption.

### 2.1 Effective Distance & Source Magnitude Scaling
To avoid near-field singularity as $R \to 0$, distance is bounded by a reference distance:

\begin{equation}
R_{\text{eff}} = \max(R, R_{\text{ref}})
\label{eq:effective_distance}
\end{equation}

Source magnitude scaling is defined by:

\begin{equation}
A_{\text{source}}(M) = A_{\text{ref}} \cdot 10^{\gamma (M - M_{\text{ref}})}
\label{eq:source_amplitude}
\end{equation}

where:
- $M$: Earthquake magnitude (empirical catalog range: $[1.5, 8.6]$, $\hat{M}_c = 3.7$).
- $M_{\text{ref}} = 4.0$: Reference normalization magnitude.
- $A_{\text{ref}} = 1.0$: Reference amplitude.
- $\gamma = 0.50$: Empirical scaling exponent.
- $R_{\text{ref}} = 10.0\text{ km}$: Reference near-field boundary distance.

### 2.2 Geometric Spreading
For spherical body wave propagation in a crustal half-space:

\begin{equation}
\mathcal{G}(R) = \left( \frac{R_{\text{ref}}}{R_{\text{eff}}} \right)^n
\label{eq:geometric_spreading}
\end{equation}

where $n = 1.0$ (body wave spherical spreading exponent).

### 2.3 Frequency-Dependent Anelastic Quality Factor $Q(f)$
In Phase 8 of the simulation pipeline, constant quality factors were replaced with standard seismological power-law frequency dependence:

\begin{equation}
Q_P(f) = Q_{P0} \left( \frac{f}{f_0} \right)^\eta, \quad Q_S(f) = Q_{S0} \left( \frac{f}{f_0} \right)^\eta
\label{eq:crustal_q}
\end{equation}

where:
- $Q_{P0} = 350.0$: P-wave reference quality factor at $f_0 = 1.0\text{ Hz}$.
- $Q_{S0} = 300.0$: S-wave reference quality factor at $f_0 = 1.0\text{ Hz}$.
- $\eta = 0.70$: Regional crustal frequency dependence power-law exponent.
- $f_0 = 1.0\text{ Hz}$: Reference frequency.
- $f_P = 5.0\text{ Hz}$: Dominant compressional wave frequency ($Q_P(5.0) \approx 1,080$).
- $f_S = 2.2\text{ Hz}$: Dominant shear wave frequency ($Q_S(2.2) \approx 520$).

### 2.4 Anelastic Attenuation Factor
\begin{equation}
\mathcal{K}_P(R, f_P) = \exp\left( - \frac{\pi f_P R_{\text{eff}}}{Q_P(f_P) V_P} \right)
\label{eq:kappa_p}
\end{equation}

\begin{equation}
\mathcal{K}_S(R, f_S) = \exp\left( - \frac{\pi f_S R_{\text{eff}}}{Q_S(f_S) V_S} \right)
\label{eq:kappa_s}
\end{equation}

### 2.5 Vertical-Component (BHZ) Wave Amplitudes
On vertical-component seismometers, particle motion of compressional waves is nearly vertical, whereas shear motion is predominantly horizontal. The vertical S/P amplitude ratio is calibrated to $r_{\text{BHZ}} = 1.35$:

\begin{equation}
A_P = A_{\text{source}}(M) \cdot \mathcal{G}(R) \cdot \mathcal{K}_P(R, f_P)
\label{eq:amp_p}
\end{equation}

\begin{equation}
A_S = A_{\text{source}}(M) \cdot \mathcal{G}(R) \cdot \mathcal{K}_S(R, f_S) \cdot r_{\text{BHZ}}
\label{eq:amp_s}
\end{equation}

with $r_{\text{BHZ}} = 1.35$.

---

## 3. Dynamic Wave Packet Envelope & Multi-Harmonic Carrier Model

### 3.1 Causal $C^1$-Continuous Gamma Onset Envelope
To prevent unphysical step discontinuities at the arrival onset, wave packets employ a $C^1$-continuous Gamma envelope:

\begin{equation}
E(\tau; \tau_r, \tau_d) = \begin{cases}
0, & \tau < 0 \\
\left( \frac{\tau}{\tau_r} \right)^2 \exp\left( - \frac{\tau}{\tau_d} \right), & \tau \ge 0
\end{cases}
\label{eq:gamma_envelope}
\end{equation}

where $\tau = t - t_{\text{arrival}}$ is elapsed time since phase arrival, $\tau_r$ is the source rise time, and $\tau_d$ is the coda decay parameter.

The function achieves its maximum at $\tau_{\text{peak}} = 2\tau_d$ with value $E_{\text{max}} = (2\tau_d / \tau_r)^2 \exp(-2)$. The peak-normalized envelope is:

\begin{equation}
\bar{E}(\tau; \tau_r, \tau_d) = \frac{E(\tau; \tau_r, \tau_d)}{\max_{\tau \ge 0} E(\tau; \tau_r, \tau_d)} = \left( \frac{\tau}{2\tau_d} \right)^2 \exp\left( 2 - \frac{\tau}{\tau_d} \right)
\label{eq:norm_gamma_envelope}
\end{equation}

### 3.2 Dynamic Scaling of Rise Time and Coda Decay
Source rupture duration scales with magnitude, while scattering dispersion broadens coda decay with distance:

\begin{equation}
\tau_{d,P}(M, R) = \max\left( \tau_{P0} [1.0 + 0.12 (M - M_{\text{ref}})] [1.0 + 0.0010 R_{\text{eff}}], \; 0.60\text{ s} \right)
\label{eq:tau_dp}
\end{equation}

\begin{equation}
\tau_{r,P}(M, R) = 0.28 \cdot \tau_{d,P}(M, R)
\label{eq:tau_rp}
\end{equation}

\begin{equation}
\tau_{d,S}(M, R) = \max\left( \tau_{S0} [1.0 + 0.18 (M - M_{\text{ref}})] [1.0 + 0.0015 R_{\text{eff}}], \; 1.50\text{ s} \right)
\label{eq:tau_ds}
\end{equation}

\begin{equation}
\tau_{r,S}(M, R) = 0.22 \cdot \tau_{d,S}(M, R)
\label{eq:tau_rs}
\end{equation}

where $\tau_{P0} = 1.50\text{ s}$ and $\tau_{S0} = 4.00\text{ s}$.

### 3.3 Multi-Harmonic Carrier Waveforms
To prevent monochromatic synthetic signatures, carrier signals combine fundamental and harmonic components:

\begin{equation}
\Phi_P(\tau) = 0.85 \sin(2\pi f_P \tau) + 0.15 \sin(4\pi f_P \tau)
\label{eq:carrier_p}
\end{equation}

\begin{equation}
\Phi_S(\tau) = 0.80 \sin\left(2\pi f_S \tau + \frac{\pi}{4}\right) + 0.20 \sin(3\pi f_S \tau)
\label{eq:carrier_s}
\end{equation}

The composite clean synthetic seismic signal is:

\begin{equation}
s_{\text{clean}}(t) = A_P \bar{E}(t - t_P; \tau_{r,P}, \tau_{d,P}) \Phi_P(t - t_P) + A_S \bar{E}(t - t_S; \tau_{r,S}, \tau_{d,S}) \Phi_S(t - t_S)
\label{eq:clean_signal}
\end{equation}

---

## 4. Pre-P Noise Floor & Mathematical SNR Calibration

To eliminate systemic sample-window bias, signal and noise windows are strictly defined.

### 4.1 Window Definitions
- **Pre-P Noise Reference Window ($W_{\text{noise}}$):**
  \begin{equation}
  W_{\text{noise}} = [0, \; \max(P_{\text{sample}} - 5, \; 50)]
  \label{eq:noise_window}
  \end{equation}
- **Seismic Signal Arrival Window ($W_{\text{signal}}$):**
  \begin{equation}
  W_{\text{signal}} = [P_{\text{sample}}, \; N_{\text{samples}}]
  \label{eq:signal_window}
  \end{equation}

### 4.2 Clean Signal Peak & Target Noise Scaling
\begin{equation}
A_{\text{signal}} = \max_{t \in W_{\text{signal}}} |s_{\text{clean}}(t)|
\label{eq:signal_peak}
\end{equation}

Given the target catalog signal-to-noise ratio $\text{SNR}_{\text{target, dB}} \in [1.50, 39.07]\text{ dB}$ (note: in the strictly causal `eew_causal_v3` benchmark, per-trace catalog SNR scaling was replaced by a fixed ambient noise floor $\sigma_{\text{noise}} = 0.05$ normalized via impulse response gain `GAINS['band']` to prevent pre-$P$ variance shortcut leakage):

\begin{equation}
\sigma_{\text{target}} = \frac{A_{\text{signal}}}{10^{\text{SNR}_{\text{target, dB}} / 20}}
\label{eq:sigma_target}
\end{equation}

### 4.3 Colored Noise Generation & Exact Pre-P Calibration
Raw zero-mean Gaussian white noise $w(t) \sim \mathcal{N}(0, 1)$ is bandpass-filtered through a 4th-order Butterworth digital filter $\mathcal{H}_{\text{BP}}$ with cutoff frequencies $[0.2, 25.0]\text{ Hz}$:

\begin{equation}
\tilde{n}(t) = \mathcal{H}_{\text{BP}}\{ w(t) \}
\label{eq:filtered_noise}
\end{equation}

The empirical standard deviation over the pre-P reference window is measured:

\begin{equation}
\hat{\sigma}_{\text{pre-P}} = \sqrt{ \frac{1}{|W_{\text{noise}}|} \sum_{t \in W_{\text{noise}}} \tilde{n}(t)^2 }
\label{eq:pre_p_rms}
\end{equation}

The calibrated noise time series is scaled so that its pre-P window RMS identically matches $\sigma_{\text{target}}$:

\begin{equation}
n(t) = \tilde{n}(t) \cdot \left( \frac{\sigma_{\text{target}}}{\hat{\sigma}_{\text{pre-P}}} \right)
\label{eq:calibrated_noise}
\end{equation}

The final synthetic earthquake waveform is:

\begin{equation}
y(t) = s_{\text{clean}}(t) + n(t)
\label{eq:total_earthquake_waveform}
\end{equation}

By construction, $|\text{SNR}_{\text{measured}} - \text{SNR}_{\text{target}}| < 0.002\text{ dB}$, completely eliminating the $+6.21\text{ dB}$ sample-crest bias identified in Phase 7.

---

## 5. Multi-Category Stochastic Noise Waveform Models (Class 0)

To construct a balanced classification task ($N_{\text{noise}} = 8,089$), four realistic non-seismic background noise models are implemented in `phase10_generate_classification_dataset.py`, assigned cyclically (25% each). Each noise trace is scaled to match the empirical distribution of pre-P noise RMS $\hat{\sigma}_{\text{pre-P}}$ sampled from the earthquake dataset.

### 5.1 Category 1: Band-Limited Gaussian Ambient Noise (`gaussian_bandlimited`)
Standard Gaussian white noise filtered through 4th-order Butterworth bandpass $[0.2, 25.0]\text{ Hz}$:

\begin{equation}
w(t) \sim \mathcal{N}(0, 1), \quad \tilde{n}_1(t) = \mathcal{H}_{\text{BP}}\{ w(t) \}
\label{eq:noise_cat1}
\end{equation}

\begin{equation}
y_{\text{noise}, 1}(t) = \tilde{n}_1(t) \cdot \frac{\sigma_{\text{target}}}{\text{RMS}_{t}(\tilde{n}_1(t))}
\end{equation}

### 5.2 Category 2: Pink $1/f$ Crustal Background Noise (`colored_pink`)
Low-frequency crustal ambient noise generated in the frequency domain with spectral density $S(f) \propto 1/f$:

\begin{equation}
X_k = \frac{Z_{r,k} + j Z_{i,k}}{\sqrt{k}}, \quad Z_{r,k}, Z_{i,k} \sim \mathcal{N}(0, 1), \quad k = 1, \dots, \frac{N_{\text{samples}}}{2}
\label{eq:pink_freq}
\end{equation}

with $X_0 = 0$. Inverted to the time domain and bandpass-filtered:

\begin{equation}
\tilde{n}_2(t) = \mathcal{H}_{\text{BP}}\left\{ \text{Re}\left[ \mathcal{F}^{-1}\{ X \} \right] \right\}
\label{eq:noise_cat2}
\end{equation}

\begin{equation}
y_{\text{noise}, 2}(t) = \tilde{n}_2(t) \cdot \frac{\sigma_{\text{target}}}{\text{RMS}_{t}(\tilde{n}_2(t))}
\end{equation}

### 5.3 Category 3: Modulated Non-Stationary Noise (`modulated_nonstationary`)
Models diurnal and meteorological environmental fluctuations via low-frequency sinusoidal modulation:

\begin{equation}
\mathcal{M}(t) = 1.0 + A_{\text{mod}} \sin(2\pi f_{\text{mod}} t + \phi_{\text{mod}})
\label{eq:modulation_envelope}
\end{equation}

where:
- $f_{\text{mod}} \sim \mathcal{U}(0.01, 0.04)\text{ Hz}$: Diurnal modulation frequency.
- $\phi_{\text{mod}} \sim \mathcal{U}(0, 2\pi)$: Random phase offset.
- $A_{\text{mod}} \sim \mathcal{U}(0.25, 0.45)$: Modulation depth.

\begin{equation}
\tilde{n}_3(t) = \mathcal{H}_{\text{BP}}\{ \mathcal{N}(0, 1) \} \cdot \mathcal{M}(t)
\label{eq:noise_cat3}
\end{equation}

\begin{equation}
y_{\text{noise}, 3}(t) = \tilde{n}_3(t) \cdot \frac{\sigma_{\text{target}}}{\text{RMS}_{t}(\tilde{n}_3(t))}
\end{equation}

### 5.4 Category 4: Ambient Noise with Transient Disturbances (`transient_disturbances`)
Models cultural noise, anthropogenic microseisms, wind gusts, and sensor glitches via smooth, non-dispersive wavelets:

\begin{equation}
\tilde{n}_4(t) = \mathcal{H}_{\text{BP}}\{ \mathcal{N}(0, 1) \} + \sum_{k=1}^{K} A_k \exp\left( - \left(\frac{t - t_{0,k}}{\tau_{\text{trans},k}}\right)^2 \right) \sin(2\pi f_{\text{trans},k} (t - t_{0,k}))
\label{eq:noise_cat4}
\end{equation}

where:
- $K \sim \text{UniformInteger}(1, 3)$: Number of transient events.
- $t_{0,k} \sim \mathcal{U}(10.0, 90.0)\text{ s}$: Center arrival time.
- $\tau_{\text{trans},k} \sim \mathcal{U}(1.2, 4.0)\text{ s}$: Transient duration parameter.
- $f_{\text{trans},k} \sim \mathcal{U}(1.5, 6.0)\text{ Hz}$: Carrier frequency.
- $A_k \sim \mathcal{U}(1.8, 3.2) \cdot \text{std}(\tilde{n}_4)$: Wavelet peak amplitude.

\begin{equation}
y_{\text{noise}, 4}(t) = \tilde{n}_4(t) \cdot \frac{\sigma_{\text{target}}}{\text{RMS}_{t}(\tilde{n}_4(t))}
\end{equation}

---

## 6. Preprocessing, Waveform Transformation & Normalization

### 6.1 Strictly Causal Pre-P Context Standardization (`eew_causal_v3`)
To prevent future S-wave and late P-coda energy from leaking into pre-S causal observation windows via full-trace statistics, each observation window in `CausalEEWGeneratorV3` is standardized exclusively using moments computed over the $1.0\text{ s}$ ($N_{\text{pre}} = 100$ samples) pre-P context interval $W_{\text{pre}} = [P_{\text{sample}} - 100, P_{\text{sample}})$:

\begin{equation}
y_{\text{norm}}(t) = \frac{y(t) - \mu_{\text{pre}}}{\sigma_{\text{pre}} + \epsilon}
\label{eq:z_score}
\end{equation}

where $\mu_{\text{pre}} = \frac{1}{N_{\text{pre}}}\sum_{t \in W_{\text{pre}}} y(t)$, $\sigma_{\text{pre}} = \sqrt{\frac{1}{N_{\text{pre}}}\sum_{t \in W_{\text{pre}}} (y(t) - \mu_{\text{pre}})^2}$, and $\epsilon = 10^{-8}$.

### 6.2 Pointwise Signed-Logarithmic Transform (CNN Input)
To compress dynamic range while preserving polarity across orders of seismic amplitude:

\begin{equation}
\tilde{x}(t) = \text{sign}(y_{\text{norm}}(t)) \cdot \ln(1 + |y_{\text{norm}}(t)|)
\label{eq:signed_log}
\end{equation}

### 6.3 Causal Oracle-P Aligned Observation Windows
For observation horizon $h \in \{1.0, 2.0, 3.0\}\text{ s}$, each window consists of $1.0\text{ s}$ pre-P context ($100$ samples) and $h$ seconds post-P observation:

\begin{equation}
W_h = [P_{\text{sample}} - 100, \quad P_{\text{sample}} + h \cdot f_s)
\label{eq:eew_window}
\end{equation}

- $h = 1.0\text{ s}$: Window length $= 200$ samples ($2.0\text{ s}$).
- $h = 2.0\text{ s}$: Window length $= 300$ samples ($3.0\text{ s}$).
- $h = 3.0\text{ s}$: Window length $= 400$ samples ($4.0\text{ s}$).

### 6.4 Pre-S Causality Eligibility Mask
To ensure an earthquake record satisfies strictly pre-S EEW observation conditions:

\begin{equation}
\mathcal{M}_{\text{eligible}, h}^{(i)} = \begin{cases}
1, & \text{if Class } 0 \text{ (Noise)} \\
\mathbb{I}\left( t_S^{(i)} - t_P^{(i)} \ge h \right), & \text{if Class } 1 \text{ (Earthquake)}
\end{cases}
\label{eq:eligibility_mask}
\end{equation}

Records with $t_S - t_P < h$ are excluded from evaluation at horizon $h$.

---

## 7. Causal Tabular Feature Extraction (BiLSTM Inputs)

Within window $W_h$, normalized samples are partitioned into contiguous, non-overlapping $10$-sample bins ($0.1\text{ s}$ per bin at $100\text{ Hz}$). The first 10 bins represent pre-P reference context; subsequent $10 \cdot h$ bins represent the post-P observation. The total number of bins is $T_h \in \{20, 30, 40\}$ for $h \in \{1, 2, 3\}\text{ s}$.

For each bin $k \in \{0, \dots, T_h - 1\}$ containing samples $\mathbf{x} = [x_0, x_1, \dots, x_9]^T$, seven deterministic features are extracted:

\begin{equation}
f_1(\mathbf{x}) = \text{RMS} = \sqrt{\frac{1}{10} \sum_{i=0}^9 x_i^2}
\label{eq:feat_rms}
\end{equation}

\begin{equation}
f_2(\mathbf{x}) = \text{Mean Absolute Amplitude (MAA)} = \frac{1}{10} \sum_{i=0}^9 |x_i|
\label{eq:feat_maa}
\end{equation}

\begin{equation}
f_3(\mathbf{x}) = \text{Peak Absolute Amplitude (PAA)} = \max_{i \in \{0, \dots, 9\}} |x_i|
\label{eq:feat_paa}
\end{equation}

\begin{equation}
f_4(\mathbf{x}) = \text{Population Variance} = \frac{1}{10} \sum_{i=0}^9 (x_i - \bar{x})^2, \quad \bar{x} = \frac{1}{10}\sum_{i=0}^9 x_i
\label{eq:feat_var}
\end{equation}

\begin{equation}
f_5(\mathbf{x}) = \text{Signal Energy} = \sum_{i=0}^9 x_i^2 = 10 \cdot f_1(\mathbf{x})^2
\label{eq:feat_energy}
\end{equation}

\begin{equation}
f_6(\mathbf{x}) = \text{Zero-Crossing Rate (ZCR)} = \frac{1}{9} \sum_{i=0}^8 \mathbb{I}(x_i \cdot x_{i+1} < 0)
\label{eq:feat_zcr}
\end{equation}

\begin{equation}
f_7(\mathbf{x}) = \text{First-Difference Energy (FDE)} = \sum_{i=0}^8 (x_{i+1} - x_i)^2
\label{eq:feat_fde}
\end{equation}

### 7.1 Train-Only Standardization of Tabular Features
Each feature $j \in \{1, \dots, 7\}$ is standardized across all time bins using population statistics fitted exclusively on eligible training records:

\begin{equation}
\tilde{f}_j(\mathbf{x}) = \frac{f_j(\mathbf{x}) - \mu_{j, \text{train}}}{\sigma_{j, \text{train}} + \epsilon}
\label{eq:feat_standardize}
\end{equation}

Validation and test sets are transformed using frozen training scalers $(\mu_{j, \text{train}}, \sigma_{j, \text{train}})$.

---

## 8. Multi-Task Loss Formulations

For input $\mathbf{x}_i$, binary label $y_i \in \{0, 1\}$, and earthquake magnitude $M_i$:

### 8.1 Binary Detection Loss (Binary Cross-Entropy)
\begin{equation}
\mathcal{L}_{\text{BCE}}(z_i, y_i) = - \left[ y_i \ln \sigma(z_i) + (1 - y_i) \ln (1 - \sigma(z_i)) \right]
\label{eq:loss_bce}
\end{equation}

where $z_i$ is the predicted detection logit and $\sigma(z) = 1 / (1 + e^{-z})$.

### 8.2 Standardized Magnitude Huber Loss
Magnitude regression is evaluated exclusively on eligible earthquake records ($y_i = 1$). Targets are standardized using training earthquake mean $\mu_{M, \text{train}}$ and population standard deviation $\sigma_{M, \text{train}}$:

\begin{equation}
\tilde{M}_i = \frac{M_i - \mu_{M, \text{train}}}{\sigma_{M, \text{train}}}
\label{eq:mag_standardize}
\end{equation}

\begin{equation}
e_i = \hat{m}_i - \tilde{M}_i
\label{eq:mag_error}
\end{equation}

\begin{equation}
\mathcal{L}_{\text{Huber}}(e_i; \delta) = \begin{cases}
\frac{1}{2} e_i^2, & |e_i| \le \delta \\
\delta \left( |e_i| - \frac{1}{2}\delta \right), & |e_i| > \delta
\end{cases}
\label{eq:loss_huber}
\end{equation}

with threshold $\delta = 1.0$.

### 8.3 Joint Multi-Task Objective
\begin{equation}
\mathcal{L}_{\text{total}} = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{\text{BCE}}(z_i, y_i) + \lambda_M \cdot \frac{1}{\sum_{i=1}^N y_i} \sum_{i: y_i=1} \mathcal{L}_{\text{Huber}}(e_i; \delta=1.0)
\label{eq:loss_total}
\end{equation}

with magnitude loss weight $\lambda_M = 1.0$. Sentinel magnitude values for noise ($M = -1$) are strictly excluded from regression gradients.
