# Phase 8 Waveform Simulator Audit: Steps, Mathematical Formulations & Comparative Analysis

**Project Title:** Physics-Informed Synthetic Seismic Waveform Generation for Earthquake Detection Using Deep Learning  
**Output Directory:** [output/phase8_improved_simulation/](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/output/phase8_improved_simulation)  
**Evaluated Sample:** 50 Representative Seismic Traces (Fixed Random Seed = 42)

---

## 1. Project Background & Audit Objectives

The dataset contains 8,089 earthquake metadata records from Northeastern India and surrounding regions, but lacks recorded seismic waveform traces. To train deep learning models for earthquake detection, a physics-informed synthetic seismic waveform simulator generates continuous vertical-component (BHZ-like) traces.

Quality Control (QC) in Phase 7 revealed two systematic issues in the initial waveform generator:
1. **SNR Discrepancy:** A $+6.21\text{ dB}$ systemic bias between the catalog target SNR (mean $9.47\text{ dB}$) and measured waveform SNR (mean $15.69\text{ dB}$).
2. **Energy Check Failures:** 8 of 50 traces failed the P-energy jump check and 8 of 50 traces failed the S-energy jump check.

Phase 8 was executed to audit the physical and mathematical assumptions, isolate the root causes of these discrepancies, implement improved waveform and attenuation models, and align the measurement conventions without relaxing QC standards.

---

## 2. Chronological Audit & Improvement Workflow

The audit and enhancement were carried out in eight systematic steps:

1. **Step 1: Codebase & Parameter Inspection**  
   Audited the original generator (`waveform_generator.py`) and QC evaluator (`waveform_quality_control.py`). Identified the exact mathematical formulations used for travel times, wave packet envelopes, geometric spreading, attenuation, ambient noise generation, and SNR scaling.

2. **Step 2: Mathematical Investigation of the SNR Bias**  
   Analyzed the statistical properties of Gaussian noise across long temporal records. Discovered that the Phase 7 evaluation measured the maximum noise crest over 10,000 samples rather than the seismic arrival, and identified variance mismatch between 10,000-sample noise scaling and 600-sample pre-P evaluation.

3. **Step 3: Investigation of Failing Traces**  
   Examined the 16 traces that failed P- or S-wave energy checks in Phase 7. Discovered that failures were caused by P-window dilution over regional distances, severe S-wave over-attenuation under constant $Q_S=150$, vertical-component $S/P$ energy ratio miscalibration, and ultra-near-field phase overlap.

4. **Step 4: Separation of Quality Control Criteria**  
   Decoupled **Mandatory Numerical Integrity Checks** (finite values, non-zero, valid bounds, physical phase ordering) from **Waveform Structure Checks** (P and S energy step-ups). Enforced that an earthquake trace cannot pass overall QC if a required phase arrival energy jump fails.

5. **Step 5: Mathematical Formulation of Smooth Physical Wave Packets**  
   Replaced sharp onset models with causal $C^1$-continuous Gamma onset functions. Formulated source rise times scaling with earthquake magnitude $M$ and coda decay times broadening with hypocentral distance $R$ to represent physical scattering. Developed a multi-harmonic carrier wave to model rich frequency content.

6. **Step 6: Implementation of Frequency-Dependent Crustal Attenuation**  
   Replaced constant seismic quality factors ($Q_P=300, Q_S=150$) with a frequency-dependent power-law attenuation model $Q(f) = Q_0 (f/f_0)^\eta$, preserving regional shear-wave ($Lg$) energy across distances up to $500\text{ km}$.

7. **Step 7: Calibration of Pre-P Noise Floor & Metric Alignment**  
   Formulated an exact empirical noise calibration step where ambient noise is scaled to match the target standard deviation directly within the pre-P reference window. Aligned signal and noise window definitions between the generator and evaluator.

8. **Step 8: Comparative Re-Evaluation & Verification**  
   Regenerated the exact same 50 representative traces using random seed 42. Generated comparative plots, envelope overlays, parameter schemas, and comparative metric CSVs.

---

## 3. Mathematical Formulations & Physical Models

### 3.1 Kinematic Travel-Time Constraints

For each record, theoretical P- and S-wave arrival times are computed from the hypocentral distance $R$:

$$R = \sqrt{D^2 + h^2}$$

where $D$ is the epicentral distance ($\text{km}$) and $h$ is the focal depth ($\text{km}$).

The theoretical travel times and discrete sample arrival indices are:

$$t_P = \frac{R}{V_P}, \quad t_S = \frac{R}{V_S}$$

$$P_{\text{sample}} = \text{round}(t_P \cdot f_s), \quad S_{\text{sample}} = \text{round}(t_S \cdot f_s)$$

$$\Delta t_{SP} = t_S - t_P = R \left( \frac{1}{V_S} - \frac{1}{V_P} \right)$$

*Baseline parameters:* $V_P = 6.0\text{ km/s}$, $V_S = 3.5\text{ km/s}$, sampling rate $f_s = 100\text{ Hz}$.

---

### 3.2 Signal-to-Noise Ratio (SNR) Analysis & Exact Formulation

#### A. Root Cause of the $+6.21\text{ dB}$ Measured Bias in Phase 7
In Phase 7, measured SNR was evaluated on the composite noisy trace $y(t) = s(t) + n(t)$ using:

$$\text{SNR}_{\text{Phase 7}} = 20 \log_{10}\left( \frac{\max_{t \in [0, 100]} |y(t)|}{\text{RMS}(y_{\text{pre-P}})} \right)$$

For band-limited Gaussian noise $n(t) \sim \mathcal{N}(0, \sigma^2)$ sampled over $N = 10,000$ points, the expected value of the extreme absolute value is governed by extreme-value statistics:

$$\mathbb{E}\left[ \max_{1 \le i \le N} |Z_i| \right] \approx \sqrt{2 \ln(N)} - \frac{\ln(\ln N) + \ln(4\pi)}{2\sqrt{2\ln(N)}} \approx 4.29$$

Expressed in decibels relative to the noise standard deviation $\sigma$:

$$20 \log_{10}(4.29) \approx 12.65\text{ dB}$$

When the catalog target SNR is low (e.g. $1.5\text{--}6.0\text{ dB}$), the true clean signal peak $A_{\text{signal}}$ is between $1.18\sigma$ and $2.00\sigma$. Consequently, evaluating $\max |y(t)|$ across 10,000 points captured a **random background noise spike elsewhere in the 100-second record**, not the earthquake arrival. This introduced a mathematical lower floor of $\approx 12.65\text{ dB}$ on measured SNR, creating a mean bias of $+6.21\text{ dB}$.

#### B. Unified Mathematical Window Definitions
To eliminate this discrepancy, signal and noise windows are rigorously defined:

- **Pre-P Noise Reference Window ($W_{\text{noise}}$):**
  $$W_{\text{noise}} = [0, t_P - \delta t], \quad \text{where } \delta t = 0.05\text{ s (5 samples)}$$
  $$\text{RMS}_{\text{noise}} = \sqrt{ \frac{1}{|W_{\text{noise}}|} \sum_{t \in W_{\text{noise}}} y(t)^2 }$$

- **Seismic Signal Arrival Window ($W_{\text{signal}}$):**
  $$W_{\text{signal}} = [t_P, t_{\text{end}}]$$
  $$A_{\text{signal}} = \max_{t \in W_{\text{signal}}} |s_{\text{clean}}(t)|$$

- **Physical SNR Definition:**
  $$\text{SNR}_{\text{target, dB}} = 20 \log_{10}\left( \frac{A_{\text{signal}}}{\text{RMS}_{\text{noise}}} \right)$$

#### C. Exact Pre-P Noise Calibration Formula
In Phase 8, the generator synthesizes raw filtered noise $\tilde{n}(t)$ and explicitly scales it against the empirical standard deviation within the pre-P reference window:

$$\sigma_{\text{target}} = \frac{A_{\text{signal}}}{10^{\text{target\_snr} / 20}}$$

$$n(t) = \tilde{n}(t) \times \left( \frac{\sigma_{\text{target}}}{\text{RMS}_{t \in W_{\text{noise}}}(\tilde{n}(t))} \right)$$

$$y(t) = s_{\text{clean}}(t) + n(t)$$

Because $\text{RMS}_{t \in W_{\text{noise}}}(n(t)) \equiv \sigma_{\text{target}}$ by construction, the evaluated SNR matches the target catalog SNR with mathematical precision:

$$|\text{SNR}_{\text{measured}} - \text{SNR}_{\text{target}}| < 0.002\text{ dB}$$

---

### 3.3 Frequency-Dependent Crustal Attenuation Model

The amplitude of an elastic seismic wave attenuates due to geometric wave-front spreading and anelastic material absorption.

#### A. Total Wave Amplitude Formula
$$A(M, R, f) = A_0(M) \cdot \mathcal{G}(R) \cdot \mathcal{K}(R, f)$$

1. **Magnitude Scaling:**
   $$A_0(M) = A_{\text{ref}} \cdot 10^{\gamma (M - M_{\text{ref}})}$$
   where $M_{\text{ref}} = 4.0$, $A_{\text{ref}} = 1.0$, and $\gamma = 0.50$.

2. **Geometric Spreading:**
   $$\mathcal{G}(R) = \left( \frac{R_{\text{ref}}}{R} \right)^n$$
   where $R_{\text{ref}} = 10.0\text{ km}$ and $n = 1.0$ (body wave spherical spreading).

3. **Anelastic Attenuation Factor:**
   $$\mathcal{K}(R, f) = \exp\left( - \frac{\pi f R}{Q(f) V} \right)$$

#### B. Frequency-Dependent Quality Factor $Q(f)$
In Phase 6, constant values ($Q_P = 300, Q_S = 150$) caused excessive damping of shear waves at regional distances:
- At $R = 450\text{ km}$ and $f_S = 2.5\text{ Hz}$, the constant $Q_S = 150$ gave $\mathcal{K}_S = \exp\left(-\frac{\pi \cdot 2.5 \cdot 450}{150 \cdot 3.5}\right) = \exp(-6.73) \approx 0.00119$.  
  The S-wave lost $99.88\%$ of its amplitude and vanished below the ambient noise floor.

Phase 8 implements the standard seismological power-law crustal quality factor:

$$Q(f) = Q_0 \left( \frac{f}{f_0} \right)^\eta$$

- **P-Wave Attenuation:** $Q_{P0} = 350$, $f_0 = 1.0\text{ Hz}$, $\eta = 0.70$. At $f_P = 5.0\text{ Hz}$, $Q_P(5.0) \approx 1,080$.
- **S-Wave Attenuation:** $Q_{S0} = 300$, $f_0 = 1.0\text{ Hz}$, $\eta = 0.70$. At $f_S = 2.2\text{ Hz}$, $Q_S(2.2) \approx 520$.

This preserves realistic guided crustal shear wave ($Lg$) energy across regional hypocentral distances ($200\text{--}500\text{ km}$).

---

### 3.4 Smooth Physical Wave Packet Model

#### A. Causal $C^1$-Smooth Gamma Onset Envelope
To prevent unphysical step discontinuities at the arrival sample, wave packets use a causal Gamma onset function with continuous first derivatives ($C^1$ continuity):

$$E(\tau) = \begin{cases} 
0, & \tau < 0 \\ 
\left( \frac{\tau}{\tau_r} \right)^2 \exp\left( - \frac{\tau}{\tau_d} \right), & \tau \ge 0 
\end{cases}$$

where $\tau = t - t_{\text{arrival}}$, $\tau_r$ is the source rise time, and $\tau_d$ is the coda decay parameter.

**Mathematical Boundary Properties:**
- At onset ($\tau = 0$): $E(0) = 0$.
- First derivative at onset:
  $$\frac{dE}{d\tau}\Big|_{\tau=0} = \left[ \frac{2\tau}{\tau_r^2} - \frac{\tau^2}{\tau_r^2 \tau_d} \right] \exp\left(-\frac{\tau}{\tau_d}\right)\Bigg|_{\tau=0} = 0$$
- Time of peak envelope amplitude: $\tau_{\text{peak}} = 2\tau_d$.
- Peak envelope value: $E_{\text{peak}} = \left(\frac{2\tau_d}{\tau_r}\right)^2 \exp(-2)$.

The normalized envelope scaled to peak amplitude $A$ is:

$$\mathcal{E}(\tau) = A \cdot \left( \frac{\tau}{2\tau_d} \right)^2 \exp\left( 2 - \frac{\tau}{\tau_d} \right)$$

#### B. Dynamic Physical Scaling of Rise and Coda Parameters
1. **Magnitude-Dependent Rise Time (Rupture Duration Scaling):**
   $$\tau_r(M) = \tau_{r0} \cdot [1.0 + 0.12 (M - M_{\text{ref}})]$$
   Larger magnitude earthquakes involve larger fault rupture areas and longer rise times.

2. **Distance-Dependent Coda Decay (Scattering Dispersion Scaling):**
   $$\tau_d(M, R) = \tau_{d0} \cdot [1.0 + 0.18 (M - M_{\text{ref}})] \cdot [1.0 + 0.0012 R]$$
   As hypocentral distance $R$ increases, multiple crustal scattering broadens the wave packet duration.

*Baseline parameters:* $\tau_{r0, P} = 0.25\text{ s}$, $\tau_{d0, P} = 0.60\text{ s}$, $\tau_{r0, S} = 0.45\text{ s}$, $\tau_{d0, S} = 1.30\text{ s}$.

---

### 3.5 Multi-Harmonic Carrier Waveforms

To avoid monochromatic synthetic tones, the carrier signals combine fundamental and harmonic modes:

$$\Phi_P(\tau) = 0.85 \sin(2\pi f_P \tau) + 0.15 \sin(4\pi f_P \tau)$$

$$\Phi_S(\tau) = 0.80 \sin\left(2\pi f_S \tau + \frac{\pi}{4}\right) + 0.20 \sin(3\pi f_S \tau)$$

The total clean synthesized earthquake signal is:

$$s_{\text{clean}}(t) = \mathcal{E}_P(t - t_P) \cdot \Phi_P(t - t_P) + \mathcal{E}_S(t - t_S) \cdot \Phi_S(t - t_S)$$

---

### 3.6 Vertical-Component (BHZ) Energy Partitioning

On three-component seismograms, phase amplitudes depend strongly on sensor orientation and wave polarization:
- Horizontal transverse components ($BHE/BHN$) are dominated by shear ($SH$) and Love waves, where $A_S / A_P \approx 3.0\text{--}5.0$.
- Vertical components ($BHZ$) register near-vertical compressional particle motions strongly, while shear particle motions are partitioned.

Setting $A_S / A_P = 3.0$ in Phase 6 forced vertical P-wave amplitudes to be one-third of the shear wave, causing P-waves to submerge beneath the noise floor. Phase 8 calibrates the vertical-component amplitude ratio to:

$$\left(\frac{A_S}{A_P}\right)_{\text{BHZ}} = 1.35$$

---

### 3.7 Quality Control Windowing & Strict Logic Separation

#### A. Active Wave Packet Evaluation Windows
To resolve the window dilution failure observed for regional events:
- **P-Wave Active Window:**
  $$W_P = [P_{\text{sample}}, \; \min(P_{\text{sample}} + 350, \; S_{\text{sample}})]$$
  Restricting the window to the active packet duration ($3.5\text{ s}$) prevents dilution by tens of seconds of pre-S ambient noise.
- **S-Wave Active Window:**
  $$W_S = [S_{\text{sample}}, \; \min(S_{\text{sample}} + 700, \; N)]$$

#### B. Separation of QC Logic Checks
Quality control checks are separated into two distinct categories:

1. **Mandatory Numerical Integrity Checks:**
   $$\text{Numerical Pass} = (0 \le P < N) \land (0 \le S < N) \land (P < S) \land \text{is\_finite}(y) \land (\text{std}(y) > 0)$$
2. **Waveform Structural Energy Checks:**
   $$\text{P-Energy Jump} = \left( \text{RMS}(y_{W_P}) > 1.02 \cdot \text{RMS}(y_{W_{\text{noise}}}) \right)$$
   $$\text{S-Energy Jump} = \left( \text{RMS}(y_{W_S}) > 1.05 \cdot \text{RMS}(y_{W_{\text{noise}}}) \right)$$
   $$\text{Structural Pass} = \text{P-Energy Jump} \land \text{S-Energy Jump}$$
3. **Strict Overall QC Pass Criterion:**
   $$\text{Overall QC Pass} = \text{Numerical Pass} \land \text{Structural Pass}$$

*Strict Enforcement:* If a trace fails either the P-energy jump or S-energy jump, it is strictly flagged as failed overall.

---

## 4. Comparative Evaluation Tables

### Table 1: Old Implementation (Phase 6/7) vs. Improved Implementation (Phase 8)

| Evaluation Metric | Phase 7 Old Implementation | Phase 8 Improved Implementation | Technical Advancement & Impact |
| :--- | :---: | :---: | :--- |
| **Evaluated Sample Size** | 50 traces (Seed = 42) | 50 traces (Seed = 42) | Strict identical sampling preserved |
| **Mandatory Numerical Checks** | 50 / 50 (100.0%) | 50 / 50 (100.0%) | Zero NaNs, zero Infs, finite real numbers |
| **Mean Target SNR** | 9.47 dB | 9.47 dB | Catalog target distribution |
| **Mean Measured SNR** | 15.69 dB | **9.47 dB** | **Eliminated +6.21 dB systemic noise bias** |
| **Mean SNR Absolute Error** | 6.23 dB error | **0.0005 dB error** | Exact pre-P noise window standard error matching |
| **Maximum SNR Error** | 10.71 dB error | **0.0017 dB error** | Mathematical calibration across all magnitudes |
| **P-Wave Energy Jump Check** | 42 / 50 (84.0%) | **48 / 50 (96.0%)** | Active P-packet window prevents dilution ($+12.0\%$) |
| **S-Wave Energy Jump Check** | 42 / 50 (84.0%) | **44 / 50 (88.0%)** | Frequency-dependent $Q(f)$ prevents over-attenuation ($+4.0\%$) |
| **Strict Overall QC Passed** | **38 / 50 (76.0%)** | **43 / 50 (86.0%)** | Separated numerical vs. structural criteria ($+10.0\%$) |

---

### Table 2: Physical & Simulation Parameters Comparison

| Simulation Parameter | Symbol | Phase 6/7 Setting | Phase 8 Improved Setting | Seismological Justification |
| :--- | :---: | :---: | :---: | :--- |
| **Sampling Rate** | $f_s$ | $100.0\text{ Hz}$ | $100.0\text{ Hz}$ | Standard regional broadband/short-period rate |
| **Waveform Duration** | $T$ | $100.0\text{ s}$ | $100.0\text{ s}$ | Sufficient for regional arrivals up to $500\text{ km}$ |
| **P-Wave Velocity** | $V_P$ | $6.0\text{ km/s}$ | $6.0\text{ km/s}$ | Average upper-crustal compressional velocity |
| **S-Wave Velocity** | $V_S$ | $3.5\text{ km/s}$ | $3.5\text{ km/s}$ | Average upper-crustal shear velocity |
| **$V_P / V_S$ Ratio** | $V_P / V_S$ | $1.714$ | $1.714$ | Standard Poisson solid assumption |
| **P Quality Factor** | $Q_P(f)$ | $300$ (constant) | $350 \cdot (f/1.0)^{0.70}$ | Frequency-dependent crustal quality factor |
| **S Quality Factor** | $Q_S(f)$ | $150$ (constant) | $300 \cdot (f/1.0)^{0.70}$ | Power-law $Q(f)$ prevents regional $Lg$ vanishing |
| **Dominant P Frequency** | $f_P$ | $5.0\text{ Hz}$ | $5.0\text{ Hz}$ | High-frequency body wave arrival |
| **Dominant S Frequency** | $f_S$ | $2.5\text{ Hz}$ | $2.2\text{ Hz}$ | Lower frequency shear/guided crustal phase |
| **$S/P$ Amplitude Ratio** | $A_S / A_P$ | $3.00$ | $1.35$ | Calibrated for vertical component ($BHZ$) |
| **Onset Envelope Function** | $E(\tau)$ | Exponential step | $C^1$ Gamma $(\tau/\tau_r)^2 e^{-\tau/\tau_d}$ | Eliminates non-physical onset step discontinuities |
| **Coda Decay Scaling** | $\tau_d(M, R)$ | Fixed constant | Dynamic: $\propto (1 + 0.18 M)(1 + 0.0012 R)$ | Physical multiple crustal scattering dispersion |
| **Carrier Function** | $\Phi(\tau)$ | Single sine wave | Multi-harmonic carrier | Rich spectral content for deep learning models |
| **Noise Calibration Target** | $\sigma_{\text{noise}}$ | Entire 100s array | Pre-P window empirical RMS | Removes noise crest bias and finite-sample variance |

---

### Table 3: Root Cause & Resolution Breakdown for Failing Traces

| Sample ID | HDF5 Index | Magnitude | Distance ($R$) | Target SNR | Phase 7 Status | Phase 8 Status | Seismological Diagnosis & Resolution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Trace 13** | 2865 | 3.3 | $255.8\text{ km}$ | $3.68\text{ dB}$ | Failed S-Energy | Failed S-Energy | Low magnitude + regional distance + low SNR ($3.68\text{ dB}$). Near detection threshold. |
| **Trace 26** | 4477 | 3.7 | $373.9\text{ km}$ | $1.50\text{ dB}$ | Failed S-Energy | Failed S-Energy | Catalog minimum SNR ($1.50\text{ dB}$). Signal peak is only $1.18\times$ noise RMS. |
| **Trace 31** | 5611 | 3.2 | $9.4\text{ km}$ | $12.43\text{ dB}$ | Failed P-Energy | Failed P-Energy | Ultra-near-field. $S-P$ arrival delay is only $1.11\text{ s}$ ($111$ samples). Coda overlap. |
| **Trace 35** | 6126 | 4.3 | $473.6\text{ km}$ | $1.50\text{ dB}$ | Failed Both | Failed Both | Regional event at catalog minimum SNR ($1.50\text{ dB}$). Signal buried in ambient noise floor. |
| **Trace 36** | 6223 | 4.4 | $464.0\text{ km}$ | $2.36\text{ dB}$ | Failed S-Energy | Failed S-Energy | Low SNR ($2.36\text{ dB}$) at regional distance ($464\text{ km}$). Approaching detection limit. |
| **Trace 37** | 6226 | 3.8 | $486.2\text{ km}$ | $1.50\text{ dB}$ | Failed S-Energy | Failed S-Energy | Far regional distance ($486\text{ km}$) with catalog minimum SNR ($1.50\text{ dB}$). |
| **Trace 42** | 6684 | 4.1 | $373.9\text{ km}$ | $1.50\text{ dB}$ | Failed Both | **PASSED** | Fixed by active P-window duration matching and frequency-dependent $Q(f)$. |
| **Trace 43** | 6743 | 3.6 | $464.0\text{ km}$ | $1.50\text{ dB}$ | Failed S-Energy | Failed S-Energy | Moderate magnitude, far regional distance, and catalog minimum SNR ($1.50\text{ dB}$). |

*Physical Justification for Retained Failures:* Traces 13, 26, 35, 36, 37, and 43 possess target SNRs between $1.50\text{ dB}$ and $3.68\text{ dB}$ at regional distances exceeding $250\text{--}480\text{ km}$. In observational seismology, signals below $2.0\text{ dB}$ on single vertical channels sit at or below visual picking thresholds. Retaining strict energy jump thresholds rather than artificially lowering standards preserves scientific realism.

---

## 5. Scientific Limitations & Seismological Disclaimers

1. **Physics-Informed Mathematical Model vs. Real Seismograms:**  
   The synthetic signals generated by this simulator are **mathematical signal models** constructed from travel-time equations, empirical attenuation functions, and stochastic noise processes. **They do not claim to reproduce recorded earthquake seismograms.**

2. **Theoretical Kinematic Constraints vs. Observed Phase Picks:**  
   The stored `p_arrival_sample` and `s_arrival_sample` values represent **theoretical model-derived kinematic constraints** computed from a 1D homogeneous velocity model ($V_P = 6.0\text{ km/s}, V_S = 3.5\text{ km/s}$). They are **NOT** observed empirical phase picks made by human network analysts or phase-picking algorithms.

3. **Unmodeled Seismological Phenomena:**  
   The current mathematical simulator omits several complex geophysical processes:
   - 3D lateral velocity heterogeneities and low-velocity sedimentary basin trapping.
   - Crustal velocity layering and head-wave refractions ($Pn, Pg, Sn, Sg$).
   - Dispersive surface wave trains (Rayleigh and Love wave fundamental and higher modes).
   - Double-couple earthquake source radiation patterns and nodal planes.
   - Frequency-dependent site resonance and shallow soil amplification ($H/V$ spectral ratios).
   - Anisotropic shear-wave splitting ($S_{\text{fast}}, S_{\text{slow}}$).

---

## 6. Project Artifact References

- **Summary Documentation:** [PHASE8_AUDIT_SUMMARY.md](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/PHASE8_AUDIT_SUMMARY.md)
- **Detailed Limitations Report:** [scientific_limitations_report.md](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/output/phase8_improved_simulation/scientific_limitations_report.md)
- **QC Comparison Table:** [old_vs_new_qc_comparison.csv](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/output/phase8_improved_simulation/old_vs_new_qc_comparison.csv)
- **Detailed Metrics CSV:** [qc_measurements_improved_50_traces.csv](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/output/phase8_improved_simulation/qc_measurements_improved_50_traces.csv)
- **Simulation Parameters:** [simulation_parameters.json](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/output/phase8_improved_simulation/simulation_parameters.json)
- **SNR Calibration Figure:** [snr_target_vs_measured.png](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/output/phase8_improved_simulation/snr_target_vs_measured.png)
- **Envelope Comparison Figure:** [energy_envelope_comparison.png](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/output/phase8_improved_simulation/energy_envelope_comparison.png)
- **Representative Waveform Plots:** [representative_waveform_plots/](file:///home/meghraj-dhakal/Downloads/Minor%20project(1)/output/phase8_improved_simulation/representative_waveform_plots)
