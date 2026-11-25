"""
Improved Physics-Informed Synthetic Seismic Waveform Generator
Module: improved_waveform_generator.py

Key Scientific Improvements (Phase 8):
1. Frequency-dependent crustal quality factor Q(f) = Q_0 * (f / f_0)^eta to prevent
   unphysical over-attenuation of regional shear waves.
2. Physically scaled wave packet envelopes: rise time and coda decay constants dynamically
   scale with earthquake magnitude M (source rupture duration) and hypocentral distance R
   (scattering and dispersion).
3. Multi-harmonic causal wave packet formulation with C^1 continuous gamma onset envelopes.
4. Vertical-component (BHZ) realistic phase amplitude partitioning (compressional P-wave
   vertical particle motion).
5. Exact mathematical Signal-to-Noise Ratio (SNR) calibration across the pre-P measurement window.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.signal import butter, sosfilt

@dataclass
class ImprovedSimulationParameters:
    sampling_rate_hz: float = 100.0
    duration_seconds: float = 100.0
    vp_km_s: float = 6.0
    vs_km_s: float = 3.5
    ref_magnitude: float = 4.0
    ref_amplitude: float = 1.0
    mag_scaling_coeff: float = 0.5
    geometric_spreading_exp: float = 1.0
    ref_distance_km: float = 10.0
    
    # Crustal Q parameters (frequency-dependent: Q(f) = Q_0 * (f / f_0)^eta)
    qp_0: float = 350.0
    qs_0: float = 300.0
    q_eta: float = 0.70
    q_ref_freq_hz: float = 1.0

    # Spectral parameters
    p_dominant_freq_hz: float = 5.0
    s_dominant_freq_hz: float = 2.2
    sp_amp_ratio_bhz: float = 1.35  # Realistic vertical component (BHZ) S/P amplitude ratio

    # Dynamic envelope parameters
    tau_p_base_s: float = 1.5
    tau_s_base_s: float = 4.0
    noise_band_hz: Tuple[float, float] = (0.2, 25.0)
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)


class ImprovedWaveformGenerator:
    """
    Advanced Physics-Informed Seismic Waveform Simulator.
    """
    def __init__(self, params: Optional[ImprovedSimulationParameters] = None):
        self.params = params if params is not None else ImprovedSimulationParameters()
        self.fs = self.params.sampling_rate_hz
        self.dt = 1.0 / self.fs
        self.n_samples = int(np.round(self.params.duration_seconds * self.fs))
        self.time_axis = np.linspace(0.0, (self.n_samples - 1) * self.dt, self.n_samples)
        
        # Precompute bandpass filter for seismic ambient noise
        self.noise_sos = butter(
            4,
            [self.params.noise_band_hz[0], self.params.noise_band_hz[1]],
            btype="bandpass",
            fs=self.fs,
            output="sos"
        )
        self.rng = np.random.default_rng(self.params.random_seed)

    def calculate_effective_amplitudes(
        self,
        magnitude: float,
        hypocentral_distance_km: float
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Calculates source-scaled, geometrically spread, and anelastically attenuated amplitudes
        incorporating frequency-dependent crustal quality factor Q(f).
        """
        R = max(hypocentral_distance_km, self.params.ref_distance_km)

        # 1. Source magnitude scaling: A_source = A_0 * 10^(c * (M - M_ref))
        source_factor = self.params.ref_amplitude * (
            10.0 ** (self.params.mag_scaling_coeff * (magnitude - self.params.ref_magnitude))
        )

        # 2. Body wave geometric spreading: G(R) = (R_0 / R)^gamma
        geom_factor = (self.params.ref_distance_km / R) ** self.params.geometric_spreading_exp

        # 3. Frequency-dependent Q(f) = Q_0 * (f / f_0)^eta
        qp = self.params.qp_0 * (self.params.p_dominant_freq_hz / self.params.q_ref_freq_hz) ** self.params.q_eta
        qs = self.params.qs_0 * (self.params.s_dominant_freq_hz / self.params.q_ref_freq_hz) ** self.params.q_eta

        # 4. Anelastic attenuation kappa = exp(-pi * f * R / (Q * v))
        kappa_p = np.exp(- (np.pi * self.params.p_dominant_freq_hz * R) / (qp * self.params.vp_km_s))
        kappa_s = np.exp(- (np.pi * self.params.s_dominant_freq_hz * R) / (qs * self.params.vs_km_s))

        amp_p = source_factor * geom_factor * kappa_p
        amp_s = source_factor * geom_factor * kappa_s * self.params.sp_amp_ratio_bhz

        diag = {
            "source_factor": float(source_factor),
            "geom_factor": float(geom_factor),
            "qp": float(qp),
            "qs": float(qs),
            "kappa_p": float(kappa_p),
            "kappa_s": float(kappa_s),
            "amp_p": float(amp_p),
            "amp_s": float(amp_s)
        }
        return float(amp_p), float(amp_s), diag

    def generate_dynamic_envelopes(
        self,
        magnitude: float,
        hypocentral_distance_km: float
    ) -> Tuple[float, float, float, float]:
        """
        Computes dynamic rise times and coda decay constants as functions of
        earthquake magnitude M (source rupture duration) and distance R (dispersion/scattering).
        """
        R = max(hypocentral_distance_km, self.params.ref_distance_km)
        m_diff = magnitude - self.params.ref_magnitude

        # P-wave envelope parameters
        tau_d_p = self.params.tau_p_base_s * (1.0 + 0.12 * m_diff) * (1.0 + 0.0010 * R)
        tau_d_p = max(tau_d_p, 0.6)
        tau_r_p = tau_d_p * 0.28  # Rapid compressional rise

        # S-wave envelope parameters
        tau_d_s = self.params.tau_s_base_s * (1.0 + 0.18 * m_diff) * (1.0 + 0.0015 * R)
        tau_d_s = max(tau_d_s, 1.5)
        tau_r_s = tau_d_s * 0.22  # Shear wave rise

        return float(tau_r_p), float(tau_d_p), float(tau_r_s), float(tau_d_s)

    def generate_p_wave(
        self,
        p_arrival_sec: float,
        amplitude: float,
        tau_r: float,
        tau_d: float
    ) -> np.ndarray:
        """
        Generates P-wave packet with C^1 continuous gamma envelope and harmonic carrier.
        """
        pw = np.zeros(self.n_samples, dtype=np.float64)
        mask = self.time_axis >= p_arrival_sec
        tau = self.time_axis[mask] - p_arrival_sec

        envelope = (tau / tau_r) ** 2 * np.exp(-tau / tau_d)
        max_env = np.max(envelope) if len(envelope) > 0 and np.max(envelope) > 0 else 1.0
        envelope = envelope / max_env

        # Multi-frequency carrier with primary tone and subtle dispersion harmonic
        fp = self.params.p_dominant_freq_hz
        carrier = 0.85 * np.sin(2.0 * np.pi * fp * tau) + 0.15 * np.sin(4.0 * np.pi * fp * tau)

        pw[mask] = amplitude * envelope * carrier
        return pw

    def generate_s_wave(
        self,
        s_arrival_sec: float,
        amplitude: float,
        tau_r: float,
        tau_d: float
    ) -> np.ndarray:
        """
        Generates S-wave packet with lower shear frequency and extended dispersive coda.
        """
        sw = np.zeros(self.n_samples, dtype=np.float64)
        mask = self.time_axis >= s_arrival_sec
        tau = self.time_axis[mask] - s_arrival_sec

        envelope = (tau / tau_r) ** 2 * np.exp(-tau / tau_d)
        max_env = np.max(envelope) if len(envelope) > 0 and np.max(envelope) > 0 else 1.0
        envelope = envelope / max_env

        fs = self.params.s_dominant_freq_hz
        carrier = 0.80 * np.sin(2.0 * np.pi * fs * tau + np.pi / 4.0) + 0.20 * np.sin(3.0 * np.pi * fs * tau)

        sw[mask] = amplitude * envelope * carrier
        return sw

    def generate_synthesized_trace(
        self,
        p_arrival_sample: int,
        s_arrival_sample: int,
        magnitude: float,
        hypocentral_distance_km: float,
        target_snr_db: float,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes a complete seismic trace with exact mathematical pre-P SNR calibration.
        """
        rng = np.random.default_rng(seed if seed is not None else self.params.random_seed)

        tp_sec = p_arrival_sample / self.fs
        ts_sec = s_arrival_sample / self.fs

        # 1. Calculate physical amplitudes and dynamic envelope constants
        amp_p, amp_s, diag = self.calculate_effective_amplitudes(magnitude, hypocentral_distance_km)
        tau_r_p, tau_d_p, tau_r_s, tau_d_s = self.generate_dynamic_envelopes(magnitude, hypocentral_distance_km)

        # 2. Synthesize clean P and S waves
        pw = self.generate_p_wave(tp_sec, amp_p, tau_r_p, tau_d_p)
        sw = self.generate_s_wave(ts_sec, amp_s, tau_r_s, tau_d_s)
        clean_signal = pw + sw

        # 3. Exact Mathematical SNR Calibration
        # Signal peak evaluated in the valid signal region [p_arrival_sample, end]
        signal_peak = float(np.max(np.abs(clean_signal[p_arrival_sample:])))
        if signal_peak <= 0:
            signal_peak = 1.0

        # Required pre-P noise standard deviation to strictly achieve target SNR
        target_noise_sigma = signal_peak / (10.0 ** (target_snr_db / 20.0))

        # Generate band-limited colored noise
        raw_noise = rng.normal(0.0, 1.0, self.n_samples)
        filtered_noise = sosfilt(self.noise_sos, raw_noise)

        # Measure noise strictly over pre-P noise window to eliminate window sample discrepancy
        pre_p_samples = max(p_arrival_sample - 5, 50)
        pre_p_raw_std = np.std(filtered_noise[:pre_p_samples])
        if pre_p_raw_std <= 0:
            pre_p_raw_std = 1.0

        # Scale noise such that its pre-P window RMS is IDENTICAL to target_noise_sigma
        calibrated_noise = filtered_noise * (target_noise_sigma / pre_p_raw_std)

        # Combine
        total_waveform = (clean_signal + calibrated_noise).astype(np.float32)

        return {
            "time_axis": self.time_axis,
            "waveform": total_waveform,
            "clean_signal": clean_signal.astype(np.float32),
            "p_wave": pw.astype(np.float32),
            "s_wave": sw.astype(np.float32),
            "calibrated_noise": calibrated_noise.astype(np.float32),
            "signal_peak": signal_peak,
            "target_noise_sigma": float(target_noise_sigma),
            "amp_p": amp_p,
            "amp_s": amp_s,
            "tau_d_p": tau_d_p,
            "tau_d_s": tau_d_s,
            "tp_sec": tp_sec,
            "ts_sec": ts_sec,
            "diag": diag
        }
