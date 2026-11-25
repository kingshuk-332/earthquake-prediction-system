"""
Physics-Informed Synthetic Seismic Waveform Generator
Module: waveform_generator.py

Implements:
1. Continuous time-axis generation.
2. Damped oscillatory P-wave and S-wave packets with smooth onset envelopes.
3. Magnitude scaling, geometric spreading, and anelastic attenuation (Q-factor).
4. Band-limited colored ambient seismic noise.
5. Calibrated Signal-to-Noise Ratio (SNR) scaling.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from typing import Tuple, Dict, Any, Optional

class WaveformGenerator:
    """
    Physics-Informed Seismic Waveform Generator for simulating vertical-component (BHZ)
    seismic traces from earthquake metadata.
    """
    def __init__(
        self,
        vp_km_s: float = 6.0,
        vs_km_s: float = 3.5,
        sampling_rate_hz: float = 100.0,
        ref_magnitude: float = 4.0,
        ref_amplitude: float = 1.0,
        mag_scaling_coeff: float = 0.5,
        geometric_spreading_exp: float = 1.0,
        ref_distance_km: float = 10.0,
        qp: float = 300.0,
        qs: float = 150.0,
        p_dominant_freq_hz: float = 6.0,
        s_dominant_freq_hz: float = 2.5,
        p_envelope_tau_s: float = 1.2,
        s_envelope_tau_s: float = 3.5,
        s_to_p_amplitude_ratio: float = 3.0,
        noise_band: Tuple[float, float] = (0.2, 25.0),
        random_seed: Optional[int] = 42
    ):
        self.vp = vp_km_s
        self.vs = vs_km_s
        self.fs = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        self.ref_mag = ref_magnitude
        self.ref_amp = ref_amplitude
        self.c_mag = mag_scaling_coeff
        self.gamma = geometric_spreading_exp
        self.r0 = ref_distance_km
        self.qp = qp
        self.qs = qs
        self.fp = p_dominant_freq_hz
        self.fs_wave = s_dominant_freq_hz
        self.tau_p = p_envelope_tau_s
        self.tau_s = s_envelope_tau_s
        self.sp_amp_ratio = s_to_p_amplitude_ratio
        self.noise_band = noise_band
        self.seed = random_seed
        self.rng = np.random.default_rng(self.seed)

    def generate_time_axis(self, duration_seconds: float) -> np.ndarray:
        """Generates continuous time axis array."""
        n_samples = int(np.round(duration_seconds * self.fs))
        return np.linspace(0.0, (n_samples - 1) * self.dt, n_samples)

    def calculate_effective_amplitudes(
        self,
        magnitude: float,
        hypocentral_distance_km: float
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Calculates physically scaled P and S wave amplitudes incorporating:
        - Source magnitude scaling: A(M) = A_0 * 10^(c * (M - M_ref))
        - Geometric spreading: G(R) = (R_0 / max(R, R_0))^gamma
        - Anelastic attenuation: kappa = exp(-pi * f * R / (Q * v))
        """
        R = max(hypocentral_distance_km, self.r0)

        # 1. Source magnitude scaling
        source_factor = self.ref_amp * (10.0 ** (self.c_mag * (magnitude - self.ref_mag)))

        # 2. Geometric spreading for body waves (1 / R^gamma)
        geometric_factor = (self.r0 / R) ** self.gamma

        # 3. Anelastic attenuation (Q-factor damping)
        # kappa_p = exp(-pi * f_p * R / (Q_p * v_p))
        kappa_p = np.exp(- (np.pi * self.fp * R) / (self.qp * self.vp))
        # kappa_s = exp(-pi * f_s * R / (Q_s * v_s))
        kappa_s = np.exp(- (np.pi * self.fs_wave * R) / (self.qs * self.vs))

        amp_p = source_factor * geometric_factor * kappa_p
        amp_s = source_factor * geometric_factor * kappa_s * self.sp_amp_ratio

        diagnostic = {
            "source_factor": source_factor,
            "geometric_factor": geometric_factor,
            "kappa_p": kappa_p,
            "kappa_s": kappa_s,
            "amp_p": amp_p,
            "amp_s": amp_s
        }
        return float(amp_p), float(amp_s), diagnostic

    def generate_p_wave(
        self,
        t: np.ndarray,
        p_arrival_seconds: float,
        amplitude: float,
        dominant_freq: Optional[float] = None,
        envelope_tau: Optional[float] = None
    ) -> np.ndarray:
        """
        Generates a smooth, continuous P-wave packet using a gamma-variate envelope
        modulated by a damped sinusoid.
        """
        f = dominant_freq if dominant_freq is not None else self.fp
        tau_d = envelope_tau if envelope_tau is not None else self.tau_p
        tau_r = tau_d * 0.35  # Fast rise time

        p_wave = np.zeros_like(t)
        mask = t >= p_arrival_seconds
        tau = t[mask] - p_arrival_seconds

        # Gamma-envelope: (tau / tau_r)^2 * exp(-tau / tau_d)
        # C^1 smooth onset at arrival (zero initial value and zero first derivative)
        envelope = (tau / tau_r)**2 * np.exp(-tau / tau_d)
        # Normalize peak envelope to 1.0 before scaling by amplitude
        peak_env = np.max(envelope) if len(envelope) > 0 and np.max(envelope) > 0 else 1.0
        envelope = envelope / peak_env

        oscillation = np.sin(2.0 * np.pi * f * tau)
        p_wave[mask] = amplitude * envelope * oscillation
        return p_wave

    def generate_s_wave(
        self,
        t: np.ndarray,
        s_arrival_seconds: float,
        amplitude: float,
        dominant_freq: Optional[float] = None,
        envelope_tau: Optional[float] = None
    ) -> np.ndarray:
        """
        Generates a smooth, continuous S-wave packet with lower frequency and
        longer coda decay.
        """
        f = dominant_freq if dominant_freq is not None else self.fs_wave
        tau_d = envelope_tau if envelope_tau is not None else self.tau_s
        tau_r = tau_d * 0.25  # Smooth onset

        s_wave = np.zeros_like(t)
        mask = t >= s_arrival_seconds
        tau = t[mask] - s_arrival_seconds

        envelope = (tau / tau_r)**2 * np.exp(-tau / tau_d)
        peak_env = np.max(envelope) if len(envelope) > 0 and np.max(envelope) > 0 else 1.0
        envelope = envelope / peak_env

        oscillation = np.sin(2.0 * np.pi * f * tau + np.pi / 4.0)
        s_wave[mask] = amplitude * envelope * oscillation
        return s_wave

    def generate_background_noise(
        self,
        n_samples: int,
        target_sigma: float = 1.0,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generates band-limited colored seismic ambient noise filtered between
        noise_band[0] and noise_band[1] Hz.
        """
        gen = rng if rng is not None else self.rng
        raw_noise = gen.normal(0.0, 1.0, n_samples)

        # 4th-order Butterworth bandpass filter
        sos = butter(4, [self.noise_band[0], self.noise_band[1]], btype='bandpass', fs=self.fs, output='sos')
        filtered_noise = sosfilt(sos, raw_noise)

        # Scale to target standard deviation
        current_std = np.std(filtered_noise)
        if current_std > 0:
            filtered_noise = (filtered_noise / current_std) * target_sigma
        return filtered_noise

    def combine_wave_components(
        self,
        p_wave: np.ndarray,
        s_wave: np.ndarray,
        target_snr_db: float,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Combines P-wave, S-wave, and calibrates background noise to match target SNR (dB).
        SNR is defined as: SNR_dB = 20 * log10(Peak_Signal_Amplitude / Noise_RMS)
        Returns: (total_waveform, noise_array, noise_sigma)
        """
        clean_signal = p_wave + s_wave
        peak_amp = np.max(np.abs(clean_signal))

        if peak_amp <= 0:
            peak_amp = 1.0

        # Noise RMS required for target SNR
        noise_sigma = peak_amp / (10.0 ** (target_snr_db / 20.0))
        noise = self.generate_background_noise(len(clean_signal), target_sigma=noise_sigma, rng=rng)

        total_waveform = clean_signal + noise
        return total_waveform, noise, float(noise_sigma)

    def generate_single_event(
        self,
        p_arrival_seconds: float,
        s_arrival_seconds: float,
        magnitude: float,
        hypocentral_distance_km: float,
        target_snr_db: float,
        duration_seconds: float = 100.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full pipeline to generate one complete synthetic waveform trace for a given event.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        t = self.generate_time_axis(duration_seconds)

        # 1. Physics-based amplitudes
        amp_p, amp_s, diag = self.calculate_effective_amplitudes(magnitude, hypocentral_distance_km)

        # 2. Synthesize individual wave packets
        p_wave = self.generate_p_wave(t, p_arrival_seconds, amp_p)
        s_wave = self.generate_s_wave(t, s_arrival_seconds, amp_s)

        # 3. Combine with calibrated ambient noise
        clean_signal = p_wave + s_wave
        waveform, noise, noise_sigma = self.combine_wave_components(p_wave, s_wave, target_snr_db, rng=rng)

        return {
            "time_axis": t,
            "clean_signal": clean_signal,
            "p_wave": p_wave,
            "s_wave": s_wave,
            "noise": noise,
            "waveform": waveform,
            "noise_sigma": noise_sigma,
            "amp_p": amp_p,
            "amp_s": amp_s,
            "p_arrival_seconds": p_arrival_seconds,
            "s_arrival_seconds": s_arrival_seconds,
            "p_arrival_sample": int(np.round(p_arrival_seconds * self.fs)),
            "s_arrival_sample": int(np.round(s_arrival_seconds * self.fs)),
            "diagnostic": diag
        }
