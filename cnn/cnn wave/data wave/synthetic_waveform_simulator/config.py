"""
Centralized Configuration Module for Physics-Informed Synthetic Seismic Waveform Generator
Research Title: Physics-Informed Synthetic Seismic Waveform Generation for Earthquake Detection Using Deep Learning
"""

import os
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # -------------------------------------------------------------
    # 1. File and Directory Paths
    # -------------------------------------------------------------
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT: str = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", ".."))
    
    # Metadata paths (dynamically resolved to repository root dataset_2.csv)
    INPUT_METADATA_PATH: str = os.path.join(PROJECT_ROOT, "dataset_2.csv")
    OUTPUT_BASE_DIR: str = os.path.join(BASE_DIR, "output")
    
    # Subdirectories for phases
    PHASE1_DIR: str = os.path.join(OUTPUT_BASE_DIR, "phase1_eda")
    PHASE2_DIR: str = os.path.join(OUTPUT_BASE_DIR, "phase2_travel_time")
    PHASE3_DIR: str = os.path.join(OUTPUT_BASE_DIR, "phase3_clean_synthesis")
    PHASE4_DIR: str = os.path.join(OUTPUT_BASE_DIR, "phase4_attenuation_noise")
    PHASE5_DIR: str = os.path.join(OUTPUT_BASE_DIR, "phase5_prototype_10")
    PHASE6_DIR: str = os.path.join(OUTPUT_BASE_DIR, "phase6_full_dataset")
    PHASE7_DIR: str = os.path.join(OUTPUT_BASE_DIR, "phase7_quality_control")

    # -------------------------------------------------------------
    # 2. Baseline Seismic Propagation Assumptions
    # -------------------------------------------------------------
    VP_KM_S: float = 6.0          # Baseline P-wave velocity in km/s (configurable)
    VS_KM_S: float = 3.5          # Baseline S-wave velocity in km/s (configurable)
    SAMPLING_RATE_HZ: float = 100.0  # Sampling frequency in Hz
    PRE_P_ARRIVAL_BUFFER_S: float = 6.0  # Fixed window lead time before P arrival (seconds)

    # -------------------------------------------------------------
    # 3. Wave Packet Frequency and Duration Parameters
    # -------------------------------------------------------------
    P_DOMINANT_FREQ_HZ: float = 6.0   # Higher frequency for P-wave packet (Hz)
    S_DOMINANT_FREQ_HZ: float = 2.5   # Lower frequency for S-wave packet (Hz)
    P_ENVELOPE_TAU_S: float = 1.2     # P-wave envelope decay constant (seconds)
    S_ENVELOPE_TAU_S: float = 3.5     # S-wave envelope decay constant (seconds)
    S_TO_P_AMPLITUDE_RATIO: float = 3.0  # Typical regional vertical component S/P amplitude ratio

    # -------------------------------------------------------------
    # 4. Amplitude Scaling & Attenuation Model
    # -------------------------------------------------------------
    REFERENCE_MAGNITUDE: float = 4.0      # Pivot magnitude M_ref
    REFERENCE_AMPLITUDE: float = 1.0      # Base amplitude A_0
    MAGNITUDE_SCALING_COEFF: float = 0.5  # c in A(M) = A_0 * 10^(c * (M - M_ref))
    GEOMETRIC_SPREADING_EXP: float = 1.0  # gamma in 1 / R^gamma for body waves
    REFERENCE_DISTANCE_KM: float = 10.0   # Reference minimum distance R_0 (avoids singularity)
    QP_QUALITY_FACTOR: float = 300.0      # P-wave crustal quality factor Q_P
    QS_QUALITY_FACTOR: float = 150.0      # S-wave crustal quality factor Q_S (Q_S ~ Q_P / 2)

    # -------------------------------------------------------------
    # 5. Noise & Window Settings
    # -------------------------------------------------------------
    DEFAULT_SNR_DB: float = 15.0          # Fallback SNR if not present in record
    NOISE_FILTER_LOW_HZ: float = 0.2      # Microseismic noise highpass filter cut
    NOISE_FILTER_HIGH_HZ: float = 25.0    # Ambient noise lowpass filter cut
    TOTAL_DURATION_SECONDS: float = 100.0 # Total trace duration (100.0 seconds = 10,000 samples)
    TOTAL_SAMPLES: int = 10000            # Total samples per waveform (100.0 s * 100 Hz)
    RANDOM_SEED: int = 42                 # Base seed for reproducibility

    def create_directories(self) -> None:
        """Create all output directories if they do not exist."""
        for path in [
            self.OUTPUT_BASE_DIR,
            self.PHASE1_DIR,
            self.PHASE2_DIR,
            self.PHASE3_DIR,
            self.PHASE4_DIR,
            self.PHASE5_DIR,
            self.PHASE6_DIR,
            self.PHASE7_DIR
        ]:
            os.makedirs(path, exist_ok=True)

# Global default configuration instance
DEFAULT_CONFIG = SimulationConfig()
