"""
Unified Pipeline Orchestrator for Physics-Informed Synthetic Seismic Waveform Generation
Script: run_pipeline.py

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --phase 1
    python run_pipeline.py --phase 2
    python run_pipeline.py --phase 3
    python run_pipeline.py --phase 4
    python run_pipeline.py --phase 5
    python run_pipeline.py --phase 6
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthetic_waveform_simulator.config import DEFAULT_CONFIG
from synthetic_waveform_simulator.phase1_inspect_metadata import run_phase1
from synthetic_waveform_simulator.phase2_travel_time import run_phase2
from synthetic_waveform_simulator.phase3_clean_synthesis import run_phase3
from synthetic_waveform_simulator.phase4_attenuation_noise import run_phase4
from synthetic_waveform_simulator.phase5_prototype_evaluation import run_phase5
from synthetic_waveform_simulator.phase6_generate_full_dataset import run_phase6
from synthetic_waveform_simulator.waveform_quality_control import main as run_phase7

def main():
    parser = argparse.ArgumentParser(description="Physics-Informed Synthetic Seismic Waveform Simulator Pipeline")
    parser.add_argument("--all", action="store_true", help="Execute all phases (1 through 7) sequentially.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6, 7], help="Execute a specific phase (1-7).")
    args = parser.parse_args()

    if not args.all and args.phase is None:
        parser.print_help()
        sys.exit(1)

    phases_to_run = [1, 2, 3, 4, 5, 6, 7] if args.all else [args.phase]

    print("=" * 80)
    print("PHYSICS-INFORMED SYNTHETIC SEISMIC WAVEFORM GENERATOR")
    print(f"Phases selected for execution: {phases_to_run}")
    print("=" * 80)

    if 1 in phases_to_run:
        run_phase1(DEFAULT_CONFIG)
    if 2 in phases_to_run:
        run_phase2(DEFAULT_CONFIG)
    if 3 in phases_to_run:
        run_phase3(DEFAULT_CONFIG)
    if 4 in phases_to_run:
        run_phase4(DEFAULT_CONFIG)
    if 5 in phases_to_run:
        run_phase5(DEFAULT_CONFIG)
    if 6 in phases_to_run:
        run_phase6(DEFAULT_CONFIG)
    if 7 in phases_to_run:
        run_phase7()

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Outputs located in: {DEFAULT_CONFIG.OUTPUT_BASE_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
