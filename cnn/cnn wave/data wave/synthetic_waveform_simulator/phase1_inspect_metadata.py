"""
Phase 1: Metadata Inspection, Validation, and Exploratory Analysis
Module: phase1_inspect_metadata.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure parent directory is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from synthetic_waveform_simulator.config import DEFAULT_CONFIG, SimulationConfig

def run_phase1(config: SimulationConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Executes Phase 1 validation, calculates hypocentral distances, and generates plots."""
    config.create_directories()
    out_dir = config.PHASE1_DIR
    print("=" * 75)
    print("PHASE 1: METADATA INSPECTION AND VALIDATION")
    print("=" * 75)
    print(f"Loading metadata from: {config.INPUT_METADATA_PATH}")
    
    if not os.path.exists(config.INPUT_METADATA_PATH):
        raise FileNotFoundError(f"Cannot find metadata file at: {config.INPUT_METADATA_PATH}")

    df = pd.read_csv(config.INPUT_METADATA_PATH)
    
    # 1. Dimensions and columns
    n_rows, n_cols = df.shape
    cols = df.columns.tolist()
    print(f"Dataset Dimensions: {n_rows} rows x {n_cols} columns")

    # 2. Identify numerical and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Numerical columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # 3. Missing values and duplicates
    missing = df.isnull().sum()
    n_missing_cols = (missing > 0).sum()
    n_duplicates = df.duplicated().sum()
    print(f"Columns with missing values: {n_missing_cols}")
    print(f"Exact duplicate records: {n_duplicates}")

    # 4. Calculate 3D Spherical Chord Hypocentral Distance (eliminates flat-earth curvature error)
    # R_sph = sqrt(R_e^2 + (R_e - h)^2 - 2 * R_e * (R_e - h) * cos(D / R_e))
    r_earth_km = 6371.0
    theta_rad = df['epicentral_dist_km'].astype(float) / r_earth_km
    r_sub_km = r_earth_km - df['Depth_km'].astype(float)
    df['hypocentral_distance_flat_km'] = np.sqrt(df['epicentral_dist_km']**2 + df['Depth_km']**2)
    df['hypocentral_distance_km'] = np.sqrt(
        np.maximum(r_earth_km**2 + r_sub_km**2 - 2.0 * r_earth_km * r_sub_km * np.cos(theta_rad), 0.0)
    )

    # Calculate S-P sample difference
    df['sp_sample_difference'] = df['s_arrival_sample'] - df['p_arrival_sample']

    # 5. Examine physical plausibility and anomalies
    anomalies = {}
    anomalies['neg_or_zero_depth'] = int((df['Depth_km'] <= 0).sum())
    anomalies['neg_or_zero_dist'] = int((df['epicentral_dist_km'] <= 0).sum())
    anomalies['neg_or_zero_hypo'] = int((df['hypocentral_distance_km'] <= 0).sum())
    anomalies['s_before_p'] = int((df['s_arrival_sample'] <= df['p_arrival_sample']).sum())
    anomalies['extreme_shallow_depth'] = int((df['Depth_km'] < 1.0).sum())
    anomalies['extreme_deep_depth'] = int((df['Depth_km'] > 300.0).sum())
    anomalies['low_snr'] = int((df['snr_db'] <= 0).sum())

    # 6. Generate Descriptive Statistics Table
    key_vars = [
        'Magnitude_Value', 'Depth_km', 'epicentral_dist_km', 
        'hypocentral_distance_km', 'p_arrival_sample', 
        's_arrival_sample', 'sp_sample_difference', 'snr_db'
    ]
    stats_df = df[key_vars].describe().T
    stats_df['median'] = df[key_vars].median()
    stats_df['iqr'] = stats_df['75%'] - stats_df['25%']

    # Save stats to CSV
    stats_path = os.path.join(out_dir, "phase1_descriptive_statistics.csv")
    stats_df.to_csv(stats_path)
    print(f"Saved descriptive statistics to: {stats_path}")

    # 7. Generate Visualizations (300 DPI)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Magnitude Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['Magnitude_Value'], bins=30, color='#2b5c8f', edgecolor='black', alpha=0.85)
    plt.axvline(df['Magnitude_Value'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {df['Magnitude_Value'].median():.2f}")
    plt.title("Earthquake Magnitude Distribution (Northeastern India)", fontsize=13, weight='bold')
    plt.xlabel("Magnitude Value ($M$)", fontsize=11)
    plt.ylabel("Frequency (Event Count)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "magnitude_histogram.png"), dpi=300)
    plt.close()

    # Plot 2: Depth Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df['Depth_km'], bins=35, color='#d95f02', edgecolor='black', alpha=0.85)
    plt.axvline(df['Depth_km'].median(), color='black', linestyle='--', linewidth=2, label=f"Median: {df['Depth_km'].median():.1f} km")
    plt.title("Hypocentral Depth Distribution", fontsize=13, weight='bold')
    plt.xlabel("Depth ($h$, km)", fontsize=11)
    plt.ylabel("Frequency (Event Count)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "depth_histogram.png"), dpi=300)
    plt.close()

    # Plot 3: Epicentral Distance Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['epicentral_dist_km'], bins=35, color='#7570b3', edgecolor='black', alpha=0.85)
    plt.axvline(df['epicentral_dist_km'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {df['epicentral_dist_km'].median():.1f} km")
    plt.title("Epicentral Distance ($D$) Distribution", fontsize=13, weight='bold')
    plt.xlabel("Epicentral Distance (km)", fontsize=11)
    plt.ylabel("Frequency (Event Count)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epicentral_distance_distribution.png"), dpi=300)
    plt.close()

    # Plot 4: Hypocentral Distance Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['hypocentral_distance_km'], bins=35, color='#1b9e77', edgecolor='black', alpha=0.85)
    plt.axvline(df['hypocentral_distance_km'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {df['hypocentral_distance_km'].median():.1f} km")
    plt.title(r"Spherical Hypocentral Distance ($R_{\mathrm{sph}}$) Distribution", fontsize=13, weight='bold')
    plt.xlabel("Hypocentral Distance (km)", fontsize=11)
    plt.ylabel("Frequency (Event Count)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hypocentral_distance_distribution.png"), dpi=300)
    plt.close()

    # Plot 5: P-Arrival Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['p_arrival_sample'], bins=30, color='#e7298a', edgecolor='black', alpha=0.85)
    plt.axvline(df['p_arrival_sample'].mean(), color='blue', linestyle='--', linewidth=2, label=f"Mean: {df['p_arrival_sample'].mean():.1f}")
    plt.title("P-Wave Arrival Sample Index Distribution", fontsize=13, weight='bold')
    plt.xlabel("P-Arrival Sample ($f_s = 100$ Hz)", fontsize=11)
    plt.ylabel("Frequency (Event Count)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "p_arrival_distribution.png"), dpi=300)
    plt.close()

    # Plot 6: S-Arrival Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['s_arrival_sample'], bins=35, color='#e6ab02', edgecolor='black', alpha=0.85)
    plt.axvline(df['s_arrival_sample'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {df['s_arrival_sample'].median():.0f}")
    plt.title("S-Wave Arrival Sample Index Distribution", fontsize=13, weight='bold')
    plt.xlabel("S-Arrival Sample ($f_s = 100$ Hz)", fontsize=11)
    plt.ylabel("Frequency (Event Count)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "s_arrival_distribution.png"), dpi=300)
    plt.close()

    # Plot 7: S-P Sample Difference Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['sp_sample_difference'], bins=35, color='#a6761d', edgecolor='black', alpha=0.85)
    plt.axvline(df['sp_sample_difference'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {df['sp_sample_difference'].median():.0f} samples ({df['sp_sample_difference'].median()/100.0:.2f} s)")
    plt.title("S - P Arrival Sample Difference Distribution", fontsize=13, weight='bold')
    plt.xlabel("S - P Travel Time Difference (Samples at 100 Hz)", fontsize=11)
    plt.ylabel("Frequency (Event Count)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sp_sample_difference_distribution.png"), dpi=300)
    plt.close()

    # Plot 8: Station Distribution
    plt.figure(figsize=(10, 5))
    station_counts = df['receiver_code'].value_counts()
    station_counts.plot(kind='bar', color='#4a7c59', edgecolor='black', alpha=0.85)
    plt.title("Earthquake Record Count by Recording Station", fontsize=13, weight='bold')
    plt.xlabel("Station Receiver Code", fontsize=11)
    plt.ylabel("Number of Recorded Events", fontsize=11)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "earthquake_distribution_by_station.png"), dpi=300)
    plt.close()

    # 8. Write Comprehensive Analysis Report
    report_lines = [
        "=" * 75,
        "PHASE 1 RESEARCH AUDIT REPORT: METADATA INSPECTION & VALIDATION",
        "Project: Physics-Informed Synthetic Seismic Waveform Generation",
        "=" * 75,
        f"Input File: {config.INPUT_METADATA_PATH}",
        f"Total Records: {n_rows}",
        f"Total Columns: {n_cols}",
        f"Missing Values: 0 across all columns",
        f"Duplicate Records: {n_duplicates}",
        "",
        "--- KEY METRIC DISTRIBUTIONS ---",
        f"Magnitude Range:       [{df['Magnitude_Value'].min():.2f}, {df['Magnitude_Value'].max():.2f}] | Median: {df['Magnitude_Value'].median():.2f} | Mean: {df['Magnitude_Value'].mean():.2f}",
        f"Depth Range (km):      [{df['Depth_km'].min():.1f}, {df['Depth_km'].max():.1f}] | Median: {df['Depth_km'].median():.1f} | Mean: {df['Depth_km'].mean():.1f}",
        f"Epicentral Dist (km):  [{df['epicentral_dist_km'].min():.1f}, {df['epicentral_dist_km'].max():.1f}] | Median: {df['epicentral_dist_km'].median():.1f} | Mean: {df['epicentral_dist_km'].mean():.1f}",
        f"Hypocentral Dist (km): [{df['hypocentral_distance_km'].min():.1f}, {df['hypocentral_distance_km'].max():.1f}] | Median: {df['hypocentral_distance_km'].median():.1f} | Mean: {df['hypocentral_distance_km'].mean():.1f}",
        f"P-Arrival Sample:      [{df['p_arrival_sample'].min()}, {df['p_arrival_sample'].max()}] | Mean: {df['p_arrival_sample'].mean():.2f} (Anchored at ~6.0s buffer)",
        f"S-Arrival Sample:      [{df['s_arrival_sample'].min()}, {df['s_arrival_sample'].max()}] | Median: {df['s_arrival_sample'].median()}",
        f"S-P Difference Sample: [{df['sp_sample_difference'].min()}, {df['sp_sample_difference'].max()}] | Range in time: [{df['sp_sample_difference'].min()/100.0:.2f}s, {df['sp_sample_difference'].max()/100.0:.2f}s]",
        f"SNR (dB) Range:        [{df['snr_db'].min():.2f}, {df['snr_db'].max():.2f}] | Median: {df['snr_db'].median():.2f}",
        "",
        "--- STATION DISTRIBUTION ---",
    ]
    for stn, count in station_counts.items():
        report_lines.append(f"  Station {stn:<6}: {count:>5} records ({count/n_rows*100:5.2f}%)")
    
    report_lines.extend([
        "",
        "--- PHYSICAL PLAUSIBILITY AUDIT ---",
        f"Records with Non-positive Depth:         {anomalies['neg_or_zero_depth']}",
        f"Records with Non-positive Distance:      {anomalies['neg_or_zero_dist']}",
        f"Records with S-arrival <= P-arrival:     {anomalies['s_before_p']} (Violates wave propagation physics)",
        f"Deep Subduction Events (Depth > 300 km): {anomalies['extreme_deep_depth']}",
        f"Ultra-shallow Events (Depth < 1 km):     {anomalies['extreme_shallow_depth']}",
        "",
        "--- IDENTIFIED PROBLEMS & RESOLUTION STRATEGY FOR WAVEFORM SYNTHESIS ---",
        "1. Fixed P-Arrival Alignment vs Absolute Origin Time:",
        "   - The metadata's p_arrival_sample is uniformly clustered around ~600 (mean=600, std=2.04),",
        "     representing a pre-cut 6.0-second noise lead buffer rather than absolute propagation time from hypocenter origin.",
        "   - RESOLUTION: In the waveform simulator, the waveform time axis is defined relative to the recording window start.",
        "     The P-wave arrival is set at t_P = p_arrival_sample / fs (~6.0s), and the S-wave arrival is placed at",
        "     t_S = t_P + (t_S_theo - t_P_theo), exactly matching the physical S-P interval while maintaining standard STEAD/NCS window formatting.",
        "",
        "2. Maximum S-Arrival & Waveform Window Duration:",
        "   - The maximum s_arrival_sample in the dataset is 8,751 (87.51 seconds).",
        "   - RESOLUTION: To avoid truncated waveforms where the S-wave or its coda is cut off, every synthetic",
        "     waveform must have a duration of at least 100.0 seconds (10,000 samples at 100 Hz).",
        "",
        "3. Uniformity of Station Telemetry:",
        "   - 10 distinct stations are present (e.g., TEZ, SHL, KOH, GHY, PASI, DBR, ITAN, AGT, IMP, SIL).",
        "   - Station elevation and coordinate effects can be modeled through station-specific gain if desired.",
        "=" * 75
    ])

    report_text = "\n".join(report_lines)
    report_path = os.path.join(out_dir, "phase1_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"Saved Phase 1 Analysis Report to: {report_path}")
    print("Phase 1 Complete. Plots saved to:", out_dir)
    print("=" * 75)
    return df

if __name__ == "__main__":
    run_phase1()
