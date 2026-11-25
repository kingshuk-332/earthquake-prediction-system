"""
Script: paper/figures/generate_figures.py
Description: Generates publication-quality figures for the IEEEtran paper from verified result files.
Source Data: final_comparison/four_model_eew_v1/final_metrics_table.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "final_comparison", "four_model_eew_v1", "final_metrics_table.csv")

df = pd.read_csv(CSV_PATH)

# Filter for fixed_0.5 policy on common cohort
df_fixed = df[df['threshold_policy'] == 'fixed_0.5'].copy()

# Palette
colors = {
    'CNN': '#1f77b4',       # Blue
    'BiLSTM': '#ff7f0e',    # Orange
    'Fusion': '#2ca02c',    # Green
    'Attention': '#d62728'  # Red
}
model_order = ['CNN', 'BiLSTM', 'Fusion', 'Attention']

# -----------------------------------------------------------------------------
# Figure 1: 3-Second Detection Performance (Accuracy, Precision, Recall, F1, ROC-AUC)
# -----------------------------------------------------------------------------
df_3s = df_fixed[df_fixed['horizon_seconds'] == 3].set_index('model').loc[model_order]

fig, ax = plt.subplots(figsize=(7, 3.8))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
n_metrics = len(metrics)
n_models = len(model_order)

x = np.arange(n_metrics)
width = 0.18

for i, model in enumerate(model_order):
    values = [df_3s.loc[model, m] for m in metrics]
    rects = ax.bar(x + (i - 1.5) * width, values, width, label=model, color=colors[model], edgecolor='black', linewidth=0.6)
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2),  # 2 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=0)

ax.set_ylabel('Score')
ax.set_title('Detection Performance Comparison at 3-Second Observation Horizon (Fixed Threshold 0.5)')
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0.50, 0.95)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', framealpha=0.9)
plt.tight_layout()
fig1_path = os.path.join(SCRIPT_DIR, "figure1_detection_metrics_3s.png")
fig1_pdf = os.path.join(SCRIPT_DIR, "figure1_detection_metrics_3s.pdf")
plt.savefig(fig1_path)
plt.savefig(fig1_pdf)
plt.close()
print(f"Saved: {fig1_path}")

# -----------------------------------------------------------------------------
# Figure 2: Magnitude Estimation Error (MAE and RMSE) across Horizons (1s, 2s, 3s)
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))
horizons = [1, 2, 3]

for model in model_order:
    sub = df_fixed[df_fixed['model'] == model].sort_values('horizon_seconds')
    ax1.plot(sub['horizon_seconds'], sub['mae'], marker='o', linewidth=1.8, markersize=5, label=model, color=colors[model])
    ax2.plot(sub['horizon_seconds'], sub['rmse'], marker='s', linewidth=1.8, markersize=5, label=model, color=colors[model])

ax1.set_title('(a) Mean Absolute Error (MAE)')
ax1.set_xlabel('Observation Horizon (seconds)')
ax1.set_ylabel('MAE (Magnitude Units)')
ax1.set_xticks(horizons)
ax1.set_xticklabels(['1.0 s', '2.0 s', '3.0 s'])
ax1.set_ylim(0.58, 0.69)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', framealpha=0.9)

ax2.set_title('(b) Root Mean Square Error (RMSE)')
ax2.set_xlabel('Observation Horizon (seconds)')
ax2.set_ylabel('RMSE (Magnitude Units)')
ax2.set_xticks(horizons)
ax2.set_xticklabels(['1.0 s', '2.0 s', '3.0 s'])
ax2.set_ylim(0.73, 0.84)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
fig2_path = os.path.join(SCRIPT_DIR, "figure2_magnitude_error_progression.png")
fig2_pdf = os.path.join(SCRIPT_DIR, "figure2_magnitude_error_progression.pdf")
plt.savefig(fig2_path)
plt.savefig(fig2_pdf)
plt.close()
print(f"Saved: {fig2_path}")

# -----------------------------------------------------------------------------
# Figure 3: Confusion Matrices for All 4 Models at 3s Horizon (Fixed Threshold 0.5)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.5))
# Exact confusion matrices from reports:
# CNN 3s: [[971, 243], [320, 880]]
# BiLSTM 3s: [[939, 275], [415, 785]]
# Fusion 3s: [[1050, 164], [386, 814]]
# Attention 3s: [[1075, 139], [401, 799]]
cms = {
    'CNN': np.array([[971, 243], [320, 880]]),
    'BiLSTM': np.array([[939, 275], [415, 785]]),
    'Fusion': np.array([[1050, 164], [386, 814]]),
    'Attention': np.array([[1075, 139], [401, 799]])
}

for ax, model in zip(axes, model_order):
    cm = cms[model]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=100, vmax=1100)
    ax.set_title(f'{model} (3s)', fontsize=10, fontweight='bold')
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Noise', 'EQ'], fontsize=8)
    ax.set_yticklabels(['Noise', 'EQ'], fontsize=8)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}\n({cm[i, j]/1200 if i==1 else cm[i,j]/1214:.1%})',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7.5)
    ax.set_ylabel('Actual' if model == 'CNN' else '', fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9)

plt.tight_layout()
fig3_path = os.path.join(SCRIPT_DIR, "figure3_confusion_matrices_3s.png")
fig3_pdf = os.path.join(SCRIPT_DIR, "figure3_confusion_matrices_3s.pdf")
plt.savefig(fig3_path)
plt.savefig(fig3_pdf)
plt.close()
print(f"Saved: {fig3_path}")
print("All publication figures generated successfully!")
