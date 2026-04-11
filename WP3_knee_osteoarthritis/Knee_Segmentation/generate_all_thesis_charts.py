import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import glob

# ==========================================
# CONFIGURATION
# ==========================================
models = ['ResNet-18', 'EfficientNet-B0', 'Swin-Tiny']
eval_dirs = [
    'classification_results/evaluations',
    'classification_results_efficientnet/evaluations',
    'classification_results_swin/evaluations'
]
train_logs = [
    'classification_results/baseline/training_log.csv',
    'classification_results_efficientnet/baseline/training_log.csv',
    'classification_results_swin/baseline/training_log.csv'
]
conditions = [
    ('baseline_on_baseline.json', 'Original'),
    ('baseline_on_lateral_masked.json', 'Lateral'),
    ('baseline_on_medial_masked.json', 'Medial'),
    ('baseline_on_blackout.json', 'Blackout')
]

# ==========================================
# 1. INDIVIDUAL ARCHITECTURE PLOTS
# ==========================================
def generate_cm_plot(eval_dir, output_filename, model_name):
    if not os.path.exists(eval_dir): return
    full_conditions = [
        ('baseline_on_baseline.json', 'Baseline (Original)'),
        ('baseline_on_lateral_masked.json', 'Lateral Masked'),
        ('baseline_on_medial_masked.json', 'Medial Masked'),
        ('baseline_on_blackout.json', 'Blackout (No Joint Space)')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (filename, title) in zip(axes, full_conditions):
        filepath = os.path.join(eval_dir, filename)
        if not os.path.exists(filepath): continue
        with open(filepath) as f:
            data = json.load(f)
        cm = np.array(data.get('confusion_matrix', [[0,0],[0,0]]))
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred KL0 (Healthy)', 'Pred KL2 (OA)'], fontsize=11)
        ax.set_yticklabels(['Actual KL0', 'Actual KL2'], fontsize=11)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=16, fontweight='bold')
        acc = (cm[0,0] + cm[1,1]) / max(cm.sum(), 1)
        ax.set_xlabel(f'Accuracy: {acc:.1%}', fontsize=12, fontweight='bold')
    plt.suptitle(f'{model_name} Cross-Evaluation Confusion Matrices', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_filename}")

def generate_confidence_bar_plot(eval_dir, output_filename, model_name):
    if not os.path.exists(eval_dir): return
    labels, mean_all, mean_correct, mean_incorrect = [], [], [], []
    for filename, title in conditions:
        filepath = os.path.join(eval_dir, filename)
        if not os.path.exists(filepath): continue
        with open(filepath) as f:
            data = json.load(f)
        conf = data.get('confidence', {})
        labels.append(title)
        mean_all.append(conf.get('mean_confidence_all', 0))
        mean_correct.append(conf.get('mean_confidence_correct', 0))
        mean_incorrect.append(conf.get('mean_confidence_incorrect', 0))
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, mean_all, width, label='All Predictions', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x, mean_correct, width, label='Correct Predictions', color='lightgreen', edgecolor='black')
    rects3 = ax.bar(x + width, mean_incorrect, width, label='Incorrect Predictions', color='salmon', edgecolor='black')
    ax.set_ylabel('Mean Confidence (Softmax Prob)', fontsize=12, fontweight='bold')
    ax.set_title(f'Confidence Degradation Comparison - {model_name}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated {output_filename}")

# Run Individual Plots
for m, edir in zip(models, eval_dirs):
    filename_base = m.lower().replace('-', '').replace(' ', '_').split('(')[0]
    generate_cm_plot(edir, f'{filename_base}_confusion_matrices.png', m)
    generate_confidence_bar_plot(edir, f'{filename_base}_confidence_degradation.png', m)

# ==========================================
# 2. EXTRACT DATA FOR GLOBAL COMPARISONS
# ==========================================
data_dict = {m: {} for m in models}
for m, edir in zip(models, eval_dirs):
    for c_file, c_name in conditions:
        filepath = os.path.join(edir, c_file)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                d = json.load(f)
                metrics = d.get('metrics', {})
                acc = metrics.get('accuracy', 0)
                oa_recall = metrics.get('2_recall', 0)
                f1 = metrics.get('macro_f1', 0)
                unc = d.get('confidence', {}).get('uncertain_correct_pct', 0)
                data_dict[m][c_name] = {'acc': acc, 'oa_recall': oa_recall, 'f1': f1, 'uncertainty': unc}

# ==========================================
# 3. GLOBAL CHART 1: ABLATION DELTA
# ==========================================
plt.figure(figsize=(12, 7))
width = 0.25
x = np.arange(3)
for i, m in enumerate(models):
    base_acc = data_dict[m].get('Original', {}).get('acc', 0)
    lat_drop = (base_acc - data_dict[m].get('Lateral', {}).get('acc', 0)) * 100
    med_drop = (base_acc - data_dict[m].get('Medial', {}).get('acc', 0)) * 100
    blk_drop = (base_acc - data_dict[m].get('Blackout', {}).get('acc', 0)) * 100
    vals = [lat_drop, med_drop, blk_drop]
    bars = plt.bar(x + (i-1)*width, vals, width, label=m, alpha=0.9, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.xticks(x, ['Lateral Masked', 'Medial Masked', 'Blackout'], fontsize=14, fontweight='bold')
plt.ylabel('Accuracy Drop vs Baseline (Percentage Points)', fontsize=12, fontweight='bold')
plt.title('Ablation Penalty: How Much Accuracy is Lost?', fontsize=16, fontweight='bold', pad=15)
plt.axhline(0, color='black', linewidth=1)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('chart1_ablation_delta.png', dpi=300)
plt.close()
print("Generated chart1_ablation_delta.png")

# ==========================================
# 4. GLOBAL CHART 2: UNCERTAINTY SURGE
# ==========================================
plt.figure(figsize=(10, 6))
x_conds = np.arange(len(conditions))
markers = ['o', 's', '^']
for i, m in enumerate(models):
    y = [data_dict[m][c[1]]['uncertainty'] for c in conditions]
    plt.plot(x_conds, y, marker=markers[i], linewidth=2.5, markersize=8, label=m)
for j in range(len(conditions)):
    c_name = conditions[j][1]
    y_vals = [data_dict[m][c_name]['uncertainty'] for m in models]
    order = np.argsort(y_vals)
    offsets = {order[2]: (0, 8), order[1]: (15, 0), order[0]: (0, -12)}
    alignments = {order[2]: ('center', 'bottom'), order[1]: ('left', 'center'), order[0]: ('center', 'top')}
    for i, m in enumerate(models):
        val = y_vals[i]
        plt.annotate(f'{val:.1f}%', (x_conds[j], val), textcoords="offset points", xytext=offsets[i],
                    ha=alignments[i][0], va=alignments[i][1], fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
plt.xticks(x_conds, [c[1] for c in conditions], fontsize=12, fontweight='bold')
plt.ylabel('Uncertain Correct Predictions (%)', fontsize=12, fontweight='bold')
plt.title('Uncertainty Surge (<0.7 Confidence)', fontsize=16, fontweight='bold', pad=15)
y_flattened = [data_dict[m][c[1]]['uncertainty'] for m in models for c in conditions]
if y_flattened: plt.ylim([max(0, min(y_flattened) - 5), max(y_flattened) + 15])
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('chart2_uncertainty_surge.png', dpi=300)
plt.close()
print("Generated chart2_uncertainty_surge.png")

# ==========================================
# 5. GLOBAL CHART 3: SENSITIVITY ANALYSIS
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics_to_plot = ['acc', 'oa_recall', 'f1']
metric_names = ['Overall Accuracy', 'OA Recall (Sensitivity)', 'Macro F1-Score']
for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
    ax = axes[idx]
    for i, m in enumerate(models):
        y = [data_dict[m][c[1]][metric] for c in conditions]
        ax.plot(x_conds, y, marker=markers[i], label=m, linewidth=2.5, markersize=8)
    for j in range(len(conditions)):
        c_name = conditions[j][1]
        y_vals = [data_dict[m][c_name][metric] for m in models]
        order = np.argsort(y_vals)
        offsets = {order[2]: (0, 8), order[1]: (12, 0), order[0]: (0, -10)}
        alignments = {order[2]: ('center', 'bottom'), order[1]: ('left', 'center'), order[0]: ('center', 'top')}
        for i, m in enumerate(models):
            val = y_vals[i]
            ax.annotate(f'{val:.2f}', (x_conds[j], val), textcoords="offset points", xytext=offsets[i],
                        ha=alignments[i][0], va=alignments[i][1], fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax.set_xticks(x_conds)
    ax.set_xticklabels([c[1] for c in conditions], rotation=30, ha='right', fontsize=12, fontweight='bold')
    ax.set_title(name, fontsize=14, fontweight='bold', pad=10)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--')
axes[0].set_ylabel('Metric Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
axes[-1].legend(fontsize=11)
plt.suptitle('Medical Assessment Triad: Overall Metric Degradation', fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('chart3_sensitivity_analysis.png', dpi=300)
plt.close()
print("Generated chart3_sensitivity_analysis.png")

# ==========================================
# 6. GLOBAL CHART 4: LEARNING CURVES
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, (m, log_path) in enumerate(zip(models, train_logs)):
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        axes[0, i].plot(df['epoch'], df['train_loss'], label='Train')
        axes[0, i].plot(df['epoch'], df['val_loss'], label='Val')
        axes[0, i].set_title(f'{m} Loss', fontweight='bold', fontsize=14)
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        if i == 0: axes[0, i].set_ylabel('Loss', fontweight='bold')
        axes[1, i].plot(df['epoch'], df['train_acc'], label='Train')
        axes[1, i].plot(df['epoch'], df['val_acc'], label='Val')
        axes[1, i].set_title(f'{m} Accuracy', fontweight='bold', fontsize=14)
        axes[1, i].set_xlabel('Epoch', fontweight='bold')
        if i == 0: axes[1, i].set_ylabel('Accuracy', fontweight='bold')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
plt.suptitle('Training Convergence / Learning Curves', fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('chart4_learning_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated chart4_learning_curves.png")

