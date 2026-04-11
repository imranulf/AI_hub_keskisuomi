import json
import os
from scipy.stats import wilcoxon

def load_data(arch_dir):
    json_path = os.path.join(arch_dir, 'robustness', 'robustness_summary.json')
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        return json.load(f)

print("=== PAIRED WILCOXON SIGNED-RANK TESTS across 5 Seeds ===\n")

architectures = {
    'ResNet-18': 'classification_results',
    'EfficientNet-B0': 'classification_results_efficientnet',
    'Swin-Tiny': 'classification_results_swin'
}

for name, d in architectures.items():
    data = load_data(d)
    if not data:
        print(f"[{name}] Data not found.\n")
        continue
    
    print(f"--- Architecture: {name} ---")
    
    # 1. Baseline vs Blackout accuracy
    # Self-evaluation accuracy
    val_base_acc = data['self_evaluation']['baseline']['accuracy']['values']
    val_black_acc = data['self_evaluation']['blackout']['accuracy']['values']
    stat, p = wilcoxon(val_base_acc, val_black_acc)
    print(f"1a. Self-Eval Accuracy (Baseline vs Blackout): p={p:.4f} (W={stat}) | Base: {sum(val_base_acc)/5:.4f}, Blackout: {sum(val_black_acc)/5:.4f}")

    # Cross-evaluation accuracy
    val_cb_acc = data['cross_evaluation']['baseline_on_baseline']['accuracy']['values']
    val_cblack_acc = data['cross_evaluation']['baseline_on_blackout']['accuracy']['values']
    stat, p = wilcoxon(val_cb_acc, val_cblack_acc)
    print(f"1b. Cross-Eval Accuracy (Baseline vs Blackout): p={p:.4f} (W={stat}) | Base: {sum(val_cb_acc)/5:.4f}, Blackout: {sum(val_cblack_acc)/5:.4f}")

    # 2. Medial vs Lateral masked accuracy
    # Self-evaluation 
    val_med_acc = data['self_evaluation']['medial_masked']['accuracy']['values']
    val_lat_acc = data['self_evaluation']['lateral_masked']['accuracy']['values']
    stat, p = wilcoxon(val_med_acc, val_lat_acc)
    print(f"2a. Self-Eval Accuracy (Medial vs Lateral): p={p:.4f} (W={stat}) | Medial: {sum(val_med_acc)/5:.4f}, Lateral: {sum(val_lat_acc)/5:.4f}")

    # Cross-evaluation
    val_cmed_acc = data['cross_evaluation']['baseline_on_medial_masked']['accuracy']['values']
    val_clat_acc = data['cross_evaluation']['baseline_on_lateral_masked']['accuracy']['values']
    stat, p = wilcoxon(val_cmed_acc, val_clat_acc)
    print(f"2b. Cross-Eval Accuracy (Medial vs Lateral): p={p:.4f} (W={stat}) | Medial: {sum(val_cmed_acc)/5:.4f}, Lateral: {sum(val_clat_acc)/5:.4f}")

    # 3. Confidence baseline vs blackout
    try:
        val_cb_conf = data['cross_evaluation']['baseline_on_baseline']['mean_confidence']['values']
        val_cblack_conf = data['cross_evaluation']['baseline_on_blackout']['mean_confidence']['values']
        stat, p = wilcoxon(val_cb_conf, val_cblack_conf)
        print(f"3.  Cross-Eval Confidence (Baseline vs Blackout): p={p:.4f} (W={stat}) | Base: {sum(val_cb_conf)/5:.4f}, Blackout: {sum(val_cblack_conf)/5:.4f}")
    except KeyError:
        print("3.  Cross-Eval Confidence metrics not natively present in structure or keys differ.")
        
    print("\n")
