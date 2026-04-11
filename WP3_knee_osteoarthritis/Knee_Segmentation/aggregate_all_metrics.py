import os
import glob
import json
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
directories = {
    'ResNet-18': 'classification_results',
    'EfficientNet-B0': 'classification_results_efficientnet',
    'Swin-Tiny': 'classification_results_swin'
}
seeds = [42, 179, 316, 453, 590]

# List to hold the rows of data
all_data = []

# Output paths
csv_out = 'Comprehensive_Experiment_Results.csv'
excel_out = 'Comprehensive_Experiment_Results.xlsx'

# ==========================================
# PARSERS
# ==========================================

print("Aggregating Experiment Results...")

for model_name, base_dir in directories.items():
    if not os.path.exists(base_dir):
        print(f"Skipping {model_name}, directory '{base_dir}' not found.")
        continue
    
    # Track the evaluation directories we want to parse
    eval_targets = []
    
    # 1. Parse initial 'single' pass runs. Done before the robustness sequence
    initial_dir = os.path.join(base_dir, 'evaluations')
    if os.path.exists(initial_dir):
        eval_targets.append((initial_dir, 'Initial Single Run (Baseline Model)'))
        
    # 2. Parse robust 5-seed sequences
    for seed in seeds:
        seed_dir = os.path.join(base_dir, 'evaluations', 'robustness', f'seed_{seed}')
        if os.path.exists(seed_dir):
            eval_targets.append((seed_dir, str(seed)))
    
    for current_eval_dir, seed_label in eval_targets:
        
        # Glob all JSON evaluations for this specific path
        json_files = glob.glob(os.path.join(current_eval_dir, '*.json'))
        
        for jpath in json_files:
            filename = os.path.basename(jpath).replace('.json', '')
            
            # Determine Evaluation Type & Setup
            eval_type = 'Unknown'
            condition = 'Unknown'
            
            if filename.endswith('_self'):
                eval_type = 'Self-Evaluation'
                condition = filename.replace('_self', '')
            elif filename.startswith('baseline_on_'):
                eval_type = 'Cross-Evaluation (Baseline tests on Ablation)'
                condition = filename.replace('baseline_on_', '')
            
            # Read JSON
            with open(jpath, 'r') as f:
                data = json.load(f)
                
            metrics = data.get('metrics', {})
            conf = data.get('confidence', {})
            
            # Build Row Dictionary
            row = {
                'Architecture': model_name,
                'Seed': seed_label,
                'Evaluation Mode': eval_type,
                'Test Condition': condition.capitalize().replace('_', ' '),
                'Accuracy': metrics.get('accuracy', None),
                'Macro F1': metrics.get('macro_f1', None),
                'Macro Precision': metrics.get('macro_precision', None),
                'Macro Recall': metrics.get('macro_recall', None),
                
                'Healthy (KL0) Recall': metrics.get('0_recall', None),
                'Healthy (KL0) Precision': metrics.get('0_precision', None),
                'Healthy (KL0) F1': metrics.get('0_f1', None),
                
                'OA (KL2) Recall': metrics.get('2_recall', None),
                'OA (KL2) Precision': metrics.get('2_precision', None),
                'OA (KL2) F1': metrics.get('2_f1', None),
                
                'True Positives (OA)': metrics.get('2_tp', None),
                'True Negatives (Healthy)': metrics.get('0_tp', None),  # 0_tp in binary config usually maps to TN 
                'False Positives (Healthy labeled as OA)': metrics.get('2_fp', None),
                'False Negatives (OA labeled as Healthy)': metrics.get('2_fn', None),
                
                'Total Validated Samples': metrics.get('total', 1086),
                
                'Mean Prediction Confidence': conf.get('mean_confidence_all', None),
                'Mean Confidence (Correct)': conf.get('mean_confidence_correct', None),
                'Mean Confidence (Incorrect)': conf.get('mean_confidence_incorrect', None),
                'Uncertain Correct (%)': conf.get('uncertain_correct_pct', None),
                'Source File': filename
            }
            
            all_data.append(row)

# ==========================================
# DATAFRAME COMPILATION & EXPORT
# ==========================================

df = pd.DataFrame(all_data)

# Sort logically for readability in Excel
df.sort_values(by=['Architecture', 'Evaluation Mode', 'Test Condition', 'Seed'], inplace=True)

# 1. Export to CSV
df.to_csv(csv_out, index=False)
print(f"Successfully generated CSV at: {csv_out}")

# 2. Try Excel Export (Requires openpyxl)
try:
    with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Experiment Results')
        
        # Auto-adjust column widths for Excel aesthetics
        worksheet = writer.sheets['Experiment Results']
        for idx, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            # Set minimum boundary to prevent tiny columns, max 50 for very long text
            max_len = min(max_len, 50)
            # Excel columns correspond to A, B, C.. handled by idx + 1
            col_letter = chr(65 + idx) if idx < 26 else chr(65 + (idx // 26) - 1) + chr(65 + (idx % 26))
            worksheet.column_dimensions[col_letter].width = max_len
            
    print(f"Successfully generated Excel file at: {excel_out}")
except ModuleNotFoundError:
    print("openpyxl is not installed. Excel format (.xlsx) was skipped. Found all data accurately contained inside the compiled CSV file.")
except Exception as e:
    print(f"Failed to generate Excel file: {e}. Check the CSV instead.")

print(f"Aggregated {len(df)} total evaluation matrix runs across architectures representing individual seed evaluations.")
