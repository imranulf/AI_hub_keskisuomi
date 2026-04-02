"""
run_robustness_swin.py — Run Swin-Tiny classification experiments with multiple
random seeds for statistical robustness analysis.

Identical protocol to run_robustness.py (ResNet-18) but using Swin Transformer Tiny.

Trains each condition N times with different seeds and reports mean +/- std
for accuracy, F1, and confidence.

Usage:
    # Run 5 seeds for all conditions (full robustness test)
    python run_robustness_swin.py --seeds 5 --epochs 25

    # Run 3 seeds for baseline only (quick test)
    python run_robustness_swin.py --seeds 3 --epochs 25 --conditions baseline

    # Resume from a specific seed (if interrupted)
    python run_robustness_swin.py --seeds 5 --start-seed 3

    # Summary only (after training complete)
    python run_robustness_swin.py --summary-only

Output:
    classification_results_swin/robustness/
        seed_{N}/{condition}/best_model.pth
        seed_{N}/evaluations/{condition}_self.json
        seed_{N}/evaluations/baseline_on_{condition}.json
        robustness_summary.json
        robustness_summary.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev


# Conditions and their data directories
CONDITIONS = {
    "baseline": "knee_osteoarthritis_dataset",
    "blackout": "classification_datasets/blackout",
    "medial_masked": "classification_datasets/medial_masked",
    "lateral_masked": "classification_datasets/lateral_masked",
}

RESULTS_BASE = Path("classification_results_swin")
TRAIN_SCRIPT = "classify_train_swin.py"
EVAL_SCRIPT = "classify_evaluate_swin.py"


def get_python():
    return sys.executable


def set_seed_env(seed: int) -> dict:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    return env


def run_command(cmd: list, env: dict = None):
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"  WARNING: Command exited with code {result.returncode}")
    return result.returncode


def train_with_seed(condition: str, data_dir: str, seed: int,
                     epochs: int, batch_size: int, output_base: Path):
    output_dir = output_base / f"seed_{seed}" / condition
    output_dir.mkdir(parents=True, exist_ok=True)

    python = get_python()
    script = str(Path(__file__).parent / TRAIN_SCRIPT)

    cmd = [
        python, script,
        "--data-dir", str(data_dir),
        "--name", f"robustness/seed_{seed}/{condition}",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", "0.0001",
        "--patience", "7",
        "--gpu", "0",
    ]

    env = set_seed_env(seed)

    seed_file = output_dir / "seed.txt"
    with open(seed_file, "w") as f:
        f.write(str(seed))

    return run_command(cmd, env=env)


def evaluate_with_seed(model_path: str, test_dir: str, eval_name: str,
                        seed: int, output_base: Path):
    python = get_python()
    script = str(Path(__file__).parent / EVAL_SCRIPT)

    full_name = f"robustness/seed_{seed}/{eval_name}"

    cmd = [
        python, script,
        "--model", str(model_path),
        "--test-dir", str(test_dir),
        "--name", full_name,
        "--batch-size", "32",
        "--gpu", "0",
    ]

    return run_command(cmd)


def load_eval_results(eval_path: Path) -> dict:
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)
    return None


def compute_robustness_stats(all_results: dict) -> dict:
    stats = {}
    conditions = set()
    for seed_results in all_results.values():
        conditions.update(seed_results.keys())

    for condition in sorted(conditions):
        metrics_by_name = {}
        for seed, seed_results in all_results.items():
            if condition in seed_results:
                for metric, value in seed_results[condition].items():
                    if isinstance(value, (int, float)):
                        if metric not in metrics_by_name:
                            metrics_by_name[metric] = []
                        metrics_by_name[metric].append(value)

        stats[condition] = {}
        for metric, values in metrics_by_name.items():
            if len(values) > 1:
                stats[condition][metric] = {
                    "mean": round(mean(values), 4),
                    "std": round(stdev(values), 4),
                    "values": [round(v, 4) for v in values],
                    "n": len(values),
                }
            elif len(values) == 1:
                stats[condition][metric] = {
                    "mean": round(values[0], 4),
                    "std": 0.0,
                    "values": [round(values[0], 4)],
                    "n": 1,
                }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed robustness experiments (Swin-Tiny)"
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conditions", nargs="+", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    output_base = RESULTS_BASE / "robustness"
    output_base.mkdir(parents=True, exist_ok=True)

    conditions = args.conditions or list(CONDITIONS.keys())
    seed_list = list(range(args.start_seed, args.start_seed + args.seeds))
    actual_seeds = [42 + i * 137 for i in seed_list]

    print("=" * 70)
    print(f"  ROBUSTNESS EXPERIMENT (Swin-Tiny)")
    print(f"  Seeds: {len(actual_seeds)} ({actual_seeds})")
    print(f"  Conditions: {conditions}")
    print(f"  Epochs: {args.epochs}")
    print("=" * 70)

    # ---- Phase 1: Training ----
    if not args.eval_only and not args.summary_only:
        print(f"\n{'#'*70}")
        print(f"  PHASE 1: TRAINING ({len(actual_seeds)} seeds x {len(conditions)} conditions)")
        print(f"{'#'*70}")

        for seed_idx, seed in enumerate(actual_seeds):
            for condition in conditions:
                data_dir = Path(CONDITIONS[condition])
                if not data_dir.exists():
                    print(f"\n  SKIP {condition} seed {seed}: data dir not found")
                    continue

                model_path = output_base / f"seed_{seed}" / condition / "best_model.pth"
                if model_path.exists():
                    print(f"\n  SKIP {condition} seed {seed}: model already exists")
                    continue

                print(f"\n{'='*70}")
                print(f"  Training: {condition} (seed {seed}, {seed_idx+1}/{len(actual_seeds)})")
                print(f"{'='*70}")

                train_with_seed(condition, data_dir, seed, args.epochs,
                                args.batch_size, output_base)

    # ---- Phase 2: Evaluation ----
    if not args.summary_only:
        print(f"\n{'#'*70}")
        print(f"  PHASE 2: EVALUATION (Swin-Tiny)")
        print(f"{'#'*70}")

        for seed in actual_seeds:
            for condition in conditions:
                model_path = output_base / f"seed_{seed}" / condition / "best_model.pth"
                if not model_path.exists():
                    continue

                test_dir = Path(CONDITIONS[condition]) / "test"
                if test_dir.exists():
                    eval_name = f"{condition}_self"
                    eval_path = RESULTS_BASE / "evaluations" / "robustness" / f"seed_{seed}" / f"{eval_name}.json"
                    if not eval_path.exists():
                        print(f"\n  Evaluating: {condition} seed {seed} (self)")
                        evaluate_with_seed(str(model_path), str(test_dir),
                                           eval_name, seed, output_base)

                if condition == "baseline":
                    for cross_cond in conditions:
                        cross_test = Path(CONDITIONS[cross_cond]) / "test"
                        if cross_test.exists():
                            cross_name = f"baseline_on_{cross_cond}"
                            cross_path = RESULTS_BASE / "evaluations" / "robustness" / f"seed_{seed}" / f"{cross_name}.json"
                            if not cross_path.exists():
                                print(f"\n  Cross-eval: baseline seed {seed} -> {cross_cond}")
                                evaluate_with_seed(str(model_path), str(cross_test),
                                                   cross_name, seed, output_base)

    # ---- Phase 3: Summary ----
    print(f"\n{'#'*70}")
    print(f"  PHASE 3: ROBUSTNESS SUMMARY (Swin-Tiny)")
    print(f"{'#'*70}")

    self_eval_results = {}
    cross_eval_results = {}

    for seed in actual_seeds:
        self_eval_results[seed] = {}
        cross_eval_results[seed] = {}

        for condition in conditions:
            eval_path = RESULTS_BASE / "evaluations" / "robustness" / f"seed_{seed}" / f"{condition}_self.json"
            data = load_eval_results(eval_path)
            if data:
                conf_data = data.get("confidence", {})
                self_eval_results[seed][condition] = {
                    "accuracy": data.get("metrics", {}).get("accuracy", 0),
                    "macro_f1": data.get("metrics", {}).get("macro_f1", 0),
                    "mean_confidence": conf_data.get("mean_confidence_all", 0),
                    "uncertain_correct_pct": conf_data.get("uncertain_correct_pct", 0),
                }

            cross_path = RESULTS_BASE / "evaluations" / "robustness" / f"seed_{seed}" / f"baseline_on_{condition}.json"
            data = load_eval_results(cross_path)
            if data:
                conf_data = data.get("confidence", {})
                cross_eval_results[seed][f"baseline_on_{condition}"] = {
                    "accuracy": data.get("metrics", {}).get("accuracy", 0),
                    "macro_f1": data.get("metrics", {}).get("macro_f1", 0),
                    "mean_confidence": conf_data.get("mean_confidence_all", 0),
                    "uncertain_correct_pct": conf_data.get("uncertain_correct_pct", 0),
                }

    self_stats = compute_robustness_stats(self_eval_results)
    cross_stats = compute_robustness_stats(cross_eval_results)

    print(f"\n{'='*70}")
    print(f"  SELF-EVALUATION (mean +/- std across {len(actual_seeds)} seeds)")
    print(f"{'='*70}")
    print(f"{'Condition':<20} {'Accuracy':<20} {'Macro F1':<20} {'Confidence':<20}")
    print("-" * 80)
    for condition in conditions:
        if condition in self_stats:
            s = self_stats[condition]
            acc = s.get("accuracy", {})
            f1 = s.get("macro_f1", {})
            conf = s.get("mean_confidence", {})
            print(f"{condition:<20} "
                  f"{acc.get('mean', 0):.4f}+/-{acc.get('std', 0):.4f}  "
                  f"{f1.get('mean', 0):.4f}+/-{f1.get('std', 0):.4f}  "
                  f"{conf.get('mean', 0):.4f}+/-{conf.get('std', 0):.4f}")

    print(f"\n{'='*70}")
    print(f"  CROSS-EVALUATION (baseline model, mean +/- std)")
    print(f"{'='*70}")
    print(f"{'Test Condition':<25} {'Accuracy':<20} {'Macro F1':<20} {'Confidence':<20}")
    print("-" * 85)
    for condition in conditions:
        key = f"baseline_on_{condition}"
        if key in cross_stats:
            s = cross_stats[key]
            acc = s.get("accuracy", {})
            f1 = s.get("macro_f1", {})
            conf = s.get("mean_confidence", {})
            print(f"{key:<25} "
                  f"{acc.get('mean', 0):.4f}+/-{acc.get('std', 0):.4f}  "
                  f"{f1.get('mean', 0):.4f}+/-{f1.get('std', 0):.4f}  "
                  f"{conf.get('mean', 0):.4f}+/-{conf.get('std', 0):.4f}")

    summary = {
        "architecture": "swin_tiny",
        "seeds": actual_seeds,
        "n_seeds": len(actual_seeds),
        "conditions": conditions,
        "epochs": args.epochs,
        "self_evaluation": self_stats,
        "cross_evaluation": cross_stats,
    }

    summary_path = output_base / "robustness_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    csv_path = output_base / "robustness_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Condition", "Metric", "Mean", "Std", "N", "Values"])
        for condition, metrics in self_stats.items():
            for metric, vals in metrics.items():
                writer.writerow(["self_eval", condition, metric,
                                 vals["mean"], vals["std"], vals["n"],
                                 ";".join(str(v) for v in vals["values"])])
        for condition, metrics in cross_stats.items():
            for metric, vals in metrics.items():
                writer.writerow(["cross_eval", condition, metric,
                                 vals["mean"], vals["std"], vals["n"],
                                 ";".join(str(v) for v in vals["values"])])
    print(f"CSV saved to: {csv_path}")

    print(f"\n{'#'*70}")
    print(f"  ROBUSTNESS ANALYSIS COMPLETE (Swin-Tiny)")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
