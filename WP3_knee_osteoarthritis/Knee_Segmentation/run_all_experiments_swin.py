"""
run_all_experiments_swin.py — Run all 4 classification experiment phases using Swin-Tiny

Identical protocol to run_all_experiments.py (ResNet-18) but using Swin Transformer Tiny.

Phases:
    1. Train 4 models (baseline, blackout, medial_masked, lateral_masked)
    2. Self-evaluation (each model on its own test set)
    3. Cross-evaluation (baseline model on all 4 test sets) — confidence analysis for RQ3
    4. Comparison summary with RQ-specific analysis

Usage:
    python run_all_experiments_swin.py --epochs 25 --batch-size 32
    python run_all_experiments_swin.py --phase 2      # evaluation only
    python run_all_experiments_swin.py --phase 4      # summary only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# Experimental conditions and their data directories
EXPERIMENTS = {
    "baseline": "knee_osteoarthritis_dataset",
    "blackout": "classification_datasets/blackout",
    "medial_masked": "classification_datasets/medial_masked",
    "lateral_masked": "classification_datasets/lateral_masked",
}

RESULTS_DIR = Path("classification_results_swin")
TRAIN_SCRIPT = "classify_train_swin.py"
EVAL_SCRIPT = "classify_evaluate_swin.py"


def get_python():
    return sys.executable


def run_command(cmd, description=""):
    """Run subprocess command with output."""
    if description:
        print(f"\n  {description}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: Command exited with code {result.returncode}")
    return result.returncode


def phase1_train(epochs, batch_size):
    """Phase 1: Train 4 Swin-Tiny models."""
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: TRAINING (Swin-Tiny)")
    print(f"  4 conditions x {epochs} epochs")
    print(f"{'#'*70}")

    python = get_python()
    script = str(Path(__file__).parent / TRAIN_SCRIPT)

    for name, data_dir in EXPERIMENTS.items():
        model_path = RESULTS_DIR / name / "best_model.pth"
        if model_path.exists():
            print(f"\n  SKIP {name}: model already exists at {model_path}")
            continue

        if not Path(data_dir).exists():
            print(f"\n  SKIP {name}: data directory not found at {data_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"  Training: {name}")
        print(f"  Data: {data_dir}")
        print(f"{'='*70}")

        cmd = [
            python, script,
            "--data-dir", str(data_dir),
            "--name", name,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", "0.0001",
            "--patience", "7",
            "--gpu", "0",
        ]
        run_command(cmd)


def phase2_self_eval(batch_size):
    """Phase 2: Evaluate each model on its own test set."""
    print(f"\n{'#'*70}")
    print(f"  PHASE 2: SELF-EVALUATION (Swin-Tiny)")
    print(f"{'#'*70}")

    python = get_python()
    script = str(Path(__file__).parent / EVAL_SCRIPT)

    for name, data_dir in EXPERIMENTS.items():
        model_path = RESULTS_DIR / name / "best_model.pth"
        test_dir = Path(data_dir) / "test"
        eval_name = f"{name}_self"

        if not model_path.exists():
            print(f"\n  SKIP {name}: no model found")
            continue
        if not test_dir.exists():
            print(f"\n  SKIP {name}: test dir not found at {test_dir}")
            continue

        eval_path = RESULTS_DIR / "evaluations" / f"{eval_name}.json"
        if eval_path.exists():
            print(f"\n  SKIP {name}: evaluation already exists")
            continue

        print(f"\n  Evaluating: {name} (self)")
        cmd = [
            python, script,
            "--model", str(model_path),
            "--test-dir", str(test_dir),
            "--name", eval_name,
            "--batch-size", str(batch_size),
            "--gpu", "0",
        ]
        run_command(cmd)


def phase3_cross_eval(batch_size):
    """Phase 3: Cross-evaluate baseline model on all test sets."""
    print(f"\n{'#'*70}")
    print(f"  PHASE 3: CROSS-EVALUATION (Swin-Tiny)")
    print(f"  Baseline model -> all 4 test sets")
    print(f"{'#'*70}")

    python = get_python()
    script = str(Path(__file__).parent / EVAL_SCRIPT)
    baseline_model = RESULTS_DIR / "baseline" / "best_model.pth"

    if not baseline_model.exists():
        print("  ERROR: Baseline model not found. Run Phase 1 first.")
        return

    for name, data_dir in EXPERIMENTS.items():
        test_dir = Path(data_dir) / "test"
        eval_name = f"baseline_on_{name}"

        if not test_dir.exists():
            print(f"\n  SKIP {name}: test dir not found")
            continue

        eval_path = RESULTS_DIR / "evaluations" / f"{eval_name}.json"
        if eval_path.exists():
            print(f"\n  SKIP {eval_name}: evaluation already exists")
            continue

        print(f"\n  Cross-eval: baseline -> {name}")
        cmd = [
            python, script,
            "--model", str(baseline_model),
            "--test-dir", str(test_dir),
            "--name", eval_name,
            "--batch-size", str(batch_size),
            "--gpu", "0",
        ]
        run_command(cmd)


def phase4_summary():
    """Phase 4: Generate comparison summary with RQ analysis."""
    print(f"\n{'#'*70}")
    print(f"  PHASE 4: COMPARISON SUMMARY (Swin-Tiny)")
    print(f"{'#'*70}")

    eval_dir = RESULTS_DIR / "evaluations"
    if not eval_dir.exists():
        print("  ERROR: No evaluations found. Run Phases 2-3 first.")
        return

    # Load all evaluation results
    results = {}
    for json_file in sorted(eval_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
            results[json_file.stem] = data

    # Self-evaluation comparison
    print(f"\n{'='*70}")
    print(f"  SELF-EVALUATION COMPARISON (Swin-Tiny)")
    print(f"{'='*70}")
    print(f"{'Condition':<20} {'Accuracy':<12} {'Macro F1':<12} {'Confidence':<12} {'Uncertain':<12}")
    print("-" * 68)

    for name in EXPERIMENTS:
        key = f"{name}_self"
        if key in results:
            r = results[key]
            acc = r["metrics"]["accuracy"]
            f1 = r["metrics"]["macro_f1"]
            conf = r["confidence"]["mean_confidence_all"]
            unc = r["confidence"]["uncertain_correct_pct"]
            print(f"{name:<20} {acc:<12.4f} {f1:<12.4f} {conf:<12.4f} {unc:<12.1f}%")

    # Cross-evaluation comparison
    print(f"\n{'='*70}")
    print(f"  CROSS-EVALUATION COMPARISON (Swin-Tiny baseline model)")
    print(f"{'='*70}")
    print(f"{'Test Condition':<25} {'Accuracy':<12} {'Macro F1':<12} {'Confidence':<12} {'Uncertain':<12}")
    print("-" * 73)

    baseline_acc = None
    for name in EXPERIMENTS:
        key = f"baseline_on_{name}"
        if key in results:
            r = results[key]
            acc = r["metrics"]["accuracy"]
            f1 = r["metrics"]["macro_f1"]
            conf = r["confidence"]["mean_confidence_all"]
            unc = r["confidence"]["uncertain_correct_pct"]
            if name == "baseline":
                baseline_acc = acc
            drop = f"({acc - baseline_acc:+.4f})" if baseline_acc is not None else ""
            print(f"{key:<25} {acc:<12.4f} {f1:<12.4f} {conf:<12.4f} {unc:<12.1f}% {drop}")

    # RQ-specific analysis
    print(f"\n{'='*70}")
    print(f"  RQ-SPECIFIC ANALYSIS (Swin-Tiny)")
    print(f"{'='*70}")

    if "baseline_self" in results and "blackout_self" in results:
        base_acc = results["baseline_self"]["metrics"]["accuracy"]
        black_acc = results["blackout_self"]["metrics"]["accuracy"]
        print(f"\n  RQ1 (Joint region): Blackout drops accuracy by {base_acc - black_acc:.4f} "
              f"({base_acc:.4f} -> {black_acc:.4f})")

    if "medial_masked_self" in results and "lateral_masked_self" in results:
        med_acc = results["medial_masked_self"]["metrics"]["accuracy"]
        lat_acc = results["lateral_masked_self"]["metrics"]["accuracy"]
        if "baseline_self" in results:
            base_acc = results["baseline_self"]["metrics"]["accuracy"]
            med_drop = base_acc - med_acc
            lat_drop = base_acc - lat_acc
            ratio = med_drop / lat_drop if lat_drop > 0 else float('inf')
            print(f"  RQ2 (Compartmental): Medial drop={med_drop:.4f}, Lateral drop={lat_drop:.4f}, "
                  f"Ratio={ratio:.1f}x")

    if "baseline_on_baseline" in results:
        print(f"\n  RQ3 (Confidence degradation -- cross-eval):")
        base_conf = results["baseline_on_baseline"]["confidence"]["mean_confidence_all"]
        for name in EXPERIMENTS:
            key = f"baseline_on_{name}"
            if key in results:
                conf = results[key]["confidence"]["mean_confidence_all"]
                unc = results[key]["confidence"]["uncertain_correct_pct"]
                print(f"    {key}: confidence={conf:.4f} (delta={conf-base_conf:+.4f}), "
                      f"uncertain correct={unc:.1f}%")

    # Save summary JSON
    summary = {
        "architecture": "swin_tiny",
        "self_evaluation": {},
        "cross_evaluation": {},
    }
    for name in EXPERIMENTS:
        key = f"{name}_self"
        if key in results:
            summary["self_evaluation"][name] = {
                "accuracy": results[key]["metrics"]["accuracy"],
                "macro_f1": results[key]["metrics"]["macro_f1"],
                "mean_confidence": results[key]["confidence"]["mean_confidence_all"],
                "uncertain_correct_pct": results[key]["confidence"]["uncertain_correct_pct"],
                "confusion_matrix": results[key].get("confusion_matrix", [])
            }
        key = f"baseline_on_{name}"
        if key in results:
            summary["cross_evaluation"][name] = {
                "accuracy": results[key]["metrics"]["accuracy"],
                "macro_f1": results[key]["metrics"]["macro_f1"],
                "mean_confidence": results[key]["confidence"]["mean_confidence_all"],
                "uncertain_correct_pct": results[key]["confidence"]["uncertain_correct_pct"],
                "confusion_matrix": results[key].get("confusion_matrix", [])
            }

    summary_path = RESULTS_DIR / "experiment_summary.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    print(f"\n{'#'*70}")
    print(f"  ALL EXPERIMENTS COMPLETE (Swin-Tiny)")
    print(f"{'#'*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all Swin-Tiny classification experiments (4 phases)"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase only (1-4). 0=all phases.")
    args = parser.parse_args()

    if args.phase == 0 or args.phase == 1:
        phase1_train(args.epochs, args.batch_size)
    if args.phase == 0 or args.phase == 2:
        phase2_self_eval(args.batch_size)
    if args.phase == 0 or args.phase == 3:
        phase3_cross_eval(args.batch_size)
    if args.phase == 0 or args.phase == 4:
        phase4_summary()


if __name__ == "__main__":
    main()
