"""
run_all_experiments.py — Master script to run all classification experiments

Experiments:
    Phase 1: Train 4 models (one per condition)
        1. Baseline   — train on original, validate on original
        2. Blackout   — train on blackout, validate on blackout
        3. Medial     — train on medial_masked, validate on medial_masked
        4. Lateral    — train on lateral_masked, validate on lateral_masked

    Phase 2: Evaluate all 4 models on their own test sets

    Phase 3: Cross-evaluation (RQ3 — confidence analysis)
        - Baseline model tested on all 4 test sets

    Phase 4: Generate comparison summary

Usage:
    python run_all_experiments.py --epochs 25 --batch-size 32
    python run_all_experiments.py --phase 2  # run only evaluation
    python run_all_experiments.py --phase 3  # run only cross-evaluation
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).parent.resolve()

EXPERIMENTS = {
    "baseline": str(BASE_DIR / "knee_osteoarthritis_dataset"),
    "blackout": str(BASE_DIR / "classification_datasets" / "blackout"),
    "medial_masked": str(BASE_DIR / "classification_datasets" / "medial_masked"),
    "lateral_masked": str(BASE_DIR / "classification_datasets" / "lateral_masked"),
}


def run_command(cmd: list, description: str):
    """Run a command and print its output."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  ERROR: Command failed with return code {result.returncode}")
        return False
    return True


def phase1_train(epochs: int, batch_size: int, lr: float, patience: int, gpu: int):
    """Phase 1: Train all 4 models."""
    print("\n" + "#" * 70)
    print("  PHASE 1: TRAINING")
    print("#" * 70)

    for name, data_dir in EXPERIMENTS.items():
        model_path = BASE_DIR / "classification_results" / name / "best_model.pth"
        if model_path.exists():
            print(f"\n  Skipping {name} — model already exists at {model_path}")
            print(f"  (Delete it to retrain)")
            continue

        cmd = [
            sys.executable, str(BASE_DIR / "classify_train.py"),
            "--data-dir", data_dir,
            "--name", name,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--patience", str(patience),
            "--gpu", str(gpu),
        ]
        success = run_command(cmd, f"Training: {name}")
        if not success:
            print(f"  Training failed for {name}. Stopping.")
            return False
    return True


def phase2_evaluate(batch_size: int, gpu: int):
    """Phase 2: Evaluate each model on its own test set."""
    print("\n" + "#" * 70)
    print("  PHASE 2: EVALUATION (each model on its own test set)")
    print("#" * 70)

    for name, data_dir in EXPERIMENTS.items():
        model_path = BASE_DIR / "classification_results" / name / "best_model.pth"
        if not model_path.exists():
            print(f"\n  Skipping {name} — no model found at {model_path}")
            continue

        test_dir = str(Path(data_dir) / "test")
        eval_name = f"{name}_self"

        cmd = [
            sys.executable, str(BASE_DIR / "classify_evaluate.py"),
            "--model", str(model_path),
            "--test-dir", test_dir,
            "--name", eval_name,
            "--batch-size", str(batch_size),
            "--gpu", str(gpu),
            "--save-per-image",
        ]
        run_command(cmd, f"Evaluating: {name} model on {name} test set")
    return True


def phase3_cross_evaluate(batch_size: int, gpu: int):
    """Phase 3: Cross-evaluation — baseline model on all ablated test sets (RQ3)."""
    print("\n" + "#" * 70)
    print("  PHASE 3: CROSS-EVALUATION (baseline model on all test sets)")
    print("#" * 70)

    baseline_model = BASE_DIR / "classification_results" / "baseline" / "best_model.pth"
    if not baseline_model.exists():
        print("  ERROR: Baseline model not found. Run Phase 1 first.")
        return False

    for name, data_dir in EXPERIMENTS.items():
        test_dir = str(Path(data_dir) / "test")
        eval_name = f"baseline_on_{name}"

        cmd = [
            sys.executable, str(BASE_DIR / "classify_evaluate.py"),
            "--model", str(baseline_model),
            "--test-dir", test_dir,
            "--name", eval_name,
            "--batch-size", str(batch_size),
            "--gpu", str(gpu),
            "--save-per-image",
        ]
        run_command(cmd, f"Cross-eval: baseline model → {name} test set")
    return True


def phase4_summary():
    """Phase 4: Generate comparison summary from all evaluation JSONs."""
    print("\n" + "#" * 70)
    print("  PHASE 4: COMPARISON SUMMARY")
    print("#" * 70)

    eval_dir = BASE_DIR / "classification_results" / "evaluations"
    if not eval_dir.exists():
        print("  No evaluation results found.")
        return

    results = {}
    for json_file in sorted(eval_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        results[data["name"]] = data

    if not results:
        print("  No evaluation JSON files found.")
        return

    # Print summary table
    print(f"\n{'Evaluation':<30s} {'Accuracy':>10s} {'Macro F1':>10s} {'Confidence':>12s}")
    print("-" * 65)

    for name, data in sorted(results.items()):
        acc = data["metrics"]["accuracy"]
        f1 = data["metrics"]["macro_f1"]
        conf = data["confidence"]["mean_confidence_all"]
        print(f"{name:<30s} {acc:>10.4f} {f1:>10.4f} {conf:>12.4f}")

    # RQ-specific analysis
    print(f"\n{'='*65}")
    print("RQ ANALYSIS")
    print(f"{'='*65}")

    # RQ1: Blackout vs Baseline
    if "baseline_self" in results and "blackout_self" in results:
        base_acc = results["baseline_self"]["metrics"]["accuracy"]
        blk_acc = results["blackout_self"]["metrics"]["accuracy"]
        drop = base_acc - blk_acc
        print(f"\nRQ1 (Joint space vs bone margins):")
        print(f"  Baseline accuracy:  {base_acc:.4f}")
        print(f"  Blackout accuracy:  {blk_acc:.4f}")
        print(f"  Accuracy drop:      {drop:+.4f} ({drop/base_acc*100:+.1f}%)")

    # RQ2: Medial vs Lateral
    if "medial_masked_self" in results and "lateral_masked_self" in results:
        med_acc = results["medial_masked_self"]["metrics"]["accuracy"]
        lat_acc = results["lateral_masked_self"]["metrics"]["accuracy"]
        print(f"\nRQ2 (Medial vs Lateral compartment):")
        print(f"  Medial masked accuracy:  {med_acc:.4f}")
        print(f"  Lateral masked accuracy: {lat_acc:.4f}")
        if "baseline_self" in results:
            base_acc = results["baseline_self"]["metrics"]["accuracy"]
            print(f"  Medial drop from baseline:  {base_acc - med_acc:+.4f}")
            print(f"  Lateral drop from baseline: {base_acc - lat_acc:+.4f}")

    # RQ3: Confidence analysis (cross-evaluation)
    cross_keys = [k for k in results if k.startswith("baseline_on_")]
    if cross_keys:
        print(f"\nRQ3 (Confidence shift under ablation):")
        for key in sorted(cross_keys):
            conf = results[key]["confidence"]
            acc = results[key]["metrics"]["accuracy"]
            print(f"  {key}:")
            print(f"    Accuracy: {acc:.4f}, Mean confidence: {conf['mean_confidence_all']:.4f}, "
                  f"Uncertain correct: {conf['uncertain_correct_count']} ({conf['uncertain_correct_pct']:.1f}%)")

    # Save summary
    summary_path = BASE_DIR / "classification_results" / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all classification experiments")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Run specific phase only (default: all phases)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    phases = [args.phase] if args.phase else [1, 2, 3, 4]

    if 1 in phases:
        if not phase1_train(args.epochs, args.batch_size, args.lr, args.patience, args.gpu):
            print("Phase 1 failed. Exiting.")
            return

    if 2 in phases:
        phase2_evaluate(args.batch_size, args.gpu)

    if 3 in phases:
        phase3_cross_evaluate(args.batch_size, args.gpu)

    if 4 in phases:
        phase4_summary()

    print("\n" + "#" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
