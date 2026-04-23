#!/usr/bin/env python3
"""
compute_auc_metrics.py

Compute ROC AUC and PR AUC from per-image evaluation CSVs produced by
classify_evaluate*.py with --save-per-image.

Outputs:
  1. per_run_auc.csv          — one row per evaluation file
  2. auc_summary_mean_std.csv — mean ± std per (architecture, evaluation mode,
                                 test condition) across seeds.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ARCH_DIRS = {
    "ResNet-18": "classification_results",
    "EfficientNet-B0": "classification_results_efficientnet",
    "Swin-Tiny": "classification_results_swin",
}

CONDITIONS = ["baseline", "blackout", "medial_masked", "lateral_masked"]

# per-image filename stem examples:
#   baseline_self
#   baseline_on_blackout
#   seed_42_baseline_self
#   seed_42_baseline_on_blackout
NAME_RE = re.compile(r"^(?P<name>.+?)_per_image\.csv$")
SEED_PREFIX_RE = re.compile(r"^seed_(?P<seed>\d+)_(?P<core>.+)$")


class RunRecord(TypedDict):
    architecture: str
    seed: str
    evaluation_mode: str
    test_condition: str
    csv: Path


def parse_eval_name(name: str) -> tuple[str, str, str]:
    """Return (seed_tag, evaluation_mode, test_condition) from filename stem."""
    seed_tag = "Initial Single Run"
    core = name

    match = SEED_PREFIX_RE.match(name)
    if match:
        seed_tag = match.group("seed")
        core = match.group("core")

    if core.endswith("_self"):
        cond = core[: -len("_self")]
        mode = "Self-Evaluation"
    elif core.startswith("baseline_on_"):
        cond = core[len("baseline_on_") :]
        mode = "Cross-Evaluation"
    else:
        raise ValueError(f"Unrecognized evaluation name: {name}")

    if cond not in CONDITIONS:
        raise ValueError(f"Unrecognized condition '{cond}' from name: {name}")

    return seed_tag, mode, cond


def _to_binary_kl2_labels(series: pd.Series) -> np.ndarray:
    # Handles both string labels ('0','2') and numeric labels (0,2).
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return (numeric.astype(int) == 2).astype(int).to_numpy()
    return (series.astype(str).str.strip() == "2").astype(int).to_numpy()


def auc_from_csv(csv_path: Path) -> tuple[float, float, int]:
    df = pd.read_csv(csv_path)
    y_true = _to_binary_kl2_labels(df["true_label"])
    y_score = pd.to_numeric(df["prob_2"], errors="coerce").to_numpy()

    valid = ~np.isnan(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), int(len(df))

    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)
    return float(roc), float(pr), int(len(df))


def discover_runs(results_root: Path) -> list[RunRecord]:
    rows: list[RunRecord] = []
    seen_files: set[Path] = set()

    for arch, arch_dir in ARCH_DIRS.items():
        arch_root = results_root / arch_dir
        if not arch_root.exists():
            continue

        # Initial single-run evaluations in top-level evaluations folder.
        init_eval_dir = arch_root / "evaluations"
        if init_eval_dir.exists():
            for csv_file in sorted(init_eval_dir.glob("*_per_image.csv")):
                if csv_file in seen_files:
                    continue
                match = NAME_RE.match(csv_file.name)
                if not match:
                    continue
                name = match.group("name")

                # Seed-prefixed files here are temporary by-products from robustness re-eval.
                if name.startswith("seed_"):
                    continue

                try:
                    seed_tag, mode, cond = parse_eval_name(name)
                except ValueError:
                    continue

                if seed_tag != "Initial Single Run":
                    continue

                rows.append(
                    {
                        "architecture": arch,
                        "seed": seed_tag,
                        "evaluation_mode": mode,
                        "test_condition": cond,
                        "csv": csv_file,
                    }
                )
                seen_files.add(csv_file)

        # Robustness evaluations in per-seed folders.
        rob_root = arch_root / "robustness"
        if not rob_root.exists():
            continue

        for seed_dir in sorted(rob_root.glob("seed_*")):
            seed = seed_dir.name.replace("seed_", "")
            for cond_dir in CONDITIONS:
                eval_dir = seed_dir / cond_dir / "evaluations"
                if not eval_dir.exists():
                    continue

                for csv_file in sorted(eval_dir.glob("*_per_image.csv")):
                    if csv_file in seen_files:
                        continue
                    match = NAME_RE.match(csv_file.name)
                    if not match:
                        continue

                    name = match.group("name")
                    try:
                        name_seed, mode, test_cond = parse_eval_name(name)
                    except ValueError:
                        continue

                    # Accept canonical names (no seed prefix) and optional matching seed-prefixed names.
                    if name_seed not in ("Initial Single Run", seed):
                        continue

                    rows.append(
                        {
                            "architecture": arch,
                            "seed": seed,
                            "evaluation_mode": mode,
                            "test_condition": test_cond,
                            "csv": csv_file,
                        }
                    )
                    seen_files.add(csv_file)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ROC AUC and PR AUC from per-image evaluation CSVs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Path containing classification_results, classification_results_efficientnet, classification_results_swin.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for per_run_auc.csv and auc_summary_mean_std.csv.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.results_root)
    if not runs:
        print("No per-image CSVs found. Re-run classify_evaluate*.py with --save-per-image first.")
        return

    long_rows: list[dict[str, object]] = []
    for run in runs:
        roc_auc, pr_auc, n = auc_from_csv(run["csv"])
        long_rows.append(
            {
                "architecture": run["architecture"],
                "seed": run["seed"],
                "evaluation_mode": run["evaluation_mode"],
                "test_condition": run["test_condition"],
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "n": n,
                "source_file": str(run["csv"]),
            }
        )

    long_df = pd.DataFrame(long_rows)
    long_path = args.out_dir / "per_run_auc.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(long_df)} rows)")

    seed_mask = long_df["seed"] != "Initial Single Run"
    summary = (
        long_df[seed_mask]
        .groupby(["architecture", "evaluation_mode", "test_condition"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            n_seeds=("roc_auc", "count"),
        )
        .sort_values(["architecture", "evaluation_mode", "test_condition"])
    )

    summary_path = args.out_dir / "auc_summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary)} rows)")

    if not summary.empty:
        print("\nMean ± std across seeds:\n")
        for _, row in summary.iterrows():
            print(
                f"  {row['architecture']:<16} "
                f"{row['evaluation_mode']:<18} "
                f"{row['test_condition']:<16} "
                f"ROC AUC = {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}  "
                f"PR AUC = {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}  "
                f"(n={int(row['n_seeds'])})"
            )


if __name__ == "__main__":
    main()
