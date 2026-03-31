"""
prepare_classification_data.py — Create classification-ready directory structures
from flat ablated image folders by copying files (cross-platform compatible).

Creates:
    classification_datasets/
        blackout/
            train/0/  → copies from blackout_train_0_extralarge_horiz/*.png
            train/2/  → copies from blackout_train_2_extralarge_horiz/*.png
            val/0/    → ...
            val/2/
            test/0/
            test/2/
        medial_masked/
            train/0/  → copies from medial_masked_train_0_extralarge_horiz/*.png
            ...
        lateral_masked/
            train/0/  → copies from lateral_masked_train_0_extralarge_horiz/*.png
            ...

Usage:
    python prepare_classification_data.py --base-dir . --output-dir classification_datasets
"""

import argparse
import os
import shutil
from pathlib import Path


# Mapping: condition name → folder prefix
CONDITIONS = {
    "blackout": "blackout",
    "medial_masked": "medial_masked",
    "lateral_masked": "lateral_masked",
}

SPLITS = ["train", "val", "test"]
GRADES = ["0", "2"]


def copy_dataset(base_dir: Path, output_dir: Path, condition: str, prefix: str):
    """Create a classification-ready directory by copying source images."""

    for split in SPLITS:
        for grade in GRADES:
            # Source folder: e.g., blackout_train_0_extralarge_horiz/
            src_folder = base_dir / f"{prefix}_{split}_{grade}_extralarge_horiz"

            if not src_folder.exists():
                print(f"  WARNING: Source not found: {src_folder}")
                continue

            # Target folder: e.g., classification_datasets/blackout/train/0/
            dst_folder = output_dir / condition / split / grade
            dst_folder.mkdir(parents=True, exist_ok=True)

            # Copy all PNG files
            count = 0
            skipped = 0
            for img_file in sorted(src_folder.glob("*.png")):
                dst_path = dst_folder / img_file.name
                if not dst_path.exists():
                    shutil.copy2(img_file, dst_path)
                    count += 1
                else:
                    skipped += 1

            msg = f"  {condition}/{split}/{grade}: {count} images copied"
            if skipped > 0:
                msg += f" ({skipped} already existed, skipped)"
            print(msg)


def main():
    parser = argparse.ArgumentParser(description="Prepare classification datasets from ablated folders")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base directory containing ablated folders (default: current dir)")
    parser.add_argument("--output-dir", type=str, default="classification_datasets",
                        help="Output directory for classification-ready datasets")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()

    for condition, prefix in CONDITIONS.items():
        print(f"Creating dataset: {condition}")
        copy_dataset(base_dir, output_dir, condition, prefix)
        print()

    # Verify counts
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    for condition in CONDITIONS:
        total = 0
        for split in SPLITS:
            for grade in GRADES:
                folder = output_dir / condition / split / grade
                if folder.exists():
                    count = len(list(folder.glob("*.png")))
                    total += count
        print(f"  {condition}: {total} total images")

    # Also check original dataset exists
    orig_dir = base_dir / "knee_osteoarthritis_dataset"
    if orig_dir.exists():
        total = 0
        for split in SPLITS:
            for grade in GRADES:
                folder = orig_dir / split / grade
                if folder.exists():
                    total += len(list(folder.glob("*.png")))
        print(f"  original: {total} total images")
    else:
        print(f"  WARNING: Original dataset not found at {orig_dir}")

    print(f"\nDone! Datasets ready in: {output_dir}")
    print(f"\nTo train, run:")
    print(f"  python classify_train.py --data-dir {output_dir}/blackout --name blackout --epochs 25")


if __name__ == "__main__":
    main()
