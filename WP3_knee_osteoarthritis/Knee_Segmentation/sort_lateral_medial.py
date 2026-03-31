"""
Sort left_masked and right_masked knee X-ray images into lateral/ and medial/
subfolders based on knee side (L/R suffix in filename).

Anatomy mapping (confirmed):
- Left knee ('L'): left side of X-ray = medial, right side = lateral
- Right knee ('R'): left side of X-ray = lateral, right side = medial

Therefore:
- left_masked + L-ending → medial/
- left_masked + R-ending → lateral/
- right_masked + L-ending → lateral/
- right_masked + R-ending → medial/

Files are COPIED (originals remain untouched).
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path("/sessions/zealous-ecstatic-einstein/mnt/Knee_Segmentation_Python")

FOLDERS = [
    "left_masked_train_0_extralarge_horiz",
    "left_masked_train_2_extralarge_horiz",
    "left_masked_test_0_extralarge_horiz",
    "left_masked_test_2_extralarge_horiz",
    "left_masked_val_0_extralarge_horiz",
    "left_masked_val_2_extralarge_horiz",
    "right_masked_train_0_extralarge_horiz",
    "right_masked_train_2_extralarge_horiz",
    "right_masked_test_0_extralarge_horiz",
    "right_masked_test_2_extralarge_horiz",
    "right_masked_val_0_extralarge_horiz",
    "right_masked_val_2_extralarge_horiz",
]


def get_target_subfolder(folder_name: str, filename: str) -> str:
    """Determine lateral or medial based on folder type and knee side."""
    is_left_masked = folder_name.startswith("left_masked")
    is_left_knee = filename.rstrip(".png").endswith("L")
    is_right_knee = filename.rstrip(".png").endswith("R")

    if not (is_left_knee or is_right_knee):
        return "unknown"  # skip files without L/R suffix

    if is_left_masked:
        # Left side of X-ray is masked
        # Left knee (L): left = medial → medial masked
        # Right knee (R): left = lateral → lateral masked
        return "medial" if is_left_knee else "lateral"
    else:
        # Right side of X-ray is masked
        # Left knee (L): right = lateral → lateral masked
        # Right knee (R): right = medial → medial masked
        return "lateral" if is_left_knee else "medial"


def process_folders():
    stats = {}

    for folder_name in FOLDERS:
        folder_path = BASE_DIR / folder_name
        if not folder_path.exists():
            print(f"SKIP: {folder_name} not found")
            continue

        lateral_dir = folder_path / "lateral"
        medial_dir = folder_path / "medial"
        lateral_dir.mkdir(exist_ok=True)
        medial_dir.mkdir(exist_ok=True)

        counts = {"lateral": 0, "medial": 0, "unknown": 0}

        for f in sorted(folder_path.iterdir()):
            if not f.is_file() or f.suffix.lower() != ".png":
                continue

            target = get_target_subfolder(folder_name, f.name)
            if target == "unknown":
                counts["unknown"] += 1
                continue

            dest_dir = lateral_dir if target == "lateral" else medial_dir
            shutil.copy2(f, dest_dir / f.name)
            counts[target] += 1

        stats[folder_name] = counts
        total = counts["lateral"] + counts["medial"]
        print(f"{folder_name}: lateral={counts['lateral']}, medial={counts['medial']}, "
              f"unknown={counts['unknown']}, total={total}")

    print("\n=== SUMMARY ===")
    total_lat = sum(s["lateral"] for s in stats.values())
    total_med = sum(s["medial"] for s in stats.values())
    total_unk = sum(s["unknown"] for s in stats.values())
    print(f"Total lateral: {total_lat}")
    print(f"Total medial:  {total_med}")
    print(f"Total unknown: {total_unk}")
    print(f"Grand total:   {total_lat + total_med + total_unk}")


if __name__ == "__main__":
    process_folders()
