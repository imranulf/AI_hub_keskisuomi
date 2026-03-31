"""
Apply horizontal masks to create three types of blackout images:
1. Left side blacked out (left half of horizontal band)
2. Right side blacked out (right half of horizontal band)
3. Full blackout (entire horizontal band)
"""
import argparse
import os
import cv2
import numpy as np


def apply_horizontal_blackout(original_path, mask_path, output_left, output_right, output_full):
    """
    Apply horizontal mask to create left, right, and full blackout versions.

    Args:
        original_path: Path to original image
        mask_path: Path to horizontal mask
        output_left: Path to save left-blackout image
        output_right: Path to save right-blackout image
        output_full: Path to save full-blackout image

    Returns:
        (success, message)
    """
    # Load original image
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        return False, "Could not read original image"

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, "Could not read mask"

    # Ensure mask and image have same size
    if original.shape != mask.shape:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    height, width = original.shape
    mid_col = width // 2

    # Create left mask (left half of horizontal band)
    left_mask = np.zeros_like(mask)
    left_mask[:, :mid_col] = mask[:, :mid_col]

    # Create right mask (right half of horizontal band)
    right_mask = np.zeros_like(mask)
    right_mask[:, mid_col:] = mask[:, mid_col:]

    # Apply left blackout
    result_left = original.copy()
    result_left[left_mask > 0] = 0
    cv2.imwrite(output_left, result_left)

    # Apply right blackout
    result_right = original.copy()
    result_right[right_mask > 0] = 0
    cv2.imwrite(output_right, result_right)

    # Apply full blackout
    result_full = original.copy()
    result_full[mask > 0] = 0
    cv2.imwrite(output_full, result_full)

    return True, "Success"


def get_args():
    parser = argparse.ArgumentParser(
        description='Apply horizontal masks to create left, right, and full blackout images')
    parser.add_argument('--original-dir', '-i', required=True,
                        help='Directory with original images')
    parser.add_argument('--mask-dir', '-m', required=True,
                        help='Directory with horizontal masks')
    parser.add_argument('--output-left', '-l', required=True,
                        help='Output directory for left-blackout images')
    parser.add_argument('--output-right', '-r', required=True,
                        help='Output directory for right-blackout images')
    parser.add_argument('--output-full', '-f', required=True,
                        help='Output directory for full-blackout images')
    parser.add_argument('--mask-suffix', '-s', default='_mask',
                        help='Suffix in mask filenames (default: _mask)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Create output directories
    os.makedirs(args.output_left, exist_ok=True)
    os.makedirs(args.output_right, exist_ok=True)
    os.makedirs(args.output_full, exist_ok=True)

    # Validate input directories
    if not os.path.exists(args.original_dir):
        print(f"Error: Original directory not found: {args.original_dir}")
        exit(1)

    if not os.path.exists(args.mask_dir):
        print(f"Error: Mask directory not found: {args.mask_dir}")
        exit(1)

    # Get list of original images
    original_files = [f for f in os.listdir(args.original_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not original_files:
        print(f"Error: No images found in {args.original_dir}")
        exit(1)

    print(f"{'='*60}")
    print(f"HORIZONTAL MASK BLACKOUT - THREE TYPES")
    print(f"{'='*60}")
    print(f"Original images: {args.original_dir}")
    print(f"Horizontal masks: {args.mask_dir}")
    print(f"Output (left):  {args.output_left}")
    print(f"Output (right): {args.output_right}")
    print(f"Output (full):  {args.output_full}")
    print(f"{'='*60}\n")

    success_count = 0
    failed_count = 0

    for i, orig_filename in enumerate(original_files, 1):
        # Construct paths
        original_path = os.path.join(args.original_dir, orig_filename)

        # Determine mask filename
        name_without_ext = os.path.splitext(orig_filename)[0]
        ext = os.path.splitext(orig_filename)[1]
        mask_filename = f"{name_without_ext}{args.mask_suffix}{ext}"
        mask_path = os.path.join(args.mask_dir, mask_filename)

        # Output paths
        output_left = os.path.join(args.output_left, orig_filename)
        output_right = os.path.join(args.output_right, orig_filename)
        output_full = os.path.join(args.output_full, orig_filename)

        print(f"{i}/{len(original_files)}: {orig_filename} ", end='', flush=True)

        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"[FAIL] Mask not found")
            failed_count += 1
            continue

        # Apply blackouts
        success, message = apply_horizontal_blackout(
            original_path, mask_path, output_left, output_right, output_full
        )

        if success:
            print("[OK] (left, right, full)")
            success_count += 1
        else:
            print(f"[FAIL] {message}")
            failed_count += 1

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"[OK] Success: {success_count}/{len(original_files)}")
    print(f"[FAIL] Failed:  {failed_count}/{len(original_files)}")
    print(f"\nOutput directories:")
    print(f"  Left blackout:  {args.output_left}/")
    print(f"  Right blackout: {args.output_right}/")
    print(f"  Full blackout:  {args.output_full}/")
    print(f"{'='*60}")
