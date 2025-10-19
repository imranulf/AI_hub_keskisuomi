"""
Split mask vertically and apply left/right halves separately to original images.
Creates two versions: one with left half masked, one with right half masked.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def apply_split_mask(original_img_path, mask_path, output_left_path, output_right_path):
    """
    Split mask vertically and create two versions of the image:
    - Left version: Only left half of mask applied (darkened)
    - Right version: Only right half of mask applied (darkened)
    
    Args:
        original_img_path: Path to original image
        mask_path: Path to mask image
        output_left_path: Path to save image with left half masked
        output_right_path: Path to save image with right half masked
    """
    # Read images
    original = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if original is None:
        raise ValueError(f"Could not read original image: {original_img_path}")
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    
    # Ensure same dimensions
    if original.shape != mask.shape:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # Get the center column for splitting
    mid_col = original.shape[1] // 2
    
    # Create left-masked version (only left half of mask applied)
    left_masked = original.copy()
    left_mask = np.zeros_like(mask)
    left_mask[:, :mid_col] = mask[:, :mid_col]  # Copy only left half of mask
    left_masked[left_mask > 0] = 0  # Apply left half mask
    
    # Create right-masked version (only right half of mask applied)
    right_masked = original.copy()
    right_mask = np.zeros_like(mask)
    right_mask[:, mid_col:] = mask[:, mid_col:]  # Copy only right half of mask
    right_masked[right_mask > 0] = 0  # Apply right half mask
    
    # Save results
    cv2.imwrite(output_left_path, left_masked)
    cv2.imwrite(output_right_path, right_masked)


def process_directory(input_dir, mask_dir, output_left_dir, output_right_dir):
    """
    Process all images in directory and create left/right masked versions.
    
    Args:
        input_dir: Directory containing original images
        mask_dir: Directory containing mask images
        output_left_dir: Directory to save left-masked images
        output_right_dir: Directory to save right-masked images
    """
    # Create output directories
    os.makedirs(output_left_dir, exist_ok=True)
    os.makedirs(output_right_dir, exist_ok=True)
    
    # Get list of images
    input_path = Path(input_dir)
    image_files = sorted([f for f in input_path.glob('*.png')])
    
    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return
    
    print(f"Original images: {len(image_files)}")
    print(f"Mask directory: {mask_dir}")
    print(f"Output left-masked: {output_left_dir}")
    print(f"Output right-masked: {output_right_dir}")
    print()
    
    success_count = 0
    failed_count = 0
    
    for idx, img_file in enumerate(image_files, 1):
        try:
            # Get corresponding mask file
            mask_filename = img_file.stem + '_mask.png'
            mask_path = os.path.join(mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                print(f"{idx}/{len(image_files)}: {img_file.name} - Mask not found, skipping")
                failed_count += 1
                continue
            
            # Output paths
            output_left_path = os.path.join(output_left_dir, img_file.name)
            output_right_path = os.path.join(output_right_dir, img_file.name)
            
            # Apply split masks
            apply_split_mask(str(img_file), mask_path, output_left_path, output_right_path)
            
            print(f"{idx}/{len(image_files)}: {img_file.name} ✓")
            success_count += 1
            
        except Exception as e:
            print(f"{idx}/{len(image_files)}: {img_file.name} ✗ Error: {str(e)}")
            failed_count += 1
    
    print()
    print("=" * 50)
    print(f"✓ Success: {success_count}/{len(image_files)}")
    print(f"✗ Failed: {failed_count}/{len(image_files)}")
    print(f"✓ Left-masked saved to: {output_left_dir}/")
    print(f"✓ Right-masked saved to: {output_right_dir}/")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Split masks vertically and apply left/right halves separately'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing original images')
    parser.add_argument('-m', '--masks', required=True,
                        help='Directory containing mask images')
    parser.add_argument('-l', '--output-left', required=True,
                        help='Output directory for left-masked images')
    parser.add_argument('-r', '--output-right', required=True,
                        help='Output directory for right-masked images')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return
    
    if not os.path.exists(args.masks):
        print(f"Error: Mask directory not found: {args.masks}")
        return
    
    # Process all images
    process_directory(args.input, args.masks, args.output_left, args.output_right)


if __name__ == '__main__':
    main()
