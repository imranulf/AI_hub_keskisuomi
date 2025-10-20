"""
Expand masks to cover more area using morphological dilation.
This will enlarge the masked regions to cover more of the bone area.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def expand_mask(mask_path, output_path, kernel_size=5, iterations=2):
    """
    Expand mask using morphological dilation.
    
    Args:
        mask_path: Path to original mask
        output_path: Path to save expanded mask
        kernel_size: Size of dilation kernel (larger = more expansion)
        iterations: Number of dilation iterations (more = larger expansion)
    """
    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    
    # Create dilation kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply dilation to expand the mask
    expanded_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    # Save expanded mask
    cv2.imwrite(output_path, expanded_mask)


def process_directory(input_dir, output_dir, kernel_size=5, iterations=2):
    """
    Process all masks in directory and create expanded versions.
    
    Args:
        input_dir: Directory containing original masks
        output_dir: Directory to save expanded masks
        kernel_size: Size of dilation kernel (default: 5)
        iterations: Number of dilation iterations (default: 2)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of mask files
    input_path = Path(input_dir)
    mask_files = sorted([f for f in input_path.glob('*_mask.png')])
    
    if not mask_files:
        print(f"No mask files (*_mask.png) found in {input_dir}")
        return
    
    print(f"Original masks: {len(mask_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Expansion settings: kernel_size={kernel_size}, iterations={iterations}")
    print()
    
    success_count = 0
    failed_count = 0
    
    for idx, mask_file in enumerate(mask_files, 1):
        try:
            # Output path
            output_path = os.path.join(output_dir, mask_file.name)
            
            # Expand mask
            expand_mask(str(mask_file), output_path, kernel_size, iterations)
            
            print(f"{idx}/{len(mask_files)}: {mask_file.name} ✓")
            success_count += 1
            
        except Exception as e:
            print(f"{idx}/{len(mask_files)}: {mask_file.name} ✗ Error: {str(e)}")
            failed_count += 1
    
    print()
    print("=" * 50)
    print(f"✓ Success: {success_count}/{len(mask_files)}")
    print(f"✗ Failed: {failed_count}/{len(mask_files)}")
    print(f"✓ Expanded masks saved to: {output_dir}/")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Expand masks to cover more area using morphological dilation'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing mask images')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for expanded masks')
    parser.add_argument('-k', '--kernel-size', type=int, default=5,
                        help='Size of dilation kernel (default: 5, larger = more expansion)')
    parser.add_argument('-n', '--iterations', type=int, default=2,
                        help='Number of dilation iterations (default: 2, more = larger expansion)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return
    
    # Process all masks
    process_directory(args.input, args.output, args.kernel_size, args.iterations)


if __name__ == '__main__':
    main()
