"""
Apply segmentation masks to original images by blacking out the masked region.
Creates new images where the segmented joint space is replaced with black pixels.
"""
import argparse
import os
import cv2
import numpy as np

def apply_mask_blackout(original_img_path, mask_path, output_path):
    """
    Load original image and mask, black out the masked region, save result.
    
    Args:
        original_img_path: Path to original image
        mask_path: Path to segmentation mask
        output_path: Where to save the result
    
    Returns:
        True if successful, False otherwise
    """
    # Load original image
    original = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        return False, "Could not read original image"
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, "Could not read mask"
    
    # Ensure mask and image have same size
    if original.shape != mask.shape:
        # Resize mask to match original if needed
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # Create output image (copy of original)
    result = original.copy()
    
    # Black out masked regions (where mask > 0)
    result[mask > 0] = 0
    
    # Save result
    cv2.imwrite(output_path, result)
    
    return True, "Success"

def get_args():
    parser = argparse.ArgumentParser(
        description='Apply segmentation masks to original images by blacking out masked regions')
    parser.add_argument('--original-dir', '-i', required=True,
                        help='Directory with original images')
    parser.add_argument('--mask-dir', '-m', required=True,
                        help='Directory with segmentation masks')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Output directory for processed images')
    parser.add_argument('--mask-suffix', '-s', default='_mask',
                        help='Suffix added to mask filenames (default: _mask)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of original images
    if not os.path.exists(args.original_dir):
        print(f"Error: Original directory not found: {args.original_dir}")
        exit(1)
    
    if not os.path.exists(args.mask_dir):
        print(f"Error: Mask directory not found: {args.mask_dir}")
        exit(1)
    
    original_files = [f for f in os.listdir(args.original_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not original_files:
        print(f"Error: No images found in {args.original_dir}")
        exit(1)
    
    print(f"Original images: {len(original_files)}")
    print(f"Mask directory: {args.mask_dir}")
    print(f"Output directory: {args.output_dir}\n")
    
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
        
        # Output filename (same as original)
        output_path = os.path.join(args.output_dir, orig_filename)
        
        print(f"{i}/{len(original_files)}: {orig_filename} ", end='', flush=True)
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"✗ Mask not found: {mask_filename}")
            failed_count += 1
            continue
        
        # Apply mask
        success, message = apply_mask_blackout(original_path, mask_path, output_path)
        
        if success:
            print("✓")
            success_count += 1
        else:
            print(f"✗ {message}")
            failed_count += 1
    
    print(f"\n{'='*50}")
    print(f"✓ Success: {success_count}/{len(original_files)}")
    print(f"✗ Failed: {failed_count}/{len(original_files)}")
    print(f"✓ Output saved to: {args.output_dir}/")
    print(f"{'='*50}")
