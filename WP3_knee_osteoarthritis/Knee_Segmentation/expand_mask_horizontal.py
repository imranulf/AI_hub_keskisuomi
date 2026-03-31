"""
Expand segmentation masks horizontally to the edges of the image.
Preserves the curved shape of the joint space but extends it to left and right edges.

The top and bottom boundaries of the mask are maintained, but extended horizontally.
Filters out noise and uses the main mask region for robust extension.
"""
import argparse
import os
import cv2
import numpy as np


def expand_mask_horizontal(mask_path, output_path, sample_percent=25, min_thickness_percent=70):
    """
    Expand mask horizontally to image edges while preserving the shape.

    - Filters out small isolated noise regions
    - Finds the main contiguous mask region
    - Extends boundaries smoothly to edges
    - Ensures minimum thickness throughout

    Args:
        mask_path: Path to input mask
        output_path: Where to save expanded mask
        sample_percent: Percentage of mask width to sample from each side (default: 25)
        min_thickness_percent: Minimum thickness as percent of max thickness (default: 50)

    Returns:
        (success, message)
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, "Could not read mask"

    height, width = mask.shape

    # Find connected components and filter out small noise (keep all significant components)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        # No foreground pixels
        cv2.imwrite(output_path, mask)
        return True, "No white pixels to expand"

    # Filter components using multiple criteria:
    # 1. Position: joint space is in central 60% of image (y: 20%-80%)
    # 2. Aspect ratio: joint space is horizontal (width > height)
    # 3. Size: relative to the largest valid component

    middle_zone_top = height // 5  # ~45 for 224px
    middle_zone_bottom = 4 * height // 5  # ~179 for 224px

    # Score each component based on how likely it is to be joint space
    component_scores = []

    for label_idx in range(1, num_labels):
        y_top = stats[label_idx, cv2.CC_STAT_TOP]
        comp_height = stats[label_idx, cv2.CC_STAT_HEIGHT]
        comp_width = stats[label_idx, cv2.CC_STAT_WIDTH]
        y_center = y_top + comp_height // 2
        area = stats[label_idx, cv2.CC_STAT_AREA]

        # Score based on position (0 if outside zone, 1 if in center)
        if y_center < middle_zone_top or y_center > middle_zone_bottom:
            position_score = 0
        else:
            # Higher score for being closer to image center
            dist_from_center = abs(y_center - height // 2)
            position_score = 1 - (dist_from_center / (height // 2))

        # Score based on aspect ratio (horizontal bands score higher)
        aspect_ratio = comp_width / max(1, comp_height)
        aspect_score = min(1.0, aspect_ratio / 5)  # Ratio of 5:1 or more gets full score

        # Combined score - position is more important than aspect ratio
        total_score = position_score * 0.6 + aspect_score * 0.4

        component_scores.append((label_idx, area, total_score, y_center))

    # Create clean mask
    clean_mask = np.zeros_like(mask)

    # Filter: keep components with score > 0.5 AND area > 30% of max valid area
    valid_components = [(idx, area, score, yc) for idx, area, score, yc in component_scores if score > 0.5]

    if valid_components:
        max_valid_area = max(area for _, area, _, _ in valid_components)
        min_area_threshold = max_valid_area * 0.30  # Reduced to 30% to keep more legitimate components

        for label_idx, area, score, yc in valid_components:
            if area >= min_area_threshold:
                clean_mask[labels == label_idx] = 255
    else:
        # Fallback: no valid components, use largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean_mask[labels == largest_label] = 255

    # Find columns that have white pixels in the clean mask
    cols_with_white = []
    top_boundary = {}
    bottom_boundary = {}

    for x in range(width):
        column = clean_mask[:, x]
        white_pixels = np.where(column > 0)[0]
        if len(white_pixels) > 0:
            cols_with_white.append(x)
            top_boundary[x] = white_pixels.min()
            bottom_boundary[x] = white_pixels.max()

    if not cols_with_white:
        cv2.imwrite(output_path, mask)
        return True, "No white pixels to expand"

    # Sort columns
    cols_with_white.sort()

    # Get leftmost and rightmost columns
    left_col = cols_with_white[0]
    right_col = cols_with_white[-1]

    # Calculate sample size (percentage of mask width)
    num_cols = len(cols_with_white)
    sample_size = max(20, int(num_cols * sample_percent / 100))

    # Left side: sample from interior (skip first few thin edge columns)
    skip_edge = min(5, num_cols // 10)  # Skip potentially thin edge columns
    left_start = skip_edge
    left_end = min(left_start + sample_size, num_cols)
    left_samples = cols_with_white[left_start:left_end]
    if not left_samples:
        left_samples = cols_with_white[:sample_size]

    left_top = min([top_boundary[c] for c in left_samples])
    left_bottom = max([bottom_boundary[c] for c in left_samples])

    # Right side: sample from interior (skip last few thin edge columns)
    right_end = num_cols - skip_edge
    right_start = max(0, right_end - sample_size)
    right_samples = cols_with_white[right_start:right_end]
    if not right_samples:
        right_samples = cols_with_white[-sample_size:]

    right_top = min([top_boundary[c] for c in right_samples])
    right_bottom = max([bottom_boundary[c] for c in right_samples])

    # Calculate thickness at each column and find the maximum
    thicknesses = {c: bottom_boundary[c] - top_boundary[c] + 1 for c in cols_with_white}
    max_thickness = max(thicknesses.values())
    # Minimum thickness: at least 12 pixels OR 70% of max, whichever is larger
    min_thickness = max(12, int(max_thickness * min_thickness_percent / 100))

    # Ensure extension boundaries are at least as thick as the mask edges
    # Get actual boundaries at the edge columns
    left_edge_top = top_boundary[left_col]
    left_edge_bottom = bottom_boundary[left_col]
    right_edge_top = top_boundary[right_col]
    right_edge_bottom = bottom_boundary[right_col]

    # Extension should use the more generous boundary (wider thickness)
    # For top: use the smaller value (higher up), for bottom: use larger value (lower down)
    left_ext_top = min(left_top, left_edge_top)
    left_ext_bottom = max(left_bottom, left_edge_bottom)
    right_ext_top = min(right_top, right_edge_top)
    right_ext_bottom = max(right_bottom, right_edge_bottom)

    # Create output mask
    expanded = np.zeros_like(mask)

    # For smooth extension, we interpolate from edge of image to actual mask edge
    # Left side: interpolate from (0, left_ext_boundaries) to (left_col, actual_edge_boundaries)
    # Right side: interpolate from (right_col, actual_edge_boundaries) to (width-1, right_ext_boundaries)

    # Fill each column from 0 to width
    for x in range(width):
        if x < left_col:
            # Left extension - interpolate from extension boundaries to actual mask edge
            if left_col > 0:
                t = x / left_col  # 0 at x=0, 1 at x=left_col
                top = int(left_ext_top + t * (left_edge_top - left_ext_top))
                bottom = int(left_ext_bottom + t * (left_edge_bottom - left_ext_bottom))
            else:
                top = left_ext_top
                bottom = left_ext_bottom
        elif x > right_col:
            # Right extension - interpolate from actual mask edge to extension boundaries
            remaining = width - 1 - right_col
            if remaining > 0:
                t = (x - right_col) / remaining  # 0 at x=right_col, 1 at x=width-1
                top = int(right_edge_top + t * (right_ext_top - right_edge_top))
                bottom = int(right_edge_bottom + t * (right_ext_bottom - right_edge_bottom))
            else:
                top = right_ext_top
                bottom = right_ext_bottom
        elif x in top_boundary:
            # Inside mask - use original boundaries
            top = top_boundary[x]
            bottom = bottom_boundary[x]
        else:
            # Gap in mask - interpolate
            left_neighbors = [c for c in cols_with_white if c < x]
            right_neighbors = [c for c in cols_with_white if c > x]

            if left_neighbors and right_neighbors:
                nl = max(left_neighbors)
                nr = min(right_neighbors)
                ratio = (x - nl) / max(1, nr - nl)
                top = int(top_boundary[nl] + ratio * (top_boundary[nr] - top_boundary[nl]))
                bottom = int(bottom_boundary[nl] + ratio * (bottom_boundary[nr] - bottom_boundary[nl]))
            elif left_neighbors:
                nl = max(left_neighbors)
                top = top_boundary[nl]
                bottom = bottom_boundary[nl]
            else:
                nr = min(right_neighbors)
                top = top_boundary[nr]
                bottom = bottom_boundary[nr]

        # Ensure minimum thickness
        current_thickness = bottom - top + 1
        if current_thickness < min_thickness:
            # Expand equally from center
            center = (top + bottom) // 2
            half_min = min_thickness // 2
            top = max(0, center - half_min)
            bottom = min(height - 1, center + half_min)

        # Fill column
        expanded[top:bottom+1, x] = 255

    # Save result
    cv2.imwrite(output_path, expanded)

    return True, "Success"


def get_args():
    parser = argparse.ArgumentParser(
        description='Expand masks horizontally to image edges while preserving shape')
    parser.add_argument('--input', '-i', required=True,
                        help='Input mask directory or single mask file')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for expanded masks')
    parser.add_argument('--sample-percent', '-p', type=int, default=25,
                        help='Percentage of mask width to sample from each side (default: 25)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single file
        filename = os.path.basename(args.input)
        output_path = os.path.join(args.output, filename)
        success, message = expand_mask_horizontal(args.input, output_path, args.sample_percent)
        if success:
            print(f"[OK] Expanded: {filename}")
        else:
            print(f"[FAIL] Failed: {filename} - {message}")

    elif os.path.isdir(args.input):
        # Directory of masks
        mask_files = [f for f in os.listdir(args.input)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not mask_files:
            print(f"Error: No mask files found in {args.input}")
            exit(1)

        print(f"Expanding {len(mask_files)} masks horizontally (preserving shape)...")
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Sample percent: {args.sample_percent}%\n")

        success_count = 0
        failed_count = 0

        for i, filename in enumerate(mask_files, 1):
            input_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, filename)

            print(f"{i}/{len(mask_files)}: {filename} ", end='', flush=True)

            success, message = expand_mask_horizontal(input_path, output_path, args.sample_percent)

            if success:
                print("[OK]")
                success_count += 1
            else:
                print(f"[FAIL] {message}")
                failed_count += 1

        print(f"\n{'='*50}")
        print(f"[OK] Success: {success_count}/{len(mask_files)}")
        print(f"[FAIL] Failed: {failed_count}/{len(mask_files)}")
        print(f"Output saved to: {args.output}/")
        print(f"{'='*50}")

    else:
        print(f"Error: {args.input} not found")
        exit(1)
