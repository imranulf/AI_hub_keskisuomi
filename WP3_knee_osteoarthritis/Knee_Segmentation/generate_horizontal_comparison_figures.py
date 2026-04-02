"""
Generate comparative visualization figures for HORIZONTAL expansion showing:
1. Original image
2. Segmentation Mask (original)
3. Horizontal Expanded Mask
4. Horizontal Full Blackout
5. Horizontal Left Masked
6. Horizontal Right Masked
"""

from PIL import Image, ImageDraw, ImageFont
import os
import argparse


def generate_horizontal_comparison_figure(image_id, dataset, grade, output_name=None):
    """
    Generate a 2x3 comparison figure for horizontal masks.

    Args:
        image_id: Image filename without extension (e.g., "9003175L")
        dataset: "train", "val", or "test"
        grade: "0" or "2"
        output_name: Optional custom output filename
    """
    print(f"Generating horizontal figure for {image_id} ({dataset}/{grade})...")

    # Define paths based on dataset and grade
    base_data = f"./data/{dataset}/data/{grade}"

    paths = {
        "Original": f"{base_data}/{image_id}.png",
        "Segmentation Mask": f"results_{dataset}_{grade}/{image_id}_mask.png",
        "Horizontal Mask": f"results_{dataset}_{grade}_horizontal/{image_id}_mask.png",
        "Horizontal Full": f"blackedout_{dataset}_{grade}_horizontal_full/{image_id}.png",
        "Horizontal Left": f"blackedout_{dataset}_{grade}_horizontal_left/{image_id}.png",
        "Horizontal Right": f"blackedout_{dataset}_{grade}_horizontal_right/{image_id}.png"
    }

    images = []
    titles = []

    for title, path in paths.items():
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            images.append(img)
            titles.append(title)
            print(f"  [OK] {title}")
        else:
            print(f"  [FAIL] {title}: {path}")

    if len(images) < 6:
        print(f"  Warning: Only {len(images)}/6 images found")
        if len(images) == 0:
            print("  No images loaded. Skipping.")
            return None

    # Create a 2x3 grid
    cols = 3
    rows = 2

    # Get image size from first image
    w, h = images[0].size
    padding = 40
    title_height = 30

    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * (h + title_height) + (rows + 1) * padding

    # Create white background
    new_img = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(new_img)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except:
            font = ImageFont.load_default()

    # Place images in grid
    for i, (img, title) in enumerate(zip(images, titles)):
        r = i // cols
        c = i % cols

        x = c * w + (c + 1) * padding
        y = r * (h + title_height) + (r + 1) * padding

        # Resize if needed to match first image size
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)

        new_img.paste(img, (x, y + title_height))
        draw.text((x, y), title, fill=(0, 0, 0), font=font)

    # Save figure
    if output_name is None:
        output_name = f"horizontal_comparison_{image_id}_{dataset}_{grade}.png"

    os.makedirs("comparison_figures_horizontal", exist_ok=True)
    output_path = os.path.join("comparison_figures_horizontal", output_name)
    new_img.save(output_path)
    print(f"  [OK] Saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison figures for horizontal knee segmentation pipeline'
    )
    parser.add_argument('--samples', '-s', type=int, default=3,
                        help='Number of samples per category (default: 3)')
    args = parser.parse_args()

    # Sample images from different datasets and grades
    samples = {
        ("test", "0"): ["9003175L", "9220406L", "9410231L"],
        ("test", "2"): ["9003316R", "9007827L", "9236300R"],
        ("val", "0"): ["9006140L", "9009067L", "9015402L"],
        ("val", "2"): ["9015402R", "9023348R", "9031141R"],
    }

    generated = []

    for (dataset, grade), image_ids in samples.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}, Grade: {grade}")
        print('='*50)

        for image_id in image_ids[:args.samples]:
            result = generate_horizontal_comparison_figure(image_id, dataset, grade)
            if result:
                generated.append(result)

    print(f"\n{'='*50}")
    print(f"[OK] Generated {len(generated)} horizontal comparison figures")
    print(f"[OK] Saved to: comparison_figures_horizontal/")
    print('='*50)


if __name__ == "__main__":
    main()
