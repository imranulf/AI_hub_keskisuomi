from PIL import Image, ImageDraw, ImageFont
import os

def generate_figure(image_id="9003175L"):
    print(f"Starting figure generation for {image_id} using PIL...")
    # Define paths
    paths = {
        "Original": f"./data/test/data/0/{image_id}.png",
        "Segmentation Mask": f"results_test_0/{image_id}_mask.png",
        "Expanded Mask": f"out_test_expanded/{image_id}_mask.png",
        "Full Blackout": f"blackedout_test_0/{image_id}.png",
        "Left Masked": f"left_masked_test_0/{image_id}.png",
        "Right Masked": f"right_masked_test_0/{image_id}.png"
    }

    images = []
    titles = []
    
    for title, path in paths.items():
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            images.append(img)
            titles.append(title)
            print(f"  [OK] Loaded {title}")
        else:
            print(f"  [FAIL] File not found: {title}")

    if not images:
        print("No images loaded. Exiting.")
        return

    # Create a grid
    cols = 3
    rows = (len(images) + cols - 1) // cols
    
    # Assume all images are same size
    w, h = images[0].size
    padding = 40
    title_height = 30
    
    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * (h + title_height) + (rows + 1) * padding
    
    new_img = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(new_img)
    
    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, (img, title) in enumerate(zip(images, titles)):
        r = i // cols
        c = i % cols
        
        x = c * w + (c + 1) * padding
        y = r * (h + title_height) + (r + 1) * padding
        
        new_img.paste(img, (x, y + title_height))
        draw.text((x, y), title, fill=(0, 0, 0), font=font)

    output_path = "pipeline_visualization.png"
    new_img.save(output_path)
    print(f"Figure saved successfully to {output_path}")

if __name__ == "__main__":
    generate_figure()

