import os
import glob
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# CONFIGURATION
# ==========================================
models = ['ResNet-18', 'EfficientNet-B0', 'Swin-Tiny']
base_dirs = [
    'gradcam_results',
    'gradcam_results_efficientnet',
    'gradcam_results_swin'
]
conditions = [
    ('baseline', 'Baseline (Original)'),
    ('baseline_on_lateral', 'Lateral Masked'),
    ('baseline_on_medial', 'Medial Masked'),
    ('baseline_on_blackout', 'Blackout (No Joint Space)')
]

output_dir = 'gradcam_comparative'
os.makedirs(output_dir, exist_ok=True)

# Helper function to find all samples (e.g., patient IDs) based on baseline directory
def get_sample_ids():
    baseline_dir = os.path.join(base_dirs[0], 'baseline')
    files = glob.glob(os.path.join(baseline_dir, '*_sidebyside.png'))
    return [os.path.basename(f).split('_sidebyside')[0] for f in files]

sample_ids = get_sample_ids()
if not sample_ids:
    print("No samples found in gradcam_results/baseline. Exiting.")
    exit()

print(f"Found {len(sample_ids)} unique samples: {sample_ids}")

# ==========================================
# GENERATE COMPARATIVE GRAD-CAM (Pure PIL for Full Resolution)
# ==========================================

print(f"Generating Comparative Grad-CAM Collages for all {len(sample_ids)} image types...")

def create_text_image(text, width, height=50, font_size=24, bg_color='white', text_color='black'):
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    # Using default font and scaling simply (fallback for missing TTF fonts)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    draw.text(((width - text_w) / 2, (height - text_h) / 2), text, font=font, fill=text_color)
    return img

for sample_id in sample_ids:
    print(f"Processing sample: {sample_id}...")
    
    # Track condition blocks for the global matrix
    condition_blocks = []
    
    # 1. Condition-Specific Isolated Collages (Models stacked vertically 'Top Down')
    for cond_folder, cond_name in conditions:
        model_imgs = []
        for m, bdir in zip(models, base_dirs):
            gdir = os.path.join(bdir, cond_folder)
            img_path = os.path.join(gdir, f"{sample_id}_sidebyside.png")
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                
                # Add model label on top or left? Let's add it on top of each side-by-side
                label_img = create_text_image(f"{m} - {cond_name}", img.width, height=40, font_size=20)
                
                # Stack label and image
                combined = Image.new('RGB', (img.width, img.height + label_img.height))
                combined.paste(label_img, (0, 0))
                combined.paste(img, (0, label_img.height))
                
                model_imgs.append(combined)

        if not model_imgs:
            continue
            
        # Stack the 3 models top-down for this condition
        cond_width = max(img.width for img in model_imgs) + 20  # 20px padding
        cond_height = sum(img.height for img in model_imgs) + (len(model_imgs) * 20) + 60
        
        cond_canvas = Image.new('RGB', (cond_width, cond_height), color='white')
        
        # Super title for condition
        title_img = create_text_image(f"Condition: {cond_name}", cond_width, height=60, font_size=28)
        cond_canvas.paste(title_img, (0, 0))
        
        y_offset = 60
        for img in model_imgs:
            cond_canvas.paste(img, (10, y_offset))
            y_offset += img.height + 20
            
        # Save isolated top-down condition collage
        out_path = os.path.join(output_dir, f'gradcam_{cond_folder}_{sample_id}.png')
        cond_canvas.save(out_path)
        
        condition_blocks.append(cond_canvas)
        
    # 2. Global Matrix: Stitch the top-down condition blocks horizontally
    if condition_blocks:
        global_width = sum(block.width for block in condition_blocks)
        global_height = max(block.height for block in condition_blocks) + 80
        
        global_canvas = Image.new('RGB', (global_width, global_height), color='white')
        
        global_title = create_text_image(
            f"GLOBAL GRAD-CAM MATRIX across Architectures & Conditions | Sample: {sample_id}", 
            global_width, height=80, font_size=36
        )
        global_canvas.paste(global_title, (0, 0))
        
        x_offset = 0
        for block in condition_blocks:
            global_canvas.paste(block, (x_offset, 80))
            x_offset += block.width
            
        global_out_path = os.path.join(output_dir, f'gradcam_global_matrix_{sample_id}.png')
        global_canvas.save(global_out_path)

print(f"Successfully moved all Grad-CAM generation to '{output_dir}/'!")
