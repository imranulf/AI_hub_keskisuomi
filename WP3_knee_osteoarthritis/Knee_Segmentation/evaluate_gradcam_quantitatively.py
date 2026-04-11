import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

# Import building functions and GradCAM classes from the existing pipeline scripts
from generate_gradcam import build_model as build_resnet, GradCAM as ResNetGradCAM
from generate_gradcam_efficientnet import build_model as build_effnet, GradCAM as EffNetGradCAM
from generate_gradcam_swin import build_model as build_swin, GradCAMSwin

def evaluate_activations(model_name, build_fn, cam_class, target_layer_str, weight_path):
    print(f"\nEvaluating {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model and load the baseline weights
    model = build_fn(num_classes=2)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()

    # Hook the registered feature mapping layer dynamically
    if target_layer_str == 'layer4[-1]':
        target_layer = model.layer4[-1]
    elif target_layer_str == 'features[-1]':
        target_layer = model.features[-1]
    elif target_layer_str == 'norm':
        target_layer = model.norm
        
    grad_cam = cam_class(model, target_layer)
    
    # Transformation standard for inference
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Locate tracking datasets
    test_dir = Path("knee_osteoarthritis_dataset/test")
    # For masks, using the primary expanded horizontal versions
    mask_dirs = {
        "0": Path("results_test_0_extralarge_horiz"),
        "2": Path("results_test_2_extralarge_horiz")
    }
    
    inside_means = []
    outside_means = []
    
    for grade in ["0", "2"]:
        grade_dir = test_dir / grade
        mask_dir = mask_dirs[grade]
        
        for img_path in grade_dir.glob("*.png"):
            # 1. Load original test image
            img_arr = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_arr is None: continue
            
            # 2. Preprocess
            input_tensor = transform(img_arr).unsqueeze(0).to(device)
            
            # 3. Generate CAM values dynamically
            heatmap, pred, conf = grad_cam.generate(input_tensor)
            
            # 4. Load the generated boundary masks
            mask_name = img_path.stem + "_mask.png"
            mask_path = mask_dir / mask_name
            if not mask_path.exists(): continue
            
            mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask_arr = cv2.resize(mask_arr, (224, 224))
            
            # 5. Extract strict segmentation limits (boundary pixel inclusion)
            inside_mask = (mask_arr > 127)
            outside_mask = (mask_arr <= 127)
            
            # 6. Calculate independent scalar targets per model
            if np.any(inside_mask):
                inside_means.append(heatmap[inside_mask].mean())
            if np.any(outside_mask):
                outside_means.append(heatmap[outside_mask].mean())
                
    mean_in = np.mean(inside_means) if inside_means else 0
    mean_out = np.mean(outside_means) if outside_means else 0
    ratio = mean_in / mean_out if mean_out > 0 else 0
    
    print(f"  Mean Activation Inside Joint Mask:  {mean_in:.4f}")
    print(f"  Mean Activation Outside Joint Mask: {mean_out:.4f}")
    print(f"  Ratio (Inside/Outside):             {ratio:.2f}x")
    return mean_in, mean_out, ratio

if __name__ == "__main__":
    print("=== Grad-CAM Quantitative Evaluation (Inside vs Outside Joint Space) ===")
    
    configs = [
        ("ResNet-18", build_resnet, ResNetGradCAM, "layer4[-1]", "classification_results/baseline/best_model.pth"),
        ("EfficientNet-B0", build_effnet, EffNetGradCAM, "features[-1]", "classification_results_efficientnet/baseline/best_model.pth"),
        ("Swin-Tiny", build_swin, GradCAMSwin, "norm", "classification_results_swin/baseline/best_model.pth"),
    ]
    
    for name, b_fn, cam_cls, layer_str, w_path in configs:
        if os.path.exists(w_path):
            evaluate_activations(name, b_fn, cam_cls, layer_str, w_path)
        else:
            print(f"Weights for {name} not found at {w_path}")
