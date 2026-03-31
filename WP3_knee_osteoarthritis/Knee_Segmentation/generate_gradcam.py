"""
generate_gradcam.py — Generate Grad-CAM heatmaps for trained ResNet-18 classifiers

Produces Grad-CAM visualizations showing which regions the model focuses on for
KL0 vs KL2 classification. Supports all 4 experimental conditions (baseline,
blackout, medial_masked, lateral_masked) for RQ4 comparison.

Based on: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" — confirmed compatible with ResNet-18 (Figure 1,
panels f, l of original paper [16]).

Usage:
    # Generate Grad-CAM for baseline model on original test images
    python generate_gradcam.py --model classification_results/baseline/best_model.pth \
        --test-dir knee_osteoarthritis_dataset/test --name baseline --num-samples 10

    # Generate Grad-CAM for all 4 conditions (comparison figures for RQ4)
    python generate_gradcam.py --all-conditions --num-samples 10

    # Generate for specific images by filename
    python generate_gradcam.py --model classification_results/baseline/best_model.pth \
        --test-dir knee_osteoarthritis_dataset/test --name baseline \
        --image-ids 9003175L 9003175R

Output:
    gradcam_results/{name}/
        {image_id}_gradcam.png        — heatmap overlay on original image
        {image_id}_heatmap.png        — raw heatmap only
        {image_id}_sidebyside.png     — original | heatmap | overlay (3-panel)
        summary.json                  — metadata and confidence scores per image
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


# ---------------------------------------------------------------------------
# Model building (must match classify_train.py exactly)
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2) -> nn.Module:
    """Build ResNet-18 with single-channel input (weights loaded separately)."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Grad-CAM implementation for ResNet-18
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM for ResNet-18.

    Hooks into the last convolutional layer (layer4[-1]) to capture:
    1. Forward activations (feature maps)
    2. Backward gradients

    Then computes the weighted combination to produce the class activation map.

    Reference: Selvaraju et al. (2017), Section 3.1
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Store forward pass activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Store backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: preprocessed image tensor [1, 1, H, W]
            target_class: class index to visualize (None = predicted class)

        Returns:
            heatmap: numpy array [H, W] in range [0, 1]
            prediction: predicted class index
            confidence: softmax probability for predicted class
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        prediction = output.argmax(dim=1).item()
        confidence = probs[0, prediction].item()

        # Backward pass for target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # Grad-CAM computation
        # Global average pooling of gradients → channel weights (alpha_k)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of forward activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU — only keep positive contributions
        cam = F.relu(cam)

        # Resize to input image dimensions
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, prediction, confidence

    def remove_hooks(self):
        """Clean up registered hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def load_raw_image(image_path: str) -> np.ndarray:
    """Load original grayscale image as numpy array [H, W] in range [0, 255]."""
    img = Image.open(image_path).convert("L")
    return np.array(img)


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess image for model input (must match classify_train.py val transforms)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = Image.open(image_path).convert("L")
    tensor = transform(img).unsqueeze(0)  # [1, 1, H, W]
    return tensor


def create_heatmap_overlay(raw_image: np.ndarray, heatmap: np.ndarray,
                            alpha: float = 0.4) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on grayscale image.

    Args:
        raw_image: grayscale image [H, W], uint8
        heatmap: normalized heatmap [H, W], float [0, 1]
        alpha: overlay transparency (0 = only image, 1 = only heatmap)

    Returns:
        overlay: BGR image [H, W, 3], uint8
    """
    # Convert heatmap to colormap (jet)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert grayscale to BGR for overlay
    image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)

    # Blend
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def create_sidebyside(raw_image: np.ndarray, heatmap: np.ndarray,
                       overlay: np.ndarray, prediction: int, confidence: float,
                       true_label: int) -> np.ndarray:
    """
    Create a 3-panel side-by-side figure:
    [Original] [Heatmap] [Overlay]
    With prediction info below.
    """
    h, w = raw_image.shape

    # Convert all to BGR
    img_bgr = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Create canvas (3 panels + text bar)
    gap = 4
    text_h = 40
    canvas_w = w * 3 + gap * 2
    canvas_h = h + text_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place panels
    canvas[:h, :w] = img_bgr
    canvas[:h, w + gap:2 * w + gap] = heatmap_bgr
    canvas[:h, 2 * w + 2 * gap:3 * w + 2 * gap] = overlay

    # Labels above each panel
    labels = ["Original", "Grad-CAM", "Overlay"]
    for i, label in enumerate(labels):
        x = i * (w + gap) + 5
        cv2.putText(canvas, label, (x, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Prediction info at bottom
    class_names = {0: "KL0 (Healthy)", 1: "KL2 (OA)"}
    pred_text = f"Pred: {class_names.get(prediction, prediction)} ({confidence:.3f})"
    true_text = f"True: {class_names.get(true_label, true_label)}"
    correct = "CORRECT" if prediction == true_label else "INCORRECT"
    color = (0, 255, 0) if prediction == true_label else (0, 0, 255)

    info_text = f"{pred_text}  |  {true_text}  |  {correct}"
    cv2.putText(canvas, info_text, (5, h + 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_single_image(grad_cam: GradCAM, image_path: str, true_label: int,
                          output_dir: Path, device: torch.device) -> dict:
    """Generate Grad-CAM for a single image and save visualizations."""
    image_id = Path(image_path).stem

    # Load and preprocess
    raw_image = load_raw_image(image_path)
    input_tensor = preprocess_image(image_path).to(device)

    # Generate Grad-CAM
    heatmap, prediction, confidence = grad_cam.generate(input_tensor)

    # Create visualizations
    overlay = create_heatmap_overlay(raw_image, heatmap)
    sidebyside = create_sidebyside(raw_image, heatmap, overlay, prediction,
                                     confidence, true_label)

    # Save outputs
    heatmap_uint8 = np.uint8(255 * heatmap)
    cv2.imwrite(str(output_dir / f"{image_id}_heatmap.png"), heatmap_uint8)
    cv2.imwrite(str(output_dir / f"{image_id}_gradcam.png"), overlay)
    cv2.imwrite(str(output_dir / f"{image_id}_sidebyside.png"), sidebyside)

    result = {
        "image_id": image_id,
        "image_path": str(image_path),
        "true_label": true_label,
        "predicted_label": prediction,
        "confidence": round(confidence, 4),
        "correct": prediction == true_label,
    }

    return result


def create_comparison_grid(results_by_condition: dict, output_dir: Path,
                            image_id: str):
    """
    Create a multi-condition comparison figure for a single image.
    Shows Grad-CAM overlay from each condition side-by-side for RQ4 analysis.

    Layout: 1 row × N conditions
    """
    panels = []
    labels = []

    for condition, data in results_by_condition.items():
        overlay_path = data["output_dir"] / f"{image_id}_gradcam.png"
        if overlay_path.exists():
            panel = cv2.imread(str(overlay_path))
            if panel is not None:
                panels.append(panel)
                labels.append(f"{condition}\n({data['confidence']:.3f})")

    if len(panels) == 0:
        return

    h, w = panels[0].shape[:2]
    gap = 4
    text_h = 50
    canvas_w = w * len(panels) + gap * (len(panels) - 1)
    canvas_h = h + text_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, (panel, label) in enumerate(zip(panels, labels)):
        x_start = i * (w + gap)
        canvas[:h, x_start:x_start + w] = panel
        # Label
        lines = label.split("\n")
        for j, line in enumerate(lines):
            cv2.putText(canvas, line, (x_start + 5, h + 18 + j * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(output_dir / f"{image_id}_comparison.png"), canvas)


def run_gradcam(model_path: str, test_dir: str, name: str,
                num_samples: int = 10, image_ids: list = None,
                device: torch.device = torch.device("cpu")) -> dict:
    """
    Run Grad-CAM for a single model/condition.

    Returns:
        dict with results metadata
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    trained_epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_acc", "?")
    print(f"Loaded model: {name} (epoch {trained_epoch}, val_acc {val_acc})")

    # Setup Grad-CAM on layer4 (last residual block)
    # ResNet-18 layer4 = final conv block, standard target for Grad-CAM
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # Collect test images
    test_path = Path(test_dir)
    image_list = []
    for grade_folder in sorted(test_path.iterdir()):
        if grade_folder.is_dir() and grade_folder.name in ("0", "2"):
            label = 0 if grade_folder.name == "0" else 1
            for img_file in sorted(grade_folder.glob("*.png")):
                if image_ids is None or img_file.stem in image_ids:
                    image_list.append((str(img_file), label))

    # Subsample if needed
    if image_ids is None and num_samples > 0 and len(image_list) > num_samples:
        # Take balanced samples: half from each class
        class_0 = [(p, l) for p, l in image_list if l == 0]
        class_1 = [(p, l) for p, l in image_list if l == 1]
        n_per_class = num_samples // 2

        # Deterministic sampling (every Nth image)
        step_0 = max(1, len(class_0) // n_per_class)
        step_1 = max(1, len(class_1) // n_per_class)
        sampled = class_0[::step_0][:n_per_class] + class_1[::step_1][:n_per_class]
        image_list = sampled

    print(f"Processing {len(image_list)} images for condition '{name}'...")

    # Output directory
    output_dir = Path("gradcam_results") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    results = []
    for i, (img_path, true_label) in enumerate(image_list):
        result = process_single_image(grad_cam, img_path, true_label, output_dir, device)
        results.append(result)

        status = "✓" if result["correct"] else "✗"
        print(f"  [{i+1}/{len(image_list)}] {result['image_id']}: "
              f"pred={result['predicted_label']} conf={result['confidence']:.3f} {status}")

    # Save summary
    summary = {
        "condition": name,
        "model_path": model_path,
        "test_dir": test_dir,
        "num_images": len(results),
        "accuracy": sum(1 for r in results if r["correct"]) / len(results) if results else 0,
        "mean_confidence": sum(r["confidence"] for r in results) / len(results) if results else 0,
        "results": results,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved {len(results)} Grad-CAM visualizations to: {output_dir}")

    # Cleanup hooks
    grad_cam.remove_hooks()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for RQ4 analysis"
    )
    parser.add_argument("--model", type=str,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--test-dir", type=str,
                        help="Test directory with 0/ and 2/ subdirs")
    parser.add_argument("--name", type=str,
                        help="Condition name for output directory")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of sample images to process (default: 10, 0=all)")
    parser.add_argument("--image-ids", nargs="+", type=str, default=None,
                        help="Specific image IDs to process (e.g., 9003175L 9003175R)")
    parser.add_argument("--all-conditions", action="store_true",
                        help="Run Grad-CAM for all 4 experimental conditions")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all_conditions:
        # Run all 4 conditions with the same sample images
        conditions = {
            "baseline": {
                "model": "classification_results/baseline/best_model.pth",
                "test_dir": "knee_osteoarthritis_dataset/test",
            },
            "blackout": {
                "model": "classification_results/blackout/best_model.pth",
                "test_dir": "classification_datasets/blackout/test",
            },
            "medial_masked": {
                "model": "classification_results/medial_masked/best_model.pth",
                "test_dir": "classification_datasets/medial_masked/test",
            },
            "lateral_masked": {
                "model": "classification_results/lateral_masked/best_model.pth",
                "test_dir": "classification_datasets/lateral_masked/test",
            },
        }

        # Also run cross-evaluation Grad-CAMs: baseline model on ablated data
        cross_conditions = {
            "baseline_on_blackout": {
                "model": "classification_results/baseline/best_model.pth",
                "test_dir": "classification_datasets/blackout/test",
            },
            "baseline_on_medial": {
                "model": "classification_results/baseline/best_model.pth",
                "test_dir": "classification_datasets/medial_masked/test",
            },
            "baseline_on_lateral": {
                "model": "classification_results/baseline/best_model.pth",
                "test_dir": "classification_datasets/lateral_masked/test",
            },
        }

        all_conditions = {**conditions, **cross_conditions}

        print(f"\n{'='*60}")
        print(f"  Generating Grad-CAM for {len(all_conditions)} conditions")
        print(f"{'='*60}\n")

        all_summaries = {}
        for cond_name, cond_config in all_conditions.items():
            model_path = cond_config["model"]
            test_dir = cond_config["test_dir"]

            if not os.path.exists(model_path):
                print(f"\n  SKIP {cond_name}: model not found at {model_path}")
                continue
            if not os.path.exists(test_dir):
                print(f"\n  SKIP {cond_name}: test dir not found at {test_dir}")
                continue

            print(f"\n{'='*60}")
            print(f"  Condition: {cond_name}")
            print(f"{'='*60}")

            summary = run_gradcam(
                model_path=model_path,
                test_dir=test_dir,
                name=cond_name,
                num_samples=args.num_samples,
                image_ids=args.image_ids,
                device=device,
            )
            all_summaries[cond_name] = summary

        # Save combined summary
        combined_dir = Path("gradcam_results")
        combined_dir.mkdir(exist_ok=True)
        with open(combined_dir / "all_conditions_summary.json", "w") as f:
            json.dump({
                name: {k: v for k, v in s.items() if k != "results"}
                for name, s in all_summaries.items()
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  ALL GRAD-CAM GENERATION COMPLETE")
        print(f"  Results in: gradcam_results/")
        print(f"{'='*60}")

    else:
        # Single condition
        if not args.model or not args.test_dir or not args.name:
            parser.error("--model, --test-dir, and --name are required (or use --all-conditions)")

        run_gradcam(
            model_path=args.model,
            test_dir=args.test_dir,
            name=args.name,
            num_samples=args.num_samples,
            image_ids=args.image_ids,
            device=device,
        )


if __name__ == "__main__":
    main()
