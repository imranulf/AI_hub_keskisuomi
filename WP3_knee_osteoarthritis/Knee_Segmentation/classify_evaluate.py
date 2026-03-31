"""
classify_evaluate.py — Evaluate a trained ResNet-18 classifier on any test set

Produces:
    - Accuracy, Precision, Recall, F1-score
    - Confusion matrix
    - Per-image confidence scores (softmax probabilities)
    - Confidence distribution analysis (for RQ3)

Usage:
    python classify_evaluate.py --model classification_results/baseline/best_model.pth \
        --test-dir knee_osteoarthritis_dataset/test --name baseline_on_original

    # Cross-evaluation (RQ3): baseline model on ablated test set
    python classify_evaluate.py --model classification_results/baseline/best_model.pth \
        --test-dir blackout_dataset/test --name baseline_on_blackout
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image


class GrayscaleImageFolder(ImageFolder):
    """ImageFolder that loads grayscale PNGs and returns (image, label, filepath)."""

    def __init__(self, root: str, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, target, path


def build_model(num_classes: int = 2) -> nn.Module:
    """Build ResNet-18 with single-channel input (no pretrained — weights loaded separately)."""
    model = models.resnet18(weights=None)
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_test_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Run inference and collect per-image predictions, confidences, and labels.

    Returns:
        all_labels: ground truth labels
        all_preds: predicted labels
        all_probs: softmax probabilities [N, 2]
        all_paths: file paths
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    for images, labels, paths in loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(paths)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_paths,
    )


def compute_metrics(labels, preds, class_names):
    """Compute accuracy, precision, recall, F1, and confusion matrix."""
    n = len(labels)
    correct = (labels == preds).sum()
    accuracy = correct / n

    metrics = {"accuracy": accuracy, "total": n, "correct": int(correct)}

    # Per-class metrics
    for i, name in enumerate(class_names):
        tp = ((preds == i) & (labels == i)).sum()
        fp = ((preds == i) & (labels != i)).sum()
        fn = ((preds != i) & (labels == i)).sum()
        tn = ((preds != i) & (labels != i)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"{name}_precision"] = precision
        metrics[f"{name}_recall"] = recall
        metrics[f"{name}_f1"] = f1
        metrics[f"{name}_tp"] = int(tp)
        metrics[f"{name}_fp"] = int(fp)
        metrics[f"{name}_fn"] = int(fn)
        metrics[f"{name}_tn"] = int(tn)

    # Macro averages
    precisions = [metrics[f"{n}_precision"] for n in class_names]
    recalls = [metrics[f"{n}_recall"] for n in class_names]
    f1s = [metrics[f"{n}_f1"] for n in class_names]
    metrics["macro_precision"] = np.mean(precisions)
    metrics["macro_recall"] = np.mean(recalls)
    metrics["macro_f1"] = np.mean(f1s)

    # Confusion matrix [actual x predicted]
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def confidence_analysis(labels, preds, probs, class_names):
    """
    Analyze prediction confidence (RQ3).

    Returns dict with:
        - mean confidence for correct/incorrect predictions
        - mean confidence per class
        - confidence when correct but low-confidence (uncertain correct)
    """
    predicted_confidence = np.max(probs, axis=1)  # max softmax prob per sample
    correct_mask = labels == preds

    analysis = {
        "mean_confidence_all": float(np.mean(predicted_confidence)),
        "mean_confidence_correct": float(np.mean(predicted_confidence[correct_mask])) if correct_mask.any() else None,
        "mean_confidence_incorrect": float(np.mean(predicted_confidence[~correct_mask])) if (~correct_mask).any() else None,
        "std_confidence_all": float(np.std(predicted_confidence)),
    }

    # Per-class confidence
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.any():
            analysis[f"{name}_mean_confidence"] = float(np.mean(predicted_confidence[mask]))
            analysis[f"{name}_correct_confidence"] = float(
                np.mean(predicted_confidence[mask & correct_mask])
            ) if (mask & correct_mask).any() else None
            analysis[f"{name}_incorrect_confidence"] = float(
                np.mean(predicted_confidence[mask & ~correct_mask])
            ) if (mask & ~correct_mask).any() else None

    # Uncertain correct: predicted correctly but confidence < 0.7
    uncertain_correct = correct_mask & (predicted_confidence < 0.7)
    analysis["uncertain_correct_count"] = int(uncertain_correct.sum())
    analysis["uncertain_correct_pct"] = float(uncertain_correct.sum() / len(labels) * 100)

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Test directory with 0/ and 2/ subdirs")
    parser.add_argument("--name", type=str, required=True,
                        help="Evaluation name (used for output files)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-per-image", action="store_true",
                        help="Save per-image predictions CSV")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    out_dir = Path("classification_results") / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    model = build_model(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded model from: {args.model}")
    print(f"  Trained epoch: {checkpoint.get('epoch', '?')}, Val acc: {checkpoint.get('val_acc', '?'):.4f}")

    # Load test data
    test_dataset = GrayscaleImageFolder(args.test_dir, transform=get_test_transform())
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    class_names = test_dataset.classes
    print(f"Test set: {len(test_dataset)} images, classes: {class_names}")

    # Run evaluation
    labels, preds, probs, paths = evaluate(model, test_loader, device)

    # Compute metrics
    metrics = compute_metrics(labels, preds, class_names)
    confidence = confidence_analysis(labels, preds, probs, class_names)

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.name}")
    print(f"{'='*60}")
    print(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"            {class_names[0]:>8s}  {class_names[1]:>8s}")
    cm = metrics["confusion_matrix"]
    for i, name in enumerate(class_names):
        print(f"  {name:>8s}  {cm[i][0]:>8d}  {cm[i][1]:>8d}")
    print(f"\nConfidence Analysis:")
    print(f"  Mean confidence (all):       {confidence['mean_confidence_all']:.4f}")
    print(f"  Mean confidence (correct):   {confidence['mean_confidence_correct']:.4f}")
    if confidence['mean_confidence_incorrect'] is not None:
        print(f"  Mean confidence (incorrect): {confidence['mean_confidence_incorrect']:.4f}")
    print(f"  Uncertain correct (<0.7):    {confidence['uncertain_correct_count']} "
          f"({confidence['uncertain_correct_pct']:.1f}%)")

    # Save results
    result = {
        "name": args.name,
        "model_path": args.model,
        "test_dir": args.test_dir,
        "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics["confusion_matrix"],
        "confidence": confidence,
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    result_path = out_dir / f"{args.name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=convert)
    print(f"\nResults saved to: {result_path}")

    # Optional: save per-image predictions
    if args.save_per_image:
        per_image_path = out_dir / f"{args.name}_per_image.csv"
        with open(per_image_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "true_label", "pred_label", "prob_0", "prob_2",
                             "confidence", "correct"])
            for path, label, pred, prob in zip(paths, labels, preds, probs):
                writer.writerow([
                    os.path.basename(path),
                    class_names[label],
                    class_names[pred],
                    f"{prob[0]:.6f}",
                    f"{prob[1]:.6f}",
                    f"{max(prob):.6f}",
                    int(label == pred),
                ])
        print(f"Per-image predictions saved to: {per_image_path}")


if __name__ == "__main__":
    main()
