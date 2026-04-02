"""
classify_train_swin.py — Train a Swin-Tiny binary classifier (KL0 vs KL2)

Identical protocol to classify_train.py (ResNet-18) but using Swin Transformer Tiny
architecture from torchvision (pretrained on ImageNet).

Swin-Tiny key details:
    - Patch size: 4x4, Window size: 7x7
    - Embed dim: 96, Depths: [2, 2, 6, 2], Heads: [3, 6, 12, 24]
    - Input: 224x224 expected (matches our data)
    - First layer: features[0][0] is the patch embedding Conv2d(3, 96, kernel_size=4, stride=4)
    - Classifier: head = Linear(768, num_classes)

Usage:
    python classify_train_swin.py --data-dir knee_osteoarthritis_dataset --name baseline --epochs 25
    python classify_train_swin.py --data-dir classification_datasets/blackout --name blackout --epochs 25

Output:
    classification_results_swin/{name}/
        best_model.pth          — best model weights (by val accuracy)
        training_log.csv        — epoch-level metrics
"""

import argparse
import os
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image


class GrayscaleImageFolder(ImageFolder):
    """ImageFolder that loads grayscale PNGs and converts to 1-channel tensors."""

    def __init__(self, root: str, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("L")  # ensure grayscale
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Build a Swin-Tiny classifier adapted for single-channel grayscale input.

    Strategy: Load ImageNet-pretrained weights, modify patch embedding layer to accept
    1 channel by averaging the 3-channel weights (preserves learned features),
    replace final head layer for binary classification.
    """
    if pretrained:
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    else:
        model = models.swin_t(weights=None)

    # Adapt patch embedding: 3-channel -> 1-channel
    # Swin-T patch embedding: features[0][0] is Conv2d(3, 96, kernel_size=4, stride=4)
    original_proj = model.features[0][0]
    new_proj = nn.Conv2d(
        1, original_proj.out_channels,
        kernel_size=original_proj.kernel_size,
        stride=original_proj.stride,
        padding=original_proj.padding,
        bias=(original_proj.bias is not None),
    )
    with torch.no_grad():
        new_proj.weight = nn.Parameter(
            original_proj.weight.mean(dim=1, keepdim=True)
        )
        if original_proj.bias is not None:
            new_proj.bias = nn.Parameter(original_proj.bias.clone())
    model.features[0][0] = new_proj

    # Replace classification head for binary classification
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model


def get_transforms(split: str) -> transforms.Compose:
    """Get transforms for train/val/test splits."""
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate, return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Swin-Tiny for KL0 vs KL2 classification")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root dataset directory containing train/val/test subdirs")
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name (used for output directory)")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs (default: 25)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (default: 7)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from scratch (no ImageNet weights)")
    args = parser.parse_args()

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: Swin-Tiny")

    # Setup output directory
    out_dir = Path("classification_results_swin") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    data_root = Path(args.data_dir)
    train_dataset = GrayscaleImageFolder(
        str(data_root / "train"), transform=get_transforms("train")
    )
    val_dataset = GrayscaleImageFolder(
        str(data_root / "val"), transform=get_transforms("val")
    )

    # Class mapping info
    print(f"Classes: {train_dataset.classes}")
    print(f"Class to idx: {train_dataset.class_to_idx}")
    print(f"Train: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")

    # Handle class imbalance with weighted loss
    class_counts = [0, 0]
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    class_weights = torch.tensor([total / (2 * c) for c in class_counts], dtype=torch.float32)
    class_weights = class_weights.to(device)
    print(f"Class counts: {dict(zip(train_dataset.classes, class_counts))}")
    print(f"Class weights: {class_weights.tolist()}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Build model
    model = build_model(num_classes=2, pretrained=not args.no_pretrained)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    log_path = out_dir / "training_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_sec"])

    print(f"\nTraining '{args.name}' for {args.epochs} epochs...")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        scheduler.step(val_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                             f"{val_loss:.6f}", f"{val_acc:.6f}",
                             f"{current_lr:.2e}", f"{elapsed:.1f}"])

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_to_idx": train_dataset.class_to_idx,
                "args": vars(args),
                "architecture": "swin_tiny",
            }, out_dir / "best_model.pth")
            patience_counter = 0
            print(f"  -> New best val accuracy: {val_acc:.4f} (saved)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {out_dir / 'best_model.pth'}")
    print(f"Log saved to:   {log_path}")


if __name__ == "__main__":
    main()
