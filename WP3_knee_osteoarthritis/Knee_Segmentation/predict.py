import argparse
import os
from operator import itemgetter
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2 as cv
import pydicom as dicom
import pandas as pd
from PIL import Image
from skimage.transform import rescale
from tqdm.auto import tqdm

from unet import UNet
from drn.drn import DRNSeg
from utils.data_loading import BasicDataset
from oa_vars import calculate_vars
from knee_localizer import find_knee_area


SUPPORTED_EXTS = {'.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def lerp(coords: list):
    """Fill gaps along x with linear interpolation of y."""
    if not coords:
        return coords
    coords.sort(key=lambda x: x[1])
    out = []
    for i in range(len(coords) - 1):
        y0, x0 = coords[i]
        y1, x1 = coords[i + 1]
        out.append((y0, x0))
        dx = x1 - x0
        if dx > 1:
            dy = (y1 - y0) / float(dx)
            for j in range(1, dx):
                out.append((int(y0 + j * dy), x0 + j))
    out.append(coords[-1])
    return out


def init_xray_image(img: np.ndarray):
    """Normalize to [0,255] and split into right/left halves (left is flipped)."""
    img = img.astype(np.float32)
    img -= img.min()
    vmax = img.max()
    if vmax > 0:
        img = img / vmax * 255.0

    mid = img.shape[1] // 2
    img_r = img[:, 100:int(img.shape[1] * 0.5) - 100]
    img_l = np.fliplr(img[:, mid + 100: img.shape[1] - 100])
    return img_r, img_l


def load_xray_image(file_path: str):
    """Load a radiograph (DICOM or standard image). Returns (image, [row_mm, col_mm])."""
    ext = Path(file_path).suffix.lower()
    if ext == ".dcm":
        dcm = dicom.dcmread(file_path)
        pixel_spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
        try:
            pixel_spacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
        except Exception:
            warnings.warn("Unexpected PixelSpacing; defaulting to 1.0 mm", RuntimeWarning)
            pixel_spacing = [1.0, 1.0]
        image = dcm.pixel_array
    else:
        with Image.open(file_path) as pil_img:
            if pil_img.mode not in ("L", "I;16", "I"):
                pil_img = pil_img.convert("L")
            image = np.array(pil_img)
        pixel_spacing = [1.0, 1.0]
        warnings.warn("Non-DICOM image -> assuming 1.0 mm/pixel.", RuntimeWarning)

    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image, pixel_spacing


def predict_img(net,
                full_img: Image.Image,
                device: torch.device,
                scale_factor: float = 1.0,
                out_threshold: float = 0.5):
    """Run model on a PIL image crop and return a binary/one-hot mask (np.ndarray)."""
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(full_img.convert("L"),
                                scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def save_mask_png(mask: np.ndarray, out_path: Path):
    """Save a binary mask (1=white, 0=black) as PNG with black background."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0).astype(np.uint8)
    black = (1 - binary) * 255  # black background, white foreground (0 vs 255 inverted)
    cv.imwrite(str(out_path), black)


def predict_one_image(net: nn.Module,
                      fpath: Path,
                      device: torch.device,
                      scale_factor: float,
                      out_threshold: float,
                      save_masks_to: dict) -> pd.DataFrame:
    """
    Returns a dataframe with OA variables for left/right sides (when present).
    save_masks_to: dict with keys 'left' and 'right' -> Path to save masks
    """
    name = fpath.stem
    img, pix = load_xray_image(str(fpath))
    img_r, img_l = init_xray_image(img)
    res = pd.DataFrame()

    for side_idx, (half_img, side_key) in enumerate(((img_l, "left"), (img_r, "right"))):
        # localize knee area and run model on the crop
        top, bottom, left, right = find_knee_area(half_img)
        crop = half_img[top:bottom, left:right]
        mask = predict_img(net, Image.fromarray(crop), device, scale_factor, out_threshold)

        # scale mask back using nearest (order=0) to preserve labels
        mask = rescale(
            mask.astype(np.float32),
            1 / scale_factor,
            order=0,
            anti_aliasing=False,
            preserve_range=True
        )

        # paste mask back into full canvas
        big_mask = np.zeros(half_img.shape, dtype=np.float32)
        h = min(big_mask.shape[0], top + mask.shape[0]) - top
        w = min(big_mask.shape[1], left + mask.shape[1]) - left
        if h > 0 and w > 0:
            big_mask[top:top + h, left:left + w] = mask[:h, :w]
        mask = big_mask

        # save per-side mask
        save_mask_png(mask, save_masks_to[side_key] / f"{name}_{side_key}_mask.png")

        # compute OA variables if segmentation exists
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            warnings.warn(f"No segmentation: {side_key} knee in {name}")
            continue

        unique_xs = np.unique(xs)
        coords_list = list(zip(ys, xs))
        tibia_coords = [max([c for c in coords_list if c[1] == x], key=itemgetter(0)) for x in unique_xs]
        femur_coords = [min([c for c in coords_list if c[1] == x], key=itemgetter(0)) for x in unique_xs]
        tibia_coords = lerp(tibia_coords)
        femur_coords = lerp(femur_coords)

        vars_dict = calculate_vars(tibia_coords, femur_coords, pix)
        vars_dict["name"] = name
        vars_dict["side"] = "r" if side_idx else "l"
        res = pd.concat([res, pd.DataFrame([vars_dict])], ignore_index=True)

    return res


def detect_split_and_class(path: Path):
    """
    Try to detect dataset split ('train'/'val'/'test') and class ('0'/'2') from a path.
    Returns (split or 'unknown', class_name or 'unknown').
    """
    parts = [p.lower() for p in path.parts]
    split = next((s for s in ("train", "val", "test") if s in parts), "unknown")
    # class folder is usually the last component ('0' or '2')
    cls = path.name if path.name in {"0", "2"} else "unknown"
    return split, cls


def iter_image_files(root: Path):
    """Yield supported image files under a directory (recursively)."""
    if root.is_file() and root.suffix.lower() in SUPPORTED_EXTS:
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def get_args():
    parser = argparse.ArgumentParser(description="Batch knee segmentation with per-image masks and progress bar")
    # model
    parser.add_argument(
        "--architecture", "-a", choices=["unet", "drn"], default="unet",
        help="Model architecture"
    )
    parser.add_argument(
        "--model", "-m",
        default=r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\Knee_Segmentation\MODEL_unet.pth",
        help="Path to model weights (.pth)"
    )
    parser.add_argument("--n-classes", "-nc", type=int, default=1, help="Number of classes in the model")
    parser.add_argument("--n-channels", "-nch", type=int, default=1, help="Input channels")
    # inference
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Downscaling factor before inference")
    parser.add_argument("--mask-threshold", "-t", type=float, default=0.5, help="Threshold for 1-class models")
    # IO
    parser.add_argument(
        "--input-dirs", "-i", nargs="+", type=str, default=[
            r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0",
            r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\2",
            r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\0",
            r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\train\data\2",
            r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\0",
            r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\val\data\2",
        ],
        help="One or more input directories (DICOM or images)"
    )
    parser.add_argument(
        "--out-root", "-o", type=str, default="outputs",
        help="Root directory where masks and CSVs will be written"
    )
    parser.add_argument(
        "--csv", type=str, default="oa_variables_all.csv",
        help="Combined CSV output filename (written under out-root)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.architecture == "unet":
        model = UNet(n_channels=args.n_channels, n_classes=args.n_classes, bilinear=True)
    else:
        model = DRNSeg(model_name="drn_d_105", n_channels=args.n_channels, n_classes=args.n_classes)
    model.to(device=device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_split_rows = {"train": [], "val": [], "test": [], "unknown": []}

    # Gather all files from all roots
    roots = [Path(p) for p in args.input_dirs]
    files = []
    for root in roots:
        files.extend(list(iter_image_files(root)))

    if not files:
        raise FileNotFoundError("No supported image files found in provided input directories.")

    pbar = tqdm(files, desc="Processing images", unit="img")
    for fpath in pbar:
        # detect split/class to build mirrored output dirs
        split, cls = detect_split_and_class(fpath.parent)
        split_dir = out_root / "masks" / split / cls
        left_dir = split_dir / "left"
        right_dir = split_dir / "right"

        save_dirs = {"left": left_dir, "right": right_dir}

        try:
            df = predict_one_image(
                net=model,
                fpath=fpath,
                device=device,
                scale_factor=args.scale,
                out_threshold=args.mask_threshold,
                save_masks_to=save_dirs
            )
            if not df.empty:
                all_rows.append(df)
                per_split_rows[split].append(df)
                # progress bar postfix
                pbar.set_postfix_str(f"OK: {fpath.name}")
            else:
                pbar.set_postfix_str(f"No seg: {fpath.name}")
        except Exception as e:
            warnings.warn(f"Failed on {fpath}: {e}", RuntimeWarning)
            pbar.set_postfix_str(f"ERR: {fpath.name}")

    # Write combined CSV
    if all_rows:
        all_df = pd.concat(all_rows, ignore_index=True)
        (out_root / "csv").mkdir(parents=True, exist_ok=True)
        all_csv_path = out_root / "csv" / args.csv
        try:
            all_df.to_csv(all_csv_path, index=False)
            print(f"[CSV] Combined OA variables -> {all_csv_path}")
        except Exception as e:
            warnings.warn(f"Could not write combined CSV: {e}", RuntimeWarning)
        # Write per-split CSVs
        for split, rows in per_split_rows.items():
            if rows:
                df = pd.concat(rows, ignore_index=True)
                sp_csv = out_root / "csv" / f"oa_variables_{split}.csv"
                try:
                    df.to_csv(sp_csv, index=False)
                    print(f"[CSV] {split} -> {sp_csv}")
                except Exception as e:
                    warnings.warn(f"Could not write {split} CSV: {e}", RuntimeWarning)
    else:
        print("No OA variables produced.")
