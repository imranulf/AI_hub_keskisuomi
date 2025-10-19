"""
Simple segmentation script for pre-cropped knee images.
Just segments and saves masks without calculating OA variables.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2 as cv
from PIL import Image

from unet import UNet
from drn.drn import DRNSeg
from utils.data_loading import BasicDataset

def predict_img(net, full_img, device, scale_factor = 1., out_threshold = 0.5):
    """Run segmentation prediction on a single image"""
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img.convert("L"),
                           scale_factor, is_mask = False))
    img = img.unsqueeze(0)
    img = img.to(device = device, dtype = torch.float32)
    with torch.no_grad():
        output = net(img)
        if net.n_classes > 1:
            probs = F.softmax(output, dim = 1)[0]
        else:
            probs = torch.sigmoid(output)[0]
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        full_mask = tf(probs.cpu()).squeeze()
    if net.n_classes == 1:
        res = (full_mask > out_threshold).numpy()
    else:
        res = F.one_hot(full_mask.argmax(dim = 0), net.n_classes).\
                        permute(2, 0, 1).numpy()
    return res

def get_args():
    parser = argparse.ArgumentParser(
        description = 'Simple segmentation for pre-cropped knee images')
    parser.add_argument('--architecture', '-a', choices = ['unet', 'drn'],
                        default = 'unet', help = 'Model architecture')
    parser.add_argument('--model', '-m', required=True, help = 'Model file path')
    parser.add_argument('--input-dir', '-i', required = True,
                        help = 'Directory with images or single image file')
    parser.add_argument('--output-dir', '-o', default = 'out',
                        help = 'Output directory for masks')
    parser.add_argument('--n-classes', '-nc', type = int, default = 1)
    parser.add_argument('--n-channels', '-nch', type = int, default = 1)
    parser.add_argument('--scale', '-s', type = float, default = 1.)
    parser.add_argument('--threshold', '-t', type = float, default = 0.5)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    if args.architecture == 'unet':
        model = UNet(n_channels = args.n_channels, n_classes = args.n_classes, bilinear = True)
    else:
        model = DRNSeg(model_name = "drn_d_105", n_channels = args.n_channels, n_classes = args.n_classes)
    model.to(device = device)
    model.load_state_dict(torch.load(args.model, map_location = device))
    print(f"Model: {args.model}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image files
    if os.path.isdir(args.input_dir):
        files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files = [os.path.join(args.input_dir, f) for f in files]
    elif os.path.isfile(args.input_dir):
        files = [args.input_dir]
    else:
        print(f"Error: {args.input_dir} not found")
        exit(1)
    
    print(f"Processing {len(files)} images...\n")
    
    # Process each image
    success = 0
    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        print(f"{i}/{len(files)}: {filename} ", end='', flush=True)
        
        try:
            # Load image
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                print("✗ Could not read")
                continue
            
            # Predict
            mask = predict_img(model, Image.fromarray(img), device, 
                             args.scale, args.threshold)
            
            # Save mask
            out_name = os.path.splitext(filename)[0] + '_mask.png'
            out_path = os.path.join(args.output_dir, out_name)
            cv.imwrite(out_path, (mask * 255).astype(np.uint8))
            
            # Count segmented pixels
            seg_pixels = np.sum(mask > 0)
            seg_percent = 100 * seg_pixels / (mask.shape[0] * mask.shape[1])
            print(f"✓ {seg_percent:.1f}% segmented")
            success += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\n✓ Completed: {success}/{len(files)} images")
    print(f"✓ Masks saved to: {args.output_dir}/")
