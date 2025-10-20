import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import cv2 as cv
import pandas as pd
from PIL import Image

from unet import UNet
from drn.drn import DRNSeg
from utils.data_loading import BasicDataset
from oa_vars import calculate_vars

def lerp(coords: list):
    """Simple linear interpolation for patching possible holes in the predicted
       knee edges
    """
    coords.sort(key = lambda x: x[1])
    temp_coords = []
    for i in range(len(coords) - 1):
        temp_coords.append(coords[i])
        x_diff = coords[i + 1][1] - coords[i][1]
        y_next = coords[i + 1][0]
        y_diff = (y_next - coords[i][0]) / x_diff if x_diff > 0 else 0
        if x_diff > 1:
            for j in range(1,x_diff):
                temp_coords.append((int(coords[i][0] + j * y_diff),
                                   coords[i][1] + j))
    return temp_coords
    
def predict_img(net, full_img, device, scale_factor = 1., out_threshold = 0.5):
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

def predict_precropped_knee(net: nn.Module, img_file_path: str, device: torch.device,
                            name, pixel_spacing: list, scale_factor = 1., 
                            out_threshold = .5, save = False) -> dict:
    """Segment a pre-cropped knee image (single knee, already localized).
    """
    # Read image
    img = cv.imread(img_file_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_file_path}")
    
    # Predict segmentation mask directly (no knee localization needed)
    mask = predict_img(net, Image.fromarray(img), device, scale_factor, out_threshold)
    
    # Extract coordinates from mask
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        # No segmentation found
        print(f"  Warning: No segmentation mask detected for {name}")
        return {
            "name": name,
            "l_em_height": np.nan,
            "m_em_height": np.nan,
            "l_em_angle": np.nan,
            "m_em_angle": np.nan,
            "l_em_x": np.nan,
            "m_em_x": np.nan,
            "l_min_jsw": np.nan,
            "m_min_jsw": np.nan,
            "l_avg_jsw": np.nan,
            "m_avg_jsw": np.nan,
            "tibia_model": [np.nan, np.nan]
        }
    
    xs = sorted(np.unique(coords[1]))
    coords_list = list(zip(coords[0], coords[1]))
    
    # Find tibia (bottom) and femur (top) edges
    tibia_coords = [max([coord for coord in coords_list if coord[1] == x]) for x in xs]
    femur_coords = [min([coord for coord in coords_list if coord[1] == x]) for x in xs]
    
    tibia_coords = lerp(tibia_coords)
    femur_coords = lerp(femur_coords)
    
    # Calculate OA variables
    try:
        vars_dict = calculate_vars(tibia_coords, femur_coords, pixel_spacing)
        vars_dict["name"] = name
    except Exception as e:
        print(f"  Warning: Could not calculate OA variables for {name}: {e}")
        vars_dict = {
            "name": name,
            "l_em_height": np.nan,
            "m_em_height": np.nan,
            "l_em_angle": np.nan,
            "m_em_angle": np.nan,
            "l_em_x": np.nan,
            "m_em_x": np.nan,
            "l_min_jsw": np.nan,
            "m_min_jsw": np.nan,
            "l_avg_jsw": np.nan,
            "m_avg_jsw": np.nan,
            "tibia_model": [np.nan, np.nan]
        }
    
    # Save mask if requested
    if save:
        os.makedirs("out", exist_ok=True)
        cv.imwrite(os.path.join("out", f"{name}.png"), (mask * 255).astype(np.uint8))
    
    return vars_dict

def get_args():
    parser = argparse.ArgumentParser(
        description = 'Predict segmentations on pre-cropped knee PNG/JPG images')
    parser.add_argument('--architecture', '-a', choices = ['unet', 'drn'],
                        default = 'unet', 
                        help = 'Architecture: unet or drn')
    parser.add_argument('--model', '-m', default = 'MODEL.pth', metavar = 'FILE',
                        help = 'Model file path')
    parser.add_argument('--save', '-sv', action = 'store_true',
                        help = 'Save segmentation masks')
    parser.add_argument('--input-dir', '-i', metavar = 'I', dest = 'input_dir',
                        type = str, required = True,
                        help = 'Directory with images or single image file path')
    parser.add_argument('--n-classes', '-nc', dest = 'n_classes', 
                        metavar = 'NC', type = int, default = 1,
                        help = 'Number of classes')
    parser.add_argument('--n-channels', '-nch', dest = 'n_channels',
                        metavar = 'NCH', type = int, default = 1, 
                        help = 'Number of input channels')
    parser.add_argument('--scale', '-s', type = float, default = 1., 
                         help = 'Downscaling factor')
    parser.add_argument('--mask-threshold', '-t', type = float, default = 0.5, 
                        help = 'Mask threshold (0-1)')
    parser.add_argument('--pixel-spacing', '-ps', type = float, nargs=2,
                        default = [0.143, 0.143],
                        help = 'Pixel spacing in mm/pixel (row col)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if args.architecture == 'unet':
        model = UNet(n_channels = args.n_channels, n_classes = args.n_classes,
                     bilinear = True)
    else:
        model = DRNSeg(model_name = "drn_d_105", n_channels = args.n_channels,
                       n_classes = args.n_classes)
    model.to(device = device)
    model.load_state_dict(torch.load(args.model, map_location = device))
    print(f"Model loaded from {args.model}\n")
    
    # Process images
    if os.path.isdir(args.input_dir):
        tree = next(os.walk(args.input_dir))
        results = []
        
        # Filter for image files only
        image_files = [f for f in tree[2] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, img_file in enumerate(image_files):
            print(f"{i+1}/{len(image_files)}: {img_file}")
            try:
                result_dict = predict_precropped_knee(
                    model, os.path.join(tree[0], img_file),
                    device, img_file, 
                    pixel_spacing = args.pixel_spacing,
                    scale_factor = args.scale,
                    out_threshold = args.mask_threshold,
                    save = args.save)
                results.append(result_dict)
            except Exception as e:
                print(f"  Error: {e}")
                if i < 3:  # Show traceback for first 3 errors
                    import traceback
                    traceback.print_exc()
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv("oa_variables.csv", index=False)
        print(f"\n✓ Processed {len(image_files)} images")
        print(f"✓ Results saved to oa_variables.csv")
        print(f"✓ {len(df)} rows in output")
        
    elif os.path.isfile(args.input_dir):
        file = args.input_dir
        print(f"Processing: {os.path.basename(file)}")
        try:
            result_dict = predict_precropped_knee(
                model, file, device, 
                name = os.path.basename(file),
                pixel_spacing = args.pixel_spacing,
                scale_factor = args.scale,
                out_threshold = args.mask_threshold,
                save = args.save)
            df = pd.DataFrame([result_dict])
            df.to_csv("oa_variables.csv", index=False)
            print(f"\n✓ Results saved to oa_variables.csv")
            print(df)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: {args.input_dir} is not a valid file or directory")
