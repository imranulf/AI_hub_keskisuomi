import argparse
import os
from operator import itemgetter

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import cv2 as cv
import pandas as pd
from PIL import Image
from skimage.transform import rescale

from unet import UNet
from drn.drn import DRNSeg
from utils.data_loading import BasicDataset
from oa_vars import calculate_vars
from knee_localizer import find_knee_area

def lerp(coords: list):
    """Simple linear interpolation for patching possible holes in the predicted
       knee edges
    Args:
        coords (list): List of coordinates in (y,x)-format
    Returns:
        list: list of coordinates with possible holes linearly interpolated.
    """
    coords.sort(key = lambda x: x[1])
    temp_coords = []
    for i in range(len(coords) - 1):
        temp_coords.append(coords[i])
        x_diff = coords[i + 1][1] - coords[i][1]
        y_next = coords[i + 1][0]
        y_diff = (y_next - coords[i][0]) / x_diff
        if x_diff > 1:
            for j in range(1,x_diff):
                temp_coords.append((int(coords[i][0] + j * y_diff),
                                   coords[i][1] + j))
    return temp_coords
    
def init_xray_image(img_array):
    """Initialize x-ray image from numpy array (PNG/JPG input)"""
    img = img_array
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img - img.min()
    img = img / img.max() * 255
    
    # Adaptive margins based on image size (avoid empty slices for small images)
    margin = min(100, int(img.shape[1] * 0.1))  # 10% margin or 100px, whichever is smaller
    mid = int(img.shape[1] / 2)
    
    # Ensure slices are valid (start < end)
    left_start = margin
    left_end = max(mid - margin, left_start + 1)
    right_start = mid + margin
    right_end = max(img.shape[1] - margin, right_start + 1)
    
    img_r = img[:, left_start:left_end]
    img_l = np.fliplr(img[:, right_start:right_end])
    return img_r, img_l
    
def predict_img(net,
                full_img,
                device,
                scale_factor = 1.,
                out_threshold = 0.5):
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
    
def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'
    return args.output or list(map(_generate_name, args.input))
    
def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis = 0) * 255
                                / mask.shape[0]).astype(np.uint8))
    return None
    
def predict_png(net: nn.Module, img_file_path: str, device: torch.device,
                name, pixel_spacing: list, scale_factor = 1., 
                out_threshold = .5, save = False) -> pd.DataFrame:
    """Finds the joint-space segments from a given PA knee x-ray image
       in PNG/JPG format using the given segmentation network. Uses the
       found segments to find the edges of the tibia and femur to
       calculate different variables related to knee osteoarthritis,
       and optionally saves the found segmentation mask as an image.
    Args:
        net (nn.Module): A segmentation network
        img_file_path (str): path to the xray-image to segment
        device (torch.device): Device to run the predictions on: cuda or cpu
        name (str): Name of the image.
        pixel_spacing (list): Pixel spacing in mm/pixel [row, col]
        scale_factor (float, optional): Downscaling factor of the image. 
        Defaults to 1.
        out_threshold (float, optional): The value above which a given pixel is
        considered to be part of the predicted segmentation mask.  Defaults to .5.
        save (bool, optional): Whether to save the segmentations as an image.
        Defaults to False.
    Returns:
        pd.DataFrame: A dataframe containing the calculated OA-variables
    """
    # Read image (PNG/JPG)
    img_array = cv.imread(img_file_path, cv.IMREAD_GRAYSCALE)
    if img_array is None:
        raise ValueError(f"Could not read image: {img_file_path}")
    
    img_r, img_l = init_xray_image(img_array)
    
    # Collect records in a list to avoid deprecated DataFrame.append
    records = []
    
    for i, img in enumerate((img_l, img_r)):
        top, bottom, left, right = find_knee_area(img)
        mask = predict_img(net, Image.fromarray(img[top : bottom, left : right]),
                           device, scale_factor, out_threshold)
        mask = rescale(mask, 1 / scale_factor)
        big_mask = np.zeros(img.shape)
        big_mask[top : bottom, left : right] = mask
        mask = big_mask

        coords = np.where(mask > 0)
        xs = sorted(np.unique(coords[1]))
        coords = list(zip(coords[0], coords[1]))
        tibia_coords = [max([coord for coord in coords if coord[1] == x]) 
                        for x in xs]
        femur_coords = [min([coord for coord in coords if coord[1] == x]) 
                        for x in xs]
        tibia_coords = lerp(tibia_coords)
        femur_coords = lerp(femur_coords)
        
        vars = calculate_vars(tibia_coords, femur_coords, pixel_spacing)
        vars["name"] = name
        vars["side"] = "r" if i else "l"
        records.append(vars)
        
        if save:
            # Ensure output directory exists
            os.makedirs("out", exist_ok=True)
            cv.imwrite(os.path.join("out", f"{name}_{vars['side']}.png"), mask * 255)
    
    # Return as DataFrame
    return pd.DataFrame.from_records(records)

def get_args():
    parser = argparse.ArgumentParser(
        description = 'Predict segmentations on PNG/JPG images')
    parser.add_argument('--architecture', '-a', choices = ['unet', 'drn'],
                        default = 'unet', 
                        help = 'Architecture of the loaded model, \
                        choices: unet, drn')
    parser.add_argument('--model', '-m', default = 'MODEL.pth', metavar = 'FILE',
                        help = 'Specify the file in which the model is stored')
    parser.add_argument('--save', '-sv', action = 'store_true',
                        help = 'save the images as they are processed')
    parser.add_argument('--input-dir', '-i', metavar = 'I', dest = 'input_dir',
                        type = str, default = 'training_data', 
                        help = 'Directory with PNG/JPG images to predict or the \
                        path to a single image file')
    parser.add_argument('--n-classes', '-nc', dest = 'n_classes', 
                        metavar = 'NC', type = int, default = 1,
                        help = 'Number of classes in the used model')
    parser.add_argument('--n-channels', '-nch', dest = 'n_channels',
                        metavar = 'NCH', type = int, default = 1, 
                        help = 'Number of channels in input images')
    parser.add_argument('--scale', '-s', type = float, default = 1., 
                         help = 'Downscaling factor of the images')
    parser.add_argument('--mask-threshold', '-t', type = float, default = 0.5, 
                        help = 'Minimum probability value to consider a mask \
                        pixel white')
    parser.add_argument('--pixel-spacing', '-ps', type = float, nargs=2,
                        default = [0.143, 0.143],
                        help = 'Pixel spacing in mm/pixel (row col). Default: 0.143 0.143')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.architecture == 'unet':
        model = UNet(n_channels = args.n_channels, n_classes = args.n_classes,
                     bilinear = True)
    else:
        model = DRNSeg(model_name = "drn_d_105", n_channels = args.n_channels,
                       n_classes = args.n_classes)
    model.to(device = device)
    model.load_state_dict(torch.load(args.model, map_location = device))
    print(f"Model loaded from {args.model}")
    
    if os.path.isdir(args.input_dir):
        tree = next(os.walk(args.input_dir))
        results = pd.DataFrame()
        
        # Filter for image files only
        image_files = [f for f in tree[2] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for i, img_file in enumerate(image_files):
            print(f"{i+1}/{len(image_files)}: {img_file}")
            try:
                variables = predict_png(model, os.path.join(tree[0], img_file),
                                       device, img_file, 
                                       pixel_spacing = args.pixel_spacing,
                                       scale_factor = args.scale, 
                                       save = args.save)
                results = pd.concat([results, variables], ignore_index=True)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                import traceback
                if i < 3:  # Show detailed traceback for first 3 errors
                    traceback.print_exc()
        
        results.to_csv("oa_variables.csv", index=False)
        print(f"\nProcessed {len(image_files)} images. Results saved to oa_variables.csv")
        
    elif os.path.isfile(args.input_dir):
        file = args.input_dir
        print(f"Processing single file: {file}")
        try:
            result = predict_png(model, file, device, 
                               name = os.path.basename(file),
                               pixel_spacing = args.pixel_spacing,
                               scale_factor = args.scale,
                               save = args.save)
            result.to_csv("oa_variables.csv", index=False)
            print(f"\nResults saved to oa_variables.csv")
            print(result)
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: {args.input_dir} is not a valid file or directory")
