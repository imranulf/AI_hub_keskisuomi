import sys
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from unet import UNet
from predict_png import predict_png

# Test with a single image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet(n_channels=1, n_classes=1, bilinear=True)
model.to(device=device)
model.load_state_dict(torch.load('MODEL_unet.pth', map_location=device))
print("Model loaded successfully")

test_image = r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0\9003175L.png"
print(f"\nTesting with: {test_image}")

try:
    result = predict_png(model, test_image, device, 
                        name="test_image.png",
                        pixel_spacing=[0.143, 0.143],
                        scale_factor=1.0,
                        out_threshold=0.5,
                        save=True)
    print("\n=== SUCCESS ===")
    print(result)
except Exception as e:
    print(f"\n=== ERROR ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    import traceback
    traceback.print_exc()
