import cv2 as cv
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from skimage.feature import canny
from skimage.filters import sobel_v, sobel_h

# Load test image
test_image = r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0\9003175L.png"
img_orig = cv.imread(test_image, cv.IMREAD_GRAYSCALE)

print(f"Original image shape: {img_orig.shape}")
print(f"Image dtype: {img_orig.dtype}")
print(f"Image range: {img_orig.min()} - {img_orig.max()}")

img = img_orig.copy()
scaled_width = 250
scale_factor = img.shape[1] / scaled_width

print(f"\nscaled_width: {scaled_width}")
print(f"img.shape[1]: {img.shape[1]}")
print(f"scale_factor: {scale_factor}")
print(f"1/scale_factor: {1/scale_factor}")

img = gaussian_filter(img, 2)
print(f"\nAfter gaussian_filter: {img.shape}")

try:
    img_rescaled = rescale(img, 1/scale_factor)
    print(f"After rescale: {img_rescaled.shape}")
except Exception as e:
    print(f"\nERROR during rescale: {e}")
    import traceback
    traceback.print_exc()
