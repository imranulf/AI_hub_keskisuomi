import cv2
import numpy as np

# Load and display image info
img = cv2.imread(r"C:\Users\imran\AI_hub_keskisuomi\WP3_knee_osteoarthritis\data\test\data\0\9003175L.png", cv2.IMREAD_GRAYSCALE)

print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")
print(f"Min/Max values: {img.min()} / {img.max()}")
print(f"Mean: {img.mean():.2f}")

# Check if it looks like a cropped knee (most edges should be dark/background)
edge_pixels = np.concatenate([
    img[0, :],  # top row
    img[-1, :],  # bottom row
    img[:, 0],  # left column
    img[:, -1]  # right column
])
edge_mean = edge_pixels.mean()
center_region = img[img.shape[0]//4:3*img.shape[0]//4, img.shape[1]//4:3*img.shape[1]//4]
center_mean = center_region.mean()

print(f"\nEdge pixels mean: {edge_mean:.2f}")
print(f"Center region mean: {center_mean:.2f}")
print(f"Ratio (center/edge): {center_mean/edge_mean if edge_mean > 0 else 'inf':.2f}")

if center_mean / edge_mean > 1.5 if edge_mean > 0 else False:
    print("\n✓ Looks like a PRE-CROPPED knee image (bright center, dark edges)")
else:
    print("\n✗ Looks like a full X-ray needing cropping")
