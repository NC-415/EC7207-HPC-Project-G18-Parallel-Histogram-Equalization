# Run: python3 input_image_processing.py

import numpy as np
import cv2
from pathlib import Path

# Navigate to project root and locate data/output directories
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
input_img = project_root / "data" / "input_image.png"
output_img = project_root / "output" / "input.pgm"

# read input image as grayscale
img = cv2.imread(str(input_img), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Input image not found or unreadable: {input_img}")

# resize image
resized = cv2.resize(img, (8192, 8192), interpolation=cv2.INTER_LINEAR)

# save as PGM
ok = cv2.imwrite(str(output_img), resized)
if not ok:
    raise RuntimeError(f"Failed to write: {output_img}")

print("Read:", input_img)
print("Saved:", output_img)
print("Output size:", resized.shape)