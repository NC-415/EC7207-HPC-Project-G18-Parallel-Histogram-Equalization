# Run: python3 scripts/preprocessing/input_image_processing.py

import numpy as np
import cv2
from pathlib import Path

# Navigate to project root and locate data/output directories
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
input_img = project_root / "data" / "input_image.png"

pgm_dir = project_root / "results" / "pgm"
png_dir = project_root / "results" / "png"

# Create both folders before writing anything
pgm_dir.mkdir(parents=True, exist_ok=True)
png_dir.mkdir(parents=True, exist_ok=True)

output_pgm = pgm_dir / "input.pgm"
png_original = png_dir / "00_original_input.png"
png_preprocessed = png_dir / "01_preprocessed_input.png"

# read input image as grayscale
img = cv2.imread(str(input_img), cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Input image not found or unreadable: {input_img}")

# resize image
resized = cv2.resize(img, (8192, 8192), interpolation=cv2.INTER_LINEAR)

# save as PGM
if not cv2.imwrite(str(output_pgm), resized):
    raise RuntimeError(f"Failed to write: {output_pgm}")

# Also save as PNG for visualization
if not cv2.imwrite(str(png_original), img):
    raise RuntimeError(f"Failed to write: {png_original}")

if not cv2.imwrite(str(png_preprocessed), resized):
    raise RuntimeError(f"Failed to write: {png_preprocessed}")

print("Read:", input_img)
print("Saved PGM:", output_pgm)
print("Saved PNG (original):", png_original)
print("Saved PNG (preprocessed):", png_preprocessed)
print("Output size:", resized.shape)
print("PGM dir:", pgm_dir)
print("PNG dir:", png_dir)