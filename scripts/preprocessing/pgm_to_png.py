# Converts all output PGM files to PNG for visual inspection.
# Run: python3 scripts/preprocessing/pgm_to_png.py

import cv2
from pathlib import Path

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

pgm_dir = project_root / "results" / "pgm"
png_dir = project_root / "results" / "png"
png_dir.mkdir(parents=True, exist_ok=True)

# Map: pgm filename → png filename
conversions = {
    "input.pgm"         : "01_preprocessed_input.png",
    "output_serial.pgm" : "02_output_serial.png",
    "output_openmp.pgm" : "03_output_openmp.png",
    "output_mpi.pgm"    : "04_output_mpi.png",
    "output_hybrid.pgm" : "05_output_hybrid.png",
}

for pgm_name, png_name in conversions.items():
    pgm_path = pgm_dir / pgm_name
    png_path = png_dir / png_name

    if not pgm_path.exists():
        print(f"Skipped (not found): {pgm_path}")
        continue

    img = cv2.imread(str(pgm_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read: {pgm_path}")
        continue

    cv2.imwrite(str(png_path), img)
    print(f"Converted: {pgm_name} → {png_name}")
