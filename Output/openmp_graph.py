# Run :  python3 openmp_graph.py

import matplotlib.pyplot as plt
from pathlib import Path

threads = []
times = []

# Resolve paths based on this script's location
output_dir = Path(__file__).resolve().parent          # .../Output
project_root = output_dir.parent                      # .../EC7207-HPC-Project-G18-Parallel-Histogram-Equalization
results_path = project_root / "Code" / "results.txt"   # .../Code/results.txt
images_dir = project_root / "Images"                  # .../Images
images_dir.mkdir(parents=True, exist_ok=True)
image_path = images_dir / "openmp_execution_time.png"

# read results
if not results_path.exists():
    raise FileNotFoundError(f"results.txt not found at: {results_path}")
with results_path.open() as f:
    for line in f:
        t, time = line.split()
        threads.append(int(t))
        times.append(float(time))

plt.figure(figsize=(8, 5))
plt.plot(threads, times, marker="o")

plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("OpenMP Execution Time vs Threads")
plt.grid(True)

plt.savefig(image_path)
plt.show()

print(f"Saved plot to: {image_path}")