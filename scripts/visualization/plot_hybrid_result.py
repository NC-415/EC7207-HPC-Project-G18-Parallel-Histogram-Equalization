# Run: python3 scripts/visualization/plot_hybrid_result.py

import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
results_path = project_root / "results" / "benchmarks" / "hybrid_benchmarks.txt"
plots_dir    = project_root / "results" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
image_path   = plots_dir / "hybrid_execution_time_plot.png"

# ── Load data ─────────────────────────────────────────────────────────────────
if not results_path.exists():
    raise FileNotFoundError(f"Benchmark file not found: {results_path}")

# Group total_s by block_size → {block_size: {np: total_s}}
data = defaultdict(dict)

with results_path.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        np_       = int(parts[0])
        block_size = int(parts[1])
        total_s    = float(parts[7])
        data[block_size][np_] = total_s

if not data:
    print("No data found in benchmark file.")
    sys.exit(1)

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))

markers = {128: "o", 256: "s", 512: "^"}

for block_size in sorted(data.keys()):
    np_vals   = sorted(data[block_size].keys())
    time_vals = [data[block_size][np_] for np_ in np_vals]
    plt.plot(np_vals, time_vals,
             marker=markers.get(block_size, "o"),
             label=f"block_size={block_size}")

plt.xlabel("Number of MPI Processes")
plt.ylabel("Execution Time (seconds)")
plt.title("CUDA+MPI Hybrid Execution Time vs Number of Processes")
plt.legend()
plt.grid(True)

plt.savefig(image_path)
plt.show()

print(f"Saved plot to: {image_path}")
