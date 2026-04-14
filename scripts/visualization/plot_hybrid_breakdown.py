# Run: python3 scripts/visualization/plot_hybrid_breakdown.py
#
# Stacked bar chart showing per-stage execution time for each (np, block_size)
# combination: H2D / hist kernel / MPI_Allreduce / remap kernel / D2H
# Output: results/plots/hybrid_time_breakdown.png

import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
results_path = project_root / "results" / "benchmarks" / "hybrid_benchmarks.txt"
plots_dir    = project_root / "results" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
image_path   = plots_dir / "hybrid_time_breakdown.png"

# ── Check file ────────────────────────────────────────────────────────────────
if not results_path.exists():
    raise FileNotFoundError(f"Benchmark file not found: {results_path}")

# ── Load data ─────────────────────────────────────────────────────────────────
# {(np, block_size): {stage: value_ms}}
groups = defaultdict(lambda: defaultdict(list))

with results_path.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        key = (int(parts[0]), int(parts[1]))
        groups[key]["h2d_ms"].append(float(parts[2]))
        groups[key]["hist_ms"].append(float(parts[3]))
        groups[key]["allreduce_ms"].append(float(parts[4]))
        groups[key]["remap_ms"].append(float(parts[5]))
        groups[key]["d2h_ms"].append(float(parts[6]))

if not groups:
    print("No data found in benchmark file.")
    sys.exit(1)

# ── Build arrays (sorted by np then block_size) ───────────────────────────────
stages = [
    ("H→D transfer",  "h2d_ms",       "#42a5f5"),
    ("Hist kernel",   "hist_ms",       "#ab47bc"),
    ("MPI_Allreduce", "allreduce_ms",  "#ef5350"),
    ("Remap kernel",  "remap_ms",      "#26a69a"),
    ("D→H transfer",  "d2h_ms",        "#8d6e63"),
]

labels = []
values = {s[1]: [] for s in stages}

for (np_, bs) in sorted(groups.keys()):
    labels.append(f"np={np_}\nblk={bs}")
    for _, key, _ in stages:
        avg = np.mean(groups[(np_, bs)][key])
        values[key].append(avg)

x     = np.arange(len(labels))
width = 0.55

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 6))

bottoms = np.zeros(len(labels))
patches = []

for label, key, color in stages:
    vals = np.array(values[key])
    ax.bar(x, vals, width, bottom=bottoms, color=color)
    bottoms += vals
    patches.append(mpatches.Patch(color=color, label=label))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel("(np, block size)")
ax.set_ylabel("Time (ms)")
ax.set_title("CUDA+MPI Hybrid: Execution Time Breakdown per Stage")
ax.legend(handles=patches, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.savefig(image_path)
plt.show()

print(f"Saved plot to: {image_path}")
