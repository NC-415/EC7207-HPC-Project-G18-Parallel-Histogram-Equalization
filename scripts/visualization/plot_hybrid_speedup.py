# Run: python3 scripts/visualization/plot_hybrid_speedup.py
#
# Plots speedup vs parallelism level for MPI-only, OpenMP, and CUDA+MPI Hybrid
# on the same axes, normalised to MPI np=1 as the baseline.
# Output: results/plots/hybrid_speedup_comparison.png

import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
bench_dir    = project_root / "results" / "benchmarks"
plots_dir    = project_root / "results" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
image_path   = plots_dir / "hybrid_speedup_comparison.png"

hybrid_path = bench_dir / "hybrid_benchmarks.txt"
mpi_path    = bench_dir / "mpi_benchmarks.txt"
omp_path    = bench_dir / "openmp_execution_results.txt"

# ── Check files exist ─────────────────────────────────────────────────────────
missing = [p for p in (hybrid_path, mpi_path, omp_path) if not p.exists()]
if missing:
    print("Missing benchmark files:")
    for p in missing:
        print(f"  {p}")
    sys.exit(1)

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_two_col(path):
    """Load two-column (key, value) benchmark file. Returns {key: value}."""
    data = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                data[int(parts[0])] = float(parts[1])
    return data

def load_hybrid_best(path):
    """Load hybrid file. Returns {np: best total_s across all block sizes}."""
    best = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            np_     = int(parts[0])
            total_s = float(parts[7])
            if np_ not in best or total_s < best[np_]:
                best[np_] = total_s
    return best

# ── Load ──────────────────────────────────────────────────────────────────────
mpi_data    = load_two_col(mpi_path)
omp_data    = load_two_col(omp_path)
hybrid_best = load_hybrid_best(hybrid_path)

if not hybrid_best:
    print(f"No data found in {hybrid_path}")
    sys.exit(1)

# ── Baseline: MPI np=1 ────────────────────────────────────────────────────────
baseline_s = mpi_data.get(1)
if baseline_s is None:
    baseline_s = mpi_data[min(mpi_data)]
    print(f"Note: MPI np=1 not found — using np={min(mpi_data)} as baseline.")

print(f"Baseline (MPI np=1): {baseline_s:.6f} s")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))

mpi_np    = sorted(mpi_data)
plt.plot(mpi_np, [baseline_s / mpi_data[n] for n in mpi_np],
         marker="s", label="MPI-only")

omp_t = sorted(omp_data)
plt.plot(omp_t, [baseline_s / omp_data[t] for t in omp_t],
         marker="^", label="OpenMP")

hyb_np = sorted(hybrid_best)
plt.plot(hyb_np, [baseline_s / hybrid_best[n] for n in hyb_np],
         marker="o", label="CUDA+MPI Hybrid (best block size)")

all_x = sorted(set(mpi_np + omp_t + hyb_np))
plt.plot(all_x, [float(x) for x in all_x],
         linestyle="--", color="grey", linewidth=1.0, label="Ideal linear speedup")

plt.xlabel("Number of Processes / Threads")
plt.ylabel("Speedup  (relative to MPI np=1)")
plt.title("Speedup Comparison: MPI vs OpenMP vs CUDA+MPI Hybrid")
plt.legend()
plt.grid(True)

plt.savefig(image_path)
plt.show()

print(f"Saved plot to: {image_path}")
