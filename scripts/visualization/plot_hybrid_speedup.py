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
mpi_exec_path = bench_dir / "mpi_execution_results.txt"
omp_exec_path = bench_dir / "openmp_execution_results.txt"

# speedup files (saved, same format as MPI/OpenMP speedup outputs)
mpi_speedup_path = bench_dir / "mpi_speedup_results.txt"
omp_speedup_path = bench_dir / "openmp_speedup_results.txt"
hybrid_speedup_path = bench_dir / "hybrid_speedup_results.txt"

# serial baseline (used to compute speedup values)
serial_time_path = bench_dir / "serial_execution_results.txt"

# ── Check files exist ─────────────────────────────────────────────────────────
missing = [p for p in (hybrid_path, mpi_exec_path, omp_exec_path) if not p.exists()]
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
hybrid_best = load_hybrid_best(hybrid_path)

# Ensure we have a serial baseline to compute speedups the same way as other scripts
serial_time = None
if serial_time_path.exists():
    try:
        with serial_time_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                serial_time = float(line.split()[0])
                break
    except Exception:
        serial_time = None

# Compute and write hybrid speedup file using serial baseline (if available)
if serial_time is not None and hybrid_best:
    with hybrid_speedup_path.open("w", encoding="utf-8") as f:
        f.write("# processes speedup\n")
        for np_ in sorted(hybrid_best):
            t = hybrid_best[np_]
            speedup = serial_time / t if t > 0 else 0.0
            f.write(f"{np_} {speedup:.6f}\n")

# Prefer reading saved speedup files; fall back to execution-time files if missing
def ensure_speedup(path_speedup, path_exec, label):
    if path_speedup.exists():
        return load_two_col(path_speedup)
    # fallback: compute from execution times if serial baseline available
    if serial_time is None:
        print(f"Missing {path_speedup} and no serial baseline available to compute {label} speedup.")
        return {}
    if not path_exec.exists():
        print(f"Missing execution file to compute {label} speedup: {path_exec}")
        return {}
    exec_data = load_two_col(path_exec)
    speedup = {k: (serial_time / v if v > 0 else 0.0) for k, v in exec_data.items()}
    # write the derived speedup file for transparency
    with path_speedup.open("w", encoding="utf-8") as f:
        f.write("# processes speedup\n")
        for k in sorted(speedup):
            f.write(f"{k} {speedup[k]:.6f}\n")
    return speedup

mpi_data = ensure_speedup(mpi_speedup_path, mpi_exec_path, "MPI")
omp_data = ensure_speedup(omp_speedup_path, omp_exec_path, "OpenMP")
hybrid_data = load_two_col(hybrid_speedup_path) if hybrid_speedup_path.exists() else {}

if not hybrid_best and not hybrid_data:
    print(f"No data found in {hybrid_path} and no hybrid speedup available")
    sys.exit(1)

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))

# Plot saved/derived speedup files
mpi_np = sorted(mpi_data)
plt.plot(mpi_np, [mpi_data[n] for n in mpi_np], marker="s", label="MPI-only")

omp_t = sorted(omp_data)
plt.plot(omp_t, [omp_data[t] for t in omp_t], marker="^", label="OpenMP")

# hybrid_data may be empty if we couldn't compute it
hyb_np = sorted(hybrid_data)
if hyb_np:
    plt.plot(hyb_np, [hybrid_data[n] for n in hyb_np], marker="o", label="CUDA+MPI Hybrid")

all_x = sorted(set(mpi_np + omp_t + hyb_np))
plt.plot(all_x, [float(x) for x in all_x], linestyle="--", color="grey", linewidth=1.0, label="Ideal linear speedup")

plt.xlabel("Number of Processes / Threads")
plt.ylabel("Speedup  (relative to MPI np=1)")
plt.title("Speedup Comparison: MPI vs OpenMP vs CUDA+MPI Hybrid")
plt.legend()
plt.grid(True)

plt.savefig(image_path)
plt.show()

print(f"Saved plot to: {image_path}")
