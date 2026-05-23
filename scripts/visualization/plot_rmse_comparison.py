#!/usr/bin/env python3
"""
Plot RMSE comparison for MPI, OpenMP, and CUDA+MPI Hybrid on one chart.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_two_col(path):
    data = {}
    with path.open() as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    data[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue
    return data


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    bench_dir = project_root / "results" / "benchmarks"
    plots_dir = project_root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mpi_path = bench_dir / "mpi_rmse_results.txt"
    omp_path = bench_dir / "openmp_rmse_results.txt"
    hybrid_path = bench_dir / "hybrid_rmse_results.txt"
    image_path = plots_dir / "rmse_comparison.png"

    missing = [p for p in (mpi_path, omp_path, hybrid_path) if not p.exists()]
    if missing:
        print("Missing RMSE files:")
        for path in missing:
            print(f"  {path}")
        sys.exit(1)

    mpi_data = load_two_col(mpi_path)
    omp_data = load_two_col(omp_path)
    hybrid_data = load_two_col(hybrid_path)

    if not mpi_data or not omp_data or not hybrid_data:
        print("One or more RMSE files do not contain numeric data.")
        sys.exit(1)

    plt.figure(figsize=(8, 5))

    mpi_x = sorted(mpi_data)
    plt.plot(mpi_x, [mpi_data[x] for x in mpi_x], marker="s", label="MPI")

    omp_x = sorted(omp_data)
    plt.plot(omp_x, [omp_data[x] for x in omp_x], marker="^", label="OpenMP")

    hybrid_x = sorted(hybrid_data)
    plt.plot(hybrid_x, [hybrid_data[x] for x in hybrid_x], marker="o", label="CUDA+MPI Hybrid")

    plt.xlabel("Number of Processes / Threads")
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison: MPI vs OpenMP vs CUDA+MPI Hybrid")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()

    print(f"Saved plot to: {image_path}")


if __name__ == "__main__":
    main()