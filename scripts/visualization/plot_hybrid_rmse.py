#!/usr/bin/env python3
"""
Plot CUDA+MPI hybrid RMSE from benchmark results.
"""
from pathlib import Path

import matplotlib.pyplot as plt


def load_two_col(path: Path) -> dict[int, float]:
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


def save_line_plot(x_values, y_values, xlabel, ylabel, title, image_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    bench_dir = project_root / "results" / "benchmarks"
    plots_dir = project_root / "results" / "plots"
    bench_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rmse_path = bench_dir / "hybrid_rmse_results.txt"
    rmse_plot_path = plots_dir / "hybrid_rmse_plot.png"

    if not rmse_path.exists():
        raise FileNotFoundError(f"Hybrid RMSE file not found: {rmse_path}")

    rmse_data = load_two_col(rmse_path)
    if not rmse_data:
        raise ValueError(f"No hybrid RMSE data found in {rmse_path}")

    processes = sorted(rmse_data)
    rmse_values = [rmse_data[p] for p in processes]

    save_line_plot(
        processes,
        rmse_values,
        "Number of MPI Processes",
        "RMSE",
        "CUDA+MPI Hybrid RMSE vs Processes",
        rmse_plot_path,
    )
    print(f"Saved plot to: {rmse_plot_path}")


if __name__ == "__main__":
    main()