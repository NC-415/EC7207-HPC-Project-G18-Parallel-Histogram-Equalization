#!/usr/bin/env python3
"""
Plot OpenMP execution time, speedup and RMSE from benchmark results.
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

def load_single_value(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open() as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                return float(line.split()[0])
            except (ValueError, IndexError):
                continue
    raise ValueError(f"No numeric value found in {path}")

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

    omp_time_path = bench_dir / "openmp_execution_results.txt"
    serial_time_path = bench_dir / "serial_execution_results.txt"
    rmse_path = bench_dir / "openmp_rmse_results.txt"

    speedup_out_path = bench_dir / "openmp_speedup_results.txt"
    execution_plot_path = plots_dir / "openmp_execution_time_plot.png"
    speedup_plot_path = plots_dir / "openmp_speedup_plot.png"
    rmse_plot_path = plots_dir / "openmp_rmse_plot.png"

    if not omp_time_path.exists():
        raise FileNotFoundError(f"OpenMP results file not found: {omp_time_path}")

    omp_data = load_two_col(omp_time_path)
    if not omp_data:
        raise ValueError(f"No OpenMP data found in {omp_time_path}")

    threads = sorted(omp_data)
    times = [omp_data[t] for t in threads]

    save_line_plot(
        threads,
        times,
        "Number of Threads",
        "Execution Time (seconds)",
        "OpenMP Execution Time vs Threads",
        execution_plot_path,
    )
    print(f"Saved plot to: {execution_plot_path}")

    try:
        serial_time = load_single_value(serial_time_path)
    except Exception as e:
        print(f"Serial baseline not found, skipping speedup: {e}")
        serial_time = None

    if serial_time is not None:
        with speedup_out_path.open("w", encoding="utf-8") as f:
            f.write("# threads speedup\n")
            for t in threads:
                speedup = serial_time / omp_data[t] if omp_data[t] > 0 else 0.0
                f.write(f"{t} {speedup:.6f}\n")

        speedups = [serial_time / omp_data[t] if omp_data[t] > 0 else 0.0 for t in threads]
        save_line_plot(
            threads,
            speedups,
            "Number of Threads",
            "Speedup (serial / openmp)",
            "OpenMP Speedup vs Threads",
            speedup_plot_path,
        )
        print(f"Saved plot to: {speedup_plot_path}")
        print(f"Wrote OpenMP speedup results to: {speedup_out_path}")
    else:
        print("Skipping speedup plot (no serial baseline).")

    if rmse_path.exists():
        rmse_data = load_two_col(rmse_path)
        if rmse_data:
            rmse_threads = sorted(rmse_data)
            rmse_values = [rmse_data[t] for t in rmse_threads]
            save_line_plot(
                rmse_threads,
                rmse_values,
                "Number of Threads",
                "RMSE",
                "OpenMP RMSE vs Threads",
                rmse_plot_path,
            )
            print(f"Saved plot to: {rmse_plot_path}")
    else:
        print(f"RMSE file not found, skipping RMSE plot: {rmse_path}")

if __name__ == "__main__":
    main()
