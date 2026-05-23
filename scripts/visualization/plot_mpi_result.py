#!/usr/bin/env python3
"""
Plot MPI execution time, speedup and RMSE from benchmark results.
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

    mpi_time_path = bench_dir / "mpi_execution_results.txt"
    serial_time_path = bench_dir / "serial_execution_results.txt"
    rmse_path = bench_dir / "mpi_rmse_results.txt"

    speedup_out_path = bench_dir / "mpi_speedup_results.txt"
    execution_plot_path = plots_dir / "mpi_execution_time_plot.png"
    speedup_plot_path = plots_dir / "mpi_speedup_plot.png"
    rmse_plot_path = plots_dir / "mpi_rmse_plot.png"

    if not mpi_time_path.exists():
        raise FileNotFoundError(f"MPI results file not found: {mpi_time_path}")

    mpi_data = load_two_col(mpi_time_path)
    if not mpi_data:
        raise ValueError(f"No MPI data found in {mpi_time_path}")

    processes = sorted(mpi_data)
    times = [mpi_data[p] for p in processes]

    save_line_plot(
        processes,
        times,
        "Number of Processes",
        "Execution Time (seconds)",
        "MPI Execution Time vs Processes",
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
            f.write("# processes speedup\n")
            for p in processes:
                speedup = serial_time / mpi_data[p] if mpi_data[p] > 0 else 0.0
                f.write(f"{p} {speedup:.6f}\n")

        speedups = [serial_time / mpi_data[p] if mpi_data[p] > 0 else 0.0 for p in processes]
        save_line_plot(
            processes,
            speedups,
            "Number of Processes",
            "Speedup (serial / mpi)",
            "MPI Speedup vs Processes",
            speedup_plot_path,
        )
        print(f"Saved plot to: {speedup_plot_path}")
        print(f"Wrote MPI speedup results to: {speedup_out_path}")
    else:
        print("Skipping speedup plot (no serial baseline).")

    if rmse_path.exists():
        rmse_data = load_two_col(rmse_path)
        if rmse_data:
            rmse_processes = sorted(rmse_data)
            rmse_values = [rmse_data[p] for p in rmse_processes]
            save_line_plot(
                rmse_processes,
                rmse_values,
                "Number of Processes",
                "RMSE",
                "MPI RMSE vs Processes",
                rmse_plot_path,
            )
            print(f"Saved plot to: {rmse_plot_path}")
    else:
        print(f"RMSE file not found, skipping RMSE plot: {rmse_path}")

if __name__ == "__main__":
    main()