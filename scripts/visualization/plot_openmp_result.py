#!/usr/bin/env python3
"""
Plot OpenMP execution time and speedup from benchmark results.

Reads:
- results/benchmarks/openmp_execution_results.txt  (threads time)
- results/benchmarks/serial_execution_results.txt  (single float)

Writes:
- results/benchmarks/openmp_speedup_results.txt  (threads speedup)
- results/plots/openmp_execution_time_plot.png
- results/plots/openmp_speedup_plot.png
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


def load_serial_time(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Serial baseline file not found: {path}")
    with path.open() as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                return float(parts[0])
            except (ValueError, IndexError):
                continue
    raise ValueError(f"No numeric serial time found in {path}")


def save_execution_time_plot(threads: list[int], times: list[float], image_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(threads, times, marker="o")
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (seconds)")
    plt.title("OpenMP Execution Time vs Threads")
    plt.grid(True)
    plt.savefig(image_path)
    plt.show()


def save_speedup_results(
    threads: list[int],
    opm_times: dict[int, float],
    serial_time: float,
    out_path: Path,
) -> list[float]:
    speedups = []
    with out_path.open("w") as file_handle:
        file_handle.write("# threads speedup\n")
        for thread_count in threads:
            opm_time = opm_times[thread_count]
            speedup = serial_time / opm_time if opm_time > 0 else 0.0
            speedups.append(speedup)
            file_handle.write(f"{thread_count} {speedup:.6f}\n")
    return speedups


def save_speedup_plot(threads: list[int], speedups: list[float], image_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(threads, speedups, marker="o")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (serial / openmp)")
    plt.title("OpenMP Speedup vs Threads")
    plt.grid(True)
    plt.savefig(image_path)
    plt.show()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    bench_dir = project_root / "results" / "benchmarks"
    plots_dir = project_root / "results" / "plots"
    bench_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    omp_path = bench_dir / "openmp_execution_results.txt"
    serial_path = bench_dir / "serial_execution_results.txt"
    speedup_out_path = bench_dir / "openmp_speedup_results.txt"
    execution_plot_path = plots_dir / "openmp_execution_time_plot.png"
    speedup_plot_path = plots_dir / "openmp_speedup.png"

    if not omp_path.exists():
        raise FileNotFoundError(f"OpenMP results file not found: {omp_path}")

    omp_data = load_two_col(omp_path)
    if not omp_data:
        raise ValueError(f"No OpenMP data found in {omp_path}")

    threads = sorted(omp_data)
    times = [omp_data[thread_count] for thread_count in threads]

    save_execution_time_plot(threads, times, execution_plot_path)
    print(f"Saved plot to: {execution_plot_path}")

    serial_time = load_serial_time(serial_path)
    speedups = save_speedup_results(threads, omp_data, serial_time, speedup_out_path)
    print(f"Wrote OpenMP speedup results to: {speedup_out_path}")

    save_speedup_plot(threads, speedups, speedup_plot_path)
    print(f"Saved plot to: {speedup_plot_path}")


if __name__ == "__main__":
    main()