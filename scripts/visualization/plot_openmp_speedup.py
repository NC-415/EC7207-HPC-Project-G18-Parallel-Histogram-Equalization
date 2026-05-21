#!/usr/bin/env python3
"""
Compute OpenMP speedup from benchmarks and optionally plot it.

Reads:
- results/benchmarks/openmp_execution_results.txt  (threads time)
- results/benchmarks/serial_execution_results.txt  (single float)

Writes:
- results/benchmarks/openmp_speedup_results.txt  (threads speedup)
- results/plots/openmp_speedup.png  (if matplotlib available)
"""
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
bench_dir = project_root / "results" / "benchmarks"
plots_dir = project_root / "results" / "plots"
bench_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

omp_path = bench_dir / "openmp_execution_results.txt"
serial_path = bench_dir / "serial_execution_results.txt"
out_path = bench_dir / "openmp_speedup_results.txt"
plot_path = plots_dir / "openmp_speedup.png"

def load_two_col(path):
	data = {}
	with path.open() as f:
		for line in f:
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

def load_serial_time(path):
	if not path.exists():
		raise FileNotFoundError(f"Serial baseline file not found: {path}")
	with path.open() as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split()
			# accept single value or first numeric token on the line
			try:
				return float(parts[0])
			except Exception:
				continue
	raise ValueError(f"No numeric serial time found in {path}")

if not omp_path.exists():
	print(f"Missing file: {omp_path}")
	sys.exit(1)

try:
	serial_time = load_serial_time(serial_path)
except Exception as e:
	print(e)
	sys.exit(1)

omp_data = load_two_col(omp_path)
if not omp_data:
	print(f"No OpenMP data found in {omp_path}")
	sys.exit(1)

# Compute speedups
with out_path.open("w") as outf:
	outf.write("# threads speedup\n")
	for t in sorted(omp_data):
		t_openmp = omp_data[t]
		speedup = serial_time / t_openmp if t_openmp > 0 else 0.0
		outf.write(f"{t} {speedup:.6f}\n")

print(f"Wrote OpenMP speedup results to: {out_path}")

# Optional plotting
try:
	import matplotlib.pyplot as plt

	threads = sorted(omp_data)
	speeds = [serial_time / omp_data[t] for t in threads]

	plt.figure(figsize=(6,4))
	plt.plot(threads, speeds, marker='o')
	plt.xlabel('Threads')
	plt.ylabel('Speedup (serial / openmp)')
	plt.title('OpenMP Speedup')
	plt.grid(True)
	plt.savefig(plot_path)
	print(f"Saved plot to: {plot_path}")
except Exception:
	print("matplotlib not available — skipping plot")

