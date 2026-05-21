#!/usr/bin/env python3
"""
Compute RMSE between the serial and OpenMP output images.

This script mirrors the other OpenMP analysis scripts:
- it reads the reference and test images from results/pgm/
- it writes the metric into results/benchmarks/
- it prints the final value for quick checking

Default inputs:
- results/pgm/output_serial.pgm
- results/pgm/output_openmp.pgm

Output:
- results/benchmarks/openmp_rmse_results.txt
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


def read_pgm(path: Path) -> list[list[int]]:
	"""Read a P2 or P5 PGM file into a 2D list of pixels."""
	with path.open("rb") as f:
		magic = f.readline().strip()
		if magic not in {b"P2", b"P5"}:
			raise ValueError(f"Unsupported PGM format in {path}: {magic!r}")

		tokens: list[bytes] = []
		while len(tokens) < 3:
			line = f.readline()
			if not line:
				raise ValueError(f"Unexpected end of file while reading header: {path}")
			line = line.strip()
			if not line or line.startswith(b"#"):
				continue
			tokens.extend(line.split())

		width = int(tokens[0])
		height = int(tokens[1])
		maxval = int(tokens[2])
		if maxval <= 0:
			raise ValueError(f"Invalid max value in {path}: {maxval}")

		if magic == b"P5":
			raw = f.read(width * height)
			if len(raw) != width * height:
				raise ValueError(f"Not enough image data in {path}")
			pixels = list(raw)
		else:
			rest = f.read().split()
			if len(rest) < width * height:
				raise ValueError(f"Not enough image data in {path}")
			pixels = [int(x) for x in rest[: width * height]]

	return [pixels[i * width : (i + 1) * width] for i in range(height)]


def rmse(reference: list[list[int]], test: list[list[int]]) -> float:
	if len(reference) != len(test) or any(len(r) != len(t) for r, t in zip(reference, test)):
		raise ValueError("Reference and test images must have the same dimensions")

	total = 0.0
	count = 0
	for row_ref, row_test in zip(reference, test):
		for ref_pixel, test_pixel in zip(row_ref, row_test):
			diff = float(ref_pixel) - float(test_pixel)
			total += diff * diff
			count += 1

	if count == 0:
		raise ValueError("Images contain no pixels")

	return math.sqrt(total / count)


def main() -> int:
	script_dir = Path(__file__).resolve().parent
	project_root = script_dir.parent.parent

	reference_path = project_root / "results" / "pgm" / "output_serial.pgm"
	test_path = project_root / "results" / "pgm" / "output_openmp.pgm"

	bench_dir = project_root / "results" / "benchmarks"
	bench_dir.mkdir(parents=True, exist_ok=True)
	out_path = bench_dir / "openmp_rmse_results.txt"

	if not reference_path.exists():
		print(f"Missing file: {reference_path}")
		return 1
	if not test_path.exists():
		print(f"Missing file: {test_path}")
		return 1

	reference = read_pgm(reference_path)
	test = read_pgm(test_path)
	value = rmse(reference, test)

	with out_path.open("w", encoding="utf-8") as f:
		f.write("# reference test rmse\n")
		f.write(f"{reference_path.name} {test_path.name} {value:.6f}\n")

	print(f"RMSE: {value:.6f}")
	print(f"Wrote RMSE results to: {out_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

