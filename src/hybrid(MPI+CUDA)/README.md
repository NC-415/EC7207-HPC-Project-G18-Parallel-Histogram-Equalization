# Hybrid CUDA + MPI Histogram Equalization

Parallel histogram equalization that combines **MPI** for inter-process distribution with **CUDA** for GPU-accelerated computation. Each MPI rank is assigned a GPU — the heavy pixel loops run on the GPU while MPI handles all inter-rank communication.

---

## How it works

### Algorithm overview

Histogram equalization redistributes pixel intensities across a grayscale image to improve contrast. The steps are:

1. Build a histogram (count of each 0–255 intensity value)
2. Compute the CDF (cumulative distribution function)
3. Build a look-up table (LUT) mapping old intensities to new ones
4. Remap every pixel through the LUT

### Parallelism strategy

```
Rank 0 reads full image
        │
        ▼  MPI_Scatterv
Each rank receives a chunk of pixels
        │
        ▼  cudaMemcpy  H→D
GPU receives chunk
        │
        ▼  histogram_kernel  (GPU)
Local 256-bin histogram
        │
        ▼  cudaMemcpy  D→H  (256 × 4 bytes = 1 KB)
        │
        ▼  MPI_Allreduce  (CPU)
Global histogram combined across all ranks
        │
        ▼  CDF + LUT  (CPU, 256 iterations)
        │
        ▼  cudaMemcpy  H→D  (LUT, 256 bytes)
        │
        ▼  remap_kernel  (GPU)
Pixels remapped using LUT
        │
        ▼  cudaMemcpy  D→H
        │
        ▼  MPI_Gatherv
Rank 0 assembles and writes output image
```

### CUDA kernels

**`histogram_kernel`**
Each thread block maintains a private 256-bin histogram in shared memory. Threads atomically increment shared bins (fast, intra-SM). At the end of each block, the shared histogram is merged into global memory. This reduces global atomics from O(n) to O(blocks × 256).

**`remap_kernel`**
Each thread reads one pixel and writes `lut[pixel]` — trivially parallel with no shared memory needed.

---

## Prerequisites

| Requirement | Version |
|---|---|
| CUDA Toolkit | 10.0 or later |
| MPI | OpenMPI or MPICH |
| GCC | 7.0 or later |
| ImageMagick | any (for PNG conversion) |
| Python | 3.6 or later |
| OpenCV (cv2) | any (for PNG conversion) |

  Check your setup:
  ```bash
  nvcc --version
  mpicc --version
  +python3 -c "import cv2; print(cv2.__version__)"
  ```

If OpenCV is not installed, set up the project virtual environment first:
```bash
python3 -m venv hpc_env
source hpc_env/bin/activate
pip install numpy opencv-python matplotlib
```

---

## Compile

Run from the **project root**:

```bash
mkdir -p build

nvcc -O2 -std=c++11 \
  "src/hybrid(MPI+CUDA)/hist_eq_cuda_mpi.cu" \
  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
  -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
  -lmpi \
  -o build/hist_eq_cuda_mpi
````

---

## Run

Run from the **project root**. Repeat for each process count to populate the benchmark file:

```bash
mpirun -np 1 ./build/hist_eq_cuda_mpi results/pgm/input.pgm results/pgm/output_hybrid.pgm
mpirun -np 2 ./build/hist_eq_cuda_mpi results/pgm/input.pgm results/pgm/output_hybrid.pgm
mpirun -np 4 ./build/hist_eq_cuda_mpi results/pgm/input.pgm results/pgm/output_hybrid.pgm
mpirun -np 6 ./build/hist_eq_cuda_mpi results/pgm/input.pgm results/pgm/output_hybrid.pgm
```

Each invocation automatically sweeps block sizes `{128, 256, 512}` and appends 3 rows to the benchmark file.

### Expected terminal output (per invocation)

```
[Config] 4 MPI rank(s), 2 GPU(s) detected — rank i → GPU (i % 2)
[Image]  8192 x 8192  (67108864 pixels)
np=4  block=128  | H2D=  1.39ms  hist=  1.01ms  allreduce= 0.025ms  remap=  0.61ms  D2H=  1.35ms  | total=  0.0048s
np=4  block=256  | H2D=  1.37ms  hist=  0.93ms  allreduce= 0.025ms  remap=  0.59ms  D2H=  1.32ms  | total=  0.0045s
np=4  block=512  | H2D=  1.38ms  hist=  0.98ms  allreduce= 0.025ms  remap=  0.60ms  D2H=  1.33ms  | total=  0.0046s
Output written: results/pgm/output_hybrid.pgm
```

---

## Benchmark file

`results/benchmarks/hybrid_benchmarks.txt` is created automatically on the first run. After all four `mpirun` invocations it contains 12 rows:

```
# np block_size h2d_ms hist_ms allreduce_ms remap_ms d2h_ms total_s
1 128 5.3120 3.8450 0.0010 2.3210 5.1340 0.016800
1 256 5.2890 3.5120 0.0010 2.1450 5.0980 0.016100
1 512 5.3340 3.6780 0.0010 2.2340 5.1120 0.016500
2 128 2.6980 1.9870 0.0120 1.1980 2.6120 0.008900
...
```

| Column | Description |
|---|---|
| `np` | Number of MPI processes |
| `block_size` | CUDA threads per block |
| `h2d_ms` | Host→Device transfer time (ms) |
| `hist_ms` | Histogram kernel time (ms) |
| `allreduce_ms` | MPI_Allreduce time (ms) |
| `remap_ms` | Remap kernel time (ms) |
| `d2h_ms` | Device→Host transfer time (ms) |
| `total_s` | Total wall-clock time (seconds) |

---

## Generate plots

```bash
python3 scripts/visualization/plot_hybrid_result.py
```

Three separate scripts, each producing one plot in `results/plots/`:

```bash
# Execution time vs np — one line per block size (matches OpenMP/MPI plot style)
python3 scripts/visualization/plot_hybrid_result.py

# Speedup comparison — MPI-only vs OpenMP vs Hybrid on the same axes
python3 scripts/visualization/plot_hybrid_speedup.py

# Time breakdown — stacked bar per stage (H2D / hist / allreduce / remap / D2H)
python3 scripts/visualization/plot_hybrid_breakdown.py
```

| Script | Output plot | Description |
|---|---|---|
| `plot_hybrid_result.py` | `hybrid_execution_time_plot.png` | Execution time vs number of MPI processes, one line per block size |
| `plot_hybrid_speedup.py` | `hybrid_speedup_comparison.png` | Speedup vs process/thread count for MPI-only, OpenMP, and Hybrid on the same axes |
| `plot_hybrid_breakdown.py` | `hybrid_time_breakdown.png` | Stacked bar showing H2D / hist / allreduce / remap / D2H per (np, block_size) |

---

## Output images

| File | Description |
|---|---|
| `results/pgm/output_hybrid.pgm` | Equalized image (binary PGM) |
| `results/png/05_output_hybrid.png` | Same image converted to PNG |

---

## GPU assignment

When there are multiple GPUs, rank `i` is assigned to `GPU (i % num_gpus)`:

```
2 GPUs, 4 ranks:   rank 0 → GPU 0,  rank 1 → GPU 1,  rank 2 → GPU 0,  rank 3 → GPU 1
1 GPU,  4 ranks:   all ranks → GPU 0  (valid for testing, ranks serialize on GPU)
```

---

## File structure

```
src/hybrid(MPI+CUDA)/
├── hist_eq_cuda_mpi.cu    — CUDA + MPI implementation
└── README.md              — this file

results/
├── benchmarks/
│   └── hybrid_benchmarks.txt     — auto-created on first run
├── pgm/
│   └── output_hybrid.pgm         — output image
└── png/
    └── 05_output_hybrid.png      — PNG version of output

scripts/visualization/
└── plot_hybrid_result.py         — generates speedup + breakdown plots
```
