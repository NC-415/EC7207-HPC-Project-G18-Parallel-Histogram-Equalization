/*
  Hybrid CUDA + MPI Histogram Equalization

  Each MPI rank is assigned one GPU. The rank:
    1. Receives a chunk of the image via MPI_Scatterv  (CPU)
    2. Copies chunk to GPU                             (H→D)
    3. Runs histogram_kernel on GPU                    (GPU)
    4. Copies 256-bin histogram back to CPU            (D→H, tiny)
    5. Combines histograms across all ranks            (MPI_Allreduce)
    6. Computes CDF + LUT on CPU                       (256 iters, negligible)
    7. Copies LUT to GPU + runs remap_kernel           (GPU)
    8. Copies remapped chunk back to CPU               (D→H)
    9. Gathers all chunks to rank 0                    (MPI_Gatherv)

  Benchmarking:
    - Sweeps block sizes {128, 256, 512} per invocation
    - CUDA events time H2D / hist kernel / remap kernel / D2H individually
    - MPI_Wtime times MPI_Allreduce and total wall clock
    - Appends one row per (np, block_size) to:
        results/benchmarks/hybrid_benchmarks.txt

  Compile (from project root):
    nvcc -O2 -std=c++11 \
      "src/hybrid(MPI+CUDA)/hist_eq_cuda_mpi.cu" \
      -Xcompiler "$(mpicc --showme:compile)" \
      $(mpicc --showme:libs) \
      -o build/hist_eq_cuda_mpi

  Run (from project root):
    mpirun -np 1 ./build/hist_eq_cuda_mpi results/pgm/input.pgm results/pgm/output_hybrid.pgm
    mpirun -np 2 ./build/hist_eq_cuda_mpi results/pgm/input.pgm results/pgm/output_hybrid.pgm
    mpirun -np 4 ./build/hist_eq_cuda_mpi results/pgm/input.pgm results/pgm/output_hybrid.pgm
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define L 256

// ── CUDA error check ──────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// ── Timing helper ─────────────────────────────────────────────────────────────
static float cuda_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ── PGM I/O ───────────────────────────────────────────────────────────────────

static void skip_comments_and_whitespace(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') {
            while ((c = fgetc(fp)) != EOF && c != '\n') {}
            continue;
        }
        ungetc(c, fp);
        break;
    }
}

static int read_int_pgm(FILE *fp, int *out) {
    skip_comments_and_whitespace(fp);
    return fscanf(fp, "%d", out) == 1;
}

static uint8_t *read_pgm_p5(const char *path, int *w, int *h) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("fopen"); return NULL; }

    char magic[3];
    if (fread(magic, 1, 2, fp) != 2) { fclose(fp); return NULL; }
    magic[2] = '\0';

    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: only binary PGM (P5) supported.\n");
        fclose(fp); return NULL;
    }

    int maxval;
    if (!read_int_pgm(fp, w) || !read_int_pgm(fp, h) || !read_int_pgm(fp, &maxval)) {
        fprintf(stderr, "Error: failed to read PGM header.\n");
        fclose(fp); return NULL;
    }
    if (maxval != 255) {
        fprintf(stderr, "Error: only 8-bit PGM (maxval=255) supported.\n");
        fclose(fp); return NULL;
    }

    fgetc(fp); // consume single whitespace after maxval

    size_t n = (size_t)(*w) * (*h);
    uint8_t *data = (uint8_t *)malloc(n);
    if (!data) { fclose(fp); return NULL; }

    if (fread(data, 1, n, fp) != n) {
        fprintf(stderr, "Error: failed to read pixel data.\n");
        free(data); fclose(fp); return NULL;
    }

    fclose(fp);
    return data;
}

static int write_pgm_p5(const char *path, const uint8_t *data, int w, int h) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror("fopen"); return 0; }
    fprintf(fp, "P5\n%d %d\n255\n", w, h);
    if (fwrite(data, 1, (size_t)w * h, fp) != (size_t)w * h) {
        fclose(fp); return 0;
    }
    fclose(fp);
    return 1;
}

// ── CUDA Kernels ──────────────────────────────────────────────────────────────

/*
  histogram_kernel
  ----------------
  Each block maintains a private 256-bin histogram in shared memory.
  Threads atomically increment shared bins (fast, intra-SM).
  At the end of the block each thread merges shared bins into the
  global histogram (one global atomic per bin per block — O(gridDim * 256)
  global atomics instead of O(n) with a naive approach).
*/
__global__ void histogram_kernel(const uint8_t *pixels,
                                  uint32_t      *hist,
                                  int            n)
{
    __shared__ uint32_t sh_hist[L];

    // Initialise shared histogram (handles blockDim < L and blockDim > L)
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        sh_hist[i] = 0u;
    __syncthreads();

    // Each thread counts one pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        atomicAdd(&sh_hist[pixels[idx]], 1u);
    __syncthreads();

    // Merge block result into global histogram
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        atomicAdd(&hist[i], sh_hist[i]);
}

/*
  remap_kernel
  ------------
  Trivially data-parallel: each thread applies the pre-computed LUT to
  one pixel. No shared memory needed — LUT fits in L1/texture cache.
*/
__global__ void remap_kernel(uint8_t       *pixels,
                              const uint8_t *lut,
                              int            n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        pixels[idx] = lut[pixels[idx]];
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            printf("Usage: mpirun -np <n> %s <input.pgm> <output.pgm>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    // ── GPU assignment ────────────────────────────────────────────────────────
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA devices found.\n");
        MPI_Finalize();
        return 1;
    }
    int my_gpu = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(my_gpu));

    if (rank == 0)
        printf("[Config] %d MPI rank(s), %d GPU(s) detected — rank i → GPU (i %% %d)\n",
               size, num_gpus, num_gpus);

    // ── Read image + compute scatter layout (rank 0 only) ────────────────────
    int      w = 0, h = 0;
    uint8_t *full_image = NULL;

    int *send_counts = (int *)malloc(size * sizeof(int));
    int *displs      = (int *)malloc(size * sizeof(int));

    if (rank == 0) {
        full_image = read_pgm_p5(argv[1], &w, &h);
        if (!full_image) {
            fprintf(stderr, "Error: could not read %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        long long total = (long long)w * h;
        int rem = (int)(total % size);
        int sum = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = (int)(total / size) + (i < rem ? 1 : 0);
            displs[i]      = sum;
            sum           += send_counts[i];
        }
        printf("[Image]  %d x %d  (%lld pixels)\n", w, h, total);
    }

    // Broadcast image dimensions and scatter layout to all ranks
    MPI_Bcast(&w,          1,    MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h,          1,    MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(send_counts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs,      size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = send_counts[rank];

    // original_chunk holds the unmodified scattered pixels so we can repeat
    // the experiment for each block size without re-scattering
    uint8_t *original_chunk = (uint8_t *)malloc(local_n);
    uint8_t *local_pixels   = (uint8_t *)malloc(local_n);

    MPI_Scatterv(full_image,    send_counts, displs, MPI_UNSIGNED_CHAR,
                 original_chunk, local_n,             MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    // ── Open benchmark file (rank 0, append mode) ─────────────────────────────
    FILE *bench_fp = NULL;
    if (rank == 0) {
        bench_fp = fopen("results/benchmarks/hybrid_benchmarks.txt", "a");
        if (bench_fp) {
            // Write column header only when creating a new file
            fseek(bench_fp, 0, SEEK_END);
            if (ftell(bench_fp) == 0)
                fprintf(bench_fp,
                        "# np block_size h2d_ms hist_ms allreduce_ms remap_ms d2h_ms total_s\n");
        } else {
            fprintf(stderr, "Warning: could not open benchmark file for writing.\n");
        }
    }

    // ── Allocate GPU buffers (once, reused across block-size sweep) ───────────
    uint8_t  *d_pixels = NULL;
    uint32_t *d_hist   = NULL;
    uint8_t  *d_lut    = NULL;

    CUDA_CHECK(cudaMalloc(&d_pixels, (size_t)local_n));
    CUDA_CHECK(cudaMalloc(&d_hist,   L * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_lut,    L));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // ── Block-size sweep ──────────────────────────────────────────────────────
    const int block_sizes[] = {128, 256, 512};
    const int num_bs        = 3;

    for (int bs_idx = 0; bs_idx < num_bs; bs_idx++) {

        int block_size = block_sizes[bs_idx];
        int grid       = (local_n + block_size - 1) / block_size;

        // Restore original (unequalized) chunk for a fair comparison each run
        memcpy(local_pixels, original_chunk, local_n);

        // ── Synchronise all ranks before starting the clock ───────────────────
        MPI_Barrier(MPI_COMM_WORLD);
        double wall_start = MPI_Wtime();

        // ── 1. H→D: copy pixel chunk to GPU ──────────────────────────────────
        CUDA_CHECK(cudaEventRecord(ev_start));
        CUDA_CHECK(cudaMemcpy(d_pixels, local_pixels,
                              (size_t)local_n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float h2d_ms = cuda_elapsed_ms(ev_start, ev_stop);

        // ── 2. GPU histogram kernel ───────────────────────────────────────────
        CUDA_CHECK(cudaMemset(d_hist, 0, L * sizeof(uint32_t)));

        CUDA_CHECK(cudaEventRecord(ev_start));
        histogram_kernel<<<grid, block_size>>>(d_pixels, d_hist, local_n);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        float hist_ms = cuda_elapsed_ms(ev_start, ev_stop);

        // Copy local 256-bin histogram back to CPU (1 KB — negligible)
        uint32_t local_hist[L]  = {0};
        uint32_t global_hist[L] = {0};
        CUDA_CHECK(cudaMemcpy(local_hist, d_hist,
                              L * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // ── 3. MPI_Allreduce: combine histograms across all ranks ─────────────
        double ar_start = MPI_Wtime();
        MPI_Allreduce(local_hist, global_hist, L,
                      MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
        float allreduce_ms = (float)((MPI_Wtime() - ar_start) * 1000.0);

        // ── 4. CDF + LUT on CPU (256 iterations — negligible) ─────────────────
        size_t T = (size_t)w * h;

        uint32_t cdf[L];
        cdf[0] = global_hist[0];
        for (int i = 1; i < L; i++)
            cdf[i] = cdf[i - 1] + global_hist[i];

        uint32_t cdf_min = 0;
        for (int i = 0; i < L; i++) {
            if (cdf[i] != 0) { cdf_min = cdf[i]; break; }
        }

        uint8_t lut[L];
        if (cdf_min == (uint32_t)T) {
            // Constant image — identity LUT
            for (int i = 0; i < L; i++) lut[i] = (uint8_t)i;
        } else {
            for (int i = 0; i < L; i++) {
                if (cdf[i] < cdf_min) { lut[i] = 0; continue; }
                double val = ((double)(cdf[i] - cdf_min) / (double)(T - cdf_min))
                             * (double)(L - 1);
                int m = (int)(val + 0.5);
                lut[i] = (uint8_t)(m < 0 ? 0 : (m > 255 ? 255 : m));
            }
        }

        // ── 5. H→D: copy LUT to GPU (256 bytes — negligible, folded into remap)
        CUDA_CHECK(cudaMemcpy(d_lut, lut, L, cudaMemcpyHostToDevice));

        // ── 6. GPU remap kernel ───────────────────────────────────────────────
        CUDA_CHECK(cudaEventRecord(ev_start));
        remap_kernel<<<grid, block_size>>>(d_pixels, d_lut, local_n);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());
        float remap_ms = cuda_elapsed_ms(ev_start, ev_stop);

        // ── 7. D→H: copy remapped chunk back to CPU ───────────────────────────
        CUDA_CHECK(cudaEventRecord(ev_start));
        CUDA_CHECK(cudaMemcpy(local_pixels, d_pixels,
                              (size_t)local_n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float d2h_ms = cuda_elapsed_ms(ev_start, ev_stop);

        // ── 8. Gather remapped chunks to rank 0 ───────────────────────────────
        MPI_Gatherv(local_pixels, local_n, MPI_UNSIGNED_CHAR,
                    full_image, send_counts, displs, MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double total_s = MPI_Wtime() - wall_start;

        // ── Report results ────────────────────────────────────────────────────
        if (rank == 0) {
            printf("np=%-2d block=%-4d | "
                   "H2D=%6.2fms  hist=%6.2fms  allreduce=%6.3fms  "
                   "remap=%6.2fms  D2H=%6.2fms  | total=%8.4fs\n",
                   size, block_size,
                   h2d_ms, hist_ms, allreduce_ms, remap_ms, d2h_ms,
                   total_s);

            if (bench_fp)
                fprintf(bench_fp, "%d %d %.4f %.4f %.4f %.4f %.4f %.6f\n",
                        size, block_size,
                        h2d_ms, hist_ms, allreduce_ms, remap_ms, d2h_ms,
                        total_s);
        }
    }

    // ── Write final output image (result from last block-size run) ────────────
    if (rank == 0) {
        if (!write_pgm_p5(argv[2], full_image, w, h)) {
            fprintf(stderr, "Error: failed to write output PGM.\n");
        } else {
            system("mkdir -p results/png && convert results/pgm/output_hybrid.pgm results/png/05_output_hybrid.png");
            printf("Output written: %s\n", argv[2]);
        }
        if (bench_fp) fclose(bench_fp);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_pixels);
    cudaFree(d_hist);
    cudaFree(d_lut);

    free(original_chunk);
    free(local_pixels);
    free(send_counts);
    free(displs);
    if (full_image) free(full_image);

    MPI_Finalize();
    return 0;
}
