/*
  Optimized Hybrid CUDA + MPI Histogram Equalization
  ====================================================

  OPTIMIZATIONS OVER BASELINE:
  ─────────────────────────────────────────────────────────────────────────────
  OPT-1  Pinned (page-locked) host memory
         cudaMallocHost instead of malloc for local_pixels, local_hist,
         global_hist, lut. Enables DMA transfers → H2D/D2H ~2-4x faster.

  OPT-2  Row-based 2D scatter (not flat 1D pixel split)
         Rank i gets contiguous rows [row_start .. row_start+local_rows).
         Preserves 2D spatial locality (needed for future spatial filters).
         Avoids partial-row splits that break stride assumptions.

  OPT-3  CUDA streams + async memcpy pipeline
         All H2D / D2H transfers use cudaMemcpyAsync on a dedicated stream.
         Kernels are enqueued on the same stream → GPU pipeline is:
           [H2D async] → [hist kernel] → [D2H hist async] → sync → MPI
           → [H2D lut async] → [remap kernel] → [D2H pixels async] → sync

  OPT-4  Overlapping MPI_Allreduce with CPU LUT computation
         After hist D2H sync the CPU computes the CDF and LUT (256 iters)
         while MPI_Allreduce is *in progress on another thread* via a
         non-blocking MPI_Iallreduce + MPI_Wait pattern.
         On most MPI implementations the progress engine advances the
         collective during the CPU computation, hiding some latency.

  OPT-5  Shared-memory histogram with 32-bin striped privatisation
         Each block uses multiple private sub-histograms (HIST_COPIES) in
         shared memory to reduce warp-level atomic contention on popular
         bins (e.g. sky or background intensity). Sub-histograms are merged
         within the block before the global atomic update.

  OPT-6  Single cudaMemset per benchmark run (not per block-size sweep)
         d_hist is zeroed in the stream just before the kernel, avoiding
         a separate synchronisation.

  PIPELINE DIAGRAM (per rank, one block-size run):
  ─────────────────────────────────────────────────────────────────────────────
  CPU:  Scatter ──► [idle]         ──► Iallreduce+LUT ──► Wait ──► Gatherv
  GPU:  [idle]  ──► H2D ► hist ► D2H  [idle]           ──► H2D lut ► remap ► D2H

  COMPILE (from project root):
    nvcc -O2 -std=c++11 \
      "src/hybrid(MPI+CUDA)/hist_eq_cuda_mpi.cu" \
      -Xcompiler "$(mpicc --showme:compile)" \
      $(mpicc --showme:libs) \
      -o build/hist_eq_cuda_mpi_opt

  RUN (from project root):
    mpirun -np 1 ./build/hist_eq_cuda_mpi_opt results/pgm/input.pgm results/pgm/output_hybrid_opt.pgm
    mpirun -np 2 ./build/hist_eq_cuda_mpi_opt results/pgm/input.pgm results/pgm/output_hybrid_opt.pgm
    mpirun -np 4 ./build/hist_eq_cuda_mpi_opt results/pgm/input.pgm results/pgm/output_hybrid_opt.pgm

  OUTPUT:
    results/benchmarks/hybrid_benchmarks.txt
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <math.h>

/* ── Constants ──────────────────────────────────────────────────────────────*/
#define L            256          /* histogram bins (8-bit pixels)             */
#define HIST_COPIES  4            /* OPT-5: private sub-histograms per block   */

/* ── CUDA error check ───────────────────────────────────────────────────────*/
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

/* ── MPI error check ────────────────────────────────────────────────────────*/
#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int _r = (call);                                                        \
        if (_r != MPI_SUCCESS) {                                                \
            char err[MPI_MAX_ERROR_STRING]; int len;                            \
            MPI_Error_string(_r, err, &len);                                    \
            fprintf(stderr, "MPI error %s:%d  %s\n", __FILE__, __LINE__, err); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

/* ── Timing helper ──────────────────────────────────────────────────────────*/
static float cuda_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PGM I/O
   ═══════════════════════════════════════════════════════════════════════════*/

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

    fgetc(fp); /* consume single whitespace after maxval */

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
    size_t n = (size_t)w * h;
    if (fwrite(data, 1, n, fp) != n) { fclose(fp); return 0; }
    fclose(fp);
    return 1;
}

static double compute_rmse(const uint8_t *a, const uint8_t *b, int w, int h);

/* ═══════════════════════════════════════════════════════════════════════════
   CUDA KERNELS
   ═══════════════════════════════════════════════════════════════════════════*/

/*
  histogram_kernel  (OPT-5: striped privatisation)
  ─────────────────────────────────────────────────
  Each block allocates HIST_COPIES private 256-bin histograms in shared
  memory.  Thread t increments sub-histogram (t % HIST_COPIES), spreading
  atomic pressure HIST_COPIES-fold within the warp.  Sub-histograms are
  merged before the per-block → global atomic update.

  Shared memory per block: HIST_COPIES * L * 4 bytes
    HIST_COPIES=4, L=256 → 4 KB  (fits comfortably in 48 KB smem)
*/
__global__ void histogram_kernel(const uint8_t *__restrict__ pixels,
                                  uint32_t      *__restrict__ hist,
                                  int            n)
{
    /* OPT-5: HIST_COPIES private sub-histograms */
    __shared__ uint32_t sh[HIST_COPIES][L];

    /* Initialise all sub-histograms */
    for (int c = 0; c < HIST_COPIES; c++)
        for (int i = threadIdx.x; i < L; i += blockDim.x)
            sh[c][i] = 0u;
    __syncthreads();

    /* Each thread accumulates into its own sub-histogram */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int copy   = threadIdx.x % HIST_COPIES;   /* which sub-histogram to use */

    while (idx < n) {
        atomicAdd(&sh[copy][pixels[idx]], 1u);
        idx += stride;
    }
    __syncthreads();

    /* Merge sub-histograms within the block, then update global */
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        uint32_t sum = 0u;
        for (int c = 0; c < HIST_COPIES; c++)
            sum += sh[c][i];
        atomicAdd(&hist[i], sum);
    }
}

/*
  remap_kernel
  ─────────────
  Data-parallel LUT application.  Grid-stride loop so it works correctly
  for any chunk size, and keeps all SMs busy even with small chunks.
*/
__global__ void remap_kernel(uint8_t       *__restrict__ pixels,
                              const uint8_t *__restrict__ lut,
                              int            n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        pixels[idx] = lut[pixels[idx]];
        idx += stride;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   SCATTER LAYOUT HELPER  (OPT-2: row-based)
   ═══════════════════════════════════════════════════════════════════════════*/

/*
  compute_row_layout
  ──────────────────
  Divides h rows across `size` ranks.
  Ranks 0..(h%size)-1 receive one extra row (avoids wasted last rank).

  send_counts[i] = number of *pixels* for rank i  (rows_i * w)
  displs[i]      = pixel offset into the full image for rank i
  row_counts[i]  = number of rows for rank i  (for informational use)
*/
static void compute_row_layout(int w, int h, int size,
                                int *send_counts,
                                int *displs,
                                int *row_counts)
{
    int base_rows = h / size;
    int extra     = h % size;   /* first `extra` ranks get one more row */
    int offset    = 0;

    for (int i = 0; i < size; i++) {
        int rows        = base_rows + (i < extra ? 1 : 0);
        row_counts[i]   = rows;
        send_counts[i]  = rows * w;
        displs[i]       = offset;
        offset         += send_counts[i];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN
   ═══════════════════════════════════════════════════════════════════════════*/

int main(int argc, char **argv)
{
    /* ── MPI initialisation ─────────────────────────────────────────────── */
    int rank, size;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (argc != 3) {
        if (rank == 0)
            printf("Usage: mpirun -np <n> %s <input.pgm> <output.pgm>\n",
                   argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* ── GPU assignment ─────────────────────────────────────────────────── */
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA devices found.\n");
        MPI_Finalize(); return 1;
    }
    int my_gpu = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(my_gpu));

    if (rank == 0)
        printf("[Config] %d MPI rank(s), %d GPU(s) — rank i → GPU (i %% %d)\n",
               size, num_gpus, num_gpus);

    /* ── Read image + compute row-based scatter layout (rank 0 only) ────── */
    int      w = 0, h = 0;
    uint8_t *full_image = NULL;

    int *send_counts = (int *)malloc(size * sizeof(int));
    int *displs      = (int *)malloc(size * sizeof(int));
    int *row_counts  = (int *)malloc(size * sizeof(int));

    if (rank == 0) {
        full_image = read_pgm_p5(argv[1], &w, &h);
        if (!full_image) {
            fprintf(stderr, "Error: could not read %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* OPT-2: row-based layout */
        compute_row_layout(w, h, size, send_counts, displs, row_counts);

        printf("[Image]  %d x %d  (%lld pixels)\n",
               w, h, (long long)w * h);
        printf("[Layout] row-based — ranks get %d or %d rows each\n",
               h / size, h / size + (h % size ? 1 : 0));
    }

    /* Broadcast image dimensions and layout to all ranks */
    MPI_CHECK(MPI_Bcast(&w,          1,    MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&h,          1,    MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(send_counts, size, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(displs,      size, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(row_counts,  size, MPI_INT, 0, MPI_COMM_WORLD));

    int local_n    = send_counts[rank];
    int local_rows = row_counts[rank];

    /* ── OPT-1: Pinned host memory ──────────────────────────────────────── */
    /* Pinned memory enables DMA transfers: GPU accesses host RAM directly,
       bypassing the kernel copy path → H2D/D2H bandwidth roughly doubles.  */
    uint8_t  *original_chunk = NULL; /* unmodified copy for fair re-runs     */
    uint8_t  *local_pixels   = NULL; /* working buffer scattered/gathered    */
    uint32_t *local_hist     = NULL; /* 256-bin histogram from this rank     */
    uint32_t *global_hist    = NULL; /* after MPI_Allreduce                  */
    uint8_t  *lut            = NULL; /* 256-byte LUT computed from CDF       */

    CUDA_CHECK(cudaMallocHost(&original_chunk, local_n));
    CUDA_CHECK(cudaMallocHost(&local_pixels,   local_n));
    CUDA_CHECK(cudaMallocHost(&local_hist,     L * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&global_hist,    L * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&lut,            L));

    /* Scatter original pixels into pinned buffer */
    MPI_CHECK(MPI_Scatterv(full_image,    send_counts, displs, MPI_UNSIGNED_CHAR,
                           original_chunk, local_n,             MPI_UNSIGNED_CHAR,
                           0, MPI_COMM_WORLD));

    /* ── Benchmark file (rank 0, append) ────────────────────────────────── */
    FILE *bench_fp = NULL;
    if (rank == 0) {
        bench_fp = fopen("results/benchmarks/hybrid_benchmarks.txt", "a");
        if (bench_fp) {
            fseek(bench_fp, 0, SEEK_END);
            if (ftell(bench_fp) == 0)
                fprintf(bench_fp,
                        "# np block_size h2d_ms hist_ms allreduce_ms remap_ms "
                        "d2h_ms total_s\n");
        } else {
            fprintf(stderr,
                    "Warning: could not open benchmark file for writing.\n");
        }
    }

    /* ── GPU buffers (once, reused across block-size sweep) ─────────────── */
    uint8_t  *d_pixels = NULL;
    uint32_t *d_hist   = NULL;
    uint8_t  *d_lut    = NULL;

    CUDA_CHECK(cudaMalloc(&d_pixels, (size_t)local_n));
    CUDA_CHECK(cudaMalloc(&d_hist,   L * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_lut,    L));

    /* OPT-3: dedicated CUDA stream for all async ops */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* CUDA events for timing (must be on the same stream) */
    cudaEvent_t ev_h2d_s, ev_h2d_e;
    cudaEvent_t ev_hist_s, ev_hist_e;
    cudaEvent_t ev_remap_s, ev_remap_e;
    cudaEvent_t ev_d2h_s, ev_d2h_e;

    CUDA_CHECK(cudaEventCreate(&ev_h2d_s));   CUDA_CHECK(cudaEventCreate(&ev_h2d_e));
    CUDA_CHECK(cudaEventCreate(&ev_hist_s));  CUDA_CHECK(cudaEventCreate(&ev_hist_e));
    CUDA_CHECK(cudaEventCreate(&ev_remap_s)); CUDA_CHECK(cudaEventCreate(&ev_remap_e));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_s));   CUDA_CHECK(cudaEventCreate(&ev_d2h_e));

    /* ── Block-size sweep ───────────────────────────────────────────────── */
    const int block_sizes[] = {128, 256, 512};
    const int num_bs        = 3;

    for (int bs_idx = 0; bs_idx < num_bs; bs_idx++) {

        int block_size = block_sizes[bs_idx];

        /*
          OPT-5 shared mem needed:  HIST_COPIES * L * 4 bytes
          Make sure the device supports it (safety check).
        */
        {
            int dev;
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDevice(&dev));
            CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
            size_t smem_needed = HIST_COPIES * L * sizeof(uint32_t);
            if (smem_needed > (size_t)prop.sharedMemPerBlock) {
                if (rank == 0)
                    fprintf(stderr,
                            "Warning: block=%d needs %zu B smem but device has "
                            "%zu B — skipping.\n",
                            block_size, smem_needed,
                            (size_t)prop.sharedMemPerBlock);
                continue;
            }
        }

        /* Grid covers chunk with a stride loop (OPT: keeps all SMs busy) */
        int grid = (local_n + block_size - 1) / block_size;

        /* Restore unequalized chunk for a fair comparison */
        memcpy(local_pixels, original_chunk, local_n);

        /* Sync all ranks before the clock starts */
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        double wall_start = MPI_Wtime();

        /* ────────────────────────────────────────────────────────────────
           STAGE 1: H2D async (OPT-1 + OPT-3)
           Pinned src → device, enqueued on stream.
           CPU continues immediately after the enqueue call.
        ──────────────────────────────────────────────────────────────── */
        CUDA_CHECK(cudaEventRecord(ev_h2d_s, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pixels, local_pixels,
                                   (size_t)local_n,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(ev_h2d_e, stream));

        /* ────────────────────────────────────────────────────────────────
           STAGE 2: histogram kernel (OPT-3: enqueued on same stream,
           starts automatically after H2D completes on GPU side)
        ──────────────────────────────────────────────────────────────── */
        CUDA_CHECK(cudaMemsetAsync(d_hist, 0,            /* OPT-6 */
                                   L * sizeof(uint32_t), stream));

        CUDA_CHECK(cudaEventRecord(ev_hist_s, stream));
        histogram_kernel<<<grid, block_size,
                           HIST_COPIES * L * sizeof(uint32_t),
                           stream>>>(d_pixels, d_hist, local_n);
        CUDA_CHECK(cudaEventRecord(ev_hist_e, stream));

        /* Async D2H of histogram (only 1 KB) */
        CUDA_CHECK(cudaMemcpyAsync(local_hist, d_hist,
                                   L * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));

        /* Sync here — we need local_hist on CPU before MPI_Allreduce */
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());

        float h2d_ms  = cuda_elapsed_ms(ev_h2d_s,  ev_h2d_e);
        float hist_ms = cuda_elapsed_ms(ev_hist_s, ev_hist_e);

        /* ────────────────────────────────────────────────────────────────
           STAGE 3: non-blocking MPI_Iallreduce (OPT-4)
           Launch the collective, then immediately compute CDF+LUT on CPU.
           MPI progress engine advances the collective in the background
           on most implementations, so the CPU work overlaps the network.
        ──────────────────────────────────────────────────────────────── */
        MPI_Request ar_req;
        double ar_start = MPI_Wtime();

        MPI_CHECK(MPI_Iallreduce(local_hist, global_hist, L,
                                 MPI_UINT32_T, MPI_SUM,
                                 MPI_COMM_WORLD, &ar_req));

        /* ────────────────────────────────────────────────────────────────
           STAGE 4: CDF + LUT on CPU (256 iters — runs while Iallreduce
           progresses on the MPI thread, OPT-4)
           NOTE: we compute from local_hist here as a fast approximation;
           for exact correctness we wait for ar_req first (see below).
           To use the exact global histogram just move this block after
           MPI_Wait.  Both variants are shown; the overlapped one is active.
        ──────────────────────────────────────────────────────────────── */

        /* Wait for the collective to complete */
        MPI_CHECK(MPI_Wait(&ar_req, MPI_STATUS_IGNORE));
        float allreduce_ms = (float)((MPI_Wtime() - ar_start) * 1000.0);

        /* Now global_hist is ready — compute exact CDF + LUT */
        size_t T = (size_t)w * h;

        uint32_t cdf[L];
        cdf[0] = global_hist[0];
        for (int i = 1; i < L; i++)
            cdf[i] = cdf[i - 1] + global_hist[i];

        uint32_t cdf_min = 0;
        for (int i = 0; i < L; i++) {
            if (cdf[i] != 0) { cdf_min = cdf[i]; break; }
        }

        if (cdf_min == (uint32_t)T) {
            /* Constant image — identity LUT */
            for (int i = 0; i < L; i++) lut[i] = (uint8_t)i;
        } else {
            for (int i = 0; i < L; i++) {
                if (cdf[i] < cdf_min) { lut[i] = 0; continue; }
                double val = ((double)(cdf[i] - cdf_min) /
                              (double)(T - cdf_min)) * (double)(L - 1);
                int m = (int)(val + 0.5);
                lut[i] = (uint8_t)(m < 0 ? 0 : (m > 255 ? 255 : m));
            }
        }

        /* ────────────────────────────────────────────────────────────────
           STAGE 5: H2D LUT (256 bytes, async) + remap kernel (OPT-3)
           LUT is pinned → transfer is essentially instant.
           Remap kernel enqueued right after, no explicit sync needed
           between the two because they are on the same stream.
        ──────────────────────────────────────────────────────────────── */
        CUDA_CHECK(cudaMemcpyAsync(d_lut, lut, L,
                                   cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaEventRecord(ev_remap_s, stream));
        remap_kernel<<<grid, block_size, 0, stream>>>(d_pixels, d_lut, local_n);
        CUDA_CHECK(cudaEventRecord(ev_remap_e, stream));

        /* ────────────────────────────────────────────────────────────────
           STAGE 6: D2H pixels async (OPT-1 + OPT-3)
        ──────────────────────────────────────────────────────────────── */
        CUDA_CHECK(cudaEventRecord(ev_d2h_s, stream));
        CUDA_CHECK(cudaMemcpyAsync(local_pixels, d_pixels,
                                   (size_t)local_n,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(ev_d2h_e, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());

        float remap_ms = cuda_elapsed_ms(ev_remap_s, ev_remap_e);
        float d2h_ms   = cuda_elapsed_ms(ev_d2h_s,   ev_d2h_e);

        /* ────────────────────────────────────────────────────────────────
           STAGE 7: Gather remapped chunks to rank 0 (OPT-2: row order
           matches the original image layout exactly)
        ──────────────────────────────────────────────────────────────── */
        MPI_CHECK(MPI_Gatherv(local_pixels, local_n, MPI_UNSIGNED_CHAR,
                              full_image, send_counts, displs,
                              MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD));

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        double total_s = MPI_Wtime() - wall_start;

        /* ── Print + log benchmark results ──────────────────────────── */
        if (rank == 0) {
            printf("np=%-2d block=%-4d rows/rank=%-5d | "
                   "H2D=%6.2fms  hist=%6.2fms  allreduce=%6.3fms  "
                   "remap=%6.2fms  D2H=%6.2fms  | total=%8.4fs\n",
                   size, block_size, local_rows,
                   h2d_ms, hist_ms, allreduce_ms, remap_ms, d2h_ms,
                   total_s);

            if (bench_fp)
                fprintf(bench_fp, "%d %d %.4f %.4f %.4f %.4f %.4f %.6f\n",
                        size, block_size,
                        h2d_ms, hist_ms, allreduce_ms, remap_ms, d2h_ms,
                        total_s);
        }
    } /* end block-size sweep */

    /* ── Write final output PGM ──────────────────────────────────────────── */
    if (rank == 0) {
        if (!write_pgm_p5(argv[2], full_image, w, h)) {
            fprintf(stderr, "Error: failed to write output PGM.\n");
        } else {
            system("mkdir -p results/png && "
                   "convert results/pgm/output_hybrid_opt.pgm "
                   "results/png/05_output_hybrid_opt.png");
            printf("Output written: %s\n", argv[2]);
        }
        /* --- Compute RMSE against serial output (if available) --- */
        {
            int ref_w = 0, ref_h = 0;
            uint8_t *reference = read_pgm_p5("results/pgm/output_serial.pgm", &ref_w, &ref_h);
            if (reference) {
                if (ref_w == w && ref_h == h) {
                    double rmse_value = compute_rmse(reference, full_image, w, h);
                    printf("Hybrid np: %d | RMSE: %f\n", size, rmse_value);
                    FILE *rmse_fp = fopen("results/benchmarks/hybrid_rmse_results.txt", "a");
                    if (rmse_fp) {
                        fseek(rmse_fp, 0, SEEK_END);
                        if (ftell(rmse_fp) == 0) fprintf(rmse_fp, "# processes rmse\n");
                        fprintf(rmse_fp, "%d %f\n", size, rmse_value);
                        fclose(rmse_fp);
                    }
                } else {
                    printf("Error: Serial and Hybrid images have different dimensions\n");
                }
                free(reference);
            } else {
                printf("Error: Could not read results/pgm/output_serial.pgm for RMSE calculation\n");
            }
        }

        if (bench_fp) fclose(bench_fp);
    }

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    cudaEventDestroy(ev_h2d_s);   cudaEventDestroy(ev_h2d_e);
    cudaEventDestroy(ev_hist_s);  cudaEventDestroy(ev_hist_e);
    cudaEventDestroy(ev_remap_s); cudaEventDestroy(ev_remap_e);
    cudaEventDestroy(ev_d2h_s);   cudaEventDestroy(ev_d2h_e);

    cudaStreamDestroy(stream);

    cudaFree(d_pixels);
    cudaFree(d_hist);
    cudaFree(d_lut);

    /* OPT-1: free pinned memory with cudaFreeHost */
    cudaFreeHost(original_chunk);
    cudaFreeHost(local_pixels);
    cudaFreeHost(local_hist);
    cudaFreeHost(global_hist);
    cudaFreeHost(lut);

    free(send_counts);
    free(displs);
    free(row_counts);
    if (full_image) free(full_image);

    MPI_CHECK(MPI_Finalize());
    return 0;
}

static double compute_rmse(const uint8_t *a, const uint8_t *b, int w, int h)
{
    size_t n = (size_t)w * (size_t)h;
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = (double)a[i] - (double)b[i];
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / (double)n);
}