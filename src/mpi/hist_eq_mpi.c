/*
  MPI Parallel Histogram Equalization

  Compile (from project root):
    mpicc -O2 -std=c11 src/mpi/hist_eq_mpi.c -o build/hist_eq_mpi

  Run (from project root):
    mpirun -np 1 ./build/hist_eq_mpi results/pgm/input.pgm results/pgm/output_mpi.pgm

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

#define L 256

// ---------- PGM Reader/Writer Helpers ----------

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

static int read_int(FILE *fp, int *out) {
    skip_comments_and_whitespace(fp);
    return fscanf(fp, "%d", out) == 1;
}

static uint8_t *read_pgm_p5(const char *path, int *w, int *h, int *maxval) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    char magic[3];
    if (fread(magic, 1, 2, fp) != 2) { fclose(fp); return NULL; }
    magic[2] = '\0';
    if (strcmp(magic, "P5") != 0) { fclose(fp); return NULL; }
    if (!read_int(fp, w) || !read_int(fp, h) || !read_int(fp, maxval)) { fclose(fp); return NULL; }
    fgetc(fp); // consume single whitespace
    size_t n = (size_t)(*w) * (*h);
    uint8_t *data = malloc(n);
    if (fread(data, 1, n, fp) != n) { free(data); fclose(fp); return NULL; }
    fclose(fp);
    return data;
}

static int write_pgm_p5(const char *path, const uint8_t *data, int w, int h) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    fprintf(fp, "P5\n%d %d\n255\n", w, h);
    fwrite(data, 1, (size_t)w * h, fp);
    fclose(fp);
    return 1;
}

// ---------- Main MPI Logic ----------

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) printf("Usage: mpirun -np <n> %s <input.pgm> <output.pgm>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int w, h, maxval;
    uint8_t *full_image = NULL;

    // FIX 1: Allocate metadata arrays on ALL ranks so MPI_Bcast has a valid destination
    int *send_counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    if (rank == 0) {
        full_image = read_pgm_p5(argv[1], &w, &h, &maxval);
        if (!full_image) {
            fprintf(stderr, "Error: Could not read image %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // FIX 2: Use long long to handle very large images (prevent overflow)
        long long total_pixels = (long long)w * h;
        int rem = total_pixels % size;
        int sum = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = (int)(total_pixels / size) + (i < rem ? 1 : 0);
            displs[i] = sum;
            sum += send_counts[i];
        }
    }

    // Broadcast metadata to all processes
    MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(send_counts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = send_counts[rank];
    uint8_t *local_pixels = malloc(local_n);

    // --- Start Timing ---
    double start_time = MPI_Wtime();

    // 1. Distribute image chunks using Scatterv (handles remainders)
    MPI_Scatterv(full_image, send_counts, displs, MPI_UNSIGNED_CHAR,
                 local_pixels, local_n, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // 2. Compute Local Histogram
    uint32_t local_hist[L] = {0};
    for (int i = 0; i < local_n; i++) {
        local_hist[local_pixels[i]]++;
    }

    // 3. Combine to Global Histogram
    uint32_t global_hist[L] = {0};
    MPI_Allreduce(local_hist, global_hist, L, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

    // 4. Compute CDF and LUT
    uint32_t cdf[L];
    size_t T = (size_t)w * h;
    cdf[0] = global_hist[0];
    for (int i = 1; i < L; i++) cdf[i] = cdf[i - 1] + global_hist[i];

    uint32_t cdf_min = 0;
    for (int i = 0; i < L; i++) {
        if (cdf[i] != 0) { cdf_min = cdf[i]; break; }
    }

    uint8_t lut[L];
    if (cdf_min != (uint32_t)T) {
        for (int i = 0; i < L; i++) {
            if (cdf[i] < cdf_min) lut[i] = 0;
            else {
                double val = ((double)(cdf[i] - cdf_min) / (T - cdf_min)) * (L - 1);
                int mapped = (int)(val + 0.5);
                lut[i] = (uint8_t)(mapped > 255 ? 255 : (mapped < 0 ? 0 : mapped));
            }
        }

        // 5. Parallel Mapping
        for (int i = 0; i < local_n; i++) {
            local_pixels[i] = lut[local_pixels[i]];
        }
    }

    // 6. Gather results back to Rank 0
    MPI_Gatherv(local_pixels, local_n, MPI_UNSIGNED_CHAR,
                full_image, send_counts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    // --- End Timing ---

   if (rank == 0) {
        printf("MPI Processes: %d | Time: %f seconds\n", size, end_time - start_time);
        
        // Open for appending ("a") to collect all experiment results in one file
        FILE *res_fp = fopen("results/benchmarks/mpi_benchmarks.txt", "a");
        if(res_fp) {
            // FORMAT: <Number of Processes> <Execution Time>
            fprintf(res_fp, "%d %f\n", size, end_time - start_time);
            fclose(res_fp);
        } else {
            fprintf(stderr, "Error: Could not open benchmarks file.\n");
        }
        
        write_pgm_p5(argv[2], full_image, w, h);

            // convert to PNG automatically
system("convert results/pgm/output_mpi.pgm results/png/04_output_mpi.png");

        free(full_image);
    }       

    free(local_pixels);
    free(send_counts);
    free(displs);
    MPI_Finalize();
    return 0;
}