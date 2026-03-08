/*
  MPI Parallel Histogram Equalization


  Compile:
    mpicc -O2 -std=c11  src/openmp/hist_eq_openmp.c -o build/hist_eq_mpi

  Run:
    mpirun -np 4 ./build/hist_eq_mpi output/input.pgm output/output_mpi.pgm
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

#define L 256

// ---------- PGM Reader/Writer (Same as your Serial/OpenMP) ----------

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
    fread(magic, 1, 2, fp);
    magic[2] = '\0';
    if (strcmp(magic, "P5") != 0) { fclose(fp); return NULL; }
    read_int(fp, w); read_int(fp, h); read_int(fp, maxval);
    fgetc(fp);
    size_t n = (size_t)(*w) * (*h);
    uint8_t *data = malloc(n);
    fread(data, 1, n, fp);
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

// ---------- MPI Logic ----------

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) printf("Usage: %s <input.pgm> <output.pgm>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int w, h, maxval;
    uint8_t *full_image = NULL;
    int *send_counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    // Rank 0 handles File I/O
    if (rank == 0) {
        full_image = read_pgm_p5(argv[1], &w, &h, &maxval);
        if (!full_image) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Calculate chunks for each process (handles remainders)
        int total_pixels = w * h;
        int rem = total_pixels % size;
        int sum = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = total_pixels / size + (i < rem ? 1 : 0);
            displs[i] = sum;
            sum += send_counts[i];
        }
    }

    // Broadcast metadata
    MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(send_counts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = send_counts[rank];
    uint8_t *local_pixels = malloc(local_n);

    // --- Start Timing ---
    double start_time = MPI_Wtime();

    // 1. Distribute image chunks
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

    // 4. Compute CDF and LUT (identical to your OpenMP logic)
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
                if (mapped < 0) mapped = 0;
                if (mapped > 255) mapped = 255;
                lut[i] = (uint8_t)mapped;
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
        printf("MPI Processes: %d Time: %f\n", size, end_time - start_time);
        
        // Save results to file similar to your OpenMP benchmarking
        FILE *res_fp = fopen("results/benchmarks/mpi_execution_results.txt", "a");
        if(res_fp) {
            fprintf(res_fp, "%d %f\n", size, end_time - start_time);
            fclose(res_fp);
        }
        
        write_pgm_p5(argv[2], full_image, w, h);
        free(full_image);
    }

    free(local_pixels);
    free(send_counts);
    free(displs);
    MPI_Finalize();
    return 0;
}