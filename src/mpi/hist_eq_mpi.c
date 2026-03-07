#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>

#define L 256

static void skip_comments_and_whitespace(FILE *fp) {
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (isspace(c)) continue;
        if (c == '#') {
            // skip to end of line
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
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    char magic[3] = {0};
    if (fread(magic, 1, 2, fp) != 2) {
        fclose(fp);
        return NULL;
    }
    magic[2] = '\0';

    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Only binary PGM (P5) supported.\n");
        fclose(fp);
        return NULL;
    }

    if (!read_int(fp, w) || !read_int(fp, h) || !read_int(fp, maxval)) {
        fprintf(stderr, "Error: Failed to read PGM header.\n");
        fclose(fp);
        return NULL;
    }

    if (*w <= 0 || *h <= 0) {
        fprintf(stderr, "Error: Invalid image dimensions.\n");
        fclose(fp);
        return NULL;
    }

    if (*maxval != 255) {
        fprintf(stderr, "Error: Only 8-bit PGM with maxval=255 supported (got %d).\n", *maxval);
        fclose(fp);
        return NULL;
    }

    // Consume single whitespace after maxval
    fgetc(fp);

    size_t n = (size_t)(*w) * (size_t)(*h);
    uint8_t *data = (uint8_t *)malloc(n);
    if (!data) {
        fprintf(stderr, "Error: malloc failed.\n");
        fclose(fp);
        return NULL;
    }

    if (fread(data, 1, n, fp) != n) {
        fprintf(stderr, "Error: Failed to read pixel data.\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return data;
}
static int write_pgm_p5(const char *path, const uint8_t *data, int w, int h) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        perror("fopen");
        return 0;
    }

    fprintf(fp, "P5\n%d %d\n255\n", w, h);
    size_t n = (size_t)w * (size_t)h;

    if (fwrite(data, 1, n, fp) != n) {
        fprintf(stderr, "Error: Failed to write pixel data.\n");
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) fprintf(stderr, "Usage: mpirun -np <p> %s input.pgm output.pgm\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int w, h, maxval;
    uint8_t *full_image = NULL;
    uint8_t *local_image = NULL;

    // --- PHASE 1: LOAD DATA (Rank 0 Only) ---
    if (rank == 0) {
        // Use your existing read_pgm_p5 function here
        full_image = read_pgm_p5(argv[1], &w, &h, &maxval);
        if (!full_image) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast image dimensions so everyone knows how much memory to allocate
    MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate chunks (Assuming h is divisible by size for simplicity)
    int rows_per_rank = h / size;
    size_t chunk_size = (size_t)w * rows_per_rank;
    local_image = (uint8_t *)malloc(chunk_size);

    // --- PHASE 2: DISTRIBUTE DATA ---
    MPI_Scatter(full_image, chunk_size, MPI_UINT8_T, 
                local_image, chunk_size, MPI_UINT8_T, 
                0, MPI_COMM_WORLD);

    // --- PHASE 3: PARALLEL HISTOGRAM ---
    uint32_t local_hist[L] = {0};
    for (size_t i = 0; i < chunk_size; i++) {
        local_hist[local_image[i]]++;
    }

    // Global Reduction: Sum all local histograms into one global histogram
    uint32_t global_hist[L] = {0};
    MPI_Allreduce(local_hist, global_hist, L, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

    // --- PHASE 4: EQUALIZATION LOGIC (Identical on all ranks) ---
    size_t T = (size_t)w * (size_t)h;
    uint32_t cdf[L] = {0};
    cdf[0] = global_hist[0];
    for (int k = 1; k < L; k++) cdf[k] = cdf[k - 1] + global_hist[k];

    uint32_t cdf_min = 0;
    for (int k = 0; k < L; k++) {
        if (cdf[k] != 0) { cdf_min = cdf[k]; break; }
    }

    uint8_t lut[L];
    for (int k = 0; k < L; k++) {
        if (cdf[k] < cdf_min) lut[k] = 0;
        else {
            double val = ((double)(cdf[k] - cdf_min) / (double)(T - cdf_min)) * (L - 1);
            int mapped = (int)(val + 0.5);
            lut[k] = (uint8_t)(mapped > 255 ? 255 : (mapped < 0 ? 0 : mapped));
        }
    }

    // Apply LUT to local chunk
    for (size_t i = 0; i < chunk_size; i++) {
        local_image[i] = lut[local_image[i]];
    }

    // --- PHASE 5: GATHER AND SAVE ---
    MPI_Gather(local_image, chunk_size, MPI_UINT8_T, 
               full_image, chunk_size, MPI_UINT8_T, 
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        write_pgm_p5(argv[2], full_image, w, h);
        free(full_image);
    }

    free(local_image);
    MPI_Finalize();
    return 0;
}