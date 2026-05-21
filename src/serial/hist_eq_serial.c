/*
  Serial Histogram Equalization (Grayscale, 8-bit) for PGM (P5) images.

  If no build folder; create it using : 
    mkdir -p build
    
  Build (Linux/macOS):
    gcc -O2 -std=c11 src/serial/hist_eq_serial.c -o build/hist_eq_serial

  Run (from project root):
    ./build/hist_eq_serial results/pgm/input.pgm results/pgm/output_serial.pgm
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <time.h>

#define L 256

// --------- Minimal PGM (P5) Reader/Writer ---------

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

// --------- Histogram Equalization (Serial) ---------

static void hist_equalize_u8(uint8_t *img, int w, int h) {
    const size_t T = (size_t)w * (size_t)h;

    // 1) Histogram
    uint32_t hist[L] = {0};
    for (size_t i = 0; i < T; i++) {
        hist[img[i]]++;
    }

    // 2) CDF (using counts to avoid float error)
    uint32_t cdf[L] = {0};
    cdf[0] = hist[0];
    for (int k = 1; k < L; k++) {
        cdf[k] = cdf[k - 1] + hist[k];
    }

    // Find CDFmin = first non-zero cdf value
    uint32_t cdf_min = 0;
    for (int k = 0; k < L; k++) {
        if (cdf[k] != 0) {
            cdf_min = cdf[k];
            break;
        }
    }

    // Edge case: image is constant (all pixels same intensity)
    if (cdf_min == (uint32_t)T) {
        // nothing to equalize, keep as-is
        return;
    }

    // 3) Build LUT using CDFmin-normalized mapping:
    // s_k = floor( (cdf[k] - cdf_min) / (T - cdf_min) * (L-1) )
    uint8_t lut[L];
    for (int k = 0; k < L; k++) {
        if (cdf[k] < cdf_min) {
            lut[k] = 0;
        } else {
            double num = (double)(cdf[k] - cdf_min);
            double den = (double)(T - cdf_min);
            double val = (num / den) * (double)(L - 1);
            int mapped = (int)(val + 0.5); // round to nearest
            if (mapped < 0) mapped = 0;
            if (mapped > 255) mapped = 255;
            lut[k] = (uint8_t)mapped;
        }
    }

    // 4) Remap pixels
    for (size_t i = 0; i < T; i++) {
        img[i] = lut[img[i]];
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s results/pgm/input.pgm results/pgm/output_serial.pgm\n", argv[0]);
        return 1;
    }

    const char *in_path  = argv[1];
    const char *out_path = argv[2];

    int w = 0, h = 0, maxval = 0;
    uint8_t *img = read_pgm_p5(in_path, &w, &h, &maxval);
    if (!img) return 1;

    // -------- Start Timing --------
    clock_t start = clock();

    hist_equalize_u8(img, w, h);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Serial Execution Time: %f seconds\n", time_spent);

    /* ensure benchmarks directory exists and persist the serial time */
    (void)system("mkdir -p results/benchmarks");
    FILE *sf = fopen("results/benchmarks/serial_execution_results.txt", "w");
    if (sf) {
        fprintf(sf, "%.6f\n", time_spent);
        fclose(sf);
    } else {
        fprintf(stderr, "Warning: could not write serial baseline file\n");
    }

    if (!write_pgm_p5(out_path, img, w, h)) {
    free(img);
    return 1;
   }

// convert to PNG automatically
system("convert results/pgm/output_serial.pgm results/png/02_output_serial.png");

    free(img);
    return 0;
}