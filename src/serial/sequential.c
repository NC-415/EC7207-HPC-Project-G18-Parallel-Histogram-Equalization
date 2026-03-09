#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Include stb_image for reading and writing PNGs in pure C
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// --- Pure C Sequential Implementation ---
void histEqSequential(const unsigned char* input, unsigned char* output, int num_pixels) {
    int hist[256] = {0};
    int cdf[256] = {0};
    int map[256] = {0};
    
    // 1. Calculate Histogram
    for (int i = 0; i < num_pixels; i++) {
        hist[input[i]]++;
    }
    
    // 2. Calculate CDF
    cdf[0] = hist[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i - 1] + hist[i];
    }
    
    // Find minimum non-zero CDF value
    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] > 0) { 
            cdf_min = cdf[i]; 
            break; 
        }
    }
    
    // 3. Create Mapping
    for (int i = 0; i < 256; i++) {
        map[i] = (int)round((float)(cdf[i] - cdf_min) / (num_pixels - cdf_min) * 255.0f);
    }
    
    // 4. Map the new pixels
    for (int i = 0; i < num_pixels; i++) {
        output[i] = map[input[i]];
    }
}

int main() {
// Use the Linux path with forward slashes
const char* base_path = "/mnt/c/Users/NC/Desktop/HPC/dataset/";
    
// Update the names to match your actual files exactly
const char* filenames[] = {
    "test_gray (1).png", 
    "test_gray (2).png", 
    "test_gray (3).png", 
    "test_gray (4).png", 
    "test_gray (5).png"
};
int num_files = 5;


    printf("Starting Pure C Sequential Evaluation...\n");
    printf("------------------------------------------------------------\n");

    for (int f = 0; f < num_files; f++) {
        // Create the full file paths
        char full_path[512];
        char output_path[512];
        snprintf(full_path, sizeof(full_path), "%s%s", base_path, filenames[f]);
        snprintf(output_path, sizeof(output_path), "%sseq_c_equalized_%s", base_path, filenames[f]);

        int width, height, original_channels;

        // 1. Load the image and force it to 1 channel (Grayscale)
        unsigned char* input_image = stbi_load(full_path, &width, &height, &original_channels, 1);
        
        if (input_image == NULL) {
            printf("Error: Cannot open %s - Check if the file exists.\n", full_path);
            continue;
        }

        int num_pixels = width * height;
        
        // Allocate memory for the output image
        unsigned char* output_image = (unsigned char*)malloc(num_pixels * sizeof(unsigned char));
        if (output_image == NULL) {
            printf("Error: Memory allocation failed!\n");
            stbi_image_free(input_image);
            continue;
        }

        printf("Processing: %s (%dx%d)\n", filenames[f], width, height);

        // 2. Start Timer using standard C time library
        clock_t start_time = clock();
        
        // 3. Run Custom Algorithm
        histEqSequential(input_image, output_image, num_pixels);
        
        // 4. End Timer
        clock_t end_time = clock();
        double seq_time_ms = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000.0;

        // 5. Print Execution Time
        printf("  Execution Time: %.2f ms\n", seq_time_ms);
        
        // 6. Save the output PNG image
        stbi_write_png(output_path, width, height, 1, output_image, width);
        
        printf("  Saved output to: seq_c_equalized_%s\n", filenames[f]);
        printf("------------------------------------------------------------\n");

        // 7. Free allocated memory
        stbi_image_free(input_image);
        free(output_image);
    }

    printf("Sequential processing complete.\n");
    return 0;
}