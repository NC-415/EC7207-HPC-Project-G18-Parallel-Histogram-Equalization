/*
 Parallel Histogram Equalization using OpenMP
 Based on the serial PGM implementation

 Compile:
   gcc -O2 -fopenmp -std=c11 hist_eq_openmp.c -o hist_eq_openmp



 Run:
   ./hist_eq_openmp input.pgm output_openmp.pgm
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#define L 256

// ---------- PGM Reader ----------

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
    fread(magic,1,2,fp);
    magic[2]='\0';

    if(strcmp(magic,"P5")!=0){
        printf("Only P5 PGM supported\n");
        fclose(fp);
        return NULL;
    }

    read_int(fp,w);
    read_int(fp,h);
    read_int(fp,maxval);

    fgetc(fp);

    size_t n=(size_t)(*w)*(*h);

    uint8_t *data=malloc(n);

    fread(data,1,n,fp);

    fclose(fp);
    return data;
}

static int write_pgm_p5(const char *path,const uint8_t *data,int w,int h){

    FILE *fp=fopen(path,"wb");
    if(!fp) return 0;

    fprintf(fp,"P5\n%d %d\n255\n",w,h);

    size_t n=(size_t)w*h;

    fwrite(data,1,n,fp);

    fclose(fp);
    return 1;
}






// ---------- OpenMP Histogram Equalization ----------

static void hist_equalize_openmp(uint8_t *img,int w,int h){

    size_t T=(size_t)w*h;

    uint32_t hist[L]={0};



    // -------- Parallel Histogram Computation --------
    int threads=omp_get_max_threads();

    uint32_t local_hist[threads][L];

    for(int t=0;t<threads;t++)
        for(int i=0;i<L;i++)
            local_hist[t][i]=0;


#pragma omp parallel
{
    int tid=omp_get_thread_num();

#pragma omp for
    for(size_t i=0;i<T;i++){
        local_hist[tid][ img[i] ]++;
    }
}


    // -------- Combine Local Histograms --------
    for(int t=0;t<threads;t++)
        for(int i=0;i<L;i++)
            hist[i]+=local_hist[t][i];



    // -------- CDF Computation (serial is fine) --------
    uint32_t cdf[L];

    cdf[0]=hist[0];

    for(int i=1;i<L;i++)
        cdf[i]=cdf[i-1]+hist[i];



    // -------- Find CDFmin --------
    uint32_t cdf_min=0;

    for(int i=0;i<L;i++)
        if(cdf[i]!=0){
            cdf_min=cdf[i];
            break;
        }



    if(cdf_min==(uint32_t)T)
        return;



    // -------- Build LUT --------
    uint8_t lut[L];

    for(int i=0;i<L;i++){

        if(cdf[i]<cdf_min)
            lut[i]=0;

        else{

            double val=((double)(cdf[i]-cdf_min)/(T-cdf_min))*(L-1);

            int mapped=(int)(val+0.5);

            if(mapped<0) mapped=0;
            if(mapped>255) mapped=255;

            lut[i]=(uint8_t)mapped;
        }
    }



    // -------- Parallel Pixel Mapping --------
#pragma omp parallel for
    for(size_t i=0;i<T;i++){
        img[i]=lut[ img[i] ];
    }

}


int main(int argc,char **argv){

    if(argc!=3){
        printf("Usage: %s input.pgm output_equalized_openmp.pgm\n",argv[0]);
        return 1;
    }

    int w,h,maxval;

    int thread_counts[] = {1,2,4,8};
    int experiments = 4;

    FILE *fp = fopen("results.txt","w");
    if(!fp){
        printf("Error: Could not open results.txt for writing\n");
        return 1;
    }

    for(int i=0;i<experiments;i++){

        int threads = thread_counts[i];

        omp_set_num_threads(threads);

        uint8_t *img = read_pgm_p5(argv[1],&w,&h,&maxval);

        double start = omp_get_wtime();

        hist_equalize_openmp(img,w,h);

        double end = omp_get_wtime();

        double time = end - start;

        printf("Threads: %d Time: %f\n",threads,time);

        fprintf(fp,"%d %f\n",threads,time);

        write_pgm_p5(argv[2],img,w,h);

        free(img);
    }

    fclose(fp);

    return 0;
}