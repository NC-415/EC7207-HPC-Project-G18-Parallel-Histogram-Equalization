To open the pgm files use the below command:  
eog output_openmp.pgm


## Parallel Histogram Equalization (OpenMP) ⚡️🖼️

This project implements **histogram equalization** for grayscale images and accelerates the heavy computations using **OpenMP** (shared-memory parallelism) 🚀.

---

## 📁 Folder Structure (OpenMP-related)

- `src/openmp/`
  - `hist_eq_openmp.c` — OpenMP implementation 🧵
- `data/`
  - `input_image.png` — sample input image 🖼️
- `results/plots/`
  - `openmp_execution_time_plot.png` — performance plot (execution time) 📈

---

## 🧠 What is OpenMP?

**OpenMP (Open Multi-Processing)** is an API for parallel programming on **shared-memory** systems (multi-core CPUs).  
In this project, OpenMP helps speed up key steps such as:

- building the **histogram** (counts per intensity) 🧮  
- computing the **CDF** (cumulative distribution function) 📊  
- transforming pixel values across the full image ✅  

Parallelism is typically introduced with directives like:

- `#pragma omp parallel`
- `#pragma omp for`
- `reduction(...)` / atomic updates (depending on the approach)

---

## 🛠️ Compile (OpenMP)

Using **GCC**:

```bash
gcc -O3 -fopenmp -o hist_eq_openmp src/openmp/hist_eq_openmp.c -lm
```

---

## ▶️ Run

Example (adjust depending on how your program accepts input):

```bash
./hist_eq_openmp
```

If your program takes an image path:

```bash
./hist_eq_openmp data/input_image.png
```

---

## 📈 Results (Performance Plot)

OpenMP execution-time results:

![OpenMP Execution Time Plot](results/plots/openmp_execution_time_plot.png)

---

## 💡 Tips

- You can control the number of threads like this:
  ```bash
  export OMP_NUM_THREADS=4
  ./hist_eq_openmp
  ```
- Performance scaling depends on CPU cores, memory bandwidth, and image size ⚙️
