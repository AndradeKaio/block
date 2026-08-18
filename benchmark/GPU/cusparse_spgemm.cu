// bench_spgemm.cu
// Benchmarks cuSPARSE SpGEMM with symbolic/numeric phase timing.
// Usage: ./bench_spgemm A.mtx [B.mtx]  (omit B to compute A*A)
//
// Phase definitions (cuSPARSE generic API, CUDA >= 11.0):
//   Symbolic : cusparseSpGEMM_workEstimation  -- determines output sparsity
//   pattern Numeric  : cusparseSpGEMM_compute         -- computes nonzero
//   values Copy     : cusparseSpGEMM_copy            -- writes result to
//   allocated CSR arrays
//
// Timings use CUDA events (device-side), not wall clock, so host Python
// overhead would not affect them even if this were called from ctypes/CuPy.

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error macros
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d  %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(_e));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t _s = (call);                                              \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuSPARSE error %s:%d  code=%d  %s\n", __FILE__,         \
              __LINE__, (int)_s, cusparseGetErrorString(_s));                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
// ---------------------------------------------------------------------------
// CUDA event timer
// ---------------------------------------------------------------------------

struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void Start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }
  void Stop(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(stop, s)); }
  float ElapsedMs() {
    float ms = 0.f;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// ---------------------------------------------------------------------------
// Matrix Market (COO) loader
// ---------------------------------------------------------------------------

struct CooMatrix {
  int rows = 0, cols = 0, nnz = 0;
  std::vector<int> row_idx, col_idx;
  std::vector<float> values;
};

// Parses "%%MatrixMarket matrix coordinate {real|integer|pattern|complex}
//          {general|symmetric|skew-symmetric|hermitian}"
// All numeric types are upcast to float.  Complex matrices use real part only.
CooMatrix load_mtx(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f)
    throw std::runtime_error(std::string("Cannot open: ") + path);

  char line[2048];

  // --- header ---
  bool is_symmetric = false;
  bool is_pattern = false;
  bool is_skew = false;

  if (!fgets(line, sizeof(line), f))
    throw std::runtime_error("Empty file");

  // %%MatrixMarket matrix coordinate <type> <structure>
  char token[64];
  // tolower all and check keywords
  for (char *p = line; *p; ++p)
    *p = (char)tolower((unsigned char)*p);

  if (!strstr(line, "matrix") || !strstr(line, "coordinate"))
    throw std::runtime_error(
        "Only coordinate (sparse) Matrix Market supported");
  if (strstr(line, "pattern"))
    is_pattern = true;
  if (strstr(line, "symmetric"))
    is_symmetric = true;
  if (strstr(line, "skew"))
    is_skew = true;

  // --- skip comment lines ---
  while (fgets(line, sizeof(line), f) && line[0] == '%')
    ;

  // --- size line ---
  CooMatrix coo;
  int nnz_file = 0;
  if (sscanf(line, "%d %d %d", &coo.rows, &coo.cols, &nnz_file) != 3)
    throw std::runtime_error("Bad size line");

  int capacity = (is_symmetric || is_skew) ? nnz_file * 2 : nnz_file;
  coo.row_idx.reserve(capacity);
  coo.col_idx.reserve(capacity);
  coo.values.reserve(capacity);

  // --- data lines ---
  for (int i = 0; i < nnz_file; i++) {
    int r, c;
    float v = 1.0f;
    if (is_pattern) {
      if (fscanf(f, "%d %d", &r, &c) != 2)
        throw std::runtime_error("Bad data line (pattern)");
    } else {
      // real / integer / complex (use real part)
      double v1, v2 = 0.0;
      int n = fscanf(f, "%d %d %lf %lf", &r, &c, &v1, &v2);
      if (n < 3)
        throw std::runtime_error("Bad data line");
      v = (float)v1;
    }

    r--;
    c--; // 1-indexed → 0-indexed

    coo.row_idx.push_back(r);
    coo.col_idx.push_back(c);
    coo.values.push_back(v);

    if ((is_symmetric || is_skew) && r != c) {
      coo.row_idx.push_back(c);
      coo.col_idx.push_back(r);
      coo.values.push_back(is_skew ? -v : v);
    }
  }
  fclose(f);

  coo.nnz = (int)coo.row_idx.size();
  return coo;
}

// ---------------------------------------------------------------------------
// Device CSR matrix
// ---------------------------------------------------------------------------

struct CsrMatrix {
  int rows = 0, cols = 0, nnz = 0;
  int *d_row_ptr = nullptr;  // size rows+1
  int *d_col_idx = nullptr;  // size nnz
  float *d_values = nullptr; // size nnz

  void free_device() {
    cudaFree(d_row_ptr);
    d_row_ptr = nullptr;
    cudaFree(d_col_idx);
    d_col_idx = nullptr;
    cudaFree(d_values);
    d_values = nullptr;
  }
};

// Sort COO by (row, col), then convert via cusparseXcoo2csr.
CsrMatrix upload_coo_as_csr(const CooMatrix &coo, cusparseHandle_t handle) {
  // Sort entries by (row, col) on host — required by cusparseXcoo2csr
  std::vector<int> order(coo.nnz);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (coo.row_idx[a] != coo.row_idx[b])
      return coo.row_idx[a] < coo.row_idx[b];
    return coo.col_idx[a] < coo.col_idx[b];
  });

  std::vector<int> sorted_row(coo.nnz), sorted_col(coo.nnz);
  std::vector<float> sorted_val(coo.nnz);
  for (int i = 0; i < coo.nnz; i++) {
    sorted_row[i] = coo.row_idx[order[i]];
    sorted_col[i] = coo.col_idx[order[i]];
    sorted_val[i] = coo.values[order[i]];
  }

  // Upload sorted COO row indices to device (only row needed for coo2csr)
  int *d_coo_row = nullptr;
  CUDA_CHECK(cudaMalloc(&d_coo_row, coo.nnz * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_coo_row, sorted_row.data(), coo.nnz * sizeof(int),
                        cudaMemcpyHostToDevice));

  CsrMatrix csr;
  csr.rows = coo.rows;
  csr.cols = coo.cols;
  csr.nnz = coo.nnz;

  CUDA_CHECK(cudaMalloc(&csr.d_row_ptr, (csr.rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&csr.d_col_idx, csr.nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&csr.d_values, csr.nnz * sizeof(float)));

  // COO row indices → CSR row pointer
  CUSPARSE_CHECK(cusparseXcoo2csr(handle, d_coo_row, csr.nnz, csr.rows,
                                  csr.d_row_ptr, CUSPARSE_INDEX_BASE_ZERO));

  // col indices and values upload directly
  CUDA_CHECK(cudaMemcpy(csr.d_col_idx, sorted_col.data(), csr.nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(csr.d_values, sorted_val.data(),
                        csr.nnz * sizeof(float), cudaMemcpyHostToDevice));

  cudaFree(d_coo_row);
  return csr;
}

// ---------------------------------------------------------------------------
// SpGEMM benchmark (one trial)
// ---------------------------------------------------------------------------

struct SpGEMMTimes {
  float symbolic_ms; // workEstimation (both calls)
  float numeric_ms;  // compute        (both calls)
  float copy_ms;     // copy to final allocation
  int nnz_C;
};

SpGEMMTimes run_spgemm(cusparseHandle_t handle, const CsrMatrix &A,
                       const CsrMatrix &B, cusparseSpGEMMAlg_t algo,
                       bool retain_C = false, CsrMatrix *out_C = nullptr) {
  // ---- Descriptors --------------------------------------------------------
  cusparseSpMatDescr_t matA, matB, matC;
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz, A.d_row_ptr,
                                   A.d_col_idx, A.d_values, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));

  CUSPARSE_CHECK(cusparseCreateCsr(&matB, B.rows, B.cols, B.nnz, B.d_row_ptr,
                                   B.d_col_idx, B.d_values, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));

  // Pre-allocate C's row pointer before creating the descriptor.
  // Newer cuSPARSE (12.x) requires a valid csrRowOffsets even for an
  // initially-empty output matrix; passing nullptr causes INVALID_VALUE on
  // the second workEstimation call when A and B are distinct matrices.
  CsrMatrix C;
  C.rows = A.rows;
  C.cols = B.cols;
  CUDA_CHECK(cudaMalloc(&C.d_row_ptr, (C.rows + 1) * sizeof(int)));

  CUSPARSE_CHECK(cusparseCreateCsr(&matC, A.rows, B.cols, 0, C.d_row_ptr,
                                   nullptr, nullptr, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F));

  cusparseSpGEMMDescr_t spgemm_desc;
  CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemm_desc));

  const float alpha = 1.f, beta = 0.f;
  GpuTimer timer;

  // ---- Phase 1: Symbolic (workEstimation) ---------------------------------
  // Two-call pattern: first call returns needed buffer size, second executes.
  size_t buf1_size = 0;
  void *buf1 = nullptr;

  timer.Start();

  CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, algo, spgemm_desc, &buf1_size, nullptr));

  CUDA_CHECK(cudaMalloc(&buf1, buf1_size ? buf1_size : 1));

  CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, algo, spgemm_desc, &buf1_size, buf1));

  timer.Stop();
  float t_symbolic = timer.ElapsedMs();

  // ---- Phase 2: Numeric (compute) -----------------------------------------
  size_t buf2_size = 0;
  void *buf2 = nullptr;

  timer.Start();

  CUSPARSE_CHECK(cusparseSpGEMM_compute(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, algo, spgemm_desc, &buf2_size, nullptr));

  CUDA_CHECK(cudaMalloc(&buf2, buf2_size ? buf2_size : 1));

  CUSPARSE_CHECK(cusparseSpGEMM_compute(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, algo, spgemm_desc, &buf2_size, buf2));

  timer.Stop();
  float t_numeric = timer.ElapsedMs();

  // ---- Get output size and allocate C col/val arrays ---------------------
  int64_t C_rows, C_cols, C_nnz;
  CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &C_rows, &C_cols, &C_nnz));
  C.nnz = (int)C_nnz;

  // d_row_ptr is already allocated; only col and value arrays are new.
  CUDA_CHECK(cudaMalloc(&C.d_col_idx, (C.nnz ? C.nnz : 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&C.d_values, (C.nnz ? C.nnz : 1) * sizeof(float)));

  CUSPARSE_CHECK(
      cusparseCsrSetPointers(matC, C.d_row_ptr, C.d_col_idx, C.d_values));

  // ---- Phase 3: Copy ------------------------------------------------------
  timer.Start();

  CUSPARSE_CHECK(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                     CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                     matA, matB, &beta, matC, CUDA_R_32F, algo,
                                     spgemm_desc));

  timer.Stop();
  float t_copy = timer.ElapsedMs();

  // ---- Cleanup ------------------------------------------------------------
  CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemm_desc));
  CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  CUSPARSE_CHECK(cusparseDestroySpMat(matB));
  CUSPARSE_CHECK(cusparseDestroySpMat(matC));
  cudaFree(buf1);
  cudaFree(buf2);

  if (retain_C && out_C) {
    *out_C = C;
  } else {
    C.free_device();
  }

  return {t_symbolic, t_numeric, t_copy, (int)C_nnz};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

// --algo= name -> cusparseSpGEMMAlg_t. ALG3 is the low-memory variant —
// useful when DEFAULT hits CUSPARSE_STATUS_INSUFFICIENT_RESOURCES on
// symbolic estimation for heavily-clustered/blocky sparsity patterns.
cusparseSpGEMMAlg_t parse_algo(const std::string &name) {
  if (name == "default")
    return CUSPARSE_SPGEMM_DEFAULT;
  if (name == "alg1")
    return CUSPARSE_SPGEMM_ALG1;
  if (name == "alg2")
    return CUSPARSE_SPGEMM_ALG2;
  if (name == "alg3")
    return CUSPARSE_SPGEMM_ALG3;
  fprintf(stderr, "Unknown --algo=%s (expected default|alg1|alg2|alg3)\n",
          name.c_str());
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  // Parse: positional args are MTX paths; --runs=N and --algo=NAME are optional
  const char *path_A = nullptr;
  const char *path_B = nullptr;
  int RUNS = 10;
  std::string algo_name = "alg1";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a.rfind("--runs=", 0) == 0)
      RUNS = std::stoi(a.substr(7));
    else if (a.rfind("--algo=", 0) == 0)
      algo_name = a.substr(7);
    else if (!path_A)
      path_A = argv[i];
    else if (!path_B)
      path_B = argv[i];
  }
  if (!path_A) {
    fprintf(stderr,
            "Usage: %s <A.mtx> [B.mtx] [--runs=N] "
            "[--algo=default|alg1|alg2|alg3]\n",
            argv[0]);
    return EXIT_FAILURE;
  }
  cusparseSpGEMMAlg_t algo = parse_algo(algo_name);

  // ---- GPU info -----------------------------------------------------------
  int dev = 0;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("Device : %s  (SM %d.%d, %.0f GiB VRAM)\n", prop.name, prop.major,
         prop.minor, prop.totalGlobalMem / 1073741824.0);

  // ---- Load ---------------------------------------------------------------
  printf("\nLoading A: %s\n", path_A);
  CooMatrix coo_A = load_mtx(path_A);
  printf("  %d x %d,  nnz = %d  (%.2f nnz/row avg)\n", coo_A.rows, coo_A.cols,
         coo_A.nnz, (double)coo_A.nnz / coo_A.rows);

  bool square_mode = (path_B == nullptr);
  CooMatrix coo_B;
  if (square_mode) {
    printf("  B = A  (squaring mode)\n");
    coo_B = coo_A;
  } else {
    printf("Loading B: %s\n", path_B);
    coo_B = load_mtx(path_B);
    printf("  %d x %d,  nnz = %d\n", coo_B.rows, coo_B.cols, coo_B.nnz);
  }

  // ---- Upload to device ---------------------------------------------------
  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));

  printf("\nUploading to device...\n");
  CsrMatrix A = upload_coo_as_csr(coo_A, handle);
  CsrMatrix B = square_mode ? A : upload_coo_as_csr(coo_B, handle);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- Warmup (1 run, discarded) ------------------------------------------
  printf("Warmup run...\n");
  auto w = run_spgemm(handle, A, B, algo);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- Benchmark ----------------------------------------------------------
  float sum_sym = 0, sum_num = 0, sum_copy = 0;
  int nnz_C = 0;

  printf("Benchmarking (%d runs)...\n", RUNS);
  for (int i = 0; i < RUNS; i++) {
    auto t = run_spgemm(handle, A, B, algo);
    sum_sym += t.symbolic_ms;
    sum_num += t.numeric_ms;
    sum_copy += t.copy_ms;
    nnz_C = t.nnz_C;
  }

  double avg_sym = sum_sym / RUNS;
  double avg_num = sum_num / RUNS;
  double avg_copy = sum_copy / RUNS;
  double avg_tot = avg_sym + avg_num + avg_copy;

  // Arithmetic intensity proxy: 2 * nnz(C) flops / bytes read
  // (Lower bound; true intermediate work is larger)
  long long bytes_A = (long long)A.nnz * (sizeof(float) + sizeof(int)) +
                      (A.rows + 1) * sizeof(int);
  long long bytes_B = (long long)B.nnz * (sizeof(float) + sizeof(int)) +
                      (B.rows + 1) * sizeof(int);
  double gflops_lower = 2.0 * nnz_C / 1e9;
  double gbytes = (bytes_A + bytes_B) / 1e9;

  printf("\n=== cuSPARSE SpGEMM  (CUDA_R_32F, algo=%s) ===\n",
         algo_name.c_str());
  printf("  nnz(A)              = %d\n", A.nnz);
  printf("  nnz(B)              = %d\n", B.nnz);
  printf("  nnz(C)              = %d\n", nnz_C);
  printf("  fill growth         = %.2fx\n", (double)nnz_C / A.nnz);
  printf("  fill density        = %.6f\n",
         (double)nnz_C / ((double)A.rows * B.cols));
  printf("\n");
  printf("  Symbolic  (avg)     = %7.3f ms  (%5.1f%%)\n", avg_sym,
         100 * avg_sym / avg_tot);
  printf("  Numeric   (avg)     = %7.3f ms  (%5.1f%%)\n", avg_num,
         100 * avg_num / avg_tot);
  printf("  Copy      (avg)     = %7.3f ms  (%5.1f%%)\n", avg_copy,
         100 * avg_copy / avg_tot);
  printf("  Total     (avg)     = %7.3f ms\n", avg_tot);
  printf("\n");
  printf("  AI (lower bound)    = %.4f FLOP/byte\n", gflops_lower / gbytes);
  printf("  GFLOP/s (on nnz_C)  = %.2f\n", gflops_lower / (avg_tot / 1e3));

  // ---- JSON output (parsed by gpu_benchmark.py) --------------------------
  printf("\nJSON_BEGIN\n{\n");
  printf("  \"symbolic_ms\" : %.4f,\n", avg_sym);
  printf("  \"numeric_ms\"  : %.4f,\n", avg_num);
  printf("  \"copy_ms\"     : %.4f,\n", avg_copy);
  printf("  \"total_ms\"    : %.4f,\n", avg_tot);
  printf("  \"gflops\"      : %.4f,\n", gflops_lower / (avg_tot / 1e3));
  printf("  \"nnz_c\"       : %d,\n", nnz_C);
  printf("  \"algo\"        : \"%s\"\n", algo_name.c_str());
  printf("}\nJSON_END\n");

  // ---- Cleanup ------------------------------------------------------------
  A.free_device();
  if (!square_mode)
    B.free_device();
  CUSPARSE_CHECK(cusparseDestroy(handle));

  return EXIT_SUCCESS;
}
