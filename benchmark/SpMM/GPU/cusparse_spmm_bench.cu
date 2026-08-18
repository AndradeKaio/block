// cusparse_spmm_bench.cu — cuSPARSE SpMM (C = S x D) benchmark on real
// .bsp matrices, wired into the SAME contract prisma_gpu_spmm_bench.cu
// uses (CLI shape, D-generation RNG, --dump-d/--dump-c semantics, JSON
// sentinels) so it plugs directly into benchmark_spmm_gpu.py /
// validate_spmm.py without either script needing a parallel code path.
//
// CLI: cusparse_spmm_bench <S.bsp> [--runs N] [--seed S]
//                          [--precision fp32|fp64] [--dump-d path]
//                          [--dump-c path] [--algo default|alg1|alg2|alg3]
//
// Unlike prisma_gpu_spmm_bench, this is a SINGLE POOLED binary (compiled
// once by benchmark_spmm_gpu.py's compile_cusparse_bench, not per-matrix
// -- cuSPARSE has no per-matrix kernel-specialization step the way
// Prisma's generated specialized kernels do) -- mirrors how
// SpMM/bench_taco_spmm.cpp is compiled once and reused across every
// matrix.
//
// S is read from .bsp (NOT .mtx): this keeps S at the exact same
// float32-truncated-then-upcast storage precision every other .bsp-based
// contender (prisma_cpu, prisma_gpu_*) reads, so validate_spmm.py can
// compare this against the SAME C_ref_bsp scipy reference those
// contenders use, not the full-precision .mtx reference TACO uses.
//
// D generation is CORRECTNESS-CRITICAL and copied VERBATIM from
// prisma_gpu_spmm_bench.cu's own contract: same seed => bit-identical D
// across every contender, CPU or GPU, cuSPARSE or Prisma, fp64 or fp32 --
// see that file's header comment and suite-sparse/validate_spmm.py's
// docstring, both of which rely on this. Always drawn in double, THEN
// narrowed to T for fp32 -- drawing directly via
// uniform_real_distribution<float> would consume the underlying rng bits
// differently and silently produce a DIFFERENT D.
//
// cuSPARSE generic API usage (CSR sparse x dense row-major, CUDA_R_32F or
// CUDA_R_64F compute type) mirrors SpGEMM/GPU/cusparse_spgemm.cu's
// CUDA_CHECK/CUSPARSE_CHECK macros and buffer-size-then-execute two-call
// pattern -- that file is SpGEMM-specific and never wired into any
// harness, this is its SpMM analog, wired in properly.
//
// Timing boundary matches every other contender in this suite: the CSR
// conversion (host-side block-walk + global sort + row_ptr prefix sum)
// and device upload and cusparseSpMM_bufferSize/_preprocess all happen
// ONCE, outside the timed run loop (see prisma_gpu_spmm_bench.cu's own
// pipe_total_ms comment for the same "structural setup, not re-timed per
// run" rationale) -- only the repeated cusparseSpMM call itself is timed
// per run, via CUDA events.

#include "block.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(x)                                                       \
  do {                                                                       \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(_e));                                  \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

#define CUSPARSE_CHECK(x)                                                    \
  do {                                                                       \
    cusparseStatus_t _s = (x);                                               \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "cuSPARSE %s:%d  code=%d  %s\n", __FILE__,        \
                   __LINE__, (int)_s, cusparseGetErrorString(_s));           \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

struct Args {
  std::string s_bsp;
  int runs = 5;
  int seed = 42;
  std::string precision = "fp64";
  std::string algo = "default";
  std::string dump_d;
  std::string dump_c;
};

void print_usage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s <S.bsp> [--runs N] [--seed S] "
               "[--precision fp32|fp64] [--dump-d path] [--dump-c path] "
               "[--algo default|alg1|alg2|alg3]\n",
               prog);
}

Args parse_args(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    std::exit(1);
  }
  Args args;
  args.s_bsp = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--runs")
      args.runs = std::atoi(next());
    else if (arg == "--seed")
      args.seed = std::atoi(next());
    else if (arg == "--precision")
      args.precision = next();
    else if (arg == "--algo")
      args.algo = next();
    else if (arg == "--dump-d")
      args.dump_d = next();
    else if (arg == "--dump-c")
      args.dump_c = next();
    else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      print_usage(argv[0]);
      std::exit(1);
    }
  }
  if (args.precision != "fp32" && args.precision != "fp64") {
    std::fprintf(stderr, "--precision must be fp32 or fp64, got: %s\n",
                 args.precision.c_str());
    std::exit(1);
  }
  return args;
}

// cusparseSpMMAlg_t enumerator names are CSR-format-specific and have
// changed across cuSPARSE/CUDA versions -- NOT the same enum family as
// SpGEMM/GPU/cusparse_spgemm.cu's cusparseSpGEMMAlg_t, don't copy those
// names by analogy. Verified against this machine's real cusparse.h
// before shipping (grep CUSPARSE_SPMM_ /usr/local/cuda/include/cusparse.h).
cusparseSpMMAlg_t parse_algo(const std::string &name) {
  if (name == "default")
    return CUSPARSE_SPMM_ALG_DEFAULT;
  if (name == "alg1")
    return CUSPARSE_SPMM_CSR_ALG1;
  if (name == "alg2")
    return CUSPARSE_SPMM_CSR_ALG2;
  if (name == "alg3")
    return CUSPARSE_SPMM_CSR_ALG3;
  std::fprintf(stderr, "unknown --algo=%s (expected default|alg1|alg2|alg3)\n",
               name.c_str());
  std::exit(1);
}

float ms_between(Clock::time_point t0, Clock::time_point t1) {
  return float(std::chrono::duration<double, std::milli>(t1 - t0).count());
}

void print_arr(const std::vector<float> &v) {
  std::printf("[");
  for (int i = 0; i < (int)v.size(); ++i) {
    if (i)
      std::printf(", ");
    std::printf("%.4f", v[i]);
  }
  std::printf("]");
}

// ---------------------------------------------------------------------------
// HostCsr: flat CSR on the host, built either from an MTX file or a BSP.

template <typename T> struct HostCsr {
  int rows = 0, cols = 0;
  long long nnz = 0;
  std::vector<int> row_ptr; // size rows+1
  std::vector<int> col_idx; // size nnz
  std::vector<T> values;    // size nnz
};

// Read a Matrix Market coordinate file into CSR.
// Handles both real/integer patterns; skips comment lines; converts 1-based
// indices to 0-based; sorts globally by (row, col).
template <typename T>
HostCsr<T> mtx_to_host_csr(const std::string &mtx_path) {
  FILE *f = std::fopen(mtx_path.c_str(), "r");
  if (!f) {
    std::fprintf(stderr, "cusparse_spmm_bench: cannot open MTX: %s\n",
                 mtx_path.c_str());
    std::exit(1);
  }

  // Skip header / comment lines (start with '%')
  char line[1024];
  bool symmetric = false;
  bool pattern   = false;
  while (std::fgets(line, sizeof(line), f)) {
    if (line[0] == '%') {
      // Check for "symmetric" and "pattern" keywords in the banner
      std::string l(line);
      for (auto &c : l) c = (char)std::tolower((unsigned char)c);
      if (l.find("symmetric") != std::string::npos) symmetric = true;
      if (l.find("pattern")   != std::string::npos) pattern   = true;
      continue;
    }
    break; // first non-comment line = "rows cols nnz"
  }

  int nrows = 0, ncols = 0;
  long long stored = 0;
  if (std::sscanf(line, "%d %d %lld", &nrows, &ncols, &stored) != 3) {
    std::fprintf(stderr, "cusparse_spmm_bench: bad MTX header in %s\n",
                 mtx_path.c_str());
    std::fclose(f);
    std::exit(1);
  }

  std::vector<int> ri, ci;
  std::vector<T>   vals;
  ri.reserve(symmetric ? stored * 2 : stored);
  ci.reserve(symmetric ? stored * 2 : stored);
  vals.reserve(symmetric ? stored * 2 : stored);

  for (long long k = 0; k < stored; ++k) {
    int r, c;
    double v = 1.0;
    if (pattern) {
      if (std::fscanf(f, "%d %d", &r, &c) != 2) break;
    } else {
      if (std::fscanf(f, "%d %d %lf", &r, &c, &v) != 3) break;
    }
    --r; --c; // 1-based → 0-based
    ri.push_back(r);
    ci.push_back(c);
    vals.push_back(static_cast<T>(v));
    if (symmetric && r != c) {
      ri.push_back(c);
      ci.push_back(r);
      vals.push_back(static_cast<T>(v));
    }
  }
  std::fclose(f);

  HostCsr<T> csr;
  csr.rows = nrows;
  csr.cols = ncols;
  csr.nnz  = (long long)ri.size();

  std::vector<int> order(ri.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return ri[a] != ri[b] ? ri[a] < ri[b] : ci[a] < ci[b];
  });

  csr.col_idx.resize((std::size_t)csr.nnz);
  csr.values .resize((std::size_t)csr.nnz);
  for (long long k = 0; k < csr.nnz; ++k) {
    csr.col_idx[k] = ci[order[k]];
    csr.values [k] = vals[order[k]];
  }

  csr.row_ptr.assign((std::size_t)nrows + 1, 0);
  for (long long k = 0; k < csr.nnz; ++k)
    csr.row_ptr[ri[order[k]] + 1]++;
  for (int r = 0; r < nrows; ++r)
    csr.row_ptr[r + 1] += csr.row_ptr[r];

  return csr;
}

// BSP fallback: block-walk → COO → sort → CSR.
template <typename T>
HostCsr<T> matrix_to_host_csr(const benchmark_core::Matrix<T> &S) {
  HostCsr<T> csr;
  csr.rows = S.M;
  csr.cols = S.N;

  std::vector<int> rows, cols;
  std::vector<T> vals;
  rows.reserve(S.n_values);
  cols.reserve(S.n_values);
  vals.reserve(S.n_values);

  for (const auto &b : S.blocks) {
    auto data = S.block_data(b);
    for (int i = 0; i < b.h; ++i)
      for (int j = 0; j < b.w; ++j) {
        T v = data[(std::size_t)i * b.w + j];
        if (v != T(0)) {
          rows.push_back(b.r + i);
          cols.push_back(b.c + j);
          vals.push_back(v);
        }
      }
  }

  csr.nnz = (long long)rows.size();
  std::vector<int> order(rows.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return rows[a] != rows[b] ? rows[a] < rows[b] : cols[a] < cols[b];
  });

  csr.col_idx.resize((std::size_t)csr.nnz);
  csr.values .resize((std::size_t)csr.nnz);
  for (long long k = 0; k < csr.nnz; ++k) {
    csr.col_idx[k] = cols[order[k]];
    csr.values [k] = vals [order[k]];
  }

  csr.row_ptr.assign((std::size_t)csr.rows + 1, 0);
  for (long long k = 0; k < csr.nnz; ++k)
    csr.row_ptr[rows[order[k]] + 1]++;
  for (int r = 0; r < csr.rows; ++r)
    csr.row_ptr[r + 1] += csr.row_ptr[r];

  return csr;
}

// Return the MTX path that sits alongside a BSP file, or "" if not found.
static std::string mtx_sibling(const std::string &bsp_path) {
  // Replace trailing ".bsp" with ".mtx"
  const std::string suffix = ".bsp";
  if (bsp_path.size() > suffix.size() &&
      bsp_path.compare(bsp_path.size() - suffix.size(), suffix.size(), suffix) == 0) {
    std::string candidate = bsp_path.substr(0, bsp_path.size() - suffix.size()) + ".mtx";
    if (FILE *f = std::fopen(candidate.c_str(), "r")) {
      std::fclose(f);
      return candidate;
    }
  }
  return {};
}

template <typename T> struct CudaTypeTraits;
template <> struct CudaTypeTraits<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};
template <> struct CudaTypeTraits<double> {
  static constexpr cudaDataType_t value = CUDA_R_64F;
};

// Runs the full benchmark for one scalar type T (float or double),
// selected at runtime by main() based on --precision. Returns the process
// exit code.
template <typename T> int run(const Args &args) {
  const std::string kernel_name = std::string("cusparse_") + args.precision;
  constexpr cudaDataType_t compute_type = CudaTypeTraits<T>::value;
  const cusparseSpMMAlg_t algo = parse_algo(args.algo);

  benchmark_core::Matrix<T> S;
  auto t_read0 = Clock::now();
  try {
    S = benchmark_core::read_matrix_binsparse<T>(args.s_bsp);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "cusparse_spmm_bench: failed to read .bsp: %s\n",
                 e.what());
    return 1;
  }
  const float bsp_read_ms = ms_between(t_read0, Clock::now());

  const int M = S.M;
  const int N = S.N; // square matrix: dense RHS is N x N, same as every
                      // other SpMM contender in this suite

  // Dense D: N x N random in [-1, 1]. CORRECTNESS-CRITICAL -- see this
  // file's header comment; must stay bit-identical to
  // prisma_gpu_spmm_bench.cu's own D generation for the same seed.
  std::vector<double> D_f64((long long)N * N);
  {
    std::mt19937_64 rng((uint64_t)args.seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::generate(D_f64.begin(), D_f64.end(), [&] { return dist(rng); });
  }
  std::vector<T> D(D_f64.size());
  for (std::size_t i = 0; i < D_f64.size(); ++i)
    D[i] = static_cast<T>(D_f64[i]);

  if (!args.dump_d.empty()) {
    FILE *fd = std::fopen(args.dump_d.c_str(), "wb");
    if (!fd) {
      std::fprintf(stderr, "cusparse_spmm_bench: cannot open dump-d: %s\n",
                   args.dump_d.c_str());
    } else {
      std::fwrite(D_f64.data(), sizeof(double), D_f64.size(), fd);
      std::fclose(fd);
    }
  }

  std::vector<T> C_host((long long)M * N);

  // --- Structural setup: CSR build + device upload + preprocess, done ONCE.
  auto t_csr0 = Clock::now();
  HostCsr<T> csr;
  std::string mtx_path = mtx_sibling(args.s_bsp);
  if (!mtx_path.empty()) {
    std::fprintf(stderr, "loading CSR from MTX: %s\n", mtx_path.c_str());
    csr = mtx_to_host_csr<T>(mtx_path);
  } else {
    csr = matrix_to_host_csr(S);
  }
  const float csr_build_ms = ms_between(t_csr0, Clock::now());

  std::fprintf(stderr,
               "S: %dx%d  S_blocks=%zu  S_nnz=%lld  precision=%s  algo=%s\n",
               M, N, S.blocks.size(), csr.nnz, args.precision.c_str(),
               args.algo.c_str());

  auto t_up0 = Clock::now();
  int *d_row_ptr = nullptr, *d_col_idx = nullptr;
  T *d_values = nullptr, *d_D = nullptr, *d_C = nullptr;
  CUDA_CHECK(cudaMalloc(&d_row_ptr, (csr.rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_idx, (csr.nnz ? csr.nnz : 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, (csr.nnz ? csr.nnz : 1) * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_row_ptr, csr.row_ptr.data(),
                        (csr.rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (csr.nnz > 0) {
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr.col_idx.data(),
                          csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, csr.values.data(), csr.nnz * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  const std::size_t n_D = (std::size_t)N * N;
  CUDA_CHECK(cudaMalloc(&d_D, n_D * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_D, D.data(), n_D * sizeof(T), cudaMemcpyHostToDevice));

  const std::size_t n_C = (std::size_t)M * N;
  CUDA_CHECK(cudaMalloc(&d_C, n_C * sizeof(T)));
  // Zeroed ONCE, before the run loop -- not strictly required for
  // correctness (cusparseSpMM's beta=0 fully overwrites C every call, so
  // it should never read this buffer), but costs nothing here and removes
  // any doubt about that "don't read C" contract -- matches Prisma's own
  // "always start from a zeroed C" discipline (see
  // spmm_gpu_dispatch.cuh's own comment on why it re-zeros before every
  // run, a stricter requirement there since Prisma accumulates via
  // atomicAdd; cuSPARSE doesn't need the PER-RUN re-zero since beta=0
  // means each call is a fresh overwrite, not an accumulation).
  CUDA_CHECK(cudaMemset(d_C, 0, n_C * sizeof(T)));
  const float device_upload_ms = ms_between(t_up0, Clock::now());

  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));

  cusparseSpMatDescr_t matS;
  CUSPARSE_CHECK(cusparseCreateCsr(
      &matS, csr.rows, csr.cols, csr.nnz, d_row_ptr, d_col_idx, d_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      compute_type));

  cusparseDnMatDescr_t matD, matC;
  CUSPARSE_CHECK(cusparseCreateDnMat(&matD, N, N, N, d_D, compute_type,
                                     CUSPARSE_ORDER_ROW));
  CUSPARSE_CHECK(cusparseCreateDnMat(&matC, M, N, N, d_C, compute_type,
                                     CUSPARSE_ORDER_ROW));

  const T alpha = T(1), beta = T(0);

  auto t_pre0 = Clock::now();
  size_t buf_size = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matS, matD, &beta, matC,
      compute_type, algo, &buf_size));
  void *d_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buffer, buf_size ? buf_size : 1));
  CUSPARSE_CHECK(cusparseSpMM_preprocess(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matS, matD, &beta, matC,
      compute_type, algo, d_buffer));
  const float preprocess_ms = ms_between(t_pre0, Clock::now());
  const float pipe_total_ms = csr_build_ms + device_upload_ms + preprocess_ms;

  std::fprintf(stderr,
               "structural_setup: bsp_read=%.4fms  csr_build=%.4fms  "
               "device_upload=%.4fms  preprocess=%.4fms  "
               "total(excl. read)=%.4fms\n",
               bsp_read_ms, csr_build_ms, device_upload_ms, preprocess_ms,
               pipe_total_ms);

  // --- Timed region: cusparseSpMM only, run 0 = warmup (matches every
  // other contender's convention).
  std::vector<float> symbolic_ms_arr, compute_ms_arr;
  symbolic_ms_arr.reserve(args.runs + 1);
  compute_ms_arr.reserve(args.runs + 1);

  cudaEvent_t e_start, e_stop;
  CUDA_CHECK(cudaEventCreate(&e_start));
  CUDA_CHECK(cudaEventCreate(&e_stop));

  for (int r = 0; r <= args.runs; ++r) {
    // symbolic_ms reports csr_build_ms (host-side block-walk + sort +
    // row_ptr construction -- the structural-analysis step, cuSPARSE's
    // closest equivalent to Prisma's block classification) on every run,
    // not just once -- see prisma_gpu_spmm_bench.cu's matching correction
    // comment for the full rationale: this makes the cost visible in any
    // total_ms/symbolic_ms-based comparison instead of silently excluded,
    // and represents what a single one-shot SpMM call actually pays.
    // device_upload_ms and preprocess_ms stay separate (JSON/stderr-only,
    // not folded in here) -- preprocess_ms happens AFTER upload and has no
    // Prisma-side equivalent to be symmetric with; device_upload_ms is
    // pure data transfer on both sides, not structural analysis.
    symbolic_ms_arr.push_back(csr_build_ms);

    CUDA_CHECK(cudaEventRecord(e_start));
    CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                matS, matD, &beta, matC, compute_type, algo,
                                d_buffer));
    CUDA_CHECK(cudaEventRecord(e_stop));
    CUDA_CHECK(cudaEventSynchronize(e_stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));
    compute_ms_arr.push_back(ms);
  }

  CUDA_CHECK(cudaMemcpy(C_host.data(), d_C, n_C * sizeof(T),
                        cudaMemcpyDeviceToHost));

  std::fprintf(stderr, "runs=%d  last: compute=%.3fms\n", args.runs,
              compute_ms_arr.back());

  if (!args.dump_c.empty()) {
    std::vector<double> C_f64(C_host.size());
    for (std::size_t i = 0; i < C_host.size(); ++i)
      C_f64[i] = static_cast<double>(C_host[i]);
    FILE *fc = std::fopen(args.dump_c.c_str(), "wb");
    if (!fc) {
      std::fprintf(stderr, "cusparse_spmm_bench: cannot open dump-c: %s\n",
                   args.dump_c.c_str());
    } else {
      std::fwrite(C_f64.data(), sizeof(double), C_f64.size(), fc);
      std::fclose(fc);
    }
  }

  CUDA_CHECK(cudaEventDestroy(e_start));
  CUDA_CHECK(cudaEventDestroy(e_stop));
  CUSPARSE_CHECK(cusparseDestroySpMat(matS));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matD));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
  CUSPARSE_CHECK(cusparseDestroy(handle));
  cudaFree(d_buffer);
  cudaFree(d_row_ptr);
  cudaFree(d_col_idx);
  cudaFree(d_values);
  cudaFree(d_D);
  cudaFree(d_C);

  std::printf("\nJSON_BEGIN\n{\n");
  std::printf("  \"kernel\": \"%s\",\n", kernel_name.c_str());
  std::printf("  \"pipe_total_ms\": %.4f,\n", pipe_total_ms);
  std::printf("  \"bsp_read_ms\": %.4f,\n", bsp_read_ms);
  std::printf("  \"csr_build_ms\": %.4f,\n", csr_build_ms);
  std::printf("  \"device_upload_ms\": %.4f,\n", device_upload_ms);
  std::printf("  \"preprocess_ms\": %.4f,\n", preprocess_ms);
  std::printf("  \"S_rows\": %d, \"S_cols\": %d,\n", M, N);
  std::printf("  \"S_blocks\": %zu, \"S_nnz\": %lld,\n", S.blocks.size(),
              csr.nnz);
  std::printf("  \"algo\": \"%s\",\n", args.algo.c_str());
  std::printf("  \"symbolic_ms\": ");
  print_arr(symbolic_ms_arr);
  std::printf(",\n  \"compute_ms\": ");
  print_arr(compute_ms_arr);
  std::printf("\n}\nJSON_END\n");

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const Args args = parse_args(argc, argv);
  return args.precision == "fp32" ? run<float>(args) : run<double>(args);
}
