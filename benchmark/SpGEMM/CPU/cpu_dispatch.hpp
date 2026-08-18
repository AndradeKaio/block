#pragma once

#include "block.hpp"
#include "matrix.hpp"
#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>
#include <utility>

#ifdef HAVE_BLAS
#include <cblas.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace benchmark_core {

namespace cpu_detail {

template <int M, int K, int N, typename T>
inline void gemm_fixed(const T *__restrict__ A, int lda,
                       const T *__restrict__ B, int ldb, T *__restrict__ C,
                       int ldc) {
  // Flat layout: b_reg[p*N + j] — stride-1 inner access, fully vectorisable.
  // Size K*N is compile-time, so the compiler allocates these in registers.
  T b_reg[K * N];
  for (int p = 0; p < K; ++p)
    for (int j = 0; j < N; ++j)
      b_reg[p * N + j] = B[p * ldb + j];

  for (int i = 0; i < M; ++i) {
    T acc[N];
    for (int j = 0; j < N; ++j)
      acc[j] = C[i * ldc + j];

    for (int p = 0; p < K; ++p) {
      const T av = A[i * lda + p];
      for (int j = 0; j < N; ++j)
        acc[j] += av * b_reg[p * N + j];
    }

    for (int j = 0; j < N; ++j)
      C[i * ldc + j] = acc[j];
  }
}

#if defined(__AVX2__) && defined(__FMA__)

template <>
inline void gemm_fixed<5, 4, 4, float>(const float *__restrict__ A, int lda,
                                       const float *__restrict__ B, int ldb,
                                       float *__restrict__ C, int ldc) {
  __m128 c0 = _mm_loadu_ps(C);
  __m128 c1 = _mm_loadu_ps(C + ldc);
  __m128 c2 = _mm_loadu_ps(C + 2 * ldc);
  __m128 c3 = _mm_loadu_ps(C + 3 * ldc);
  __m128 c4 = _mm_loadu_ps(C + 4 * ldc);
  for (int k = 0; k < 4; k++) {
    __m128 bk = _mm_loadu_ps(B + k * ldb);
    c0 = _mm_fmadd_ps(_mm_set1_ps(A[k]), bk, c0);
    c1 = _mm_fmadd_ps(_mm_set1_ps(A[lda + k]), bk, c1);
    c2 = _mm_fmadd_ps(_mm_set1_ps(A[2 * lda + k]), bk, c2);
    c3 = _mm_fmadd_ps(_mm_set1_ps(A[3 * lda + k]), bk, c3);
    c4 = _mm_fmadd_ps(_mm_set1_ps(A[4 * lda + k]), bk, c4);
  }
  _mm_storeu_ps(C, c0);
  _mm_storeu_ps(C + ldc, c1);
  _mm_storeu_ps(C + 2 * ldc, c2);
  _mm_storeu_ps(C + 3 * ldc, c3);
  _mm_storeu_ps(C + 4 * ldc, c4);
}

template <>
inline void gemm_fixed<5, 4, 4, double>(const double *__restrict__ A, int lda,
                                        const double *__restrict__ B, int ldb,
                                        double *__restrict__ C, int ldc) {
  __m256d c0 = _mm256_loadu_pd(C);
  __m256d c1 = _mm256_loadu_pd(C + ldc);
  __m256d c2 = _mm256_loadu_pd(C + 2 * ldc);
  __m256d c3 = _mm256_loadu_pd(C + 3 * ldc);
  __m256d c4 = _mm256_loadu_pd(C + 4 * ldc);
  for (int k = 0; k < 4; k++) {
    __m256d bk = _mm256_loadu_pd(B + k * ldb);
    c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(A + k), bk, c0);
    c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(A + lda + k), bk, c1);
    c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(A + 2 * lda + k), bk, c2);
    c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(A + 3 * lda + k), bk, c3);
    c4 = _mm256_fmadd_pd(_mm256_broadcast_sd(A + 4 * lda + k), bk, c4);
  }
  _mm256_storeu_pd(C, c0);
  _mm256_storeu_pd(C + ldc, c1);
  _mm256_storeu_pd(C + 2 * ldc, c2);
  _mm256_storeu_pd(C + 3 * ldc, c3);
  _mm256_storeu_pd(C + 4 * ldc, c4);
}

#endif

// Per-matrix named-register specializations of gemm_fixed<M,K,N,float>.
// Must appear before gemm() so that specializations are visible at the point
// where gemm() calls gemm_fixed<m,k,n>(...).
#ifdef SPGEMM_KERNELS_H
#include SPGEMM_KERNELS_H
#endif

template <typename T>
inline void gemm_fallback(int M, int K, int N, const T *A, int lda, const T *B,
                          int ldb, T *C, int ldc) {
  // Column-tiled: for M>1 or K>1 and N in the thousands (the realistic case
  // for shapes that miss the specialized dispatch table), the untiled i,p,j
  // nest revisits a full N-wide slice of both operands on every inner-loop
  // step — C[i,:] gets read-modified-written once per p (K passes over N),
  // and worse, B[p,:] gets re-read from scratch on every i (H passes over a
  // row that's typically far bigger than L1/L2) since i is the outer loop.
  // Tiling the j/N dimension shrinks what has to stay resident across those
  // repeated passes to NB elements instead of N, so the K reuses of a given
  // C[i,tile] and the H reuses of a given B[p,tile] both land in L1/L2
  // instead of DRAM. This changes only which order tiles are visited in —
  // for any fixed (i,j) the p-accumulation still runs 0..K-1 in the same
  // order as before, so results are bit-identical to the untiled version.
  constexpr int NB = 256;
  for (int j0 = 0; j0 < N; j0 += NB) {
    const int jb = (NB < N - j0) ? NB : (N - j0);
    for (int i = 0; i < M; ++i) {
      T *__restrict__ Ci = C + i * ldc + j0;
      for (int p = 0; p < K; ++p) {
        T av = A[i * lda + p];
        const T *__restrict__ Bp = B + p * ldb + j0;
#pragma omp simd
        for (int j = 0; j < jb; ++j)
          Ci[j] += av * Bp[j];
      }
    }
  }
}

template <typename T>
inline void gemm(int M, int K, int N, const T *A, int lda, const T *B, int ldb,
                 T *C, int ldc) {
#ifdef SPGEMM_DISPATCH_H
#define DISPATCH(m, k, n)                                                      \
  if (M == m && K == k && N == n) {                                            \
    gemm_fixed<m, k, n>(A, lda, B, ldb, C, ldc);                               \
    return;                                                                    \
  }
#include SPGEMM_DISPATCH_H
#undef DISPATCH
#endif
  gemm_fallback(M, K, N, A, lda, B, ldb, C, ldc);
}

} // namespace cpu_detail

template <typename T>
std::vector<std::tuple<int, int, int>>
cpu_gemm_top_shapes(const FusionResult &fusion, const Matrix<T> &A,
                    const Matrix<T> &B, int top_n) {
  std::map<std::tuple<int, int, int>, long long> counts;
  for (int fi = 0; fi < (int)fusion.fused_blocks.size(); ++fi) {
    for (const auto &c : fusion.fused_contributions[fi]) {
      const int K = c.k.k1 - c.k.k0 + 1;
      counts[{A.blocks[c.a_index].h, K, B.blocks[c.b_index].w}]++;
    }
  }
  std::vector<std::pair<long long, std::tuple<int, int, int>>> ranked;
  ranked.reserve(counts.size());
  for (auto &[key, cnt] : counts)
    ranked.push_back({cnt, key});
  std::sort(ranked.rbegin(), ranked.rend());
  std::vector<std::tuple<int, int, int>> result;
  const int n = std::min(top_n, (int)ranked.size());
  result.reserve(n);
  for (int i = 0; i < n; ++i)
    result.push_back(ranked[i].second);
  return result;
}

template <typename T>
void cpu_gemm_histogram(const FusionResult &fusion, const Matrix<T> &A,
                        const Matrix<T> &B) {
  std::map<std::tuple<int, int, int>, long long> freq;
  long long total = 0;
  long long total_flops = 0;

  for (int fi = 0; fi < (int)fusion.fused_blocks.size(); ++fi) {
    for (const auto &c : fusion.fused_contributions[fi]) {
      const Block &ab = A.blocks[c.a_index];
      const Block &bb = B.blocks[c.b_index];
      const int K = c.k.k1 - c.k.k0 + 1;
      auto key = std::make_tuple(ab.h, K, bb.w);
      freq[key]++;
      total++;
      total_flops += (long long)ab.h * K * bb.w * 2;
    }
  }

  std::vector<std::pair<long long, std::tuple<int, int, int>>> sorted;
  sorted.reserve(freq.size());
  for (auto &[k, v] : freq)
    sorted.push_back({v, k});
  std::sort(sorted.rbegin(), sorted.rend());

  std::printf("\n── GEMM shape histogram (%lld total calls, %lld flops) ──\n",
              total, total_flops);
  std::printf("  %6s  %4s  %4s  %4s  %8s  %8s\n", "count", "M", "K", "N",
              "flops", "% calls");
  for (auto &[cnt, key] : sorted) {
    auto [M, K, N] = key;
    long long flops = (long long)M * K * N * 2;
    std::printf("  %6lld  %4d  %4d  %4d  %8lld  %7.2f%%\n", cnt, M, K, N, flops,
                100.0 * cnt / total);
  }
  std::printf("────────────────────────────────────────────────────────\n");
}

struct WorkItem {
  int fi;
  int a_index;
  int b_index;
  KRange k;
};

template <typename T> struct CpuPlan {
  Matrix<T> C;
  const FusionResult *fusion;
  const Matrix<T> *A;
  const Matrix<T> *B;
  std::vector<WorkItem> work;
};

template <typename T>
CpuPlan<T> cpu_plan_build(const FusionResult &fusion, const Matrix<T> &A,
                          const Matrix<T> &B) {
  CpuPlan<T> plan;
  plan.fusion = &fusion;
  plan.A = &A;
  plan.B = &B;

  Matrix<T> &C = plan.C;
  C.M = A.M;
  C.N = B.N;
  C.blocks = fusion.fused_blocks;
  C.n_values = static_cast<std::size_t>(assign_offsets(C.blocks));
  C.values = new T[C.n_values]();

  for (int fi = 0; fi < (int)fusion.fused_blocks.size(); ++fi)
    for (const auto &c : fusion.fused_contributions[fi])
      plan.work.push_back({fi, c.a_index, c.b_index, c.k});

  return plan;
}

template <typename T, bool Specialized> double cpu_compute(CpuPlan<T> &plan) {
  const FusionResult &fusion = *plan.fusion;
  const Matrix<T> &A = *plan.A;
  const Matrix<T> &B = *plan.B;
  Matrix<T> &C = plan.C;

  std::memset(C.values, 0, C.n_values * sizeof(T));

  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();

  if constexpr (Specialized) {
    for (const auto &w : plan.work) {
      const Block &fused = C.blocks[w.fi];
      const Block &ab = A.blocks[w.a_index];
      const Block &bb = B.blocks[w.b_index];
      const int K = w.k.k1 - w.k.k0 + 1;
      const T *A_ptr = A.values + ab.offset + (w.k.k0 - ab.c);
      const T *B_ptr =
          B.values + bb.offset + static_cast<long long>(w.k.k0 - bb.r) * bb.w;
      T *C_ptr = C.values + fused.offset +
                 static_cast<long long>(ab.r - fused.r) * fused.w +
                 (bb.c - fused.c);
      cpu_detail::gemm(ab.h, K, bb.w, A_ptr, ab.w, B_ptr, bb.w, C_ptr, fused.w);
    }
  } else {
    const int n = static_cast<int>(C.blocks.size());
#pragma omp parallel for schedule(dynamic, 16)
    for (int fi = 0; fi < n; ++fi) {
      const Block &fused = C.blocks[fi];
      for (const auto &c : fusion.fused_contributions[fi]) {
        const Block &ab = A.blocks[c.a_index];
        const Block &bb = B.blocks[c.b_index];
        const int K = c.k.k1 - c.k.k0 + 1;
        const T *A_ptr = A.values + ab.offset + (c.k.k0 - ab.c);
        const T *B_ptr =
            B.values + bb.offset + static_cast<long long>(c.k.k0 - bb.r) * bb.w;
        T *C_ptr = C.values + fused.offset +
                   static_cast<long long>(ab.r - fused.r) * fused.w +
                   (bb.c - fused.c);
        cpu_detail::gemm_fallback(ab.h, K, bb.w, A_ptr, ab.w, B_ptr, bb.w,
                                  C_ptr, fused.w);
      }
    }
  }

  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

} // namespace benchmark_core
