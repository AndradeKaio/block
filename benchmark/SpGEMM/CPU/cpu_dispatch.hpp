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

// Per-matrix named-register specializations of gemm_fixed<M,K,N,float>.
// Must appear before gemm() so that specializations are visible at the point
// where gemm() calls gemm_fixed<m,k,n>(...).
#ifdef SPGEMM_KERNELS_H
#include SPGEMM_KERNELS_H
#endif

template <typename T>
inline void gemm_fallback(int M, int K, int N, const T *__restrict__ A, int lda,
                          const T *__restrict__ B, int ldb, T *C, int ldc) {
  // Untiled: gemm_fallback only ever runs for shapes that miss the
  // specialized dispatch table, i.e. the tail of a mined block's own (M,K,N)
  // intersection sizes -- N here is a mined block's width (bb.w), not a
  // dense operand's column count. A previous version column-tiled this at
  // NB=256, reasoning from "N in the thousands" -- a real scenario for
  // SpMM's dense operand, but not for SpGEMM's mined-block N, which stays
  // small by construction (observed across real matrices: N never exceeds
  // a few dozen). With N that small, C[i,:] and B[p,:] already fit L1
  // comfortably in one untiled pass, so the tiling never actually tiled
  // anything -- it just added an outer loop, a per-call jb computation, and
  // extra pointer arithmetic that always ran exactly one iteration.
  for (int i = 0; i < M; ++i) {
    T *__restrict__ Ci = C + i * ldc;
    for (int p = 0; p < K; ++p) {
      T av = A[i * lda + p];
      const T *__restrict__ Bp = B + p * ldb;
#pragma omp simd
      for (int j = 0; j < N; ++j)
        Ci[j] += av * Bp[j];
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

// Diagnostic only -- not called by default (see prisma_cpu_bench.cpp's
// --histogram flag). Prints the full (M,K,N) shape distribution, not just
// the top-N cpu_gemm_top_shapes returns.
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

// Fully resolved GEMM call, no indirection needed at compute time. Building
// a flat array of these (below) instead of re-deriving A_ptr/B_ptr/C_ptr from
// Contribution{a_index,b_index,k} + a Block lookup on every access matters
// for two reasons: (1) fusion.fused_contributions is a
// vector<vector<Contribution>> -- one separate heap allocation per fused
// block (521,462 of them for pkustk08), so just walking it means chasing a
// fresh pointer for every block; (2) each contribution still needed two more
// indirections (A.blocks[a_index], B.blocks[b_index]) before the actual
// addresses were known. Neither pattern gives the hardware prefetcher
// anything to work with -- effectively random access per contribution.
// GemmWork collapses all of that into one flat, contiguous, resolved array
// built once in cpu_plan_build; cpu_compute's hot loop then just walks it
// sequentially.
template <typename T> struct GemmWork {
  const T *A_ptr;
  const T *B_ptr;
  T *C_ptr;
  int M, K, N;
  int lda, ldb, ldc;
};

template <typename T> struct CpuPlan {
  Matrix<T> C;
  std::vector<GemmWork<T>> work;  // flat, resolved, grouped by fi
  std::vector<int> fi_offsets;    // fi's work is work[fi_offsets[fi] .. fi_offsets[fi+1])
};

template <typename T>
CpuPlan<T> cpu_plan_build(const FusionResult &fusion, const Matrix<T> &A,
                          const Matrix<T> &B) {
  CpuPlan<T> plan;

  Matrix<T> &C = plan.C;
  C.M = A.M;
  C.N = B.N;
  C.blocks = fusion.fused_blocks;
  C.n_values = static_cast<std::size_t>(assign_offsets(C.blocks));
  C.values = new T[C.n_values]();

  const int n = static_cast<int>(fusion.fused_blocks.size());
  plan.fi_offsets.resize(n + 1, 0);
  for (int fi = 0; fi < n; ++fi)
    plan.fi_offsets[fi + 1] =
        plan.fi_offsets[fi] + static_cast<int>(fusion.fused_contributions[fi].size());
  plan.work.resize(plan.fi_offsets[n]);

  for (int fi = 0; fi < n; ++fi) {
    const Block &fused = C.blocks[fi];
    const int wi0 = plan.fi_offsets[fi];
    int wi = wi0;
    for (const auto &c : fusion.fused_contributions[fi]) {
      const Block &ab = A.blocks[c.a_index];
      const Block &bb = B.blocks[c.b_index];
      const int K = c.k.k1 - c.k.k0 + 1;
      plan.work[wi++] = GemmWork<T>{
          A.values + ab.offset + (c.k.k0 - ab.c),
          B.values + bb.offset + static_cast<long long>(c.k.k0 - bb.r) * bb.w,
          C.values + fused.offset +
              static_cast<long long>(ab.r - fused.r) * fused.w + (bb.c - fused.c),
          ab.h, K, bb.w,
          ab.w, bb.w, fused.w,
      };
    }

    // Sort this fi's slice by A_ptr (then B_ptr as tiebreaker) so entries
    // that read the same input block land next to each other in visit
    // order. One output block's own contribution list can span several
    // distinct A/B blocks (that's inherent to which pairs intersect it, not
    // something this reordering changes); what was missing is any guarantee
    // that a block reused by two contributions *within this same fi* was
    // still hot in cache between the two visits, since insertion order here
    // just reflects find_intersecting_pairs' query order, not locality.
    // Correctness is unaffected: entries within one fi can already target
    // overlapping C sub-regions and accumulate via +=, so this only changes
    // floating-point summation order -- the same tolerance-class effect as
    // changing thread count or OpenMP's scheduling, not which values get
    // summed.
    std::sort(plan.work.begin() + wi0, plan.work.begin() + wi,
              [](const GemmWork<T> &x, const GemmWork<T> &y) {
                if (x.A_ptr != y.A_ptr)
                  return x.A_ptr < y.A_ptr;
                return x.B_ptr < y.B_ptr;
              });
  }

  return plan;
}

// Both branches parallelize over fused blocks (fi), each thread serially
// walking that block's own contiguous slice of `work`. Fused blocks never
// overlap (see core/block_mining.cpp's overlap-prevention), so different
// fi's write disjoint C regions and this is race-free without atomics.
// Contributions *within* one fi CAN target overlapping sub-regions of that
// block (the accumulating `+=` inside gemm/gemm_fallback), so those must
// stay serial, which is why the parallel loop is over fi and not over the
// flat work array directly.
//
// Specialized previously ran this same per-fi/per-contribution loop but
// without #pragma omp parallel for -- entirely single-threaded, unlike the
// generic branch below, which silently capped every "specialized"/"top10"
// benchmark result to 1 core regardless of how many were available.
template <typename T, bool Specialized> double cpu_compute(CpuPlan<T> &plan) {
  Matrix<T> &C = plan.C;

  std::memset(C.values, 0, C.n_values * sizeof(T));

  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();

  const int n = static_cast<int>(plan.fi_offsets.size()) - 1;
#pragma omp parallel for schedule(dynamic, 16)
  for (int fi = 0; fi < n; ++fi) {
    for (int wi = plan.fi_offsets[fi]; wi < plan.fi_offsets[fi + 1]; ++wi) {
      const GemmWork<T> &w = plan.work[wi];
      if constexpr (Specialized)
        cpu_detail::gemm(w.M, w.K, w.N, w.A_ptr, w.lda, w.B_ptr, w.ldb,
                         w.C_ptr, w.ldc);
      else
        cpu_detail::gemm_fallback(w.M, w.K, w.N, w.A_ptr, w.lda, w.B_ptr,
                                  w.ldb, w.C_ptr, w.ldc);
    }
  }

  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

} // namespace benchmark_core
