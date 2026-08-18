#pragma once

// spmm_dispatch.hpp — specialized SpMM micro-kernels for fixed block shapes.
//
// For SpMM (C += A * B) where A is a small block (H×W), B is a wide dense
// matrix (W×N, N large and runtime), and C is the output (H×N).
//
// Strategy: fix H, W, and NR (column chunk width) at compile time so every
// loop inside spmm_chunk<H,W,NR> is fully unrolled — same philosophy as
// gemm_fixed<M,K,N> for SpGEMM. The j-loop over N lives in the dispatcher,
// which calls spmm_chunk once per NR-column slice and handles the tail.
//
// NR is chosen to keep register pressure within 32 ZMM (AVX-512):
//   NV = NR/8 AVX-512 vectors per row; accumulators = H*NV.
//   H ≤ 14 → NR=16 (NV=2, up to 28 ZMM accumulators)
//   H = 18 → NR= 8 (NV=1, 18 ZMM accumulators)

#include "cpu_dispatch.hpp"  // for gemm_fallback

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace spmm_detail {

// ---------------------------------------------------------------------------
// spmm_chunk<H, W, NR> — processes exactly NR columns, all loops unrolled.
// NR must be a multiple of 8 (AVX-512 double width).
// ---------------------------------------------------------------------------

template <int H, int W, int NR>
inline void spmm_chunk(const double* __restrict__ A, int lda,
                       const double* __restrict__ B, int ldb,
                       double*       __restrict__ C, int ldc) {
    static_assert(NR % 4 == 0, "NR must be a multiple of 4");
    constexpr int NV = NR / 8;  // number of AVX-512 vectors per row (NR % 8 == 0 on AVX-512)

    // Hoist A block into scalars — compiler keeps in registers across NV loop.
    double a_reg[H][W];
    for (int i = 0; i < H; ++i)
        for (int p = 0; p < W; ++p)
            a_reg[i][p] = A[i * lda + p];

#if defined(__AVX512F__)
    // H*NV accumulator registers — all loops over H, NV, W are compile-time constants.
    __m512d c[H][NV];
    for (int i = 0; i < H; ++i)
        for (int v = 0; v < NV; ++v)
            c[i][v] = _mm512_loadu_pd(C + i * ldc + v * 8);

    for (int p = 0; p < W; ++p) {
        __m512d b[NV];
        for (int v = 0; v < NV; ++v)
            b[v] = _mm512_loadu_pd(B + p * ldb + v * 8);
        for (int i = 0; i < H; ++i)
            for (int v = 0; v < NV; ++v)
                c[i][v] = _mm512_fmadd_pd(_mm512_set1_pd(a_reg[i][p]), b[v], c[i][v]);
    }

    for (int i = 0; i < H; ++i)
        for (int v = 0; v < NV; ++v)
            _mm512_storeu_pd(C + i * ldc + v * 8, c[i][v]);

#elif defined(__AVX2__) && defined(__FMA__)
    // AVX2 fallback: NR/4 YMM vectors per row (4 doubles each).
    static_assert(NR % 4 == 0, "NR must be a multiple of 4 for AVX2");
    constexpr int NV4 = NR / 4;
    __m256d c[H][NV4];
    for (int i = 0; i < H; ++i)
        for (int v = 0; v < NV4; ++v)
            c[i][v] = _mm256_loadu_pd(C + i * ldc + v * 4);

    for (int p = 0; p < W; ++p) {
        __m256d b[NV4];
        for (int v = 0; v < NV4; ++v)
            b[v] = _mm256_loadu_pd(B + p * ldb + v * 4);
        for (int i = 0; i < H; ++i)
            for (int v = 0; v < NV4; ++v)
                c[i][v] = _mm256_fmadd_pd(_mm256_set1_pd(a_reg[i][p]), b[v], c[i][v]);
    }

    for (int i = 0; i < H; ++i)
        for (int v = 0; v < NV4; ++v)
            _mm256_storeu_pd(C + i * ldc + v * 4, c[i][v]);
#else
    // Scalar fallback (non-SIMD builds).
    for (int i = 0; i < H; ++i)
        for (int p = 0; p < W; ++p)
            for (int j = 0; j < NR; ++j)
                C[i * ldc + j] += a_reg[i][p] * B[p * ldb + j];
#endif
}

// ---------------------------------------------------------------------------
// spmm_kernel<H, W, NR> — sweeps j in NR-column chunks, scalar tail.
// This is the per-shape entry point called by spmm_dispatch.
// ---------------------------------------------------------------------------

template <int H, int W, int NR>
inline void spmm_kernel(int N,
                        const double* __restrict__ A, int lda,
                        const double* __restrict__ B, int ldb,
                        double*       __restrict__ C, int ldc) {
    // Prefetch first B chunk: W separate row streams may exceed HW prefetch
    // stream capacity, so cover them explicitly.
    // __builtin_prefetch is a portable GCC/Clang built-in (x86 PREFETCHT1 or
    // ARM PRFM PLDL2KEEP); no header required, locality=2 targets L2.
    for (int p = 0; p < W; ++p)
        __builtin_prefetch(B + p * ldb, 0, 2);

    int j = 0;
    for (; j + NR <= N; j += NR) {
        // Prefetch next B chunk into L2 while computing the current one.
        // Each row of B is ldb doubles away from the previous, which can
        // exceed the HW prefetcher's stream tracking on high-W shapes.
        if (j + NR < N) {
            for (int p = 0; p < W; ++p)
                __builtin_prefetch(B + j + NR + p * ldb, 0, 2);
        }
        spmm_chunk<H, W, NR>(A, lda, B + j, ldb, C + j, ldc);
    }

    // Hoist A into registers once for the tail (A is tiny, guaranteed in L1).
    double a_reg[H][W];
    for (int i = 0; i < H; ++i)
        for (int p = 0; p < W; ++p)
            a_reg[i][p] = A[i * lda + p];

    // Vectorised tail: handle remaining 8-column chunks using the generic
    // spmm_chunk<H,W,8> template — on AVX-512 this is one ZMM per row, on
    // AVX2 two YMM, on scalar the 8-iteration body auto-vectorises (NEON/SVE).
    // Always better than the unguarded scalar residue loop below.
    for (; j + 8 <= N; j += 8)
        spmm_chunk<H, W, 8>(A, lda, B + j, ldb, C + j, ldc);

    // Scalar residue for the final < 8 columns.
    // Loop order: outer (i,p) so the inner j-stride over B[p*ldb+jj] is 1
    // (sequential), not stride-N per j as the naive j-outer ordering would give.
    if (j < N) {
        for (int i = 0; i < H; ++i)
            for (int p = 0; p < W; ++p) {
                const double av = a_reg[i][p];
                for (int jj = j; jj < N; ++jj)
                    C[i * ldc + jj] += av * B[p * ldb + jj];
            }
    }
}

// Named-register specializations for each mined shape — override the generic
// spmm_chunk template above with structurally register-allocated versions.
// Must be included inside namespace spmm_detail so the specializations find
// the primary template.
#include "spmm_kernels_generated.hpp"

} // namespace spmm_detail

// spmm_dispatch() — generated by gen_spmm_kernels.py.
// Re-run the generator to update shapes; do not edit spmm_dispatch_table.hpp directly.
#include "spmm_dispatch_table.hpp"
