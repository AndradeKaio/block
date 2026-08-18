#pragma once
// spmm_gpu_kernels.cuh — Output-centric SpMM kernels (C = S * D).
//
// Both kernels use the same dispatch: one CTA per (output_row_tile × col_tile).
// Each CTA loops over all sparse blocks in its tile, accumulates, direct-stores.
// No atomics anywhere.
//
// ── Kernel A: spmm_cuda_kernel ────────────────────────────────────────────────
//   Used for tiles with any non-TC block (or when TC is disabled).
//   dim3(TILE, TILE) = 256 threads.
//   Accumulator: sC[ROW_TILE][BN] in shmem.
//   Standard tiled GEMM: sS[ROW_TILE][TILE+1] × sD[TILE][BN] → sC.
//   Shmem: 14 KB (float) / 28 KB (double).
//
// ── Kernel B: spmm_tc_kernel ──────────────────────────────────────────────────
//   Used for tiles where every block has h%16==0 && w%16==0. T=float only.
//   dim3(256) = 8 warps; n_w_h=ROW_TILE/16=2 row-warps, n_w_n=BN/16=4 col-warps.
//   Accumulator: persistent WMMA c_frag[warp] across all blocks in the tile.
//   Each k-strip: load sS[ROW_TILE][K_STRIP+1] + sD[K_STRIP][BN] → mma_sync.
//   Final: store c_frag → warp_buf[warp][16][17] → C.
//   Shmem: ~14 KB.

#include <cuda_runtime.h>
#include <mma.h>
#include "spmm_gpu_plan.hpp"

namespace spmm_gpu {

// ── Kernel A: CUDA-core output-centric SpMM ───────────────────────────────────

template <typename T>
__global__ void spmm_cuda_kernel(
        const SpmmTask    * __restrict__ tasks,
        const int         * __restrict__ task_ids,
        const SpmmTileInfo* __restrict__ tiles,
        const int         * __restrict__ tile_blk_ids,
        const SpmmBlock   * __restrict__ blocks,
        const T           * __restrict__ S,
        const T           * __restrict__ D, int ldd, int dense_N,
        T                 *              C, int ldc)
{
    __shared__ T sC[ROW_TILE][BN];
    __shared__ T sS[ROW_TILE][TILE + 1];
    __shared__ T sD[TILE][BN];

    const SpmmTask&     task = tasks[task_ids[blockIdx.x]];
    const SpmmTileInfo& tile = tiles[task.tile_idx];
    const int j0  = task.col_tile * BN;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * TILE + tx;

    // Zero sC
    for (int i = tid; i < ROW_TILE * BN; i += TILE * TILE)
        reinterpret_cast<T*>(sC)[i] = T(0);
    __syncthreads();

    for (int bi = tile.blk_start; bi < tile.blk_start + tile.blk_count; ++bi) {
        const SpmmBlock& b         = blocks[tile_blk_ids[bi]];
        if (b.tc) continue;  // TC kernel handles TC blocks
        const int        local_row = b.row - tile.row_start;

        for (int k0 = 0; k0 < b.w; k0 += TILE) {
            // Load sS[ROW_TILE][TILE+1]: 512 elements, 256 threads → 2 each
            for (int i = tid; i < ROW_TILE * TILE; i += TILE * TILE) {
                const int r  = i / TILE;
                const int k  = i % TILE;
                const int br = r - local_row;
                sS[r][k] = (br >= 0 && br < b.h && k0 + k < b.w)
                         ? S[b.s_off + (long long)br * b.w + k0 + k]
                         : T(0);
            }
            // Load sD[TILE][BN]: 1024 elements, 256 threads → 4 each
            for (int i = tid; i < TILE * BN; i += TILE * TILE) {
                const int k = i / BN;
                const int c = i % BN;
                sD[k][c] = (k0 + k < b.w && j0 + c < dense_N)
                         ? D[(long long)(b.col + k0 + k) * ldd + j0 + c]
                         : T(0);
            }
            __syncthreads();

            // Accumulate: each thread owns 2 rows × 4 cols of sC
            #pragma unroll
            for (int ri = 0; ri < ROW_TILE / TILE; ++ri) {
                #pragma unroll
                for (int ci = 0; ci < BN / TILE; ++ci) {
                    T acc = T(0);
                    #pragma unroll
                    for (int k = 0; k < TILE; ++k)
                        acc += sS[ty + ri * TILE][k] * sD[k][tx + ci * TILE];
                    sC[ty + ri * TILE][tx + ci * TILE] += acc;
                }
            }
            __syncthreads();
        }
    }

    // Accumulate into C (TC kernel may have already written its contribution)
    #pragma unroll
    for (int ri = 0; ri < ROW_TILE / TILE; ++ri) {
        const int r = ty + ri * TILE;
        if (r < tile.row_h) {
            #pragma unroll
            for (int ci = 0; ci < BN / TILE; ++ci) {
                const int c = tx + ci * TILE;
                if (j0 + c < dense_N)
                    C[(long long)(tile.row_start + r) * ldc + j0 + c] += sC[r][c];
            }
        }
    }
}

// ── Kernel B: TC (WMMA tf32) output-centric SpMM ─────────────────────────────
//
// Only instantiated for T=float; all TC-eligible blocks in each tile accumulate
// into a persistent WMMA fragment across the block loop.
//
// Warp layout (ROW_TILE=32, BN=64):
//   n_w_h = ROW_TILE/16 = 2  warp rows
//   n_w_n = BN/16       = 4  warp cols
//   total = 8 warps = 256 threads
//
// Shmem:
//   sS  [ROW_TILE][K_STRIP+1]  = 32×9×4  = 1.1 KB   (one k-strip at a time)
//   sD  [K_STRIP][BN]          = 8×64×4  = 2.0 KB
//   warp_buf [8][16][17]       = 8704 B  = 8.5 KB   (staging for final store)
//   Total: ~12 KB

template <typename T>
__global__ void spmm_tc_kernel(
        const SpmmTask    * __restrict__ tasks,
        const int         * __restrict__ task_ids,
        const SpmmTileInfo* __restrict__ tiles,
        const int         * __restrict__ tile_blk_ids,
        const SpmmBlock   * __restrict__ blocks,
        const T           * __restrict__ S,
        const T           * __restrict__ D, int ldd, int dense_N,
        T                 *              C, int ldc)
{
    using namespace nvcuda::wmma;

    static constexpr int N_WARPS = (ROW_TILE / 16) * (BN / 16);  // = 8

    __shared__ float sS[ROW_TILE][K_STRIP + 1];
    __shared__ float sD[K_STRIP][BN];
    __shared__ float warp_buf[N_WARPS][16][17];  // ldm=17: bank-conflict-free

    const SpmmTask&     task    = tasks[task_ids[blockIdx.x]];
    const SpmmTileInfo& tile    = tiles[task.tile_idx];
    const int           j0      = task.col_tile * BN;
    const int           warp_id = threadIdx.x / 32;
    const int           tid     = threadIdx.x;
    const int           n_w_n   = BN / 16;       // = 4
    const int           wr      = warp_id / n_w_n;
    const int           wc      = warp_id % n_w_n;

    // Persistent WMMA accumulator — zero'd once, spans all blocks in tile
    fragment<accumulator, 16, 16, 8, float> c_frag;
    fill_fragment(c_frag, 0.f);

    for (int bi = tile.blk_start; bi < tile.blk_start + tile.blk_count; ++bi) {
        const SpmmBlock& b         = blocks[tile_blk_ids[bi]];
        if (!b.tc) continue;  // CUDA kernel handles non-TC blocks
        const int        local_row = b.row - tile.row_start;
        const int        n_strips  = (b.w + K_STRIP - 1) / K_STRIP;

        for (int strip = 0; strip < n_strips; ++strip) {
            const int k0 = strip * K_STRIP;

            // Collaborative load of sS[ROW_TILE][K_STRIP+1]
            for (int i = tid; i < ROW_TILE * K_STRIP; i += blockDim.x) {
                const int r  = i / K_STRIP;
                const int k  = i % K_STRIP;
                const int br = r - local_row;
                sS[r][k] = (br >= 0 && br < b.h && k0 + k < b.w)
                          ? static_cast<float>(
                                S[b.s_off + (long long)br * b.w + k0 + k])
                          : 0.f;
            }

            // Collaborative load of sD[K_STRIP][BN]
            for (int i = tid; i < K_STRIP * BN; i += blockDim.x) {
                const int k = i / BN;
                const int c = i % BN;
                sD[k][c] = (k0 + k < b.w && j0 + c < dense_N)
                          ? static_cast<float>(
                                D[(long long)(b.col + k0 + k) * ldd + j0 + c])
                          : 0.f;
            }
            __syncthreads();

            // WMMA: warp (wr, wc) computes the 16×16 tile at
            //   output rows [wr*16, wr*16+16)  ×  output cols [wc*16, wc*16+16)
            fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
            fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
            load_matrix_sync(a_frag, &sS[wr * 16][0], K_STRIP + 1);
            load_matrix_sync(b_frag, &sD[0][wc * 16], BN);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
    }

    // Store c_frag → warp_buf → C (accumulate: CUDA kernel adds non-TC contribution after)
    store_matrix_sync(&warp_buf[warp_id][0][0], c_frag, 17, mem_row_major);
    __syncwarp();

    for (int r = 0; r < 16; ++r) {
        if (wr * 16 + r >= tile.row_h) continue;
        const int abs_row = tile.row_start + wr * 16 + r;
        for (int c = 0; c < 16; ++c) {
            const int abs_col = j0 + wc * 16 + c;
            if (abs_col < dense_N)
                C[(long long)abs_row * ldc + abs_col] +=
                    static_cast<T>(warp_buf[warp_id][r][c]);
        }
    }
}

// ── dispatch_spmm ─────────────────────────────────────────────────────────────

struct SpmmTimes { float tc_ms = 0.f, cuda_ms = 0.f; };

template <typename T>
SpmmTimes dispatch_spmm(DevSpmmPlan<T>& dev) {
    // Zero C: rows in empty tiles (no contributing blocks) are never written
    SPMM_CUDA_CHECK(cudaMemset(dev.d_C, 0,
                               (std::size_t)dev.S_M * dev.dense_N * sizeof(T)));

    cudaStream_t s_tc, s_cuda;
    SPMM_CUDA_CHECK(cudaStreamCreate(&s_tc));
    SPMM_CUDA_CHECK(cudaStreamCreate(&s_cuda));

    cudaEvent_t e_tc_s, e_tc_e, e_cu_s, e_cu_e;
    SPMM_CUDA_CHECK(cudaEventCreate(&e_tc_s));
    SPMM_CUDA_CHECK(cudaEventCreate(&e_tc_e));
    SPMM_CUDA_CHECK(cudaEventCreate(&e_cu_s));
    SPMM_CUDA_CHECK(cudaEventCreate(&e_cu_e));

    // TC runs first; CUDA adds non-TC contributions after TC finishes.
    // Serialized so CUDA can do a simple += without atomics.
    if (dev.n_tc > 0) {
        SPMM_CUDA_CHECK(cudaEventRecord(e_tc_s, s_tc));
        spmm_tc_kernel<T><<<dev.n_tc, 256, 0, s_tc>>>(
            dev.d_tasks, dev.d_tc_ids, dev.d_tiles, dev.d_tile_blk_ids,
            dev.d_blocks, dev.d_S, dev.d_D, dev.ldd, dev.dense_N,
            dev.d_C, dev.ldc);
        SPMM_CUDA_CHECK(cudaEventRecord(e_tc_e, s_tc));
    }

    // Make CUDA stream wait for TC stream before writing
    SPMM_CUDA_CHECK(cudaStreamSynchronize(s_tc));

    if (dev.n_cuda > 0) {
        SPMM_CUDA_CHECK(cudaEventRecord(e_cu_s, s_cuda));
        spmm_cuda_kernel<T><<<dev.n_cuda, dim3(TILE, TILE), 0, s_cuda>>>(
            dev.d_tasks, dev.d_cuda_ids, dev.d_tiles, dev.d_tile_blk_ids,
            dev.d_blocks, dev.d_S, dev.d_D, dev.ldd, dev.dense_N,
            dev.d_C, dev.ldc);
        SPMM_CUDA_CHECK(cudaEventRecord(e_cu_e, s_cuda));
    }

    SPMM_CUDA_CHECK(cudaStreamSynchronize(s_cuda));

    SpmmTimes t;
    if (dev.n_tc   > 0) SPMM_CUDA_CHECK(cudaEventElapsedTime(&t.tc_ms,   e_tc_s, e_tc_e));
    if (dev.n_cuda > 0) SPMM_CUDA_CHECK(cudaEventElapsedTime(&t.cuda_ms, e_cu_s, e_cu_e));

    SPMM_CUDA_CHECK(cudaEventDestroy(e_tc_s)); SPMM_CUDA_CHECK(cudaEventDestroy(e_tc_e));
    SPMM_CUDA_CHECK(cudaEventDestroy(e_cu_s)); SPMM_CUDA_CHECK(cudaEventDestroy(e_cu_e));
    SPMM_CUDA_CHECK(cudaStreamDestroy(s_tc));
    SPMM_CUDA_CHECK(cudaStreamDestroy(s_cuda));
    return t;
}

} // namespace spmm_gpu
