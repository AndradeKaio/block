#pragma once
// spmm_gpu_plan.hpp — Output-centric symbolic phase for GPU SpMM (C = S * D).
//
// Dispatch: one CTA per (output_row_tile × col_tile).
// Each CTA owns a fixed ROW_TILE × BN output region, loops over every sparse
// block contributing to those rows, accumulates in shmem, direct-stores to C.
// No atomics. CTA count is independent of the number of sparse blocks.
//
// Two kernel paths:
//   TC   — all blocks in the tile satisfy h%16==0 && w%16==0; uses WMMA tf32.
//   CUDA — any other tile; uses CUDA-core scalar GEMM.
//
// Tiles with mixed blocks (some TC-eligible, some not) go to the CUDA kernel.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "block.hpp"
#include "matrix.hpp"

namespace spmm_gpu {

static constexpr int BN       = 64;   // output columns per CTA
static constexpr int K_STRIP  = 8;    // WMMA contraction strip depth
static constexpr int TILE     = 16;   // CUDA-core tile edge
static constexpr int ROW_TILE = 32;   // output rows per CTA (must be multiple of 16)

// ── Block classification ───────────────────────────────────────────────────────
//
// Default threshold (16×16): matches the old strict criterion for typical FEM
// matrices where TC-eligible blocks are exactly 16-multiple-sized.
// Set tc_min_h/tc_min_w lower (e.g. 4×4) to pad smaller blocks into TC path.

inline bool is_tc_block(int h, int w, int tc_min_h = 16, int tc_min_w = 16) noexcept {
    return h >= tc_min_h && w >= tc_min_w;
}

// ── Data structures ───────────────────────────────────────────────────────────

struct SpmmBlock {
    int       h, w;
    int       row;    // first output row  (= b.r)
    int       col;    // first S column    (= b.c; contraction-dim start in D)
    long long s_off;  // offset into S.values[]
    bool      tc;     // TC-eligible
};

struct SpmmTileInfo {
    int  row_start;   // = rt * ROW_TILE
    int  row_h;       // = min(ROW_TILE, M - row_start)
    int  blk_start;   // first index in tile_blk_ids[]
    int  blk_count;
    bool has_tc;      // tile has at least one TC-eligible block
    bool has_cuda;    // tile has at least one non-TC block
};

struct SpmmTask {
    int tile_idx;
    int col_tile;   // j0 = col_tile * BN
};

template <typename T>
struct SpmmGpuPlan {
    std::vector<SpmmBlock>    blocks;
    std::vector<SpmmTileInfo> tiles;
    std::vector<int>          tile_blk_ids;  // flat: block indices per tile
    std::vector<SpmmTask>     tasks;
    std::vector<int>          tc_task_ids;   // indices into tasks[] for TC kernel
    std::vector<int>          cuda_task_ids; // indices into tasks[] for CUDA kernel
    int dense_N, S_M, S_N;
    int n_row_tiles, n_col_tiles;
    int n_tasks, n_tc_tasks, n_cuda_tasks;
};

// ── build_spmm_plan ───────────────────────────────────────────────────────────

template <typename T>
SpmmGpuPlan<T> build_spmm_plan(const benchmark_core::Matrix<T>& S,
                                int dense_N, bool use_tc = true,
                                int tc_min_h = 16, int tc_min_w = 16) {
    SpmmGpuPlan<T> plan;
    plan.S_M         = S.M;
    plan.S_N         = S.N;
    plan.dense_N     = dense_N;
    plan.n_row_tiles = (S.M + ROW_TILE - 1) / ROW_TILE;
    plan.n_col_tiles = (dense_N + BN - 1) / BN;

    plan.blocks.reserve(S.blocks.size());
    for (const auto& b : S.blocks)
        plan.blocks.push_back({b.h, b.w, b.r, b.c, b.offset,
                               use_tc && is_tc_block(b.h, b.w, tc_min_h, tc_min_w)});

    // For each block, find every row tile it overlaps.
    // Block overlaps tile rt if: b.row < (rt+1)*ROW_TILE && b.row+b.h > rt*ROW_TILE
    // → rt in [ b.row/ROW_TILE , (b.row+b.h-1)/ROW_TILE ]
    std::vector<std::vector<int>> tile_blks(plan.n_row_tiles);
    for (int bi = 0; bi < (int)plan.blocks.size(); ++bi) {
        const auto& b = plan.blocks[bi];
        int rt0 = std::max(0,                     b.row / ROW_TILE);
        int rt1 = std::min(plan.n_row_tiles - 1, (b.row + b.h - 1) / ROW_TILE);
        for (int rt = rt0; rt <= rt1; ++rt)
            tile_blks[rt].push_back(bi);
    }

    plan.tiles.resize(plan.n_row_tiles);
    for (int rt = 0; rt < plan.n_row_tiles; ++rt) {
        plan.tiles[rt].row_start = rt * ROW_TILE;
        plan.tiles[rt].row_h     = std::min(ROW_TILE, S.M - rt * ROW_TILE);
        plan.tiles[rt].blk_start = (int)plan.tile_blk_ids.size();
        bool has_tc   = false;
        bool has_cuda = false;
        for (int bi : tile_blks[rt]) {
            plan.tile_blk_ids.push_back(bi);
            if (plan.blocks[bi].tc) has_tc   = true;
            else                    has_cuda  = true;
        }
        plan.tiles[rt].blk_count = (int)plan.tile_blk_ids.size() - plan.tiles[rt].blk_start;
        plan.tiles[rt].has_tc    = has_tc;
        plan.tiles[rt].has_cuda  = has_cuda;
    }

    // Build tasks. A mixed tile appears in both tc_task_ids and cuda_task_ids.
    for (int ti = 0; ti < plan.n_row_tiles; ++ti) {
        if (plan.tiles[ti].blk_count == 0) continue;
        for (int ct = 0; ct < plan.n_col_tiles; ++ct) {
            int task_idx = (int)plan.tasks.size();
            plan.tasks.push_back({ti, ct});
            if (plan.tiles[ti].has_tc)
                plan.tc_task_ids.push_back(task_idx);
            if (plan.tiles[ti].has_cuda)
                plan.cuda_task_ids.push_back(task_idx);
        }
    }
    plan.n_tasks      = (int)plan.tasks.size();
    plan.n_tc_tasks   = (int)plan.tc_task_ids.size();
    plan.n_cuda_tasks = (int)plan.cuda_task_ids.size();
    return plan;
}

// ── Device plan ───────────────────────────────────────────────────────────────

template <typename T>
struct DevSpmmPlan {
    T            *d_S            = nullptr;
    T            *d_D            = nullptr;
    T            *d_C            = nullptr;
    SpmmBlock    *d_blocks       = nullptr;
    SpmmTileInfo *d_tiles        = nullptr;
    int          *d_tile_blk_ids = nullptr;
    SpmmTask     *d_tasks        = nullptr;
    int          *d_tc_ids       = nullptr;
    int          *d_cuda_ids     = nullptr;
    int  n_tc = 0, n_cuda = 0;
    int  dense_N, S_M, S_N, ldd, ldc;
};

#define SPMM_CUDA_CHECK(x)                                                    \
    do {                                                                       \
        cudaError_t _e = (x);                                                 \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(_e));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

template <typename T>
DevSpmmPlan<T> upload_spmm_plan(const SpmmGpuPlan<T>& plan,
                                const benchmark_core::Matrix<T>& S,
                                const T* D_host) {
    DevSpmmPlan<T> dev;
    dev.dense_N = plan.dense_N;
    dev.S_M     = plan.S_M;
    dev.S_N     = plan.S_N;
    dev.ldd     = plan.dense_N;
    dev.ldc     = plan.dense_N;
    dev.n_tc    = plan.n_tc_tasks;
    dev.n_cuda  = plan.n_cuda_tasks;

    SPMM_CUDA_CHECK(cudaMalloc(&dev.d_S, S.n_values * sizeof(T)));
    SPMM_CUDA_CHECK(cudaMemcpy(dev.d_S, S.values,
                               S.n_values * sizeof(T), cudaMemcpyHostToDevice));

    const std::size_t n_D = (std::size_t)plan.S_N * plan.dense_N;
    SPMM_CUDA_CHECK(cudaMalloc(&dev.d_D, n_D * sizeof(T)));
    SPMM_CUDA_CHECK(cudaMemcpy(dev.d_D, D_host,
                               n_D * sizeof(T), cudaMemcpyHostToDevice));

    const std::size_t n_C = (std::size_t)plan.S_M * plan.dense_N;
    SPMM_CUDA_CHECK(cudaMalloc(&dev.d_C, n_C * sizeof(T)));
    SPMM_CUDA_CHECK(cudaMemset(dev.d_C, 0, n_C * sizeof(T)));

    SPMM_CUDA_CHECK(cudaMalloc(&dev.d_blocks,
                               plan.blocks.size() * sizeof(SpmmBlock)));
    SPMM_CUDA_CHECK(cudaMemcpy(dev.d_blocks, plan.blocks.data(),
                               plan.blocks.size() * sizeof(SpmmBlock),
                               cudaMemcpyHostToDevice));

    SPMM_CUDA_CHECK(cudaMalloc(&dev.d_tiles,
                               plan.tiles.size() * sizeof(SpmmTileInfo)));
    SPMM_CUDA_CHECK(cudaMemcpy(dev.d_tiles, plan.tiles.data(),
                               plan.tiles.size() * sizeof(SpmmTileInfo),
                               cudaMemcpyHostToDevice));

    if (!plan.tile_blk_ids.empty()) {
        SPMM_CUDA_CHECK(cudaMalloc(&dev.d_tile_blk_ids,
                                   plan.tile_blk_ids.size() * sizeof(int)));
        SPMM_CUDA_CHECK(cudaMemcpy(dev.d_tile_blk_ids, plan.tile_blk_ids.data(),
                                   plan.tile_blk_ids.size() * sizeof(int),
                                   cudaMemcpyHostToDevice));
    }

    SPMM_CUDA_CHECK(cudaMalloc(&dev.d_tasks,
                               plan.tasks.size() * sizeof(SpmmTask)));
    SPMM_CUDA_CHECK(cudaMemcpy(dev.d_tasks, plan.tasks.data(),
                               plan.tasks.size() * sizeof(SpmmTask),
                               cudaMemcpyHostToDevice));

    if (dev.n_tc > 0) {
        SPMM_CUDA_CHECK(cudaMalloc(&dev.d_tc_ids,
                                   plan.tc_task_ids.size() * sizeof(int)));
        SPMM_CUDA_CHECK(cudaMemcpy(dev.d_tc_ids, plan.tc_task_ids.data(),
                                   plan.tc_task_ids.size() * sizeof(int),
                                   cudaMemcpyHostToDevice));
    }
    if (dev.n_cuda > 0) {
        SPMM_CUDA_CHECK(cudaMalloc(&dev.d_cuda_ids,
                                   plan.cuda_task_ids.size() * sizeof(int)));
        SPMM_CUDA_CHECK(cudaMemcpy(dev.d_cuda_ids, plan.cuda_task_ids.data(),
                                   plan.cuda_task_ids.size() * sizeof(int),
                                   cudaMemcpyHostToDevice));
    }

    return dev;
}

template <typename T>
void free_dev_spmm_plan(DevSpmmPlan<T>& dev) {
    cudaFree(dev.d_S);
    cudaFree(dev.d_D);
    cudaFree(dev.d_C);
    cudaFree(dev.d_blocks);
    cudaFree(dev.d_tiles);
    cudaFree(dev.d_tile_blk_ids);
    cudaFree(dev.d_tasks);
    cudaFree(dev.d_tc_ids);
    cudaFree(dev.d_cuda_ids);
    dev = DevSpmmPlan<T>{};
}

template <typename T>
void sync_spmm_c(const DevSpmmPlan<T>& dev, T* C_host) {
    SPMM_CUDA_CHECK(cudaMemcpy(C_host, dev.d_C,
                               (std::size_t)dev.S_M * dev.dense_N * sizeof(T),
                               cudaMemcpyDeviceToHost));
}

} // namespace spmm_gpu
