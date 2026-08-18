// tc_spgemm.cu
//
// GPU implementation of the core ideas from TC_SpGEMM (IPDPS 2026):
//   "High Performance Sparse General Matrix Multiplication
//    with Tensor Core-Accelerated"
//
// What this file implements:
//   1. CSR-Dense data structure (result of preprocessing)
//   2. Block pair generation  — Algorithm 4 (two-pass: count → scan → fill)
//   3. TC_SpGEMM WMMA kernel  — Algorithm 5 (one warp per block pair)
//       ▸ Tensor Core stage : wmma::mma_sync  (16×16×16, FP16 in → FP32 acc)
//       ▸ CUDA Core stage   : ballot/popc + row_ori/col_ori index recovery
//   4. Preprocessing
//       ▸ preprocess_A: Block-wise Similarity Reordering — rows sorted by their
//         block-column-index set (lexicographic, approximating MinHash/LSH);
//         fills row_ori[] for coordinate recovery
//       ▸ preprocess_B: Row-Block Column Compression — per 16-row window,
//         active columns are deduplicated and packed into dense 16×16 tiles;
//         fills col_ori[] for coordinate recovery
//   5. COO deduplication — Thrust sort_by_key + reduce_by_key to merge
//      contributions from multiple A-blocks to the same C entry
//   6. Host driver  (tc_spgemm)
//
// Build:
//   nvcc -O3 -arch=sm_80 -std=c++17 tc_spgemm.cu -o tc_spgemm   # Ampere
//   nvcc -O3 -arch=sm_90 -std=c++17 tc_spgemm.cu -o tc_spgemm   # Hopper
//   nvcc -O3 -arch=sm_120 -std=c++17 tc_spgemm.cu -o tc_spgemm  # Blackwell

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <chrono>

using namespace nvcuda;

// Per-phase timing returned by tc_spgemm_compute
struct ComputeTimes {
    double symbolic_ms; // upload A/B + pair generation (k_count + scan + k_fill)
    double kernel_ms;   // TC WMMA kernel only
};

// ─────────────────────────────────────────────────────────────────────────────
// Error checking
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int BS   = 16;       // WMMA fragment side — must be 16 for nvcuda
static constexpr int BS2  = BS * BS;  // elements per dense block = 256
static constexpr int WARP = 32;

// ─────────────────────────────────────────────────────────────────────────────
// CSR-Dense matrix representation
//
// Two-level layout (paper Fig. 8):
//
//   Block level (sparse):
//     block_row_ptr[i]..block_row_ptr[i+1)  — range of non-empty blocks in row i
//     block_col_idx[b]                       — block-column of the b-th block
//
//   Value level (dense):
//     dense_data[b*BS2 .. (b+1)*BS2)        — 16×16 row-major FP16 tile
//                                              ready for wmma::load_matrix_sync
//
//   Index-recovery auxiliary arrays:
//     row_ori[bi*BS + r]  = original row for reordered row (bi*BS+r)   [A only]
//     col_ori[b*BS  + c]  = original col for compressed col c of block b [B only]
// ─────────────────────────────────────────────────────────────────────────────
struct CSRDense {
    int     nblk_rows;      // number of block rows  = ⌈nrows / BS⌉
    int     nblk_cols;      // number of block cols (informational)
    int     nnzb;           // number of non-empty dense blocks

    int*    block_row_ptr;  // [nblk_rows + 1]
    int*    block_col_idx;  // [nnzb]
    __half* dense_data;     // [nnzb * BS2]  row-major FP16

    int*    row_ori;        // [nblk_rows * BS]  — filled by preprocess_A; nullptr for B
    int*    col_ori;        // [nnzb * BS]        — filled by preprocess_B; nullptr for A
};

// ─────────────────────────────────────────────────────────────────────────────
// Block pair generation — Algorithm 4
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_count_pairs(
    const int* __restrict__ A_blk_col,
    const int* __restrict__ B_blk_ptr,
    int A_nnzb,
    int* __restrict__ counts
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < A_nnzb) {
        int j     = A_blk_col[a];
        counts[a] = B_blk_ptr[j+1] - B_blk_ptr[j];
    }
}

__global__ void k_blk_row_of(
    const int* __restrict__ A_blk_ptr,
    int nblk_rows,
    int* __restrict__ blk_row_of
) {
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi < nblk_rows)
        for (int a = A_blk_ptr[bi]; a < A_blk_ptr[bi+1]; ++a)
            blk_row_of[a] = bi;
}

__global__ void k_fill_pairs(
    const int* __restrict__ A_blk_col,
    const int* __restrict__ A_blk_row_of,
    const int* __restrict__ B_blk_ptr,
    int A_nnzb,
    const int* __restrict__ offsets,
    int* __restrict__ pair_a,
    int* __restrict__ pair_b,
    int* __restrict__ pair_bi
) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= A_nnzb) return;

    int j    = A_blk_col[a];
    int bi   = A_blk_row_of[a];
    int base = offsets[a];

    for (int b = B_blk_ptr[j], t = 0; b < B_blk_ptr[j+1]; ++b, ++t) {
        pair_a [base+t] = a;
        pair_b [base+t] = b;
        pair_bi[base+t] = bi;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device functors for COO key packing / unpacking (avoids --expt-extended-lambda)
// ─────────────────────────────────────────────────────────────────────────────
struct PackKeyFn {
    int ncols;
    __device__ long long operator()(int r, int c) const {
        return (long long)r * ncols + c;
    }
};
struct UnpackRowFn {
    int ncols;
    __device__ int operator()(long long k) const { return (int)(k / ncols); }
};
struct UnpackColFn {
    int ncols;
    __device__ int operator()(long long k) const { return (int)(k % ncols); }
};

// ─────────────────────────────────────────────────────────────────────────────
// TC_SpGEMM kernel — Algorithm 5
// ─────────────────────────────────────────────────────────────────────────────
template<int WPB>
__global__ void tc_spgemm_kernel(
    const __half* __restrict__ A_data,
    const __half* __restrict__ B_data,

    const int* __restrict__ pair_a,
    const int* __restrict__ pair_b,
    const int* __restrict__ pair_bi,
    int num_pairs,

    const int* __restrict__ A_row_ori,
    const int* __restrict__ B_col_ori,

    int*   __restrict__ out_row,
    int*   __restrict__ out_col,
    float* __restrict__ out_val,
    int*   __restrict__ global_nnz
) {
    const int warp_in_blk = threadIdx.x / WARP;
    const int gw          = blockIdx.x * WPB + warp_in_blk;
    const int lane        = threadIdx.x & (WARP - 1);

    if (gw >= num_pairs) return;

    const int a_blk = pair_a [gw];
    const int b_blk = pair_b [gw];
    const int bi    = pair_bi[gw];

    // ── Tensor Core stage ──────────────────────────────────────────────────
    wmma::fragment<wmma::matrix_a,    BS, BS, BS, __half, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b,    BS, BS, BS, __half, wmma::row_major> fB;
    wmma::fragment<wmma::accumulator, BS, BS, BS, float>                   fC;

    wmma::fill_fragment(fC, 0.0f);
    wmma::load_matrix_sync(fA, A_data + (long long)a_blk * BS2, BS);
    wmma::load_matrix_sync(fB, B_data + (long long)b_blk * BS2, BS);
    wmma::mma_sync(fC, fA, fB, fC);

    extern __shared__ float smem[];
    float* wbuf = smem + warp_in_blk * BS2;
    wmma::store_matrix_sync(wbuf, fC, BS, wmma::mem_row_major);
    __syncwarp();

    // ── CUDA Core stage ────────────────────────────────────────────────────
    const int* bcol = B_col_ori + b_blk * BS;

    for (int base = 0; base < BS2; base += WARP) {
        const int   t    = base + lane;
        const float v    = wbuf[t];
        const bool  nonz = (v != 0.0f);

        const unsigned ballot = __ballot_sync(0xFFFFFFFFu, nonz);
        (void)ballot;

        if (nonz) {
            const int local_r = t / BS;
            const int local_c = t % BS;

            const int orig_row = A_row_ori[bi * BS + local_r];
            const int orig_col = bcol[local_c];

            const int idx = atomicAdd(global_nnz, 1);
            out_row[idx]  = orig_row;
            out_col[idx]  = orig_col;
            out_val[idx]  = v;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing: A — Block-wise Similarity Reordering
//
// For each row of A, compute its block-column index set (the set of block-column
// positions that have at least one nonzero).  Sort rows lexicographically by
// that set so rows with similar sparsity patterns become adjacent in groups of
// BS.  This approximates the MinHash/LSH bucketing from the paper and increases
// the density of each 16×16 dense tile, giving Tensor Cores more useful work.
//
// Outputs:
//   out.block_row_ptr, out.block_col_idx  — block-level sparsity of reordered A
//   out.dense_data                         — FP16 tiles (row-major per block)
//   out.row_ori[bi*BS + r]                 — original row for reordered position
//
// Padding rows (when nrows is not a multiple of BS) get row_ori = -1 and
// produce all-zero tile entries, so the kernel filters them out via v != 0.0f.
// ─────────────────────────────────────────────────────────────────────────────
void preprocess_A(
    int          nrows,
    int          ncols,
    const int*   row_ptr,   // [nrows+1]
    const int*   col_idx,   // [nnz]
    const float* vals,      // [nnz]
    CSRDense&    out
) {
    const int nblk_rows = (nrows + BS - 1) / BS;
    const int nblk_cols = (ncols + BS - 1) / BS;

    // Build block-column set for each row
    std::vector<std::vector<int>> row_blk_cols(nrows);
    for (int i = 0; i < nrows; i++) {
        std::set<int> blk_set;
        for (int p = row_ptr[i]; p < row_ptr[i+1]; p++)
            blk_set.insert(col_idx[p] / BS);
        row_blk_cols[i].assign(blk_set.begin(), blk_set.end());
    }

    // Sort rows by their block-column sets (lexicographic)
    std::vector<int> row_perm(nrows);
    std::iota(row_perm.begin(), row_perm.end(), 0);
    std::stable_sort(row_perm.begin(), row_perm.end(), [&](int a, int b) {
        return row_blk_cols[a] < row_blk_cols[b];
    });

    // Fill row_ori: reordered index → original row (-1 for padding)
    out.row_ori = new int[nblk_rows * BS];
    std::fill(out.row_ori, out.row_ori + nblk_rows * BS, -1);
    for (int r = 0; r < nrows; r++)
        out.row_ori[r] = row_perm[r];

    // Build block structure of the reordered matrix
    std::vector<int>    blk_row_ptr(nblk_rows + 1, 0);
    std::vector<int>    blk_col_idx_vec;
    std::vector<__half> dense_data_vec;

    for (int bi = 0; bi < nblk_rows; bi++) {
        const int row_start = bi * BS;
        const int row_end   = std::min(row_start + BS, nrows);

        // Collect all block-columns touched by rows in this block-row
        std::set<int> blk_cols_in_block;
        for (int r = row_start; r < row_end; r++) {
            int orig = row_perm[r];
            for (int p = row_ptr[orig]; p < row_ptr[orig+1]; p++)
                blk_cols_in_block.insert(col_idx[p] / BS);
        }

        blk_row_ptr[bi] = (int)blk_col_idx_vec.size();

        for (int j_blk : blk_cols_in_block) {
            blk_col_idx_vec.push_back(j_blk);

            // Fill 16×16 FP16 tile (zero-initialised)
            float tile[BS][BS] = {};
            for (int r = row_start; r < row_end; r++) {
                int orig    = row_perm[r];
                int local_r = r - row_start;
                for (int p = row_ptr[orig]; p < row_ptr[orig+1]; p++) {
                    if (col_idx[p] / BS == j_blk)
                        tile[local_r][col_idx[p] % BS] += vals[p];
                }
            }

            for (int r = 0; r < BS; r++)
                for (int c = 0; c < BS; c++)
                    dense_data_vec.push_back(__float2half(tile[r][c]));
        }
    }
    blk_row_ptr[nblk_rows] = (int)blk_col_idx_vec.size();

    const int nnzb = (int)blk_col_idx_vec.size();

    out.nblk_rows     = nblk_rows;
    out.nblk_cols     = nblk_cols;
    out.nnzb          = nnzb;
    out.block_row_ptr = new int[nblk_rows + 1];
    out.block_col_idx = new int[nnzb ? nnzb : 1];
    out.dense_data    = new __half[(long long)nnzb * BS2 + 1];
    out.col_ori       = nullptr;

    std::copy(blk_row_ptr.begin(),    blk_row_ptr.end(),    out.block_row_ptr);
    std::copy(blk_col_idx_vec.begin(), blk_col_idx_vec.end(), out.block_col_idx);
    std::copy(dense_data_vec.begin(), dense_data_vec.end(), out.dense_data);
}

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessing: B — Row-Block Column Compression
//
// For each block-row j of B (rows j*BS .. j*BS+BS-1):
//   1. Collect all column indices that appear in those rows → active_cols
//   2. Assign each active column a local index 0..nc-1 (nc = |active_cols|)
//   3. Pack every group of BS local indices into one 16×16 dense tile;
//      each tile covers BS rows × BS compressed columns
//   4. Record col_ori[global_block * BS + c] = original column for local col c
//
// This removes the structural zeros within each 16-row window so WMMA sees
// actual data instead of empty columns.
//
// Padding compressed-columns (when nc is not a multiple of BS) get
// col_ori = -1 and produce all-zero tile entries.
// ─────────────────────────────────────────────────────────────────────────────
void preprocess_B(
    int          nrows,
    int          ncols,
    const int*   row_ptr,
    const int*   col_idx,
    const float* vals,
    CSRDense&    out
) {
    const int nblk_rows = (nrows + BS - 1) / BS;

    std::vector<int>    blk_row_ptr(nblk_rows + 1, 0);
    std::vector<int>    blk_col_idx_vec;
    std::vector<__half> dense_data_vec;
    std::vector<int>    col_ori_vec;

    for (int j = 0; j < nblk_rows; j++) {
        const int row_start = j * BS;
        const int row_end   = std::min(row_start + BS, nrows);

        // Collect + sort + deduplicate column indices for this window
        std::set<int> active_col_set;
        for (int r = row_start; r < row_end; r++)
            for (int p = row_ptr[r]; p < row_ptr[r+1]; p++)
                active_col_set.insert(col_idx[p]);

        std::vector<int> active_cols(active_col_set.begin(), active_col_set.end());
        const int nc = (int)active_cols.size();

        // Map original column → local index
        std::unordered_map<int,int> col_to_local;
        col_to_local.reserve(nc);
        for (int k = 0; k < nc; k++)
            col_to_local[active_cols[k]] = k;

        // Partition active_cols into blocks of BS compressed columns
        const int nblks = (nc == 0) ? 0 : (nc + BS - 1) / BS;
        blk_row_ptr[j] = (int)blk_col_idx_vec.size();

        for (int b = 0; b < nblks; b++) {
            blk_col_idx_vec.push_back(b);   // local block-column within block-row j

            // col_ori for this block (padding positions get -1)
            for (int c = 0; c < BS; c++) {
                int local_idx = b * BS + c;
                col_ori_vec.push_back(local_idx < nc ? active_cols[local_idx] : -1);
            }

            // Fill 16×16 FP16 tile (zero-initialised)
            float tile[BS][BS] = {};
            for (int r = row_start; r < row_end; r++) {
                int local_r = r - row_start;
                for (int p = row_ptr[r]; p < row_ptr[r+1]; p++) {
                    int lc = col_to_local.at(col_idx[p]);
                    if (lc / BS == b)
                        tile[local_r][lc % BS] += vals[p];
                }
            }

            for (int r = 0; r < BS; r++)
                for (int c = 0; c < BS; c++)
                    dense_data_vec.push_back(__float2half(tile[r][c]));
        }
    }
    blk_row_ptr[nblk_rows] = (int)blk_col_idx_vec.size();

    const int nnzb = (int)blk_col_idx_vec.size();

    out.nblk_rows     = nblk_rows;
    out.nblk_cols     = (ncols + BS - 1) / BS;   // informational; uncompressed view
    out.nnzb          = nnzb;
    out.block_row_ptr = new int[nblk_rows + 1];
    out.block_col_idx = new int[nnzb ? nnzb : 1];
    out.dense_data    = new __half[(long long)nnzb * BS2 + 1];
    out.col_ori       = new int[(long long)nnzb * BS + 1];
    out.row_ori       = nullptr;

    std::copy(blk_row_ptr.begin(),    blk_row_ptr.end(),    out.block_row_ptr);
    std::copy(blk_col_idx_vec.begin(), blk_col_idx_vec.end(), out.block_col_idx);
    std::copy(dense_data_vec.begin(), dense_data_vec.end(), out.dense_data);
    std::copy(col_ori_vec.begin(),    col_ori_vec.end(),    out.col_ori);
}

// Free host-side CSRDense arrays allocated by preprocess_A / preprocess_B
void free_csrdense(CSRDense& m) {
    delete[] m.block_row_ptr;
    delete[] m.block_col_idx;
    delete[] m.dense_data;
    delete[] m.row_ori;
    delete[] m.col_ori;
    m.block_row_ptr = nullptr;
    m.block_col_idx = nullptr;
    m.dense_data    = nullptr;
    m.row_ori       = nullptr;
    m.col_ori       = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw COO result from the compute phase (device arrays, before deduplication)
// ─────────────────────────────────────────────────────────────────────────────
struct RawCOO {
    int*   d_row;   // device memory, raw_nnz entries
    int*   d_col;   // device memory, raw_nnz entries
    float* d_val;   // device memory, raw_nnz entries
    int    raw_nnz; // actual entry count (already on host)
};

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1 of the host driver: upload A & B, generate block pairs, launch the
// TC_SpGEMM kernel, and synchronize.  Returns the raw (possibly duplicate) COO
// output in device memory via `out`.  All temporary device allocations (dA, dB,
// pair arrays, etc.) are freed before returning; only out.d_row/col/val remain
// allocated and must be freed by tc_spgemm_postprocess or the caller.
// ─────────────────────────────────────────────────────────────────────────────
ComputeTimes tc_spgemm_compute(
    const CSRDense& hA,
    const CSRDense& hB,
    int             ncols_C,
    RawCOO&         out
) {
    using Clock = std::chrono::steady_clock;
    using Fms   = std::chrono::duration<double, std::milli>;
    const int T = 256;
    auto t_sym_start = Clock::now();

    // ── Upload A ──────────────────────────────────────────────────────────
    CSRDense dA{};
    dA.nblk_rows = hA.nblk_rows;
    dA.nblk_cols = hA.nblk_cols;
    dA.nnzb      = hA.nnzb;

    CUDA_CHECK(cudaMalloc(&dA.block_row_ptr, (hA.nblk_rows+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dA.block_col_idx,  hA.nnzb         * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dA.dense_data,    (long long)hA.nnzb * BS2 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dA.row_ori,       (long long)hA.nblk_rows * BS * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dA.block_row_ptr, hA.block_row_ptr,
                          (hA.nblk_rows+1) * sizeof(int),                   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA.block_col_idx, hA.block_col_idx,
                          hA.nnzb * sizeof(int),                             cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA.dense_data,    hA.dense_data,
                          (long long)hA.nnzb * BS2 * sizeof(__half),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA.row_ori,       hA.row_ori,
                          (long long)hA.nblk_rows * BS * sizeof(int),        cudaMemcpyHostToDevice));

    // ── Upload B ──────────────────────────────────────────────────────────
    CSRDense dB{};
    dB.nblk_rows = hB.nblk_rows;
    dB.nblk_cols = hB.nblk_cols;
    dB.nnzb      = hB.nnzb;

    CUDA_CHECK(cudaMalloc(&dB.block_row_ptr, (hB.nblk_rows+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dB.block_col_idx,  hB.nnzb         * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dB.dense_data,    (long long)hB.nnzb * BS2 * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB.col_ori,       (long long)hB.nnzb * BS  * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dB.block_row_ptr, hB.block_row_ptr,
                          (hB.nblk_rows+1) * sizeof(int),                   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.block_col_idx, hB.block_col_idx,
                          hB.nnzb * sizeof(int),                             cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.dense_data,    hB.dense_data,
                          (long long)hB.nnzb * BS2 * sizeof(__half),         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.col_ori,       hB.col_ori,
                          (long long)hB.nnzb * BS  * sizeof(int),            cudaMemcpyHostToDevice));

    // ── Build blk_row_of ─────────────────────────────────────────────────
    int* d_blk_row_of;
    CUDA_CHECK(cudaMalloc(&d_blk_row_of, hA.nnzb * sizeof(int)));
    k_blk_row_of<<<(hA.nblk_rows + T - 1) / T, T>>>(
        dA.block_row_ptr, hA.nblk_rows, d_blk_row_of);

    // ── Count pairs per A-block ───────────────────────────────────────────
    int* d_counts;
    CUDA_CHECK(cudaMalloc(&d_counts, (hA.nnzb + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counts, 0, (hA.nnzb + 1) * sizeof(int)));
    k_count_pairs<<<(hA.nnzb + T - 1) / T, T>>>(
        dA.block_col_idx, dB.block_row_ptr, hA.nnzb, d_counts);

    // ── Exclusive prefix scan → total pairs ──────────────────────────────
    thrust::device_ptr<int> cnt_ptr(d_counts);
    thrust::exclusive_scan(thrust::device, cnt_ptr, cnt_ptr + hA.nnzb + 1, cnt_ptr);

    int total_pairs = 0;
    CUDA_CHECK(cudaMemcpy(&total_pairs, d_counts + hA.nnzb, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Total block pairs: %d\n", total_pairs);

    auto free_upload = [&]() {
        cudaFree(dA.block_row_ptr); cudaFree(dA.block_col_idx);
        cudaFree(dA.dense_data);    cudaFree(dA.row_ori);
        cudaFree(dB.block_row_ptr); cudaFree(dB.block_col_idx);
        cudaFree(dB.dense_data);    cudaFree(dB.col_ori);
        cudaFree(d_blk_row_of);
        cudaFree(d_counts);
    };

    if (total_pairs == 0) {
        auto t_sym_end_zero = Clock::now();
        free_upload();
        out = {nullptr, nullptr, nullptr, 0};
        return { Fms(t_sym_end_zero - t_sym_start).count(), 0.0 };
    }

    // ── Fill pair arrays ──────────────────────────────────────────────────
    int *d_pa, *d_pb, *d_pbi;
    CUDA_CHECK(cudaMalloc(&d_pa,  total_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pb,  total_pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pbi, total_pairs * sizeof(int)));

    k_fill_pairs<<<(hA.nnzb + T - 1) / T, T>>>(
        dA.block_col_idx, d_blk_row_of, dB.block_row_ptr,
        hA.nnzb, d_counts,
        d_pa, d_pb, d_pbi);
    CUDA_CHECK(cudaDeviceSynchronize()); // fence pair-gen; marks symbolic / kernel boundary
    auto t_sym_end = Clock::now();

    // ── Allocate raw COO output (worst case: all 256 elements nonzero per pair)
    const long long max_out = (long long)total_pairs * BS2;
    int*   d_out_row; CUDA_CHECK(cudaMalloc(&d_out_row, max_out * sizeof(int)));
    int*   d_out_col; CUDA_CHECK(cudaMalloc(&d_out_col, max_out * sizeof(int)));
    float* d_out_val; CUDA_CHECK(cudaMalloc(&d_out_val, max_out * sizeof(float)));
    int*   d_nnz;     CUDA_CHECK(cudaMalloc(&d_nnz, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nnz, 0, sizeof(int)));

    // ── Launch TC_SpGEMM kernel ───────────────────────────────────────────
    constexpr int WPB     = 4;
    const int     grid_sz = (total_pairs + WPB - 1) / WPB;
    const int     blk_sz  = WPB * WARP;
    const int     smem_sz = WPB * BS2 * sizeof(float);

    cudaEvent_t _ev0, _ev1;
    cudaEventCreate(&_ev0); cudaEventCreate(&_ev1);
    cudaEventRecord(_ev0);
    tc_spgemm_kernel<WPB><<<grid_sz, blk_sz, smem_sz>>>(
        dA.dense_data, dB.dense_data,
        d_pa, d_pb, d_pbi, total_pairs,
        dA.row_ori, dB.col_ori,
        d_out_row, d_out_col, d_out_val, d_nnz);
    cudaEventRecord(_ev1);
    cudaEventSynchronize(_ev1);
    float _kern_ms; cudaEventElapsedTime(&_kern_ms, _ev0, _ev1);
    cudaEventDestroy(_ev0); cudaEventDestroy(_ev1);

    // ── Copy raw_nnz to host, free all temporaries ────────────────────────
    int raw_nnz = 0;
    CUDA_CHECK(cudaMemcpy(&raw_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Raw COO entries (before dedup): %d\n", raw_nnz);

    free_upload();
    cudaFree(d_pa);  cudaFree(d_pb);  cudaFree(d_pbi);
    cudaFree(d_nnz);

    out.d_row   = d_out_row;
    out.d_col   = d_out_col;
    out.d_val   = d_out_val;
    out.raw_nnz = raw_nnz;

    return { Fms(t_sym_end - t_sym_start).count(), (double)_kern_ms };
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 of the host driver: COO deduplication and coordinate recovery.
// Takes the raw device COO from tc_spgemm_compute, sorts by (row,col) key,
// sums duplicate entries, unpacks original coordinates, and copies to host.
// Frees raw.d_row/col/val before returning.
// ─────────────────────────────────────────────────────────────────────────────
void tc_spgemm_postprocess(
    const RawCOO& raw,
    int           ncols_C,
    int**         h_out_row,
    int**         h_out_col,
    float**       h_out_val,
    int*          h_out_nnz
) {
    if (raw.raw_nnz == 0) {
        *h_out_nnz = 0;
        *h_out_row = nullptr;
        *h_out_col = nullptr;
        *h_out_val = nullptr;
        return;
    }

    // Pack (row, col) → long long key
    thrust::device_vector<long long> d_keys(raw.raw_nnz);
    thrust::device_ptr<int>   rptr(raw.d_row);
    thrust::device_ptr<int>   cptr(raw.d_col);
    thrust::transform(rptr, rptr + raw.raw_nnz, cptr, d_keys.begin(),
                      PackKeyFn{ncols_C});

    // Copy vals into a device_vector for sort
    thrust::device_vector<float> d_v(raw.d_val, raw.d_val + raw.raw_nnz);

    // Sort by key (moves vals in lockstep)
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_v.begin());

    // Reduce duplicate keys by summing their values
    thrust::device_vector<long long> d_keys_out(raw.raw_nnz);
    thrust::device_vector<float>     d_v_out(raw.raw_nnz);
    auto ends = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),
        d_v.begin(),
        d_keys_out.begin(),
        d_v_out.begin());
    const int final_nnz = (int)(ends.first - d_keys_out.begin());

    printf("Output NNZ (after dedup): %d\n", final_nnz);

    // Recover original row and col from reduced keys
    thrust::device_vector<int> d_row_f(final_nnz), d_col_f(final_nnz);
    thrust::transform(d_keys_out.begin(), d_keys_out.begin() + final_nnz,
                      d_row_f.begin(), UnpackRowFn{ncols_C});
    thrust::transform(d_keys_out.begin(), d_keys_out.begin() + final_nnz,
                      d_col_f.begin(), UnpackColFn{ncols_C});

    // Copy to host
    *h_out_nnz = final_nnz;
    *h_out_row = (int*)  malloc(final_nnz * sizeof(int));
    *h_out_col = (int*)  malloc(final_nnz * sizeof(int));
    *h_out_val = (float*)malloc(final_nnz * sizeof(float));
    thrust::copy(d_row_f.begin(), d_row_f.end(), *h_out_row);
    thrust::copy(d_col_f.begin(), d_col_f.end(), *h_out_col);
    thrust::copy(d_v_out.begin(), d_v_out.begin() + final_nnz, *h_out_val);

    cudaFree(raw.d_row);
    cudaFree(raw.d_col);
    cudaFree(raw.d_val);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience wrapper: compute + postprocess in one call.
// ─────────────────────────────────────────────────────────────────────────────
void tc_spgemm(
    const CSRDense& hA,
    const CSRDense& hB,
    int             ncols_C,
    int**   h_out_row,
    int**   h_out_col,
    float** h_out_val,
    int*    h_out_nnz
) {
    RawCOO raw{};
    tc_spgemm_compute(hA, hB, ncols_C, raw);
    tc_spgemm_postprocess(raw, ncols_C, h_out_row, h_out_col, h_out_val, h_out_nnz);
}

// ─────────────────────────────────────────────────────────────────────────────
// Smoke test
//
// A = 32×32 identity  (2 diagonal 16×16 blocks)
// B = 32×32 2×identity
// Expected C = A×B = 2×identity  →  32 diagonal entries of value 2.0
// ─────────────────────────────────────────────────────────────────────────────
#ifndef TC_SPGEMM_NO_MAIN
int main() {
    const int M = 32, K = 32, N = 32;

    // Build raw CSR for A = identity
    std::vector<int>   A_rptr(M + 1), A_cidx(M);
    std::vector<float> A_vals(M, 1.0f);
    for (int i = 0; i <= M; i++) A_rptr[i] = i;
    for (int i = 0;  i < M;  i++) A_cidx[i] = i;

    // Build raw CSR for B = 2×identity
    std::vector<int>   B_rptr(K + 1), B_cidx(K);
    std::vector<float> B_vals(K, 2.0f);
    for (int i = 0; i <= K; i++) B_rptr[i] = i;
    for (int i = 0;  i < K;  i++) B_cidx[i] = i;

    // Preprocess
    CSRDense hA{}, hB{};
    preprocess_A(M, K, A_rptr.data(), A_cidx.data(), A_vals.data(), hA);
    preprocess_B(K, N, B_rptr.data(), B_cidx.data(), B_vals.data(), hB);

    printf("A: %d block-rows, %d blocks\n", hA.nblk_rows, hA.nnzb);
    printf("B: %d block-rows, %d blocks\n", hB.nblk_rows, hB.nnzb);

    // Run
    int*   out_row = nullptr;
    int*   out_col = nullptr;
    float* out_val = nullptr;
    int    out_nnz = 0;

    tc_spgemm(hA, hB, N, &out_row, &out_col, &out_val, &out_nnz);

    // Verify: should have 32 diagonal entries, all value 2.0
    printf("\nOutput NNZ: %d  (expected 32)\n", out_nnz);
    int errors = 0;
    for (int i = 0; i < out_nnz; i++) {
        if (out_row[i] != out_col[i] || fabsf(out_val[i] - 2.0f) > 1e-3f) {
            printf("  UNEXPECTED: C[%d, %d] = %.4f\n", out_row[i], out_col[i], out_val[i]);
            errors++;
        }
    }
    if (errors == 0 && out_nnz == 32)
        printf("PASSED\n");
    else
        printf("FAILED (%d errors)\n", errors);

    free(out_row); free(out_col); free(out_val);
    free_csrdense(hA);
    free_csrdense(hB);
    return errors == 0 ? 0 : 1;
}
#endif // TC_SPGEMM_NO_MAIN
