#pragma once
// gpu_dispatch.cuh — device upload and kernel dispatch for the CSP GPU plan.
//
// Must be compiled by nvcc (include from a .cu file).
//
// Usage:
//   auto [C, plan] = gpu_kernel_plan(fusion, cls, A, B);
//   auto dev       = upload_plan(plan, A, B, C);
//   auto times     = dispatch(dev);
//   sync_c(dev, C);

#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector>

#include "gpu_pipeline.hpp"

namespace benchmark_core {

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(_e));                                         \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ─── Constants
// ────────────────────────────────────────────────────────────────

static constexpr int K_STRIP = 8;
static constexpr int TILE = 16;

// Approach 1: 1 CTA per output block, batch loop
static constexpr int TC_CTA_SZ = 1024; // 32 warps, hardware max

// Approach 2: 1 CTA per BM×BN tile
// BM and BN are defined in gpu_pipeline.hpp (included above).
static constexpr int TILE_WARPS = (BM / 16) * (BN / 16); // 16
static constexpr int TILE_CTA_SZ = TILE_WARPS * 32;      // 512
// ─── DevPlan<T>
// ───────────────────────────────────────────────────────────────

template <typename T> struct DevPlan {
  T *d_A = nullptr;
  std::size_t n_A = 0;
  T *d_B = nullptr;
  std::size_t n_B = 0;
  T *d_C = nullptr;
  std::size_t n_C = 0;

  KEntry<T> *d_k_entries = nullptr;
  int n_k = 0;
  TcDesc *d_tc_descs = nullptr;
  int n_tc = 0;
  CudaDesc *d_cuda_descs = nullptr;
  int n_cuda = 0;
  GemmTile *d_tc_tiles = nullptr;
  int n_tiles = 0; // ← new

  std::vector<TcDesc> h_tc_descs; // ← host copy for shmem calculation
};

template <typename T> void free_dev_plan(DevPlan<T> &dev) {
  cudaFree(dev.d_A);
  cudaFree(dev.d_B);
  cudaFree(dev.d_C);
  cudaFree(dev.d_k_entries);
  cudaFree(dev.d_tc_descs);
  cudaFree(dev.d_tc_tiles);
  cudaFree(dev.d_cuda_descs);
  dev = DevPlan<T>{};
}

// ─── Upload
// ───────────────────────────────────────────────────────────────────

template <typename T>
DevPlan<T> upload_plan(const GpuKernelPlan<T> &plan, const Matrix<T> &A,
                       const Matrix<T> &B, const Matrix<T> &C) {
  DevPlan<T> dev;

  dev.n_A = A.n_values;
  CUDA_CHECK(cudaMalloc(&dev.d_A, dev.n_A * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(dev.d_A, A.values, dev.n_A * sizeof(T),
                        cudaMemcpyHostToDevice));

  dev.n_B = B.n_values;
  CUDA_CHECK(cudaMalloc(&dev.d_B, dev.n_B * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(dev.d_B, B.values, dev.n_B * sizeof(T),
                        cudaMemcpyHostToDevice));

  dev.n_C = C.n_values;
  CUDA_CHECK(cudaMalloc(&dev.d_C, dev.n_C * sizeof(T)));
  CUDA_CHECK(cudaMemset(dev.d_C, 0, dev.n_C * sizeof(T)));

  // Rebase KEntry pointers: host → device.
  // Offset from base is preserved; only the base address changes.
  std::vector<KEntry<T>> rebased = plan.k_entries;
  for (auto &ke : rebased) {
    ke.A_ptr = dev.d_A + (ke.A_ptr - A.values);
    ke.B_ptr = dev.d_B + (ke.B_ptr - B.values);
    ke.C_ptr = dev.d_C + (ke.C_ptr - C.values);
  }

  dev.n_k = static_cast<int>(rebased.size());
  CUDA_CHECK(cudaMalloc(&dev.d_k_entries, dev.n_k * sizeof(KEntry<T>)));
  CUDA_CHECK(cudaMemcpy(dev.d_k_entries, rebased.data(),
                        dev.n_k * sizeof(KEntry<T>), cudaMemcpyHostToDevice));

  dev.n_tc = static_cast<int>(plan.tc_descs.size());
  if (dev.n_tc > 0) {
    CUDA_CHECK(cudaMalloc(&dev.d_tc_descs, dev.n_tc * sizeof(TcDesc)));
    CUDA_CHECK(cudaMemcpy(dev.d_tc_descs, plan.tc_descs.data(),
                          dev.n_tc * sizeof(TcDesc), cudaMemcpyHostToDevice));
  }

  dev.h_tc_descs = plan.tc_descs; // host copy for Approach 1 shmem calculation

  dev.n_tiles = (int)plan.tc_tiles.size();
  if (dev.n_tiles > 0) {
    CUDA_CHECK(cudaMalloc(&dev.d_tc_tiles, dev.n_tiles * sizeof(GemmTile)));
    CUDA_CHECK(cudaMemcpy(dev.d_tc_tiles, plan.tc_tiles.data(),
                          dev.n_tiles * sizeof(GemmTile),
                          cudaMemcpyHostToDevice));
  }

  dev.n_cuda = static_cast<int>(plan.cuda_descs.size());
  if (dev.n_cuda > 0) {
    CUDA_CHECK(cudaMalloc(&dev.d_cuda_descs, dev.n_cuda * sizeof(CudaDesc)));
    CUDA_CHECK(cudaMemcpy(dev.d_cuda_descs, plan.cuda_descs.data(),
                          dev.n_cuda * sizeof(CudaDesc),
                          cudaMemcpyHostToDevice));
  }

  return dev;
}

template <typename T> void sync_c(const DevPlan<T> &dev, Matrix<T> &C) {
  CUDA_CHECK(cudaMemcpy(C.values, dev.d_C, dev.n_C * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

template <typename T>
__global__ void tc_kernel(const TcDesc *__restrict__ tc_descs,
                          const KEntry<T> *__restrict__ k_entries) {
  using namespace nvcuda::wmma;

  extern __shared__ float smem[];
  const TcDesc &desc = tc_descs[blockIdx.x];
  const KEntry<T> &k0 = k_entries[desc.k_start];
  float *sA = smem;
  float *sB = smem + desc.M * (K_STRIP + 1);

  T *const C_ptr = k0.C_ptr;
  const int ldc = k0.ldc;
  const int warp_id = threadIdx.x / 32;
  const int n_col_tiles = desc.N / 16;
  const int n_tiles = desc.n_warps;
  const int n_avail = TC_CTA_SZ / 32; // 32

  for (int batch = 0; batch < n_tiles; batch += n_avail) {
    const int my_tile = batch + warp_id;
    const bool active = my_tile < n_tiles;
    const int wr = active ? my_tile / n_col_tiles : 0;
    const int wc = active ? my_tile % n_col_tiles : 0;

    fragment<accumulator, 16, 16, 8, float> c_frag;
    if (active)
      fill_fragment(c_frag, 0.f);

    for (int t = 0; t < desc.k_count; ++t) {
      const KEntry<T> &ke = k_entries[desc.k_start + t];
      const int K_pad = ((ke.K + K_STRIP - 1) / K_STRIP) * K_STRIP;

      for (int k0_off = 0; k0_off < K_pad; k0_off += K_STRIP) {
        for (int i = threadIdx.x; i < desc.M * K_STRIP; i += TC_CTA_SZ) {
          int r = i / K_STRIP, k = i % K_STRIP;
          sA[r * (K_STRIP + 1) + k] =
              (r < desc.M && k0_off + k < ke.K)
                  ? static_cast<float>(ke.A_ptr[r * ke.lda + k0_off + k])
                  : 0.f;
        }
        for (int i = threadIdx.x; i < K_STRIP * desc.N; i += TC_CTA_SZ) {
          int k = i / desc.N, c = i % desc.N;
          sB[k * (desc.N + 1) + c] =
              (k0_off + k < ke.K && c < desc.N)
                  ? static_cast<float>(ke.B_ptr[(k0_off + k) * ke.ldb + c])
                  : 0.f;
        }
        __syncthreads();

        if (active) {
          fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
          fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
          load_matrix_sync(a_frag, sA + wr * 16 * (K_STRIP + 1), K_STRIP + 1);
          load_matrix_sync(b_frag, sB + wc * 16, desc.N + 1);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
      }
    }

    if (active)
      store_matrix_sync(reinterpret_cast<float *>(C_ptr) + wr * 16 * ldc +
                            wc * 16,
                        c_frag, ldc, mem_row_major);
  }
}
// ─── CUDA kernel
// ──────────────────────────────────────────────────────────────
//
// Grid : (n_cuda_descs,)   Block : (TILE, TILE) = 256 threads
// For blocks with M < 16 or N < 16.

// Each k-entry is accumulated independently and written to its own ke.C_ptr
// via atomicAdd.  This is necessary because contributions within a fused group
// can map to different (possibly overlapping) sub-regions of the fused block
// when block positions are not snapped to tile boundaries (i.e. CUDA-only
// path). For k_count==1 the atomicAdd degenerates to a plain store (C starts at
// 0).
template <typename T>
__global__ void cuda_kernel(const CudaDesc *__restrict__ cuda_descs,
                            const KEntry<T> *__restrict__ k_entries) {
  __shared__ float sA[TILE][TILE + 1];
  __shared__ float sB[TILE][TILE + 1];

  const CudaDesc &desc = cuda_descs[blockIdx.x];
  const KEntry<T> &k0 = k_entries[desc.k_start];

  T *const C_ptr = k0.C_ptr;
  const int M = k0.M;
  const int N = k0.N;
  const int ldc = k0.ldc;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  for (int row_off = 0; row_off < M; row_off += TILE) {
    for (int col_off = 0; col_off < N; col_off += TILE) {

      const int row = row_off + ty;
      const int col = col_off + tx;
      float acc = 0.f; // ← reset per output tile

      for (int t = 0; t < desc.k_count; ++t) {
        const KEntry<T> &ke = k_entries[desc.k_start + t];

        for (int k0_off = 0; k0_off < ke.K; k0_off += TILE) {
          sA[ty][tx] =
              (row < M && k0_off + tx < ke.K)
                  ? static_cast<float>(ke.A_ptr[row * ke.lda + k0_off + tx])
                  : 0.f;
          sB[ty][tx] =
              (k0_off + ty < ke.K && col < N)
                  ? static_cast<float>(ke.B_ptr[(k0_off + ty) * ke.ldb + col])
                  : 0.f;
          __syncthreads();
#pragma unroll
          for (int k = 0; k < TILE; ++k)
            acc += sA[ty][k] * sB[k][tx];
          __syncthreads();
        }
      }

      if (row < M && col < N)
        atomicAdd(&C_ptr[row * ldc + col], static_cast<T>(acc));
    }
  }
}

template <typename T>
__global__ void tc_tile_kernel(const GemmTile *__restrict__ tiles,
                               const TcDesc *__restrict__ tc_descs,
                               const KEntry<T> *__restrict__ k_entries) {
  using namespace nvcuda::wmma;

  // Static shmem — sized for BM×BN, safe because tile dimensions are fixed
  __shared__ float sA[BM][K_STRIP + 1]; // 64×9  = 2.25 KB
  __shared__ float sB[K_STRIP][BN + 1]; // 8×65  = 2.03 KB

  const GemmTile &tile = tiles[blockIdx.x];
  const TcDesc &desc = tc_descs[tile.desc_idx];
  const KEntry<T> &k0 = k_entries[desc.k_start];

  T *const C_ptr = k0.C_ptr;
  const int ldc = k0.ldc;
  const int warp_id = threadIdx.x / 32;
  const int n_col_tiles = BN / 16; // 4 for BN=64
  const int wr = warp_id / n_col_tiles;
  const int wc = warp_id % n_col_tiles;

  // Absolute position in the output block
  const int abs_row = tile.row_off + wr * 16;
  const int abs_col = tile.col_off + wc * 16;

  fragment<accumulator, 16, 16, 8, float> c_frag;
  fill_fragment(c_frag, 0.f);

  for (int t = 0; t < desc.k_count; ++t) {
    const KEntry<T> &ke = k_entries[desc.k_start + t];
    const int K_pad = ((ke.K + K_STRIP - 1) / K_STRIP) * K_STRIP;

    for (int k0_off = 0; k0_off < K_pad; k0_off += K_STRIP) {
      // Cooperative load: all TILE_CTA_SZ threads fill BM×8 and 8×BN
      for (int i = threadIdx.x; i < BM * K_STRIP; i += TILE_CTA_SZ) {
        int r = i / K_STRIP, k = i % K_STRIP;
        int abs_r = tile.row_off + r;
        sA[r][k] =
            (abs_r < desc.M && k0_off + k < ke.K)
                ? static_cast<float>(ke.A_ptr[abs_r * ke.lda + k0_off + k])
                : 0.f;
      }
      for (int i = threadIdx.x; i < K_STRIP * BN; i += TILE_CTA_SZ) {
        int k = i / BN, c = i % BN;
        int abs_c = tile.col_off + c;
        sB[k][c] =
            (k0_off + k < ke.K && abs_c < desc.N)
                ? static_cast<float>(ke.B_ptr[(k0_off + k) * ke.ldb + abs_c])
                : 0.f;
      }
      __syncthreads();

      fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a_frag;
      fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> b_frag;
      load_matrix_sync(a_frag, &sA[wr * 16][0], K_STRIP + 1);
      load_matrix_sync(b_frag, &sB[0][wc * 16], BN + 1);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
      __syncthreads();
    }
  }

  // Only write if this warp's 16×16 sub-tile is within the output block.
  // Blocks are snapped to multiples of 16 so this is always whole-tile in or
  // out.
  if (abs_row < desc.M && abs_col < desc.N)
    store_matrix_sync(reinterpret_cast<float *>(C_ptr) + abs_row * ldc +
                          abs_col,
                      c_frag, ldc, mem_row_major);
}
// ─── Dispatch
// ─────────────────────────────────────────────────────────────────

struct DispatchTimes {
  float tc_ms = 0.f;
  float cuda_ms = 0.f;
};

enum class TcStrategy { PerBlock, PerTile };

template <typename T>
DispatchTimes dispatch(const DevPlan<T> &dev,
                       TcStrategy strategy = TcStrategy::PerTile) {
  cudaStream_t s_tc, s_cuda;
  CUDA_CHECK(cudaStreamCreate(&s_tc));
  CUDA_CHECK(cudaStreamCreate(&s_cuda));

  cudaEvent_t e_tc_s, e_tc_e, e_cu_s, e_cu_e;
  CUDA_CHECK(cudaEventCreate(&e_tc_s));
  CUDA_CHECK(cudaEventCreate(&e_tc_e));
  CUDA_CHECK(cudaEventCreate(&e_cu_s));
  CUDA_CHECK(cudaEventCreate(&e_cu_e));

  DispatchTimes t;

  const int n_tc_work =
      (strategy == TcStrategy::PerBlock) ? dev.n_tc : dev.n_tiles;
  if (n_tc_work > 0) {
    CUDA_CHECK(cudaEventRecord(e_tc_s, s_tc));

    if (strategy == TcStrategy::PerBlock) {
      // Approach 1: dynamic shmem, 1 CTA per output block
      int max_shmem = 0;
      for (const auto &d : dev.h_tc_descs) {
        int sz = d.M * (K_STRIP + 1) + K_STRIP * (d.N + 1);
        max_shmem = std::max(max_shmem, sz);
      }
      tc_kernel<T><<<dev.n_tc, TC_CTA_SZ, max_shmem * sizeof(float), s_tc>>>(
          dev.d_tc_descs, dev.d_k_entries);
    } else {
      // Approach 2: static shmem, 1 CTA per BM×BN tile
      tc_tile_kernel<T><<<dev.n_tiles, TILE_CTA_SZ, 0, s_tc>>>(
          dev.d_tc_tiles, dev.d_tc_descs, dev.d_k_entries);
    }

    CUDA_CHECK(cudaEventRecord(e_tc_e, s_tc));
  }

  if (dev.n_cuda > 0) {
    CUDA_CHECK(cudaEventRecord(e_cu_s, s_cuda));
    cuda_kernel<T><<<dev.n_cuda, dim3(TILE, TILE), 0, s_cuda>>>(
        dev.d_cuda_descs, dev.d_k_entries);
    CUDA_CHECK(cudaEventRecord(e_cu_e, s_cuda));
  }

  CUDA_CHECK(cudaStreamSynchronize(s_tc));
  CUDA_CHECK(cudaStreamSynchronize(s_cuda));

  if (n_tc_work > 0)
    CUDA_CHECK(cudaEventElapsedTime(&t.tc_ms, e_tc_s, e_tc_e));
  if (dev.n_cuda > 0)
    CUDA_CHECK(cudaEventElapsedTime(&t.cuda_ms, e_cu_s, e_cu_e));

  CUDA_CHECK(cudaEventDestroy(e_tc_s));
  CUDA_CHECK(cudaEventDestroy(e_tc_e));
  CUDA_CHECK(cudaEventDestroy(e_cu_s));
  CUDA_CHECK(cudaEventDestroy(e_cu_e));
  CUDA_CHECK(cudaStreamDestroy(s_tc));
  CUDA_CHECK(cudaStreamDestroy(s_cuda));
  return t;
}

// Convenience wrapper: upload → dispatch → sync → free in one call.
// Returns the completed host C matrix paired with kernel timing.
// C is taken by value; the caller should std::move it in.
template <typename T>
std::pair<Matrix<T>, DispatchTimes>
run(const GpuKernelPlan<T> &plan, const Matrix<T> &A, const Matrix<T> &B,
    Matrix<T> C, TcStrategy strategy = TcStrategy::PerTile) {
  auto dev = upload_plan(plan, A, B, C);
  auto times = dispatch(dev, strategy);
  sync_c(dev, C);
  free_dev_plan(dev);
  return {std::move(C), times};
}

#undef CUDA_CHECK

} // namespace benchmark_core
