#pragma once
// gpu_pipeline.hpp — GPU kernel classification (phase 3) and execution-plan
// construction (phase 4) for the CSP SpGEMM pipeline.
//
// Constraint: blocks are either exact multiples of 16×16 (→ TC)
// or fully smaller than 16×16 (→ CUDA).  No partial-tile Hybrid case.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
#endif

#include "matrix.hpp"   // Matrix<T>, DataType, scalar_traits
#include "pipeline.hpp" // FusionResult, Contribution, KRange

namespace benchmark_core {

static constexpr int BM = 64;
static constexpr int BN = 64;
// ─── Phase 3: classification
// ──────────────────────────────────────────────────

// Blocks are either full TC tiles or full CUDA tiles — no hybrid border.
enum class GpuKernelType : uint8_t { TC, CUDA };

struct BlockClass {
  GpuKernelType type;
  int M, N;    // block dimensions (M×N multiples of 16 for TC, <16 for CUDA)
  int n_warps; // (M/16)*(N/16) — 0 for CUDA
};

// Classify a single M×N block.
// Precondition: (M%16==0 && N%16==0) || (M<16 || N<16)
inline BlockClass classify_block(int M, int N) noexcept {
  // assert((M % 16 == 0 || M < 16) && "M is neither a multiple of 16 nor <
  // 16"); assert((N % 16 == 0 || N < 16) && "N is neither a multiple of 16 nor
  // < 16");
  int n_warps = 0;
  if (M % 16 == 0 && N % 16 == 0)
    n_warps = (M / 16) * (N / 16);
  else
    n_warps = 0;
  return {n_warps > 0 ? GpuKernelType::TC : GpuKernelType::CUDA, M, N, n_warps};
}

struct GpuKernelClassification {
  std::vector<BlockClass> entries; // parallel to FusionResult::fused_blocks
  int n_tc = 0;
  int n_cuda = 0;
};

// When tc_only is false every block is forced to the CUDA kernel regardless
// of its dimensions — useful for validating the CUDA path with arbitrary
// block sizes, including multiples of 16 that would otherwise go to TC.
inline GpuKernelClassification gpu_kernel_classify(const FusionResult &fusion,
                                                   bool tc_only = false) {
  GpuKernelClassification result;
  result.entries.reserve(fusion.fused_blocks.size());
  for (const Block &b : fusion.fused_blocks) {
    BlockClass bc = tc_only ? classify_block(b.h, b.w)
                            : BlockClass{GpuKernelType::CUDA, b.h, b.w, 0};
    bc.type == GpuKernelType::TC ? ++result.n_tc : ++result.n_cuda;
    result.entries.push_back(bc);
  }
  return result;
}

// ─── Phase 4: execution plan
// ──────────────────────────────────────────────────

// One (A-tile, B-tile) → C-sub-region GEMM contribution.
// C_ptr, M, N, ldc live here (not in TcDesc/CudaDesc) because contributions
// within a fused block may write to different spatial sub-regions of C.
template <typename T> struct KEntry {
  const T *A_ptr = nullptr;
  const T *B_ptr = nullptr;
  T *C_ptr = nullptr;
  int M = 0;   // output rows  (= a_block.h)
  int K = 0;   // contraction  (= k.k1 - k.k0 + 1, KRange is inclusive)
  int N = 0;   // output cols  (= b_block.w)
  int lda = 0; // A block row stride (= a_block.w)
  int ldb = 0; // B block row stride (= b_block.w)
  int ldc = 0; // C fused-block row stride (= fused.w)
};

// TC descriptor — block dimensions are exact multiples of 16.
// M_rem and N_rem are omitted: they are always 0 under the current constraint.
struct TcDesc {
  int M = 0;       // block height (multiple of 16)
  int N = 0;       // block width  (multiple of 16)
  int n_warps = 0; // (M/16) * (N/16)
  int k_start = 0;
  int k_count = 0;
};

// CUDA descriptor — block fits entirely within a 16×16 tile.
struct CudaDesc {
  int k_start = 0;
  int k_count = 0;
};

struct GemmTile {
  int desc_idx; // index into tc_descs
  int row_off;  // top-left row within the output block
  int col_off;  // top-left col within the output block
};

template <typename T> struct GpuKernelPlan {
  std::vector<KEntry<T>> k_entries;
  std::vector<TcDesc> tc_descs;
  std::vector<CudaDesc> cuda_descs;
  std::vector<GemmTile> tc_tiles;
};

// Build C and the execution plan together.
// C is allocated with fusion.fused_blocks as its block list (offsets assigned,
// values zero-initialised).  Plan pointers borrow from A, B, and C — keep all
// three alive for the plan's entire lifetime.
template <typename T>
std::pair<Matrix<T>, GpuKernelPlan<T>>
gpu_kernel_plan(const FusionResult &fusion, const GpuKernelClassification &cls,
                const Matrix<T> &A, const Matrix<T> &B, bool use_tc = true);

// ─── Implementation
// ───────────────────────────────────────────────────────────

template <typename T>
std::pair<Matrix<T>, GpuKernelPlan<T>>
gpu_kernel_plan(const FusionResult &fusion, const GpuKernelClassification &cls,
                const Matrix<T> &A, const Matrix<T> &B, bool use_tc) {
  Matrix<T> C;
  C.M = A.M;
  C.N = B.N;
  C.blocks = fusion.fused_blocks;
  C.n_values = static_cast<std::size_t>(assign_offsets(C.blocks));
  C.values = new T[C.n_values]();

  GpuKernelPlan<T> plan;
  const int n = static_cast<int>(fusion.fused_blocks.size());

  // Sub-region key: contributions within a fused block may target different
  // spatial sub-regions (A.r - fused.r, A.h, B.c - fused.c, B.w).
  struct SubRegion {
    int row_off, M_sub, col_off, N_sub;
    bool operator==(const SubRegion &o) const noexcept {
      return row_off == o.row_off && M_sub == o.M_sub &&
             col_off == o.col_off && N_sub == o.N_sub;
    }
  };
  struct SubRegionHash {
    std::size_t operator()(const SubRegion &k) const noexcept {
      auto mix = [](std::size_t s, int v) -> std::size_t {
        return s ^ (static_cast<std::size_t>(v) * 2654435761ULL +
                    0x9e3779b9ULL + (s << 6) + (s >> 2));
      };
      std::size_t s = 0;
      s = mix(s, k.row_off); s = mix(s, k.M_sub);
      s = mix(s, k.col_off); s = mix(s, k.N_sub);
      return s;
    }
  };

  // ── Helper: classify a sub-region, applying use_tc and alignment checks ──
  auto classify_sr = [&](const SubRegion &sr, long long c_off) -> BlockClass {
    BlockClass bc = classify_block(sr.M_sub, sr.N_sub);
    if (!use_tc) { bc.type = GpuKernelType::CUDA; bc.n_warps = 0; }
    // store_matrix_sync on sm_90+ requires 16-byte (4-float) aligned dest.
    if (bc.type == GpuKernelType::TC && c_off % 4 != 0) {
      bc.type = GpuKernelType::CUDA; bc.n_warps = 0;
    }
    return bc;
  };

  // ── Helper: build sorted sub-region list for fused block fi ──────────────
  // Sorted order guarantees consistent layout between pass 1 and pass 2.
  using SubList = std::vector<std::pair<SubRegion, std::vector<int>>>;
  auto build_sub_list = [&](int fi) -> SubList {
    const Block &fused = C.blocks[fi];
    const auto &contribs = fusion.fused_contributions[fi];
    std::unordered_map<SubRegion, std::vector<int>, SubRegionHash> sub_map;
    for (int ci = 0; ci < (int)contribs.size(); ++ci) {
      const auto &c = contribs[ci];
      const Block &ab = A.blocks[c.a_index];
      const Block &bb = B.blocks[c.b_index];
      sub_map[SubRegion{ab.r - fused.r, ab.h, bb.c - fused.c, bb.w}]
          .push_back(ci);
    }
    SubList sl(std::make_move_iterator(sub_map.begin()),
               std::make_move_iterator(sub_map.end()));
    std::sort(sl.begin(), sl.end(), [](const auto &a, const auto &b) {
      if (a.first.row_off != b.first.row_off) return a.first.row_off < b.first.row_off;
      if (a.first.col_off != b.first.col_off) return a.first.col_off < b.first.col_off;
      if (a.first.M_sub   != b.first.M_sub)   return a.first.M_sub   < b.first.M_sub;
      return a.first.N_sub < b.first.N_sub;
    });
    return sl;
  };

  // ── Pass 1: count per-block output sizes (parallel) ──────────────────────
  std::vector<int> n_k(n, 0), n_cuda(n, 0), n_tc(n, 0), n_tiles(n, 0);

#pragma omp parallel for schedule(dynamic, 32)
  for (int fi = 0; fi < n; ++fi) {
    const Block &fused = C.blocks[fi];
    auto sl = build_sub_list(fi);
    for (const auto &[sr, indices] : sl) {
      n_k[fi] += (int)indices.size();
      long long c_off = fused.offset + (long long)sr.row_off * fused.w + sr.col_off;
      BlockClass bc = classify_sr(sr, c_off);
      if (bc.type == GpuKernelType::CUDA) {
        n_cuda[fi]++;
      } else {
        n_tc[fi]++;
        n_tiles[fi] += (sr.M_sub / BM) * (sr.N_sub / BN);
      }
    }
  }

  // ── Prefix sum (sequential) ───────────────────────────────────────────────
  std::vector<int> k_off(n+1,0), cuda_off(n+1,0), tc_off(n+1,0), tile_off(n+1,0);
  for (int fi = 0; fi < n; ++fi) {
    k_off   [fi+1] = k_off   [fi] + n_k   [fi];
    cuda_off[fi+1] = cuda_off[fi] + n_cuda [fi];
    tc_off  [fi+1] = tc_off  [fi] + n_tc   [fi];
    tile_off[fi+1] = tile_off[fi] + n_tiles[fi];
  }
  plan.k_entries .resize(k_off   [n]);
  plan.cuda_descs.resize(cuda_off[n]);
  plan.tc_descs  .resize(tc_off  [n]);
  plan.tc_tiles  .resize(tile_off[n]);

  // ── Pass 2: fill pre-allocated slices (parallel) ─────────────────────────
  // Each fused block writes to disjoint slices — no contention.
#pragma omp parallel for schedule(dynamic, 32)
  for (int fi = 0; fi < n; ++fi) {
    const Block &fused = C.blocks[fi];
    const auto &contribs = fusion.fused_contributions[fi];
    auto sl = build_sub_list(fi);

    int local_k = 0, local_cuda = 0, local_tc = 0, local_tile = 0;

    for (auto &[sr, indices] : sl) {
      const int M_sub = sr.M_sub;
      const int N_sub = sr.N_sub;
      long long c_off = fused.offset + (long long)sr.row_off * fused.w + sr.col_off;
      T *C_ptr = C.values + c_off;

      BlockClass bc = classify_sr(sr, c_off);

      const int abs_k_start = k_off[fi] + local_k;
      for (int j = 0; j < (int)indices.size(); ++j) {
        const auto &c = contribs[indices[j]];
        const Block &ab = A.blocks[c.a_index];
        const Block &bb = B.blocks[c.b_index];
        const T *A_ptr = A.values + ab.offset + (long long)(c.k.k0 - ab.c);
        const T *B_ptr = B.values + bb.offset + (long long)(c.k.k0 - bb.r) * bb.w;
        plan.k_entries[abs_k_start + j] =
            {A_ptr, B_ptr, C_ptr, M_sub, c.k.k1 - c.k.k0 + 1, N_sub,
             ab.w, bb.w, fused.w};
      }
      local_k += (int)indices.size();

      const int k_count = (int)indices.size();
      if (bc.type == GpuKernelType::CUDA) {
        plan.cuda_descs[cuda_off[fi] + local_cuda++] = {abs_k_start, k_count};
      } else {
        const int abs_tc_idx = tc_off[fi] + local_tc++;
        plan.tc_descs[abs_tc_idx] = {M_sub, N_sub, bc.n_warps, abs_k_start, k_count};
        for (int r = 0; r < M_sub; r += BM)
          for (int c = 0; c < N_sub; c += BN)
            plan.tc_tiles[tile_off[fi] + local_tile++] = {abs_tc_idx, r, c};
      }
    }
  }

  return {std::move(C), std::move(plan)};
}

} // namespace benchmark_core
