#include "block.hpp"
#include "cpu_dispatch.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Scalar = double;
using Clock = std::chrono::steady_clock;

struct Args {
  std::string s_bsp;
  int runs = 5;
  int seed = 42;
  bool specialized = false;
  bool use_static = false;
  int threads = 0; // 0 = auto (see main(): capped default unless
                    // OMP_NUM_THREADS is set); >0 = explicit override
  std::string dump_x; // --dump-x <path>: write N-length x vector as raw doubles
  std::string dump_y; // --dump-y <path>: write M-length y vector as raw doubles
};

void print_usage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s <S.bsp> [--runs N] [--seed S] [--specialized-kernels] "
               "[--static] [--threads N] [--dump-x path] [--dump-y path]\n",
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
    if (arg == "--runs" && i + 1 < argc)
      args.runs = std::atoi(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc)
      args.seed = std::atoi(argv[++i]);
    else if (arg == "--specialized-kernels")
      args.specialized = true;
    else if (arg == "--static")
      args.use_static = true;
    else if (arg == "--threads" && i + 1 < argc)
      args.threads = std::atoi(argv[++i]);
    else if (arg == "--dump-x" && i + 1 < argc)
      args.dump_x = argv[++i];
    else if (arg == "--dump-y" && i + 1 < argc)
      args.dump_y = argv[++i];
    else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      print_usage(argv[0]);
      std::exit(1);
    }
  }
  return args;
}

float ms_since(Clock::time_point t0) {
  return float(
      std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
}

float ms_between(Clock::time_point t0, Clock::time_point t1) {
  return float(std::chrono::duration<double, std::milli>(t1 - t0).count());
}

void print_arr(const std::vector<double> &v) {
  std::printf("[");
  for (int i = 0; i < (int)v.size(); ++i) {
    if (i)
      std::printf(", ");
    std::printf("%.4f", v[i]);
  }
  std::printf("]");
}

// Row-by-row dot product: y[i] += dot(A[i,:], x) for i in [0,H). The natural
// SpMV shape -- vectorizes along W (the block's own column span, the one
// axis SpMV actually has), unlike SpMM's gemm_fallback which vectorizes
// along an N-wide output row. With N=1 (a single output value per row per
// block), gemm_fallback's AXPY loop degenerates to exactly this same
// scalar-accumulate shape anyway, so this is a dedicated version of that
// degenerate case rather than a new algorithm.
inline void spmv_fallback(int H, int W, const Scalar *__restrict__ A, int lda,
                          const Scalar *__restrict__ x, Scalar *__restrict__ y) {
  for (int i = 0; i < H; ++i) {
    const Scalar *__restrict__ Ai = A + (long long)i * lda;
    Scalar acc = 0;
#pragma omp simd reduction(+ : acc)
    for (int p = 0; p < W; ++p)
      acc += Ai[p] * x[p];
    y[i] += acc;
  }
}

// Exact-shape dispatch to a named-register gemm_fixed<H,W,1,double>
// specialization (core/gen_kernel.py), the same shared kernel/dispatch
// mechanism SpGEMM uses (see cpu_dispatch.hpp's gemm<T>()) -- x/y are
// treated as W×1/H×1 "matrices" with unit row stride (ldb=ldc=1), which is
// exactly what gemm_fixed<H,W,1,T> expects: B[p*ldb+0] == x[p],
// C[i*ldc+0] == y[i]. Falls back to spmv_fallback for shapes with no
// specialization compiled in.
inline void spmv_dispatch(int H, int W, const Scalar *__restrict__ A, int lda,
                          const Scalar *__restrict__ x, Scalar *__restrict__ y) {
#ifdef GEMM_DISPATCH_H
#define DISPATCH(h, w, n)                                                    \
  if (H == h && W == w) {                                                    \
    benchmark_core::cpu_detail::gemm_fixed<h, w, n, Scalar>(A, lda, x, 1, y,  \
                                                            1);              \
    return;                                                                  \
  }
#include GEMM_DISPATCH_H
#undef DISPATCH
#endif
  spmv_fallback(H, W, A, lda, x, y);
}

} // namespace

int main(int argc, char **argv) {
  const Args args = parse_args(argc, argv);

  benchmark_core::Matrix<Scalar> S;
  auto t_read0 = Clock::now();
  try {
    S = benchmark_core::read_matrix_binsparse<Scalar>(args.s_bsp);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "prisma_cpu_spmv_bench: failed to read .bsp: %s\n",
                 e.what());
    return 1;
  }
  const float bsp_read_ms = ms_since(t_read0);

  const int M = S.M; // y has length M
  const int N = S.N; // x has length N -- M and N need not be equal, unlike
                      // the SpMM benchmark this is ported from, whose dense
                      // operand was square only because it was itself a
                      // matrix.

#ifdef _OPENMP
  // Thread count: same measured cap as prisma_cpu_spmm_bench.cpp (see that
  // file's comment for the full profiling rationale) -- carried forward as
  // a starting default, not re-validated against SpMV's own bandwidth
  // profile, which is plausibly different (even lower arithmetic intensity
  // per byte touched than SpMM, with zero reuse of a loaded A[i,p] across
  // multiple output columns).
  if (args.threads > 0) {
    omp_set_num_threads(args.threads);
  } else if (!std::getenv("OMP_NUM_THREADS")) {
    omp_set_num_threads(std::min(16, omp_get_max_threads()));
  }
#endif

  // Dense x: N random values in [-1, 1]
  std::vector<Scalar> x((size_t)N);
  {
    std::mt19937_64 rng((uint64_t)args.seed);
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
    std::generate(x.begin(), x.end(), [&] { return dist(rng); });
  }

  // Optionally dump x (N doubles) for external validation
  if (!args.dump_x.empty()) {
    FILE *fx = std::fopen(args.dump_x.c_str(), "wb");
    if (!fx) {
      std::fprintf(stderr, "prisma_cpu_spmv_bench: cannot open dump-x: %s\n",
                   args.dump_x.c_str());
    } else {
      std::fwrite(x.data(), sizeof(Scalar), x.size(), fx);
      std::fclose(fx);
    }
  }
  // Output y: M values, zeroed each run
  std::vector<Scalar> y((size_t)M);

  std::vector<double> symbolic_ms_arr, compute_ms_arr;
  symbolic_ms_arr.reserve(args.runs + 1);
  compute_ms_arr.reserve(args.runs + 1);

  float last_row_group_setup_ms = 0.0f;

  for (int r = 0; r <= args.runs; ++r) {
    // Group blocks so each group owns a disjoint y row range -> parallel
    // across groups, serial within a group (no races). Redone from scratch
    // every run (this codebase's strict rule: a real, one-off caller pays
    // this cost every call, not once amortised across many) instead of
    // being built once and reused: sort by starting row, then sweep-merge --
    // if a block's start row falls inside the current group's extent, it
    // belongs to the same group. Depends only on S's block geometry, not on
    // x's shape, so this is identical to prisma_cpu_spmm_bench.cpp's scheme.
    auto t_setup0 = Clock::now();
    struct RowGroup {
      std::vector<int> block_ids;
    };
    std::vector<RowGroup> row_groups;
    {
      std::vector<int> order((int)S.blocks.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int a, int b) { return S.blocks[a].r < S.blocks[b].r; });
      int group_end = -1;
      for (int bi : order) {
        const auto &blk = S.blocks[bi];
        if (blk.r >= group_end) {
          row_groups.push_back({});
          group_end = blk.r + blk.h;
        } else {
          group_end = std::max(group_end, blk.r + blk.h);
        }
        row_groups.back().block_ids.push_back(bi);
      }
    }

    // Reorder for x-locality: block.c selects which elements of x a block
    // reads, and many blocks across different row groups share the same or
    // nearby column range. Re-sort groups by mean block column (a pure
    // reordering of independent, row-disjoint work items -- doesn't change
    // results) so groups worked on concurrently or back-to-back reference
    // nearby x elements, improving cache reuse. Also sort each group's own
    // blocks by column for the same reason when a single thread walks them
    // serially.
    {
      for (auto &g : row_groups)
        std::sort(g.block_ids.begin(), g.block_ids.end(),
                  [&](int a, int b) { return S.blocks[a].c < S.blocks[b].c; });
      std::sort(row_groups.begin(), row_groups.end(),
                [&](const RowGroup &a, const RowGroup &b) {
                  double ca = 0, cb = 0;
                  for (int bi : a.block_ids)
                    ca += S.blocks[bi].c;
                  for (int bi : b.block_ids)
                    cb += S.blocks[bi].c;
                  ca /= (double)a.block_ids.size();
                  cb /= (double)b.block_ids.size();
                  return ca < cb;
                });
    }
    const float row_group_setup_ms = ms_since(t_setup0);

    if (r == 0) {
      std::vector<int> sizes;
      sizes.reserve(row_groups.size());
      for (const auto &g : row_groups)
        sizes.push_back((int)g.block_ids.size());
      std::sort(sizes.begin(), sizes.end());

      long long total = 0;
      for (int s : sizes)
        total += s;

      int n = (int)sizes.size();
      std::fprintf(
          stderr,
          "row_groups: count=%d  blocks=%lld  "
          "min=%d  p25=%d  median=%d  p75=%d  p95=%d  max=%d  mean=%.2f\n",
          n, total, sizes[0], sizes[n * 25 / 100], sizes[n * 50 / 100],
          sizes[n * 75 / 100], sizes[n * 95 / 100], sizes[n - 1],
          (double)total / n);

      // Row-overlap depth: for each y row, count how many blocks cover it.
      // max_depth = minimum number of serial passes any correct parallel
      // schedule needs.
      std::vector<int> depth(M, 0);
      for (const auto &blk : S.blocks)
        for (int ri = 0; ri < blk.h; ++ri)
          depth[blk.r + ri]++;
      int max_depth = *std::max_element(depth.begin(), depth.end());
      long long contested_rows = 0;
      for (int d : depth)
        if (d > 1)
          ++contested_rows;
      int max_h = 0;
      for (const auto &blk : S.blocks)
        max_h = std::max(max_h, blk.h);
      std::fprintf(
          stderr,
          "overlap: max_block_h=%d  max_depth=%d  contested_rows=%lld/%d\n",
          max_h, max_depth, contested_rows, M);
      std::fprintf(stderr,
                   "structural_setup (per run): bsp_read=%.4fms  "
                   "row_group=%.4fms\n",
                   bsp_read_ms, row_group_setup_ms);
    }

    auto t0 = Clock::now();

    // Zero y inside the timed compute region: this is real, unavoidable
    // per-run cost (accumulating blocks need a zeroed target) and TACO pays
    // an equivalent cost inside compute() -- though TACO's own SpMV kernel
    // does a direct assignment per row (its CSR row-segments are pre-merged
    // during pack_A, so each row's dot product completes in one pass and
    // needs no pre-zeroing); Prisma's row-groups can still have multiple
    // physically distinct blocks contributing to the same output row within
    // a group (that's exactly why row-groups + the "contested rows"
    // diagnostic above exist), so Prisma genuinely needs this zero+
    // accumulate even though TACO doesn't. Excluding it here would be
    // timing the two contenders inconsistently.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i)
      y[i] = Scalar(0);

    if (args.use_static) {
      // Static schedule: consecutive row groups land on the same thread, so
      // x elements loaded for group g may still be in cache for group g+1
      // if they share column structure.
#pragma omp parallel for schedule(static)
      for (int gi = 0; gi < (int)row_groups.size(); ++gi) {
        for (int bi : row_groups[gi].block_ids) {
          const benchmark_core::Block &blk = S.blocks[bi];
          if (args.specialized)
            spmv_dispatch(blk.h, blk.w, S.values + blk.offset, blk.w,
                         x.data() + blk.c, y.data() + blk.r);
          else
            spmv_fallback(blk.h, blk.w, S.values + blk.offset, blk.w,
                          x.data() + blk.c, y.data() + blk.r);
        }
      }
    } else {
      // Default: dynamic schedule with a small chunk size to amortise
      // dispatch overhead without sacrificing load balance for irregular
      // row groups.
#pragma omp parallel for schedule(dynamic, 4)
      for (int gi = 0; gi < (int)row_groups.size(); ++gi) {
        for (int bi : row_groups[gi].block_ids) {
          const benchmark_core::Block &blk = S.blocks[bi];
          if (args.specialized)
            spmv_dispatch(blk.h, blk.w, S.values + blk.offset, blk.w,
                         x.data() + blk.c, y.data() + blk.r);
          else
            spmv_fallback(blk.h, blk.w, S.values + blk.offset, blk.w,
                          x.data() + blk.c, y.data() + blk.r);
        }
      }
    }

    auto t1 = Clock::now();
    symbolic_ms_arr.push_back((double)row_group_setup_ms);
    compute_ms_arr.push_back((double)ms_between(t0, t1));

    if (r == args.runs)
      last_row_group_setup_ms = row_group_setup_ms;
  }

  // Optionally dump y (M doubles) for external validation
  if (!args.dump_y.empty()) {
    FILE *fy = std::fopen(args.dump_y.c_str(), "wb");
    if (!fy) {
      std::fprintf(stderr, "prisma_cpu_spmv_bench: cannot open dump-y: %s\n",
                   args.dump_y.c_str());
    } else {
      std::fwrite(y.data(), sizeof(Scalar), y.size(), fy);
      std::fclose(fy);
    }
  }

  std::printf("\nJSON_BEGIN\n{\n");
  std::printf("  \"kernel\": \"prisma_cpu_spmv\",\n");
  // pipe_total_ms / row_group_setup_ms below report the LAST run's setup
  // measurement purely for diagnostic display -- the real per-run values
  // consumers should use are symbolic_ms below, one genuine measurement per
  // run, not a single one-time cost re-reported.
  std::printf("  \"pipe_total_ms\": %.4f,\n", last_row_group_setup_ms);
  std::printf("  \"bsp_read_ms\": %.4f,\n", bsp_read_ms);
  std::printf("  \"row_group_setup_ms\": %.4f,\n", last_row_group_setup_ms);
  std::printf("  \"S_blocks\": %zu,\n", S.blocks.size());
  std::printf("  \"S_rows\": %d, \"S_cols\": %d,\n", M, N);
  std::printf("  \"symbolic_ms\": ");
  print_arr(symbolic_ms_arr);
  std::printf(",\n");
  std::printf("  \"compute_ms\": ");
  print_arr(compute_ms_arr);
  std::printf("\n}\nJSON_END\n");

  return 0;
}
