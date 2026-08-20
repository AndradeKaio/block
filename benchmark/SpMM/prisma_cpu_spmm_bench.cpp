#include "block.hpp"
#include "cpu_dispatch.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"
#include "spmm_dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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
  int tile_n = 0;
  bool auto_schedule = false; // --auto: pick static vs. tiled at runtime
                              // instead of requiring the caller to know
                              // which one their matrix wants (see main()).
  int threads = 0; // 0 = auto (see main(): capped default unless
                    // OMP_NUM_THREADS is set); >0 = explicit override
  std::string dump_d; // --dump-d <path>: write N×N D matrix as raw doubles
  std::string dump_c; // --dump-c <path>: write M×N C matrix as raw doubles
};

void print_usage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s <S.bsp> [--runs N] [--seed S] [--specialized-kernels]"
               " [--static] [--tile-n T] [--auto] [--threads N] [--dump-d path]"
               " [--dump-c path]\n",
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
    else if (arg == "--tile-n" && i + 1 < argc)
      args.tile_n = std::atoi(argv[++i]);
    else if (arg == "--auto")
      args.auto_schedule = true;
    else if (arg == "--threads" && i + 1 < argc)
      args.threads = std::atoi(argv[++i]);
    else if (arg == "--dump-d" && i + 1 < argc)
      args.dump_d = argv[++i];
    else if (arg == "--dump-c" && i + 1 < argc)
      args.dump_c = argv[++i];
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

} // namespace

int main(int argc, char **argv) {
  const Args args = parse_args(argc, argv);

  benchmark_core::Matrix<Scalar> S;
  auto t_read0 = Clock::now();
  try {
    S = benchmark_core::read_matrix_binsparse<Scalar>(args.s_bsp);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "prisma_cpu_spmm_bench: failed to read .bsp: %s\n",
                 e.what());
    return 1;
  }
  const float bsp_read_ms = ms_since(t_read0);

  const int M = S.M;
  const int N = S.N; // square matrix: dense RHS is N×N

  // Structural setup: everything below that depends only on S's sparsity
  // pattern (not on D) is redone from scratch on every timed run, inside the
  // loop below — a real, one-off caller who invokes SpMM once pays this cost
  // every single time, not once amortised across many calls. Previously this
  // was built ONCE outside the run loop and that single measurement was
  // re-reported on every row (misrepresenting a one-shot cost as free on
  // every subsequent call); now symbolic_ms is a genuine per-run
  // measurement, matching the same fix applied to Prisma's SpGEMM symbolic
  // pipeline (see SpGEMM/CPU/prisma_cpu_bench.cpp) and to TACO's own
  // pack_B() (see bench_taco_spmm.cpp, which now re-packs B every run too).

#ifdef _OPENMP
  // Thread count: this workload is fork-join- and memory-bandwidth-bound
  // rather than compute-bound — measured on this 64-thread machine, going
  // from ~16 to 64 threads makes every matrix tested slower (up to ~15x on
  // small matrices dominated by parallel-region launch overhead, ~2x on
  // large bandwidth-saturated ones).
  //
  // Tried capping against the actual unit count (tasks.size() for tiled,
  // row_groups.size() for static/default) instead of a flat constant,
  // reasoning that a matrix like olm1000 — which --auto correctly routes to
  // "tiled" but which only decomposes into 2 tasks at N=1000 — shouldn't pay
  // for 16 idle threads. Measured instead of assumed: OMP_NUM_THREADS=2 on
  // that exact matrix is ~3x SLOWER than both 1 thread (0.44ms) and the old
  // flat 16-thread cap's median (0.55ms) — some pathology specific to small
  // thread counts on this machine that a bigger cap coincidentally avoids
  // more often than the "correctly sized" small cap does. Reverted: a
  // reasoning-backed change that regresses the measured benchmark doesn't
  // ship, however sound the reasoning seemed going in. Flat 16 is not
  // provably optimal, just the simplest rule that measured better than the
  // alternatives tried so far — real headroom likely remains here.
  //
  // --threads overrides explicitly; otherwise respect a caller-set
  // OMP_NUM_THREADS (they asked for a specific width on purpose); only
  // apply the cap when neither is given, i.e. when we'd otherwise silently
  // default to all cores.
  if (args.threads > 0) {
    omp_set_num_threads(args.threads);
  } else if (!std::getenv("OMP_NUM_THREADS")) {
    omp_set_num_threads(std::min(16, omp_get_max_threads()));
  }
#endif

  // Dense D: N×N random in [-1, 1] (row-major, ldd = N)
  std::vector<Scalar> D((long long)N * N);
  {
    std::mt19937_64 rng((uint64_t)args.seed);
    std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
    std::generate(D.begin(), D.end(), [&] { return dist(rng); });
  }

  // Optionally dump D (N×N row-major doubles) for external validation
  if (!args.dump_d.empty()) {
    FILE *fd = std::fopen(args.dump_d.c_str(), "wb");
    if (!fd) {
      std::fprintf(stderr, "prisma_cpu_spmm_bench: cannot open dump-d: %s\n",
                   args.dump_d.c_str());
    } else {
      std::fwrite(D.data(), sizeof(Scalar), D.size(), fd);
      std::fclose(fd);
    }
  }
  // Dense C: M×N output (row-major, ldc = N), zeroed each run
  std::vector<Scalar> C((long long)M * N);

  // Count block shape frequencies (outside timing, for metadata).
  std::map<std::pair<int, int>, int> shape_freq;
  if (args.specialized) {
    for (const auto &blk : S.blocks)
      shape_freq[{blk.h, blk.w}]++;
  }

  std::vector<double> symbolic_ms_arr, compute_ms_arr;
  symbolic_ms_arr.reserve(args.runs + 1);
  compute_ms_arr.reserve(args.runs + 1);

  float last_row_group_setup_ms = 0.0f, last_task_list_setup_ms = 0.0f,
        last_structural_setup_ms = 0.0f;

  for (int r = 0; r <= args.runs; ++r) {
    // Group blocks so each group owns a disjoint C row range → parallel
    // across groups, serial within a group (no races). Redone from scratch
    // every run (see the top-of-function comment) instead of being built
    // once and reused: sort by starting row, then sweep-merge — if a
    // block's start row falls inside the current group's extent, it
    // belongs to the same group.
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

    // Reorder for D locality: block.c selects which rows of the huge N×N
    // dense D operand a block reads (D doesn't fit in any cache for
    // realistic N), and many blocks across different row groups share the
    // same or nearby column range (mined block shapes repeat; empirically
    // 3-13x more total block width than N on the matrices this was
    // profiled against). The row-major grouping above interleaves column
    // ranges arbitrarily, so consecutive groups — which land on the same or
    // nearby threads under static/dynamic/tiled scheduling — usually pull
    // unrelated D rows. Re-sort groups by mean block column (a pure
    // reordering of independent, row-disjoint work items — doesn't change
    // results) so groups worked on concurrently or back-to-back reference
    // nearby D rows, improving L2/L3 reuse. Also sort each group's own
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

    // Strategy selection. --static and --tile-n remain explicit, independent
    // overrides; --auto picks the strategy at runtime from two cheap
    // signals (max_share, N) fit against a profiling sweep across 28
    // SuiteSparse matrices — see the version history for the full
    // rationale. Recomputed every run alongside row_groups since it depends
    // on max_share (derived from row_groups); the decision itself is
    // invariant across runs (same S every time), only the measurement is
    // now genuinely fresh.
    const double kSevereCollapseCutoff = 0.5;
    const double kImbalanceCutoff = 0.10;
    const int kImbalanceMinN = 2000;
    const int kSizeCutoffN = 4096;
    double max_share = 0.0;
    {
      size_t max_group_blocks = 0;
      for (const auto &g : row_groups)
        max_group_blocks = std::max(max_group_blocks, g.block_ids.size());
      if (!S.blocks.empty())
        max_share = (double)max_group_blocks / (double)S.blocks.size();
    }
    const bool auto_mode =
        args.auto_schedule && !args.use_static && args.tile_n == 0;
    const bool auto_picks_tiled =
        max_share > kSevereCollapseCutoff ||
        (max_share > kImbalanceCutoff && N > kImbalanceMinN) ||
        N >= kSizeCutoffN;
    const bool use_tiled = args.tile_n > 0 || (auto_mode && auto_picks_tiled);
    const bool use_static_sched =
        args.use_static || (auto_mode && !auto_picks_tiled);
    if (auto_mode && r == 0)
      std::fprintf(stderr, "auto: max_share=%.3f N=%d -> %s\n", max_share, N,
                   use_tiled ? "tiled" : "static");

    // Flat (group, column-tile) task list for the tiled strategy below.
    // Why this can't just be "for each tile, omp-for over groups" (nested
    // the other way, as a first attempt had it): row_groups only guarantees
    // *row*-disjointness, and for matrices with heavy row-overlap between
    // mined blocks, the merge sweep above can collapse almost everything
    // into one or two groups (measured: linverse collapses to a SINGLE
    // group of 5999 blocks; bundle1 puts 61% of its blocks in one group of
    // 8). Nesting an omp-for over groups *inside* a serial tile loop pins
    // every tile of such a group to the same thread (schedule(static)
    // deterministically maps a lone iteration to the same thread every
    // time), so a dominant group gets zero parallelism no matter the thread
    // count — confirmed: both matrices showed flat wall-clock time from 4
    // to 64 threads. Flattening to one task per (group, tile) and
    // parallelizing over the flat list lets a single group's column range
    // split across many threads: different tiles of the same group touch
    // disjoint C columns, so they're safe to run concurrently; only blocks
    // *within* one (group, tile) task must stay serial (they can share
    // output columns).
    struct Task {
      int gi, j, t;
    };
    std::vector<Task> tasks;
    auto t_tasklist0 = Clock::now();
    if (use_tiled) {
      const int T = args.tile_n > 0 ? args.tile_n : 512;
      int n_tiles = (N + T - 1) / T;
      tasks.reserve((size_t)row_groups.size() * n_tiles);
      for (int gi = 0; gi < (int)row_groups.size(); ++gi)
        for (int j = 0; j < N; j += T)
          tasks.push_back({gi, j, std::min(T, N - j)});
    }
    const float task_list_setup_ms = ms_since(t_tasklist0);
    const float structural_setup_ms = row_group_setup_ms + task_list_setup_ms;

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

      // Row-overlap depth: for each C row, count how many blocks cover it.
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
      std::fprintf(
          stderr,
          "structural_setup (per run): bsp_read=%.4fms  row_group=%.4fms  "
          "task_list=%.4fms  total(excl. read)=%.4fms\n",
          bsp_read_ms, row_group_setup_ms, task_list_setup_ms,
          structural_setup_ms);
    }

    auto t0 = Clock::now();

    // Zero C inside the timed compute region: this is real, unavoidable
    // per-run cost (accumulating blocks need a zeroed target) and TACO pays
    // the equivalent cost inside compute() (its first loop zeros A_vals).
    // Excluding it here would be timing the two contenders inconsistently.
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)M * N; ++i)
      C[i] = Scalar(0);

    if (use_tiled) {
      // Option 2: column tiling over a flat (group, tile) task list — see
      // the comment where `tasks` is built for why this must be flat rather
      // than a tile loop wrapped around a per-group omp-for: a single
      // dominant row group needs its column range spread across threads,
      // not pinned to one. dynamic scheduling since task cost varies with
      // both group size and (for the last tile) width.
#pragma omp parallel for schedule(dynamic, 1)
      for (int ti = 0; ti < (int)tasks.size(); ++ti) {
        const Task &task = tasks[ti];
        for (int bi : row_groups[task.gi].block_ids) {
          const benchmark_core::Block &blk = S.blocks[bi];
          const Scalar *A_ptr = S.values + blk.offset;
          const Scalar *B_ptr = D.data() + (long long)blk.c * N + task.j;
          Scalar *C_ptr = C.data() + (long long)blk.r * N + task.j;

          if (args.specialized)
            spmm_dispatch(blk.h, blk.w, task.t, A_ptr, blk.w, B_ptr, N, C_ptr,
                          N);
          else
            benchmark_core::cpu_detail::gemm_fallback(
                blk.h, blk.w, task.t, A_ptr, blk.w, B_ptr, N, C_ptr, N);
        }
      }
    } else if (use_static_sched) {
      // Option 1: static schedule — consecutive row groups land on the same
      // thread, so D rows loaded for group g may still be in L3 for group g+1
      // if they share column structure.
#pragma omp parallel for schedule(static)
      for (int gi = 0; gi < (int)row_groups.size(); ++gi) {
        for (int bi : row_groups[gi].block_ids) {
          const benchmark_core::Block &blk = S.blocks[bi];
          const Scalar *A_ptr = S.values + blk.offset;
          const Scalar *B_ptr = D.data() + (long long)blk.c * N;
          Scalar *C_ptr = C.data() + (long long)blk.r * N;

          if (args.specialized)
            spmm_dispatch(blk.h, blk.w, N, A_ptr, blk.w, B_ptr, N, C_ptr, N);
          else
            benchmark_core::cpu_detail::gemm_fallback(
                blk.h, blk.w, N, A_ptr, blk.w, B_ptr, N, C_ptr, N);
        }
      }
    } else {
      // Default: dynamic schedule with a small chunk size to amortise dispatch
      // overhead without sacrificing load balance for irregular row groups.
#pragma omp parallel for schedule(dynamic, 4)
      for (int gi = 0; gi < (int)row_groups.size(); ++gi) {
        for (int bi : row_groups[gi].block_ids) {
          const benchmark_core::Block &blk = S.blocks[bi];
          const Scalar *A_ptr = S.values + blk.offset;
          const Scalar *B_ptr = D.data() + (long long)blk.c * N;
          Scalar *C_ptr = C.data() + (long long)blk.r * N;

          if (args.specialized)
            spmm_dispatch(blk.h, blk.w, N, A_ptr, blk.w, B_ptr, N, C_ptr, N);
          else
            benchmark_core::cpu_detail::gemm_fallback(
                blk.h, blk.w, N, A_ptr, blk.w, B_ptr, N, C_ptr, N);
        }
      }
    }

    auto t1 = Clock::now();
    symbolic_ms_arr.push_back((double)structural_setup_ms);
    compute_ms_arr.push_back((double)ms_between(t0, t1));

    if (r == args.runs) {
      last_row_group_setup_ms = row_group_setup_ms;
      last_task_list_setup_ms = task_list_setup_ms;
      last_structural_setup_ms = structural_setup_ms;
    }
  }

  // Optionally dump C (M×N row-major doubles) for external validation
  if (!args.dump_c.empty()) {
    FILE *fc = std::fopen(args.dump_c.c_str(), "wb");
    if (!fc) {
      std::fprintf(stderr, "prisma_cpu_spmm_bench: cannot open dump-c: %s\n",
                   args.dump_c.c_str());
    } else {
      std::fwrite(C.data(), sizeof(Scalar), C.size(), fc);
      std::fclose(fc);
    }
  }

  std::printf("\nJSON_BEGIN\n{\n");
  std::printf("  \"kernel\": \"prisma_cpu_spmm\",\n");
  // pipe_total_ms / row_group_setup_ms / task_list_setup_ms below report the
  // LAST run's structural-setup measurement (see last_structural_setup_ms
  // etc., stashed inside the loop) purely for diagnostic display — the
  // real per-run values consumers should use are symbolic_ms below, one
  // genuine measurement per run, not a single one-time cost re-reported.
  std::printf("  \"pipe_total_ms\": %.4f,\n", last_structural_setup_ms);
  std::printf("  \"bsp_read_ms\": %.4f,\n", bsp_read_ms);
  std::printf("  \"row_group_setup_ms\": %.4f,\n", last_row_group_setup_ms);
  std::printf("  \"task_list_setup_ms\": %.4f,\n", last_task_list_setup_ms);
  std::printf("  \"S_blocks\": %zu,\n", S.blocks.size());
  std::printf("  \"S_rows\": %d, \"S_cols\": %d,\n", M, N);
  if (!shape_freq.empty()) {
    std::printf("  \"shape_freq\": {");
    bool first = true;
    for (const auto &[hw, cnt] : shape_freq) {
      if (!first)
        std::printf(", ");
      std::printf("\"%dx%d\": %d", hw.first, hw.second, cnt);
      first = false;
    }
    std::printf("},\n");
  }
  std::printf("  \"symbolic_ms\": ");
  print_arr(symbolic_ms_arr);
  std::printf(",\n");
  std::printf("  \"compute_ms\": ");
  print_arr(compute_ms_arr);
  std::printf("\n}\nJSON_END\n");

  return 0;
}
