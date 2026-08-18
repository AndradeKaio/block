// GPU/prisma_bench.cu — PRISMA benchmark on real block-sparse matrices (.bsp)
//
// CLI: prisma_bench <A.bsp> <B.bsp> [--runs N] [--tc-kernel tile|block]
//
// Reads two .bsp files (pass the same path twice for A×A squaring),
// runs the full PRISMA block-sparse GEMM pipeline, and prints JSON timings
// between JSON_BEGIN / JSON_END sentinels — compatible with benchmark_gpu.py.
//
// Exits non-zero if either .bsp file does not exist.
//
// Compile (example, run from SpGEMM/GPU/):
//   nvcc -O3 --expt-relaxed-constexpr -std=c++20 -arch=sm_120 \
//        -DHAVE_HDF5 \
//        -I../../core -I. \
//        ../../core/block.cpp ../../core/block_generator.cpp \
//        ../../core/interval_tree.cpp \
//        ../../core/matrix.cpp ../../core/matrix_io.cpp ../../core/pipeline.cpp \
//        ../../core/segment_tree.cpp \
//        prisma_bench.cu \
//        -lhdf5 -o prisma_bench

#include "block.hpp"
#include "gpu_dispatch.cuh"
#include "gpu_pipeline.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"
#include "pipeline.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Scalar = float;

struct Args {
  std::string a_bsp;
  std::string b_bsp;
  std::string tc_kernel; // "tile", "block", or "" (CUDA-only)
  std::string validate;  // if non-empty, dump C as sorted COO to this path
  int runs = 10;
};

void print_usage(const char *prog) {
  std::fprintf(
      stderr,
      "Usage: %s <A.bsp> <B.bsp> [--runs N] [--tc-kernel tile|block]"
      " [--validate FILE]\n"
      "  Reads two .bsp files and benchmarks PRISMA block-sparse GEMM.\n"
      "  For A\xc3\x97A squaring, pass the same path for both arguments.\n"
      "  --validate FILE  After the last run, dump C as sorted COO to FILE.\n"
      "  Exits non-zero if either .bsp file is missing.\n",
      prog);
}

Args parse_args(int argc, char **argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    std::exit(1);
  }
  Args args;
  args.a_bsp = argv[1];
  args.b_bsp = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--runs")
      args.runs = std::atoi(next());
    else if (arg == "--tc-kernel")
      args.tc_kernel = next();
    else if (arg == "--validate")
      args.validate = next();
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

void emit_empty_json(const std::string &kernel_name) {
  std::printf("\nJSON_BEGIN\n{\n");
  std::printf("  \"kernel\": \"%s\",\n", kernel_name.c_str());
  std::printf(
      "  \"n_pairs\": 0, \"n_groups\": 0, \"n_tc\": 0, \"n_cuda\": 0,\n");
  std::printf("  \"plan_ms\": [], \"tc_ms\": [], \"cuda_ms\": []\n");
  std::printf("}\nJSON_END\n");
}

void print_arr(const std::vector<float> &v) {
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

  const std::string kernel_name =
      args.tc_kernel.empty() ? "prisma_cuda" : ("prisma_tc_" + args.tc_kernel);

  // ── Verify both .bsp files exist ─────────────────────────────────────────
  for (const auto &path : {args.a_bsp, args.b_bsp}) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
      std::fprintf(stderr, "prisma_bench: .bsp not found: %s\n", path.c_str());
      return 1;
    }
    std::fclose(f);
  }

  // ── Read matrices from .bsp ───────────────────────────────────────────────
  benchmark_core::Matrix<Scalar> A, B;
  try {
    A = benchmark_core::read_matrix_binsparse<Scalar>(args.a_bsp);
    B = benchmark_core::read_matrix_binsparse<Scalar>(args.b_bsp);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "prisma_bench: failed to read .bsp: %s\n", e.what());
    return 1;
  }

  std::printf("A: %dx%d  blocks=%zu\n", A.M, A.N, A.blocks.size());
  std::printf("B: %dx%d  blocks=%zu\n", B.M, B.N, B.blocks.size());

  if (A.blocks.empty() || B.blocks.empty()) {
    std::fprintf(stderr, "prisma_bench: one or both matrices have no blocks\n");
    emit_empty_json(kernel_name);
    return 0;
  }

  // ── Pipeline (one-time): intersect → merge → fuse → classify → plan ────────
  using Clock = std::chrono::steady_clock;
  auto ms_since = [](Clock::time_point t0) {
    return float(
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  };

  benchmark_core::IntersectionTimings isec_timings;
  auto t0 = Clock::now();
  auto found = benchmark_core::find_intersecting_pairs(A.blocks, B.blocks, &isec_timings);
  float pipe_intersect_ms = ms_since(t0);

  if (found.contributions.empty()) {
    std::printf("No intersecting pairs.\n");
    emit_empty_json(kernel_name);
    return 0;
  }

  // merge_overlapping_output_blocks fuses the old
  // find_overlapping_output_blocks
  // + merge_groups two-step into a single sweep+union-find pass.
  t0 = Clock::now();
  auto groups =
      benchmark_core::merge_overlapping_output_blocks(found.output_blocks);
  float pipe_merge_ms = ms_since(t0);

  t0 = Clock::now();
  auto fused = benchmark_core::block_fusion(found.output_blocks,
                                            found.contributions, groups);
  float pipe_fuse_ms = ms_since(t0);

  std::printf("pairs=%zu  groups=%zu\n", found.contributions.size(),
              fused.fused_blocks.size());

  const bool use_tc = !args.tc_kernel.empty();
  const benchmark_core::TcStrategy strategy =
      (args.tc_kernel == "block") ? benchmark_core::TcStrategy::PerBlock
                                  : benchmark_core::TcStrategy::PerTile;

  // Build the execution plan once — it only depends on fused/A/B which are
  // constant across runs. C.values is zero-initialised here; upload_plan
  // uses cudaMemset to re-zero device C each run so we don't need to reset
  // the host copy.
  t0 = Clock::now();
  auto cls = benchmark_core::gpu_kernel_classify(fused, use_tc);
  float pipe_classify_ms = ms_since(t0);

  t0 = Clock::now();
  auto [C, plan] =
      benchmark_core::gpu_kernel_plan<Scalar>(fused, cls, A, B, use_tc);
  float pipe_build_ms = ms_since(t0);

  const int saved_n_tc = (int)plan.tc_tiles.size();
  const int saved_n_cuda = (int)plan.cuda_descs.size();
  std::printf("classify (fused blocks): tc=%d  cuda=%d\n", cls.n_tc,
              cls.n_cuda);
  std::printf(
      "plan (sub-regions): k_entries=%zu  tc_tiles=%zu  cuda_descs=%zu\n",
      plan.k_entries.size(), plan.tc_tiles.size(), plan.cuda_descs.size());

  float pipe_total = pipe_intersect_ms + pipe_merge_ms + pipe_fuse_ms +
                     pipe_classify_ms + pipe_build_ms;

  // ── Warmup (r=0) + timed runs (r=1..runs) ────────────────────────────────
  // plan_ms is 0 every run (plan is cached); only compute is measured here.
  std::vector<float> plan_ms_arr, tc_ms_arr, cuda_ms_arr;
  plan_ms_arr.reserve(args.runs + 1);
  tc_ms_arr.reserve(args.runs + 1);
  cuda_ms_arr.reserve(args.runs + 1);

  for (int r = 0; r <= args.runs; ++r) {
    // Compute phase: upload (device C zeroed by cudaMemset) + kernels + sync
    auto [rC, times] = benchmark_core::run(plan, A, B, std::move(C), strategy);
    C = std::move(rC);
    plan_ms_arr.push_back(0.f);
    tc_ms_arr.push_back(times.tc_ms);
    cuda_ms_arr.push_back(times.cuda_ms);
  }

  std::printf("runs=%d  last: tc=%.3fms  cuda=%.3fms\n", args.runs,
              tc_ms_arr.back(), cuda_ms_arr.back());

  // ── Optional validation dump: sorted COO of result C ─────────────────────
  if (!args.validate.empty()) {
    std::vector<std::tuple<int, int, Scalar>> entries;
    entries.reserve(C.n_values / 4); // rough pre-size
    for (const auto &blk : C.blocks) {
      for (int ri = 0; ri < blk.h; ++ri) {
        for (int ci = 0; ci < blk.w; ++ci) {
          const Scalar v =
              C.values[blk.offset + static_cast<long long>(ri) * blk.w + ci];
          if (v != Scalar(0))
            entries.emplace_back(blk.r + ri, blk.c + ci, v);
        }
      }
    }
    std::sort(entries.begin(), entries.end());
    FILE *vf = std::fopen(args.validate.c_str(), "w");
    if (!vf) {
      std::fprintf(stderr, "prisma_bench: cannot open validate file: %s\n",
                   args.validate.c_str());
    } else {
      for (const auto &[r, c, v] : entries)
        std::fprintf(vf, "%d %d %.8e\n", r, c, v);
      std::fclose(vf);
      std::printf("validate: wrote %zu non-zeros to %s\n", entries.size(),
                  args.validate.c_str());
    }
  }

  // ── Timing breakdown ──────────────────────────────────────────────────────
  auto avg = [&](const std::vector<float> &v) {
    if (v.size() <= 1)
      return v.empty() ? 0.f : v[0];
    float s = 0.f;
    for (int i = 1; i < (int)v.size(); ++i)
      s += v[i];
    return s / float(v.size() - 1);
  };
  float avg_tc = avg(tc_ms_arr);
  float avg_cuda = avg(cuda_ms_arr);
  float avg_compute = avg_tc + avg_cuda;

  std::printf("\n── symbolic breakdown ───────────────────────────────────\n");
  std::printf("  pipeline (one-time)\n");
  std::printf("    intersect_pairs  : %8.3f ms\n", pipe_intersect_ms);
  std::printf("      tree_build     : %8.3f ms\n", isec_timings.tree_build_ms);
  std::printf("      query          : %8.3f ms\n", isec_timings.query_ms);
  std::printf("    overlap+merge    : %8.3f ms\n", pipe_merge_ms);
  std::printf("    block_fusion     : %8.3f ms\n", pipe_fuse_ms);
  std::printf("    classify         : %8.3f ms\n", pipe_classify_ms);
  std::printf("    build_plan       : %8.3f ms\n", pipe_build_ms);
  std::printf("    total            : %8.3f ms\n", pipe_total);
  std::printf("  compute per-run (avg of %d timed runs)\n", args.runs);
  std::printf("    tc_kernel        : %8.3f ms\n", avg_tc);
  std::printf("    cuda_kernel      : %8.3f ms\n", avg_cuda);
  std::printf("    compute total    : %8.3f ms\n", avg_compute);
  std::printf("  wall (amortised over %d runs): %.3f ms\n", args.runs,
              pipe_total / args.runs + avg_compute);
  std::printf("────────────────────────────────────────────────────────\n");

  // ── JSON output ───────────────────────────────────────────────────────────
  std::printf("\nJSON_BEGIN\n{\n");
  std::printf("  \"kernel\": \"%s\",\n", kernel_name.c_str());
  std::printf("  \"n_pairs\": %zu, \"n_groups\": %zu,\n",
              found.contributions.size(), fused.fused_blocks.size());
  std::printf("  \"n_tc\": %d, \"n_cuda\": %d,\n", saved_n_tc, saved_n_cuda);
  std::printf("  \"pipe_intersect_ms\": %.4f,\n", pipe_intersect_ms);
  std::printf("  \"pipe_tree_build_ms\": %.4f,\n", isec_timings.tree_build_ms);
  std::printf("  \"pipe_query_ms\": %.4f,\n", isec_timings.query_ms);
  std::printf("  \"pipe_merge_ms\": %.4f,\n", pipe_merge_ms);
  std::printf("  \"pipe_fuse_ms\": %.4f,\n", pipe_fuse_ms);
  std::printf("  \"pipe_classify_ms\": %.4f,\n", pipe_classify_ms);
  std::printf("  \"pipe_build_ms\": %.4f,\n", pipe_build_ms);
  std::printf("  \"pipe_total_ms\": %.4f,\n", pipe_total);
  std::printf("  \"plan_ms\": ");
  print_arr(plan_ms_arr);
  std::printf(",\n  \"tc_ms\": ");
  print_arr(tc_ms_arr);
  std::printf(",\n  \"cuda_ms\": ");
  print_arr(cuda_ms_arr);
  std::printf("\n}\nJSON_END\n");

  return 0;
}
