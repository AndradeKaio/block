// Standalone GPU demo/benchmark: runs the block-sparse GEMM pipeline through
// actual CUDA dispatch (phase 5, gpu_dispatch.cuh).
//
// Outputs a JSON_BEGIN/JSON_END block with per-run TC and CUDA kernel timings,
// plus metadata — suitable for machine parsing by sweep.py.
//
// Uses float: tc_kernel stores via reinterpret_cast<float*>(C_ptr), only valid
// when T == float.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "block.hpp"
#include "gpu_dispatch.cuh"
#include "gpu_pipeline.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"
#include "pipeline.hpp"

namespace {

using Scalar = float;

struct Args {
  int M = 2048, K = 2048, N = 2048;
  int blocks_A = 16, blocks_B = 16;
  int block_h_min = 8, block_h_max = 14;
  int block_w_min = 8, block_w_max = 14;
  std::uint64_t seed = 42;
  std::string tc_kernel;
  int runs = 1;
  double block_density = 1.0;
  std::string output;  // write C to this .mtx path (last run only)
  std::string out_dir; // write A.mtx / B.mtx here; empty = cwd
};

void print_usage() {
  std::printf(
      "Usage: gpu_dispatch_demo [options]\n"
      "  --M N              rows of A             (default 2048)\n"
      "  --K N              cols of A / rows of B (default 2048)\n"
      "  --N N              cols of B             (default 2048)\n"
      "  --blocks-A N       number of A blocks    (default 16)\n"
      "  --blocks-B N       number of B blocks    (default 16)\n"
      "  --block-h-min N    min block height      (default 8)\n"
      "  --block-h-max N    max block height      (default 14)\n"
      "  --block-w-min N    min block width       (default 8)\n"
      "  --block-w-max N    max block width       (default 14)\n"
      "  --seed N           RNG seed              (default 42)\n"
      "  --runs N           timed repetitions     (default 1)\n"
      "  --tc-kernel         snap blocks to multiples of 16 (TC kernel). The "
      "CTA division is based on the kernel type \"tile\" or \"block\"\n"
      "Kernel only)\n"
      "  --block-density F  fraction of elements kept nonzero within each block (default: 1.0)\n"
      "  --out-dir PATH     write A.mtx/B.mtx here (default: cwd)\n"
      "  --output PATH      write result C to .mtx file\n");
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--M")
      args.M = std::atoi(next());
    else if (arg == "--K")
      args.K = std::atoi(next());
    else if (arg == "--N")
      args.N = std::atoi(next());
    else if (arg == "--blocks-A")
      args.blocks_A = std::atoi(next());
    else if (arg == "--blocks-B")
      args.blocks_B = std::atoi(next());
    else if (arg == "--block-h-min")
      args.block_h_min = std::atoi(next());
    else if (arg == "--block-h-max")
      args.block_h_max = std::atoi(next());
    else if (arg == "--block-w-min")
      args.block_w_min = std::atoi(next());
    else if (arg == "--block-w-max")
      args.block_w_max = std::atoi(next());
    else if (arg == "--seed")
      args.seed = static_cast<std::uint64_t>(std::atoll(next()));
    else if (arg == "--runs")
      args.runs = std::atoi(next());
    else if (arg == "--tc-kernel")
      args.tc_kernel = next();
    else if (arg == "--block-density")
      args.block_density = std::atof(next());
    else if (arg == "--out-dir")
      args.out_dir = next();
    else if (arg == "--output")
      args.output = next();
    else if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      print_usage();
      std::exit(1);
    }
  }
  return args;
}

std::string mtx_path(const Args &args, const char *name) {
  return args.out_dir.empty() ? std::string("./") + name
                              : args.out_dir + "/" + name;
}

} // namespace

int main(int argc, char **argv) {
  const Args args = parse_args(argc, argv);

  const auto A = benchmark_core::generate_random_matrix<Scalar>(
      args.M, args.K, args.blocks_A, {args.block_h_min, args.block_h_max},
      {args.block_w_min, args.block_w_max}, args.seed,
      (args.tc_kernel == "block" || args.tc_kernel == "tile"), args.block_density);
  const auto B = benchmark_core::generate_random_matrix<Scalar>(
      args.K, args.N, args.blocks_B, {args.block_h_min, args.block_h_max},
      {args.block_w_min, args.block_w_max}, args.seed + 1,
      args.tc_kernel != "", args.block_density);

  std::printf("%dx%d @ %dx%d  blocks=%d/%d\n", args.M, args.K, args.K, args.N,
              args.blocks_A, args.blocks_B);
  std::printf("A blocks=%zu/%d  B blocks=%zu/%d\n", A.blocks.size(),
              args.blocks_A, B.blocks.size(), args.blocks_B);

  // Always write A.mtx/B.mtx so cuSPARSE can reuse the same matrices.
  benchmark_core::write_matrix_market(A, mtx_path(args, "A.mtx"));
  benchmark_core::write_matrix_market(B, mtx_path(args, "B.mtx"));
  std::printf("wrote A.mtx / B.mtx → %s\n",
              args.out_dir.empty() ? "./" : args.out_dir.c_str());

  // --- empty-C sentinel for early exits ------------------------------------
  auto write_empty_c = [&]() {
    if (!args.output.empty()) {
      benchmark_core::Matrix<Scalar> ec;
      ec.M = args.M;
      ec.N = args.N;
      benchmark_core::write_matrix_market(ec, args.output);
      std::printf("wrote empty C → %s\n", args.output.c_str());
    }
  };

  auto emit_empty_json = [&](int n_tc, int n_cuda) {
    std::printf("\nJSON_BEGIN\n{\n");
    std::printf("  \"M\": %d, \"K\": %d, \"N\": %d,\n", args.M, args.K, args.N);
    std::printf("  \"blocks_A\": %zu, \"blocks_B\": %zu,\n", A.blocks.size(),
                B.blocks.size());
    std::printf("  \"block_h_min\": %d, \"block_h_max\": %d,\n",
                args.block_h_min, args.block_h_max);
    std::printf("  \"block_w_min\": %d, \"block_w_max\": %d,\n",
                args.block_w_min, args.block_w_max);
    std::printf("  \"tc_kernel\": \"%s\",\n", args.tc_kernel.c_str());
    std::printf("  \"block_density\": %.4f,\n", args.block_density);
    std::printf("  \"n_pairs\": 0, \"n_groups\": 0,\n");
    std::printf("  \"n_tc\": %d, \"n_cuda\": %d,\n", n_tc, n_cuda);
    std::printf("  \"plan_ms\": [], \"tc_ms\": [], \"cuda_ms\": []\n");
    std::printf("}\nJSON_END\n");
  };

  if (A.blocks.empty() || B.blocks.empty()) {
    std::printf("No blocks placed.\n");
    write_empty_c();
    emit_empty_json(0, 0);
    return 0;
  }

  auto found = benchmark_core::find_intersecting_pairs(A.blocks, B.blocks);
  auto overlap_pairs =
      benchmark_core::find_overlapping_output_blocks(found.output_blocks);
  auto groups =
      benchmark_core::merge_groups(found.output_blocks, overlap_pairs);
  auto fused = benchmark_core::block_fusion(found.output_blocks,
                                            found.contributions, groups);

  if (found.contributions.empty()) {
    std::printf("No intersecting pairs.\n");
    write_empty_c();
    emit_empty_json(0, 0);
    return 0;
  }

  std::printf("pairs=%zu  groups=%zu\n", found.contributions.size(),
              fused.fused_blocks.size());

  const benchmark_core::TcStrategy strategy =
      (args.tc_kernel == "block") ? benchmark_core::TcStrategy::PerBlock
                                  : benchmark_core::TcStrategy::PerTile;

  // --- warmup (run_id=0) + timed runs (run_id=1..runs) ----------------------
  // Planning (classify + plan) is timed per-run as the symbolic phase.
  std::vector<float> tc_ms_arr, cuda_ms_arr, plan_ms_arr;
  tc_ms_arr.reserve(args.runs + 1);
  cuda_ms_arr.reserve(args.runs + 1);
  plan_ms_arr.reserve(args.runs + 1);

  int saved_n_tc = 0, saved_n_cuda = 0;

  benchmark_core::Matrix<Scalar> C_last;
  for (int r = 0; r <= args.runs; ++r) {
    // ── Symbolic phase: classify + plan (CPU-side, wall-clock timed) ────────
    auto sym_start = std::chrono::steady_clock::now();
    auto cls = benchmark_core::gpu_kernel_classify(
        fused, (args.tc_kernel == "tile" || args.tc_kernel == "block"));
    auto [C, plan] = benchmark_core::gpu_kernel_plan<Scalar>(fused, cls, A, B);
    auto sym_end = std::chrono::steady_clock::now();
    float plan_ms = float(
        std::chrono::duration<double, std::milli>(sym_end - sym_start).count());
    plan_ms_arr.push_back(plan_ms);

    if (r == 0) {
      saved_n_tc   = cls.n_tc;
      saved_n_cuda = cls.n_cuda;
      std::printf("classify: tc=%d  cuda=%d\n", cls.n_tc, cls.n_cuda);
      std::printf("plan: k_entries=%zu  tc_descs=%zu  cuda_descs=%zu\n",
                  plan.k_entries.size(), plan.tc_descs.size(),
                  plan.cuda_descs.size());
    }

    // ── Compute phase: upload + kernel + download (CUDA-event timed) ────────
    auto [rC, times] = benchmark_core::run(plan, A, B, std::move(C), strategy);
    tc_ms_arr.push_back(times.tc_ms);
    cuda_ms_arr.push_back(times.cuda_ms);
    if (r == args.runs) C_last = std::move(rC);
  }

  std::printf("runs=%d  last: plan=%.3fms  tc=%.3fms  cuda=%.3fms\n", args.runs,
              plan_ms_arr.back(), tc_ms_arr.back(), cuda_ms_arr.back());

  if (!args.output.empty()) {
    benchmark_core::write_matrix_market(C_last, args.output);
    std::printf("wrote C → %s\n", args.output.c_str());
  }

  // --- JSON output ---------------------------------------------------------
  auto print_arr = [](const std::vector<float> &v) {
    std::printf("[");
    for (int i = 0; i < (int)v.size(); ++i) {
      if (i)
        std::printf(", ");
      std::printf("%.4f", v[i]);
    }
    std::printf("]");
  };

  std::printf("\nJSON_BEGIN\n{\n");
  std::printf("  \"M\": %d, \"K\": %d, \"N\": %d,\n", args.M, args.K, args.N);
  std::printf("  \"blocks_A\": %zu, \"blocks_B\": %zu,\n", A.blocks.size(),
              B.blocks.size());
  std::printf("  \"block_h_min\": %d, \"block_h_max\": %d,\n", args.block_h_min,
              args.block_h_max);
  std::printf("  \"block_w_min\": %d, \"block_w_max\": %d,\n", args.block_w_min,
              args.block_w_max);
  std::printf("  \"tc_kernel\": \"%s\",\n", args.tc_kernel.c_str());
  std::printf("  \"block_density\": %.4f,\n", args.block_density);
  std::printf("  \"n_pairs\": %zu, \"n_groups\": %zu,\n",
              found.contributions.size(), fused.fused_blocks.size());
  std::printf("  \"n_tc\": %d, \"n_cuda\": %d,\n", saved_n_tc, saved_n_cuda);
  std::printf("  \"plan_ms\": ");
  print_arr(plan_ms_arr);
  std::printf(",\n  \"tc_ms\": ");
  print_arr(tc_ms_arr);
  std::printf(",\n  \"cuda_ms\": ");
  print_arr(cuda_ms_arr);
  std::printf("\n}\nJSON_END\n");

  return 0;
}
