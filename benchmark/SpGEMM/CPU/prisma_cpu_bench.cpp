#include "block.hpp"
#include "cpu_dispatch.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"
#include "pipeline.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace {

// double, matching TACO's own bench_taco.c (Bv/Cv/A_t->vals are all
// `double`) -- was float; switched so the two are comparable at a tight
// tolerance instead of only "within float32 precision". See
// gen_spgemm_kernels.py's docstring for the matching codegen change.
using Scalar = double;
using Clock = std::chrono::steady_clock;

struct Args {
  std::string a_bsp;
  std::string b_bsp;
  std::string validate;
  int runs = 5;
  bool specialized = false;
  int print_shapes = 0;
};

void print_usage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s <A.bsp> <B.bsp> [--runs N] [--validate FILE] "
               "[--specialized-kernels] [--print-shapes N]\n",
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
    else if (arg == "--validate")
      args.validate = next();
    else if (arg == "--specialized-kernels")
      args.specialized = true;
    else if (arg == "--print-shapes")
      args.print_shapes = std::atoi(next());
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

  for (const auto &path : {args.a_bsp, args.b_bsp}) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
      std::fprintf(stderr, "prisma_cpu_bench: not found: %s\n", path.c_str());
      return 1;
    }
    std::fclose(f);
  }

  benchmark_core::Matrix<Scalar> A, B;
  try {
    A = benchmark_core::read_matrix_binsparse<Scalar>(args.a_bsp);
    B = benchmark_core::read_matrix_binsparse<Scalar>(args.b_bsp);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "prisma_cpu_bench: failed to read .bsp: %s\n",
                 e.what());
    return 1;
  }

  std::printf("A: %dx%d  blocks=%zu\n", A.M, A.N, A.blocks.size());
  std::printf("B: %dx%d  blocks=%zu\n", B.M, B.N, B.blocks.size());

  if (A.blocks.empty() || B.blocks.empty()) {
    std::printf("\nJSON_BEGIN\n{\n");
    std::printf("  \"kernel\": \"prisma_cpu\",\n");
    std::printf("  \"n_pairs\": 0, \"n_groups\": 0,\n");
    std::printf("  \"pipe_total_ms\": 0.0,\n");
    std::printf("  \"compute_ms\": []\n");
    std::printf("}\nJSON_END\n");
    return 0;
  }

  // Symbolic pipeline — run once
  auto t0 = Clock::now();
  auto found = benchmark_core::find_intersecting_pairs(A.blocks, B.blocks);
  float pipe_intersect_ms = ms_since(t0);

  if (found.contributions.empty()) {
    std::printf("No intersecting pairs.\n");
    std::printf("\nJSON_BEGIN\n{\n");
    std::printf("  \"kernel\": \"prisma_cpu\",\n");
    std::printf("  \"n_pairs\": 0, \"n_groups\": 0,\n");
    std::printf("  \"pipe_total_ms\": 0.0,\n");
    std::printf("  \"compute_ms\": []\n");
    std::printf("}\nJSON_END\n");
    return 0;
  }

  t0 = Clock::now();
  auto groups =
      benchmark_core::merge_overlapping_output_blocks(found.output_blocks);
  float pipe_merge_ms = ms_since(t0);

  t0 = Clock::now();
  auto fused = benchmark_core::block_fusion(found.output_blocks,
                                            found.contributions, groups);
  float pipe_fuse_ms = ms_since(t0);

  float pipe_total_ms = pipe_intersect_ms + pipe_merge_ms + pipe_fuse_ms;

  std::printf("pairs=%zu  groups=%zu\n", found.contributions.size(),
              fused.fused_blocks.size());

  benchmark_core::cpu_gemm_histogram(fused, A, B);

  if (args.print_shapes > 0) {
    t0 = Clock::now();
    auto shapes = benchmark_core::cpu_gemm_top_shapes(fused, A, B, args.print_shapes);
    float pipe_topshapes_ms = ms_since(t0);
    float pipe_shapes_total_ms = pipe_total_ms + pipe_topshapes_ms;
    std::printf("\nJSON_BEGIN\n{\n");
    std::printf("  \"top_shapes\": [");
    for (int i = 0; i < (int)shapes.size(); ++i) {
      auto [m, k, n] = shapes[i];
      if (i) std::printf(", ");
      std::printf("[%d, %d, %d]", m, k, n);
    }
    std::printf("],\n");
    std::printf("  \"pipe_total_ms\": %.4f,\n", pipe_shapes_total_ms);
    std::printf("  \"n_pairs\": %zu, \"n_groups\": %zu\n",
                found.contributions.size(), fused.fused_blocks.size());
    std::printf("}\nJSON_END\n");
    return 0;
  }

  std::printf("kernel: %s\n", args.specialized ? "specialized" : "generic");

  // Every timed run redoes the full symbolic pipeline (intersect -> merge ->
  // fuse -> plan_build) from scratch instead of reusing the `fused`/`plan`
  // computed above -- that first pass exists only for the early-exit check
  // and the histogram print. A real caller who invokes SpGEMM once pays the
  // full, un-amortised symbolic cost; reporting the mean of N independently
  // measured symbolic passes (instead of one measurement divided by N)
  // reflects that honestly.
  std::vector<double> symbolic_ms_arr, compute_ms_arr;
  symbolic_ms_arr.reserve(args.runs + 1);
  compute_ms_arr.reserve(args.runs + 1);

  for (int r = 0; r <= args.runs; ++r) {
    auto ts0 = Clock::now();
    auto found_r  = benchmark_core::find_intersecting_pairs(A.blocks, B.blocks);
    auto groups_r = benchmark_core::merge_overlapping_output_blocks(found_r.output_blocks);
    auto fused_r  = benchmark_core::block_fusion(found_r.output_blocks,
                                                  found_r.contributions, groups_r);
    auto plan     = benchmark_core::cpu_plan_build(fused_r, A, B);
    symbolic_ms_arr.push_back(double(ms_since(ts0)));

    double ms = args.specialized
                    ? benchmark_core::cpu_compute<Scalar, true>(plan)
                    : benchmark_core::cpu_compute<Scalar, false>(plan);
    compute_ms_arr.push_back(ms);

    // Optional validation dump — from the last run's output.
    if (r == args.runs && !args.validate.empty()) {
      std::vector<std::tuple<int, int, Scalar>> entries;
      for (const auto &blk : plan.C.blocks) {
        for (int ri = 0; ri < blk.h; ++ri) {
          for (int ci = 0; ci < blk.w; ++ci) {
            Scalar v = plan.C.values[blk.offset + (long long)ri * blk.w + ci];
            if (v != Scalar(0))
              entries.emplace_back(blk.r + ri, blk.c + ci, v);
          }
        }
      }
      std::sort(entries.begin(), entries.end());
      FILE *vf = std::fopen(args.validate.c_str(), "w");
      if (!vf) {
        std::fprintf(stderr, "prisma_cpu_bench: cannot open validate file: %s\n",
                     args.validate.c_str());
      } else {
        for (const auto &[rr, cc, v] : entries)
          std::fprintf(vf, "%d %d %.8e\n", rr, cc, v);
        std::fclose(vf);
        std::printf("validate: wrote %zu non-zeros to %s\n", entries.size(),
                    args.validate.c_str());
      }
    }
  }

  std::printf("runs=%d  last: symbolic=%.3fms compute=%.3fms\n", args.runs,
              symbolic_ms_arr.back(), compute_ms_arr.back());

  // Timing breakdown — mean of the N real timed-run measurements.
  auto avg = [](const std::vector<double> &v) {
    if (v.size() <= 1)
      return v.empty() ? 0.0 : v[0];
    double s = 0.0;
    for (int i = 1; i < (int)v.size(); ++i)
      s += v[i];
    return s / double(v.size() - 1);
  };
  double avg_sym  = avg(symbolic_ms_arr);
  double avg_comp = avg(compute_ms_arr);

  std::printf("\n── symbolic breakdown ───────────────────────────────────\n");
  std::printf("  symbolic per-run (avg of %d timed runs, redone cold every run)\n",
              args.runs);
  std::printf("    symbolic total   : %8.3f ms\n", avg_sym);
  std::printf("  compute per-run (avg of %d timed runs)\n", args.runs);
  std::printf("    compute total    : %8.3f ms\n", avg_comp);
  std::printf("────────────────────────────────────────────────────────\n");

  std::printf("\nJSON_BEGIN\n{\n");
  std::printf("  \"kernel\": \"prisma_cpu\",\n");
  std::printf("  \"n_pairs\": %zu, \"n_groups\": %zu,\n",
              found.contributions.size(), fused.fused_blocks.size());
  std::printf("  \"symbolic_ms\": ");
  print_arr(symbolic_ms_arr);
  std::printf(",\n");
  std::printf("  \"compute_ms\": ");
  print_arr(compute_ms_arr);
  std::printf("\n}\nJSON_END\n");

  return 0;
}
