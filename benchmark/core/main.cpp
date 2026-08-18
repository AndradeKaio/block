// Standalone demo for the block-sparse GEMM pipeline: generates two random
// matrices, runs find_intersecting_pairs -> find_overlapping_output_blocks
// -> merge_groups -> block_fusion, then feeds the fused result through the
// GPU-plan phases (gpu_kernel_classify -> gpu_kernel_plan), and reports
// timing/summary stats for both stages.
//
// This exercises the same pipeline stage benchmark.py measures as
// `pipeline_s` (blocks.py -> core/pipeline.*), plus the host-side GPU
// kernel classification/execution-plan construction added in
// core/gpu_pipeline.hpp (phases 3-4; no CUDA involved here).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "block.hpp"
#include "gpu_pipeline.hpp"
#include "matrix.hpp"
#include "pipeline.hpp"

namespace {

using Scalar = double; // matches the Python reference's --dtype f64 default

struct Args {
    int M = 2048, K = 2048, N = 2048;
    int blocks_A = 16, blocks_B = 16;
    int block_h_min = 16, block_h_max = 128;
    int block_w_min = 16, block_w_max = 128;
    std::uint64_t seed = 42;
};

void print_usage() {
    std::printf(
        "Usage: block_pipeline_demo [options]\n"
        "  --M N              rows of A            (default 2048)\n"
        "  --K N              cols of A / rows of B (default 2048)\n"
        "  --N N              cols of B             (default 2048)\n"
        "  --blocks-A N       number of A blocks    (default 16)\n"
        "  --blocks-B N       number of B blocks    (default 16)\n"
        "  --block-h-min N    min block height       (default 16)\n"
        "  --block-h-max N    max block height       (default 128)\n"
        "  --block-w-min N    min block width        (default 16)\n"
        "  --block-w-max N    max block width        (default 128)\n"
        "  --seed N           RNG seed               (default 42)\n");
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--M") {
            args.M = std::atoi(next());
        } else if (arg == "--K") {
            args.K = std::atoi(next());
        } else if (arg == "--N") {
            args.N = std::atoi(next());
        } else if (arg == "--blocks-A") {
            args.blocks_A = std::atoi(next());
        } else if (arg == "--blocks-B") {
            args.blocks_B = std::atoi(next());
        } else if (arg == "--block-h-min") {
            args.block_h_min = std::atoi(next());
        } else if (arg == "--block-h-max") {
            args.block_h_max = std::atoi(next());
        } else if (arg == "--block-w-min") {
            args.block_w_min = std::atoi(next());
        } else if (arg == "--block-w-max") {
            args.block_w_max = std::atoi(next());
        } else if (arg == "--seed") {
            args.seed = static_cast<std::uint64_t>(std::atoll(next()));
        } else if (arg == "--help" || arg == "-h") {
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

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    const auto A = benchmark_core::generate_random_matrix<Scalar>(
        args.M, args.K, args.blocks_A, {args.block_h_min, args.block_h_max},
        {args.block_w_min, args.block_w_max}, args.seed);
    const auto B = benchmark_core::generate_random_matrix<Scalar>(
        args.K, args.N, args.blocks_B, {args.block_h_min, args.block_h_max},
        {args.block_w_min, args.block_w_max}, args.seed + 1);

    std::printf("%dx%d @ %dx%d  blocks=%d/%d  h in [%d,%d]  w in [%d,%d]\n", args.M, args.K,
                args.K, args.N, args.blocks_A, args.blocks_B, args.block_h_min, args.block_h_max,
                args.block_w_min, args.block_w_max);
    std::printf("A  blocks=%zu/%d\n", A.blocks.size(), args.blocks_A);
    std::printf("B  blocks=%zu/%d\n", B.blocks.size(), args.blocks_B);

    if (A.blocks.empty() || B.blocks.empty()) {
        std::printf("No blocks placed.\n");
        return 0;
    }

    const auto t0 = std::chrono::steady_clock::now();

    auto found = benchmark_core::find_intersecting_pairs(A.blocks, B.blocks);
    auto overlap_pairs = benchmark_core::find_overlapping_output_blocks(found.output_blocks);
    auto groups = benchmark_core::merge_groups(found.output_blocks, overlap_pairs);
    auto fused = benchmark_core::block_fusion(found.output_blocks, found.contributions, groups);

    const auto t1 = std::chrono::steady_clock::now();
    const double pipeline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (found.contributions.empty()) {
        std::printf("No intersecting pairs.\n");
        return 0;
    }

    long long fused_cells = 0;
    long long fused_imperfections = 0;
    for (const auto& b : fused.fused_blocks) {
        fused_cells += b.num_cells();
        fused_imperfections += b.imperfections;
    }

    std::printf("pairs=%zu  groups=%zu  pipeline=%.3fms\n", found.contributions.size(),
                fused.fused_blocks.size(), pipeline_ms);
    std::printf("fused output cells=%lld (imperfections=%lld) across %zu block(s)\n", fused_cells,
                fused_imperfections, fused.fused_blocks.size());

    // Phase 3 (wmma/CUDA-core classification) + phase 4 (execution-plan
    // construction) — host-side only, feeds GPU kernel dispatch elsewhere.
    const auto t2 = std::chrono::steady_clock::now();

    auto cls = benchmark_core::gpu_kernel_classify(fused);
    auto [C, plan] = benchmark_core::gpu_kernel_plan<Scalar>(fused, cls, A, B);

    const auto t3 = std::chrono::steady_clock::now();
    const double gpu_plan_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::printf("gpu classify: tc=%d  cuda=%d\n", cls.n_tc, cls.n_cuda);
    std::printf("gpu plan: k_entries=%zu  tc_descs=%zu  cuda_descs=%zu  plan=%.3fms\n",
                plan.k_entries.size(), plan.tc_descs.size(), plan.cuda_descs.size(), gpu_plan_ms);
    std::printf("C  %dx%d  values=%zu\n", C.M, C.N, C.n_values);

    return 0;
}
