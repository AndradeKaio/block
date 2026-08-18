// GPU/compare_intersection.cu — compare three intersection implementations:
//   old_cpu : find_intersecting_pairs  (interval treap, current baseline)
//   new_cpu : find_intersecting_pairs_sorted  (sort + binary search, OMP)
//   new_gpu : find_intersecting_pairs_gpu     (CUB sort + CUDA kernels)
//
// CLI: compare_intersection <A.bsp> <B.bsp> [--runs N]
//
// Outputs JSON between JSON_BEGIN / JSON_END sentinels, parseable by the
// compare_intersection.py driver script.
//
// Compile (run from SpGEMM/GPU/):
//   nvcc -O3 --expt-relaxed-constexpr -std=c++20 -arch=sm_<X> \
//        -Xcompiler -fopenmp -lgomp \
//        -DHAVE_HDF5 \
//        -I../../core -I. \
//        ../../core/block.cpp ../../core/block_generator.cpp ../../core/interval_tree.cpp \
//        ../../core/matrix.cpp ../../core/matrix_io.cpp \
//        ../../core/pipeline.cpp ../../core/pipeline_sorted.cpp \
//        ../../core/segment_tree.cpp \
//        compare_intersection.cu \
//        -lhdf5 -o compare_intersection

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "block.hpp"
#include "intersection_gpu.cuh"
#include "matrix.hpp"
#include "matrix_io.hpp"
#include "pipeline.hpp"
#include "pipeline_sorted.hpp"

namespace {

using Scalar = float;

struct Args {
    std::string a_bsp;
    std::string b_bsp;
    int runs = 5;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <A.bsp> <B.bsp> [--runs N]\n"
        "  Compares old_cpu / new_cpu / new_gpu intersection implementations.\n"
        "  Pass the same path for both arguments to do A\xc3\x97""A squaring.\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); std::exit(1); }
    Args a;
    a.a_bsp = argv[1];
    a.b_bsp = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--runs") a.runs = std::atoi(next());
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "unknown argument: %s\n", arg.c_str()); print_usage(argv[0]); std::exit(1); }
    }
    return a;
}

// Canonical pair representation for comparison.
struct Pair {
    int ai, bi, k0, k1;
    bool operator<(const Pair& o) const {
        if (ai != o.ai) return ai < o.ai;
        return bi < o.bi;
    }
    bool operator==(const Pair& o) const {
        return ai == o.ai && bi == o.bi && k0 == o.k0 && k1 == o.k1;
    }
};

std::vector<Pair> to_sorted_pairs(const benchmark_core::IntersectionResult& r) {
    std::vector<Pair> ps;
    ps.reserve(r.contributions.size());
    for (const auto& c : r.contributions)
        ps.push_back({c.a_index, c.b_index, c.k.k0, c.k.k1});
    std::sort(ps.begin(), ps.end());
    return ps;
}

bool pairs_equal(const std::vector<Pair>& a, const std::vector<Pair>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!(a[i] == b[i])) return false;
    return true;
}

void print_arr(const std::vector<float>& v) {
    std::printf("[");
    for (int i = 0; i < (int)v.size(); ++i) {
        if (i) std::printf(", ");
        std::printf("%.4f", v[i]);
    }
    std::printf("]");
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    for (const auto& p : {args.a_bsp, args.b_bsp}) {
        FILE* f = std::fopen(p.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "compare_intersection: file not found: %s\n", p.c_str());
            return 1;
        }
        std::fclose(f);
    }

    benchmark_core::Matrix<Scalar> A, B;
    try {
        A = benchmark_core::read_matrix_binsparse<Scalar>(args.a_bsp);
        B = benchmark_core::read_matrix_binsparse<Scalar>(args.b_bsp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "compare_intersection: failed to read .bsp: %s\n", e.what());
        return 1;
    }

    std::printf("A: %dx%d  blocks=%zu\n", A.M, A.N, A.blocks.size());
    std::printf("B: %dx%d  blocks=%zu\n", B.M, B.N, B.blocks.size());

    if (A.blocks.empty() || B.blocks.empty()) {
        std::printf("\nJSON_BEGIN\n{\"n_pairs\":0,\"pairs_match_cpu\":true,\"pairs_match_gpu\":true,"
                    "\"old_cpu_ms\":[],\"new_cpu_ms\":[],\"new_gpu_sort_ms\":[],\"new_gpu_kernel_ms\":[]}\nJSON_END\n");
        return 0;
    }

    const int nA = static_cast<int>(A.blocks.size());
    const int nB = static_cast<int>(B.blocks.size());

    using Clock = std::chrono::steady_clock;
    auto elapsed_ms = [](Clock::time_point t0) {
        return float(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    };

    std::vector<float> old_cpu_ms, new_cpu_ms,
                       new_gpu_sort_ms, new_gpu_kernel_ms, new_gpu_result_ms;
    old_cpu_ms.reserve(args.runs + 1);
    new_cpu_ms.reserve(args.runs + 1);
    new_gpu_sort_ms.reserve(args.runs + 1);
    new_gpu_kernel_ms.reserve(args.runs + 1);
    new_gpu_result_ms.reserve(args.runs + 1);

    bool pairs_match_cpu = true;
    bool pairs_match_gpu = true;
    int n_pairs = 0;

    for (int r = 0; r <= args.runs; ++r) {
        // ── old CPU ──────────────────────────────────────────────────────────
        auto t0 = Clock::now();
        auto old_result = benchmark_core::find_intersecting_pairs(A.blocks, B.blocks);
        old_cpu_ms.push_back(elapsed_ms(t0));

        // ── new CPU ──────────────────────────────────────────────────────────
        t0 = Clock::now();
        auto new_cpu_result = benchmark_core::find_intersecting_pairs_sorted(A.blocks, B.blocks);
        new_cpu_ms.push_back(elapsed_ms(t0));

        // ── new GPU ──────────────────────────────────────────────────────────
        float g_sort = 0.f, g_kernel = 0.f, g_result = 0.f;
        auto gpu_result = benchmark_core::find_intersecting_pairs_gpu(
            A.blocks, B.blocks, &g_sort, &g_kernel, &g_result);
        new_gpu_sort_ms.push_back(g_sort);
        new_gpu_kernel_ms.push_back(g_kernel);
        new_gpu_result_ms.push_back(g_result);

        // ── correctness check (run 0 only) ───────────────────────────────────
        if (r == 0) {
            n_pairs = static_cast<int>(old_result.contributions.size());
            auto ref  = to_sorted_pairs(old_result);
            auto cpup = to_sorted_pairs(new_cpu_result);
            auto gpup = to_sorted_pairs(gpu_result);
            pairs_match_cpu = pairs_equal(ref, cpup);
            pairs_match_gpu = pairs_equal(ref, gpup);

            if (!pairs_match_cpu) {
                std::printf("MISMATCH new_cpu: ref=%zu  got=%zu\n",
                            ref.size(), cpup.size());
                int shown = 0;
                for (size_t i = 0; i < ref.size() && shown < 5; ++i)
                    if (i >= cpup.size() || !(ref[i] == cpup[i])) {
                        std::printf("  ref[%zu]=(%d,%d,%d,%d)\n",
                                    i, ref[i].ai, ref[i].bi, ref[i].k0, ref[i].k1);
                        ++shown;
                    }
            }
            if (!pairs_match_gpu) {
                std::printf("MISMATCH new_gpu: ref=%zu  got=%zu\n",
                            ref.size(), gpup.size());
                int shown = 0;
                for (size_t i = 0; i < ref.size() && shown < 5; ++i)
                    if (i >= gpup.size() || !(ref[i] == gpup[i])) {
                        std::printf("  ref[%zu]=(%d,%d,%d,%d)\n",
                                    i, ref[i].ai, ref[i].bi, ref[i].k0, ref[i].k1);
                        ++shown;
                    }
            }
        }
    }

    // ── Timing summary ────────────────────────────────────────────────────────
    auto avg = [&](const std::vector<float>& v) {
        if (v.size() <= 1) return v.empty() ? 0.f : v[0];
        float s = 0.f;
        for (int i = 1; i < (int)v.size(); ++i) s += v[i];
        return s / float(v.size() - 1);
    };

    std::printf("\n── intersection timing (avg of %d timed runs) ───────────────\n", args.runs);
    std::printf("  old_cpu (treap)   : %8.3f ms\n", avg(old_cpu_ms));
    std::printf("  new_cpu (sorted)  : %8.3f ms\n", avg(new_cpu_ms));
    std::printf("  new_gpu sort+up   : %8.3f ms\n", avg(new_gpu_sort_ms));
    std::printf("  new_gpu kernels   : %8.3f ms\n", avg(new_gpu_kernel_ms));
    std::printf("  new_gpu result    : %8.3f ms  (DtoH + parallel build)\n", avg(new_gpu_result_ms));
    std::printf("  new_gpu total     : %8.3f ms\n", avg(new_gpu_sort_ms) + avg(new_gpu_kernel_ms) + avg(new_gpu_result_ms));
    std::printf("  pairs_match_cpu   : %s\n", pairs_match_cpu ? "true" : "false");
    std::printf("  pairs_match_gpu   : %s\n", pairs_match_gpu ? "true" : "false");
    std::printf("  n_pairs           : %d\n", n_pairs);
    std::printf("  blocks_a=%d  blocks_b=%d\n", nA, nB);
    std::printf("─────────────────────────────────────────────────────────────\n");

    // ── JSON output ───────────────────────────────────────────────────────────
    std::printf("\nJSON_BEGIN\n{\n");
    std::printf("  \"n_pairs\": %d,\n", n_pairs);
    std::printf("  \"blocks_a\": %d,\n", nA);
    std::printf("  \"blocks_b\": %d,\n", nB);
    std::printf("  \"pairs_match_cpu\": %s,\n", pairs_match_cpu ? "true" : "false");
    std::printf("  \"pairs_match_gpu\": %s,\n", pairs_match_gpu ? "true" : "false");
    std::printf("  \"old_cpu_ms\": ");   print_arr(old_cpu_ms);
    std::printf(",\n  \"new_cpu_ms\": "); print_arr(new_cpu_ms);
    std::printf(",\n  \"new_gpu_sort_ms\": ");    print_arr(new_gpu_sort_ms);
    std::printf(",\n  \"new_gpu_kernel_ms\": ");  print_arr(new_gpu_kernel_ms);
    std::printf(",\n  \"new_gpu_result_ms\": ");  print_arr(new_gpu_result_ms);
    std::printf("\n}\nJSON_END\n");

    return 0;
}
