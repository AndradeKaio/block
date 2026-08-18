// prisma_gpu_spmm_bench.cu — Prisma GPU SpMM benchmark (C = S * D).
//
// Compile line (nvcc, from benchmark_spmm_gpu.py):
//   nvcc -O3 --expt-relaxed-constexpr -std=c++20 -arch=sm_80
//        -DHAVE_HDF5 -I<core> -I<SpGEMM/GPU> -I<SpMM/GPU>
//        block.cpp block_generator.cpp matrix.cpp matrix_io.cpp
//        prisma_gpu_spmm_bench.cu libhdf5.so -o prisma_gpu_spmm_bench_<name>
//
// D generation is CORRECTNESS-CRITICAL: drawn in double via mt19937_64,
// narrowed to T — identical contract to cusparse_spmm_bench.cu.
//
// JSON output fields (matching _CSV_FIELDS in benchmark_spmm_gpu.py):
//   symbolic_ms[], tc_ms[], cuda_ms[], compute_ms[], total_ms[]
//   n_tc_tiles, n_cuda_tiles, n_specialized_shapes (always 0: no codegen)

#include "block.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"
#include "spmm_gpu_kernels.cuh"   // includes spmm_gpu_plan.hpp

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static float ms_between(Clock::time_point t0, Clock::time_point t1) {
    return float(std::chrono::duration<double, std::milli>(t1 - t0).count());
}

static void print_arr(const std::vector<float>& v) {
    std::printf("[");
    for (int i = 0; i < (int)v.size(); ++i) {
        if (i) std::printf(", ");
        std::printf("%.4f", v[i]);
    }
    std::printf("]");
}

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Args {
    std::string bsp;
    int         runs          = 5;
    int         seed          = 42;
    std::string precision     = "fp64";
    bool        force_cuda    = false;
    bool        row_group     = false;
    bool        specialized   = false;
    int         tc_min_h      = 16;   // --tc-classify HxW: minimum h to use TC
    int         tc_min_w      = 16;   // minimum w to use TC (padded to 16×8)
    std::string dump_d;
    std::string dump_c;
};

static void print_usage(const char* p) {
    std::fprintf(stderr,
        "Usage: %s <S.bsp> [--runs N] [--seed S] [--precision fp32|fp64]\n"
        "       [--force-cuda-fallback] [--row-group] [--specialized-kernels]\n"
        "       [--tc-classify HxW]  (e.g. 4x4; default 16x16)\n"
        "       [--dump-d path] [--dump-c path]\n", p);
}

static Args parse_args(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); std::exit(1); }
    Args a;
    a.bsp = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if      (arg == "--runs")               a.runs       = std::atoi(next());
        else if (arg == "--seed")               a.seed       = std::atoi(next());
        else if (arg == "--precision")          a.precision  = next();
        else if (arg == "--force-cuda-fallback") a.force_cuda = true;
        else if (arg == "--row-group")          a.row_group  = true;
        else if (arg == "--specialized-kernels") a.specialized = true;
        else if (arg == "--dump-d")             a.dump_d     = next();
        else if (arg == "--dump-c")             a.dump_c     = next();
        else if (arg == "--tc-classify") {
            const char* val = next();
            if (std::sscanf(val, "%dx%d", &a.tc_min_h, &a.tc_min_w) != 2) {
                std::fprintf(stderr, "--tc-classify expects HxW format, e.g. 4x4\n");
                std::exit(1);
            }
        }
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]); std::exit(1);
        }
    }
    if (a.precision != "fp32" && a.precision != "fp64") {
        std::fprintf(stderr, "--precision must be fp32 or fp64\n");
        std::exit(1);
    }
    return a;
}

// ── Main per-precision run ────────────────────────────────────────────────────

template <typename T>
int run(const Args& args) {
    using namespace spmm_gpu;
    const std::string kernel_name =
        std::string("prisma_gpu_cuda_") + args.precision +
        (args.row_group ? "_row_group" : "");

    // 1. Read S.
    benchmark_core::Matrix<T> S;
    try {
        S = benchmark_core::read_matrix_binsparse<T>(args.bsp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prisma_gpu_spmm_bench: read error: %s\n", e.what());
        return 1;
    }
    const int M = S.M;
    const int N = S.N;  // dense RHS is N×N (square), matching cusparse_spmm_bench

    std::fprintf(stderr, "S: %d×%d  blocks=%zu  precision=%s  tc_classify=%dx%d\n",
                 M, N, S.blocks.size(), args.precision.c_str(),
                 args.tc_min_h, args.tc_min_w);

    // 2. Generate D (drawn in double, narrowed to T — same as cusparse bench).
    std::vector<double> D_f64((std::size_t)N * N);
    {
        std::mt19937_64 rng((uint64_t)args.seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::generate(D_f64.begin(), D_f64.end(), [&]{ return dist(rng); });
    }
    std::vector<T> D(D_f64.size());
    for (std::size_t i = 0; i < D_f64.size(); ++i)
        D[i] = static_cast<T>(D_f64[i]);

    if (!args.dump_d.empty()) {
        FILE* f = std::fopen(args.dump_d.c_str(), "wb");
        if (f) {
            std::fwrite(D_f64.data(), sizeof(double), D_f64.size(), f);
            std::fclose(f);
        }
    }

    // 3. Symbolic phase (timed; re-reported every run to represent one-shot cost).
    // TC path: WMMA tf32 requires T=float; disabled for double or --force-cuda-fallback.
    const bool use_tc = !args.force_cuda && std::is_same_v<T, float>;
    auto t_sym0 = Clock::now();
    auto plan = build_spmm_plan<T>(S, N, use_tc, args.tc_min_h, args.tc_min_w);
    const float symbolic_ms_val = ms_between(t_sym0, Clock::now());

    std::fprintf(stderr,
        "plan: row_tiles=%d  col_tiles=%d  tc_tasks=%d  cuda_tasks=%d  symbolic=%.3fms\n",
        plan.n_row_tiles, plan.n_col_tiles,
        plan.n_tc_tasks, plan.n_cuda_tasks, symbolic_ms_val);

    // 4. Upload plan to device (done ONCE, outside timed loop).
    DevSpmmPlan<T> dev = upload_spmm_plan<T>(plan, S, D.data());

    std::vector<T> C_host((std::size_t)M * N, T(0));

    // 5. Run loop (run 0 = warmup, not recorded).
    std::vector<float> sym_arr, tc_arr, cuda_arr, compute_arr, total_arr;
    sym_arr.reserve(args.runs + 1);
    tc_arr.reserve(args.runs + 1);
    cuda_arr.reserve(args.runs + 1);
    compute_arr.reserve(args.runs + 1);
    total_arr.reserve(args.runs + 1);

    for (int r = 0; r <= args.runs; ++r) {
        sym_arr.push_back(symbolic_ms_val);

        auto t0 = Clock::now();
        SpmmTimes kt = dispatch_spmm(dev);
        auto t1 = Clock::now();

        tc_arr.push_back(kt.tc_ms);
        cuda_arr.push_back(kt.cuda_ms);
        const float compute_ms = ms_between(t0, t1);
        compute_arr.push_back(compute_ms);
        total_arr.push_back(symbolic_ms_val + compute_ms);
    }

    // 6. Copy C back and optionally dump.
    sync_spmm_c(dev, C_host.data());

    if (!args.dump_c.empty()) {
        std::vector<double> C_f64(C_host.size());
        for (std::size_t i = 0; i < C_host.size(); ++i)
            C_f64[i] = static_cast<double>(C_host[i]);
        FILE* f = std::fopen(args.dump_c.c_str(), "wb");
        if (f) {
            std::fwrite(C_f64.data(), sizeof(double), C_f64.size(), f);
            std::fclose(f);
        }
    }

    free_dev_spmm_plan(dev);

    std::fprintf(stderr, "runs=%d  last: tc=%.3fms  cuda=%.3fms  compute=%.3fms\n",
                 args.runs, tc_arr.back(), cuda_arr.back(), compute_arr.back());

    // 7. JSON output.
    std::printf("\nJSON_BEGIN\n{\n");
    std::printf("  \"kernel\": \"%s\",\n", kernel_name.c_str());
    std::printf("  \"S_rows\": %d, \"S_cols\": %d,\n", M, N);
    std::printf("  \"S_blocks\": %zu,\n", S.blocks.size());
    std::printf("  \"n_tc_tiles\": %d,\n",   plan.n_tc_tasks);
    std::printf("  \"n_cuda_tiles\": %d,\n", plan.n_cuda_tasks);
    std::printf("  \"n_specialized_shapes\": 0,\n");
    std::printf("  \"symbolic_ms\": ");   print_arr(sym_arr);
    std::printf(",\n  \"tc_ms\": ");      print_arr(tc_arr);
    std::printf(",\n  \"cuda_ms\": ");    print_arr(cuda_arr);
    std::printf(",\n  \"compute_ms\": "); print_arr(compute_arr);
    std::printf(",\n  \"total_ms\": ");   print_arr(total_arr);
    std::printf("\n}\nJSON_END\n");

    return 0;
}

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);
    return args.precision == "fp32" ? run<float>(args) : run<double>(args);
}
