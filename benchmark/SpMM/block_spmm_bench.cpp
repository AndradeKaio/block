// block_spmm_bench.cpp — driver for block_spmm_loops / block_spmm_blas.
//
// Reads a block-sparse S from a .bsp file, constructs a random dense D (N×N),
// then times C = block_spmm(S, D).
//
// Output: JSON block (same format as prisma_cpu_spmm_bench) so benchmark_spmm.py
// can parse it without any changes to _parse_json_block.
//
// CLI: block_spmm_bench <S.bsp> [--runs R]
//
// Compile (loops variant):
//   g++ -O3 -std=c++20 -fopenmp -march=native -DHAVE_HDF5 \
//       -I<core> -I<hdf5/inc> <core>/*.cpp block_spmm_bench.cpp \
//       ../SpGEMM/CPU/block_spmm_loops.c <hdf5/lib>/libhdf5.so -o block_spmm_loops_bench
//
// Compile (blas variant):
//   same but replace block_spmm_loops.c with block_spmm_blas.c and add -lopenblas

#include "block.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <vector>

extern "C" void block_spmm(
    const double* A_data, const double* B_data,
          double* C_data, int C_size,
    int NC, int NG,
    const int*  M_v,    const int*  N_v,    const int*  K_v,
    const long* A_off,  const int*  A_lda,
    const long* B_off,  const int*  B_ldb,
    const long* C_goff, const int*  C_ldc,
    const int*  G_start);

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::string s_bsp;
    int runs = 5;
};

Args parse_args(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <S.bsp> [--runs N]\n", argv[0]);
        std::exit(1);
    }
    Args a;
    a.s_bsp = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--runs" && i + 1 < argc)
            a.runs = std::atoi(argv[++i]);
    }
    return a;
}

float ms_since(Clock::time_point t0) {
    return float(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
}

void print_arr(const std::vector<double>& v) {
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

    benchmark_core::Matrix<double> S;
    try {
        S = benchmark_core::read_matrix_binsparse<double>(args.s_bsp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "block_spmm_bench: failed to read .bsp: %s\n", e.what());
        return 1;
    }

    const int M = S.M;
    const int N = S.N;
    const int NC = (int)S.blocks.size();

    // Group blocks by blk.r (row), preserving sorted order.
    std::map<int, std::vector<int>> by_row;
    for (int bi = 0; bi < NC; ++bi)
        by_row[S.blocks[bi].r].push_back(bi);
    const int NG = (int)by_row.size();

    // Build flat metadata arrays for block_spmm.
    std::vector<int>  M_v(NC), N_v(NC), K_v(NC);
    std::vector<long> A_off(NC), B_off(NC), C_goff(NC);
    std::vector<int>  A_lda(NC), B_ldb(NC), C_ldc(NC);
    std::vector<int>  G_start(NG + 1);

    {
        int ci = 0, gi = 0;
        for (auto& [row, ids] : by_row) {
            G_start[gi++] = ci;
            for (int bi : ids) {
                const benchmark_core::Block& blk = S.blocks[bi];
                M_v[ci]    = blk.h;
                N_v[ci]    = N;
                K_v[ci]    = blk.w;
                A_off[ci]  = (long)blk.offset;
                A_lda[ci]  = blk.w;
                B_off[ci]  = (long)blk.c * N;
                B_ldb[ci]  = N;
                C_goff[ci] = (long)blk.r * N;
                C_ldc[ci]  = N;
                ++ci;
            }
        }
        G_start[NG] = NC;
    }

    // Sanity check: C_size must fit in int.
    const long long C_total = (long long)M * N;
    if (C_total > (long long)INT_MAX) {
        std::fprintf(stderr, "block_spmm_bench: M*N=%lld overflows int C_size\n", C_total);
        return 1;
    }
    const int C_size = (int)C_total;

    // Dense D: N×N random in [-1,1], fixed seed.
    std::vector<double> D((long long)N * N);
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::generate(D.begin(), D.end(), [&]{ return dist(rng); });
    }

    // Output C: zeroed by block_spmm internally at each call.
    std::vector<double> C(C_total);

    std::vector<double> compute_ms_arr;
    compute_ms_arr.reserve(args.runs + 1);

    for (int r = 0; r <= args.runs; ++r) {
        auto t0 = Clock::now();
        // block_spmm zeros C internally then computes C = S * D.
        block_spmm(
            S.values, D.data(), C.data(), C_size,
            NC, NG,
            M_v.data(), N_v.data(), K_v.data(),
            A_off.data(), A_lda.data(),
            B_off.data(), B_ldb.data(),
            C_goff.data(), C_ldc.data(),
            G_start.data()
        );
        compute_ms_arr.push_back(ms_since(t0));
    }

    std::printf("\nJSON_BEGIN\n{\n");
    std::printf("  \"kernel\": \"block_spmm\",\n");
    std::printf("  \"pipe_total_ms\": 0.0,\n");
    std::printf("  \"S_blocks\": %d,\n", NC);
    std::printf("  \"S_rows\": %d, \"S_cols\": %d,\n", M, N);
    std::printf("  \"compute_ms\": ");
    print_arr(compute_ms_arr);
    std::printf("\n}\nJSON_END\n");
    return 0;
}
