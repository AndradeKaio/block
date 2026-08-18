#include "block.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"
#include "pipeline.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Scalar = float;

void print_usage(const char *prog) {
    std::fprintf(stderr, "Usage: %s <A.bsp> [B.bsp]\n", prog);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        print_usage(argv[0]);
        return 1;
    }
    const std::string a_bsp = argv[1];
    const std::string b_bsp = argc == 3 ? argv[2] : argv[1];

    benchmark_core::Matrix<Scalar> A, B;
    try {
        A = benchmark_core::read_matrix_binsparse<Scalar>(a_bsp);
        B = benchmark_core::read_matrix_binsparse<Scalar>(b_bsp);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "histogram_bin: failed to read .bsp: %s\n", e.what());
        return 1;
    }

    if (A.blocks.empty() || B.blocks.empty()) {
        std::printf("\nJSON_BEGIN\n{\n");
        std::printf("  \"total_calls\": 0,\n");
        std::printf("  \"total_flops\": 0,\n");
        std::printf("  \"shapes\": []\n");
        std::printf("}\nJSON_END\n");
        return 0;
    }

    auto found  = benchmark_core::find_intersecting_pairs(A.blocks, B.blocks);
    if (found.contributions.empty()) {
        std::printf("\nJSON_BEGIN\n{\n");
        std::printf("  \"total_calls\": 0,\n");
        std::printf("  \"total_flops\": 0,\n");
        std::printf("  \"shapes\": []\n");
        std::printf("}\nJSON_END\n");
        return 0;
    }

    auto groups = benchmark_core::merge_overlapping_output_blocks(found.output_blocks);
    auto fused  = benchmark_core::block_fusion(found.output_blocks, found.contributions, groups);

    std::map<std::tuple<int,int,int>, long long> freq;
    long long total_calls = 0, total_flops = 0;

    for (int fi = 0; fi < (int)fused.fused_blocks.size(); ++fi) {
        for (const auto &c : fused.fused_contributions[fi]) {
            const benchmark_core::Block &ab = A.blocks[c.a_index];
            const benchmark_core::Block &bb = B.blocks[c.b_index];
            const int K = c.k.k1 - c.k.k0 + 1;
            freq[{ab.h, K, bb.w}]++;
            total_calls++;
            total_flops += (long long)ab.h * K * bb.w * 2;
        }
    }

    std::vector<std::pair<long long, std::tuple<int,int,int>>> sorted;
    sorted.reserve(freq.size());
    for (const auto &[key, cnt] : freq)
        sorted.push_back({cnt, key});
    std::sort(sorted.rbegin(), sorted.rend());

    std::printf("\nJSON_BEGIN\n{\n");
    std::printf("  \"total_calls\": %lld,\n", total_calls);
    std::printf("  \"total_flops\": %lld,\n", total_flops);
    std::printf("  \"shapes\": [\n");
    for (int i = 0; i < (int)sorted.size(); ++i) {
        const auto [cnt, key] = sorted[i];
        const auto [M, K, N] = key;
        const long long flops = (long long)M * K * N * 2;
        const double pct = 100.0 * cnt / total_calls;
        const char *comma = (i + 1 < (int)sorted.size()) ? "," : "";
        std::printf("    {\"M\": %d, \"K\": %d, \"N\": %d, \"count\": %lld,"
                    " \"flops_per_call\": %lld, \"pct\": %.4f}%s\n",
                    M, K, N, cnt, flops, pct, comma);
    }
    std::printf("  ]\n}\nJSON_END\n");
    return 0;
}
