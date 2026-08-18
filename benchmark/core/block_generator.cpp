#include "block_generator.hpp"

#include <algorithm>
#include <random>

namespace benchmark_core {

std::vector<Block> generate_random_blocks(int rows, int cols, int n_blocks,
                                           std::pair<int, int> h_range,
                                           std::pair<int, int> w_range,
                                           std::uint64_t seed,
                                           bool snap_to_tc) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> h_dist(h_range.first, h_range.second);
    std::uniform_int_distribution<int> w_dist(w_range.first, w_range.second);

    struct Placed {
        int r0, r1, c0, c1;
    };
    std::vector<Placed> placed;
    std::vector<Block> blocks;
    placed.reserve(n_blocks);
    blocks.reserve(n_blocks);

    auto snap16 = [](int x) -> int {
        return x >= 16 ? (x / 16) * 16 : x;
    };

    const int max_attempts = n_blocks * 50;
    for (int attempt = 0;
         attempt < max_attempts && static_cast<int>(blocks.size()) < n_blocks; ++attempt) {
        int h = std::min(h_dist(rng), rows);
        int w = std::min(w_dist(rng), cols);
        if (snap_to_tc) { h = snap16(h); w = snap16(w); }
        if (h == 0 || w == 0) continue;
        int r0 = std::uniform_int_distribution<int>(0, rows - h)(rng);
        int c0 = std::uniform_int_distribution<int>(0, cols - w)(rng);
        // Snap positions too: fused MBR height = max(r0+h) - min(r0); if r0
        // values are arbitrary, the MBR height can be a non-multiple of 16
        // even when every individual block height is.  Flooring r0/c0 is safe
        // because snap16 already floored h/w, so r0_snapped + h <= rows.
        if (snap_to_tc) { r0 = (r0 / 16) * 16; c0 = (c0 / 16) * 16; }
        const int r1 = r0 + h;
        const int c1 = c0 + w;

        const bool overlaps = std::any_of(placed.begin(), placed.end(), [&](const Placed& p) {
            return r0 < p.r1 && r1 > p.r0 && c0 < p.c1 && c1 > p.c0;
        });
        if (overlaps) {
            continue;
        }

        placed.push_back({r0, r1, c0, c1});
        Block b;
        b.r = r0;
        b.c = c0;
        b.h = h;
        b.w = w;
        blocks.push_back(b);
    }
    return blocks;
}

} // namespace benchmark_core
