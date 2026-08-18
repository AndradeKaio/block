#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "block.hpp"

namespace benchmark_core {

// Randomly places up to `n_blocks` non-overlapping rectangular blocks inside
// a `rows` x `cols` grid, with height/width drawn uniformly from the given
// inclusive ranges. Mirrors the block-placement loop in shared.py's
// `generate_matrices` (geometry only — no cell values).
//
// Placement is greedy with rejection sampling, capped at `n_blocks * 50`
// attempts, so fewer than `n_blocks` blocks may be returned if the grid is
// too crowded to place them all.
// When snap_to_tc is true, generated dimensions >= 16 are floored to the
// nearest multiple of 16 so they satisfy the TC-kernel requirement exactly.
// Dimensions < 16 are left as-is in both modes (they route to the CUDA kernel).
std::vector<Block> generate_random_blocks(int rows, int cols, int n_blocks,
                                           std::pair<int, int> h_range,
                                           std::pair<int, int> w_range,
                                           std::uint64_t seed,
                                           bool snap_to_tc = false);

} // namespace benchmark_core
