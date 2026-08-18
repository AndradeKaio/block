#pragma once

#include <vector>
#include "block.hpp"

namespace benchmark_core {

struct MineParams {
    float Twf    = 0.5f; // min fraction of a new row/col that must be alive to expand
    float To     = 0.3f; // max allowed imperfection ratio (zeros / area)
    int   Thslim = 50;   // max aspect ratio (longer_dim / shorter_dim)

    // When true (default): both directions are re-evaluated after every growth
    // step, so a direction that failed earlier is retried if the block grew in
    // the other dimension.  Produces fewer, larger blocks with more padding.
    // When false: matches Python mine_matrices.py — a direction is permanently
    // disabled the first time it fails, producing more, smaller blocks.
    bool retry_expand = true;
};

// Mine block structure from a CSR pattern matrix (values are ignored).
// row_ptr: length M+1, col_idx: length nnz (0-based, sorted per row).
// Returns blocks with r/c/h/w/imperfections filled; Block::offset is left 0.
// Caller must call assign_offsets(blocks) before constructing a Matrix<T>.
std::vector<Block> mine_blocks(int M, int N,
                                const int* row_ptr,
                                const int* col_idx,
                                MineParams params = {});

} // namespace benchmark_core
