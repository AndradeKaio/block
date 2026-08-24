#pragma once

#include <utility>
#include <vector>

#include "block.hpp"

namespace benchmark_core {

// One A-block/B-block pair whose k-ranges overlap, contributing a partial
// product into `output_blocks[i]` (same index as this contribution).
struct Contribution {
    int a_index = 0;
    int b_index = 0;
    KRange k;
};

struct IntersectionResult {
    std::vector<Contribution> contributions;
    std::vector<Block> output_blocks; // parallel to `contributions`
};

struct IntersectionTimings {
    float tree_build_ms = 0.f;
    float query_ms      = 0.f;
};

// For SpGEMM C = A @ B: find every (A_block, B_block) pair whose shared
// k-dimension ranges overlap, via an interval tree over B's row ranges.
// If `timings` is non-null it is filled with the tree-build and query sub-times.
IntersectionResult find_intersecting_pairs(const std::vector<Block>& a_blocks,
                                            const std::vector<Block>& b_blocks,
                                            IntersectionTimings* timings = nullptr);

// A connected component of mutually (transitively) overlapping output
// blocks. `id` is the union-find representative; `members` lists the
// original `output_blocks` indices belonging to the group, in the order
// they were first attached.
struct Group {
    int id = 0;
    std::vector<int> members;
};

// Sweep-line over `output_blocks` combined with inline union-find: finds
// every spatially overlapping pair and immediately merges the two blocks
// into the same connected component. Returns one Group per component.
// An output block's row range is always exactly its contributing A-block's
// row range (see intersect() in block.cpp), so two output blocks with
// disjoint row ranges can never end up in the same component regardless of
// column overlap -- this partitions output_blocks into row-disjoint panels
// first (same sweep as SpMM's row_groups), then runs one independent sweep +
// union-find per panel in parallel via OpenMP.
std::vector<Group> merge_overlapping_output_blocks(
    const std::vector<Block>& output_blocks);

// Kept for API compatibility; prefer merge_overlapping_output_blocks.
std::vector<std::pair<int, int>> find_overlapping_output_blocks(
    const std::vector<Block>& output_blocks);
std::vector<Group> merge_groups(const std::vector<Block>& output_blocks,
                                 const std::vector<std::pair<int, int>>& overlap_pairs);

// Minimum bounding rectangle of a set of blocks, with `imperfections` set to
// the number of cells in that rectangle not covered by any input block
// (computed via Klee's algorithm / coordinate-compressed area sweep).
Block mbr(const std::vector<Block>& blocks);

struct FusionResult {
    // fused_contributions[i] holds every contribution that lands in
    // fused_blocks[i].
    std::vector<std::vector<Contribution>> fused_contributions;
    std::vector<Block> fused_blocks;
};

// Fuse overlapping output blocks per group: the MBR of each group's output
// blocks becomes one fused output block, and every contribution landing in
// the group is accumulated together under it.
FusionResult block_fusion(const std::vector<Block>& output_blocks,
                           const std::vector<Contribution>& contributions,
                           const std::vector<Group>& groups);

} // namespace benchmark_core
