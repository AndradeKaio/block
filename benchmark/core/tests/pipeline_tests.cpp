// Small assertion-based test suite for the block pipeline (no external test
// framework, to keep the `core` project dependency-free). Every expected
// value below was hand-derived from the block geometry in the test itself.

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "block.hpp"
#include "pipeline.hpp"

using benchmark_core::Block;
using benchmark_core::Contribution;
using benchmark_core::Group;
using benchmark_core::intersect;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond, __FILE__,    \
                          __LINE__);                                                \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

namespace {

Block make_block(int r, int c, int h, int w) {
    Block b;
    b.r = r;
    b.c = c;
    b.h = h;
    b.w = w;
    return b;
}

void test_intersect_overlap() {
    const Block a = make_block(0, 0, 2, 3); // rows[0,1] cols[0,2]
    const Block b = make_block(2, 5, 3, 4); // rows[2,4] cols[5,8]

    auto result = intersect(a, b);
    CHECK(result.has_value());
    CHECK(result->k.k0 == 2);
    CHECK(result->k.k1 == 2);
    CHECK(result->output.r == 0);
    CHECK(result->output.h == 2);
    CHECK(result->output.c == 5);
    CHECK(result->output.w == 4);
}

void test_intersect_no_overlap() {
    const Block a = make_block(0, 0, 2, 3); // cols[0,2]
    const Block b = make_block(5, 5, 3, 4); // rows[5,7]

    CHECK(!intersect(a, b).has_value());
}

void test_find_intersecting_pairs() {
    const std::vector<Block> a_blocks = {make_block(0, 0, 2, 3)};
    const std::vector<Block> b_blocks = {make_block(2, 5, 3, 4)};

    auto result = benchmark_core::find_intersecting_pairs(a_blocks, b_blocks);
    CHECK(result.contributions.size() == 1);
    CHECK(result.contributions[0].a_index == 0);
    CHECK(result.contributions[0].b_index == 0);
    CHECK(result.contributions[0].k.k0 == 2);
    CHECK(result.contributions[0].k.k1 == 2);

    CHECK(result.output_blocks.size() == 1);
    const Block& out = result.output_blocks[0];
    CHECK(out.r == 0 && out.h == 2 && out.c == 5 && out.w == 4);
}

void test_find_overlapping_output_blocks() {
    const std::vector<Block> blocks = {
        make_block(0, 0, 4, 4),   // rows[0,3] cols[0,3]
        make_block(2, 2, 4, 4),   // rows[2,5] cols[2,5] -- overlaps block 0
        make_block(10, 10, 2, 2), // isolated
    };

    auto pairs = benchmark_core::find_overlapping_output_blocks(blocks);
    CHECK(pairs.size() == 1);
    CHECK(pairs[0].first == 0);
    CHECK(pairs[0].second == 1);
}

void test_merge_groups() {
    const std::vector<Block> blocks(3, make_block(0, 0, 1, 1));
    const std::vector<std::pair<int, int>> pairs = {{0, 1}};

    auto groups = benchmark_core::merge_groups(blocks, pairs);
    CHECK(groups.size() == 2);

    // group 0 contains the merged {0,1}, group 1 contains the singleton {2}.
    CHECK(groups[0].members.size() == 2);
    CHECK(groups[0].members[0] == 0);
    CHECK(groups[0].members[1] == 1);
    CHECK(groups[1].members.size() == 1);
    CHECK(groups[1].members[0] == 2);
}

void test_mbr_exact_union() {
    // Two blocks that tile a rectangle exactly, with no gaps.
    const std::vector<Block> blocks = {
        make_block(0, 0, 2, 2), // rows[0,1] cols[0,1]
        make_block(0, 2, 2, 2), // rows[0,1] cols[2,3]
    };

    Block result = benchmark_core::mbr(blocks);
    CHECK(result.r == 0 && result.c == 0 && result.h == 2 && result.w == 4);
    CHECK(result.imperfections == 0);
    CHECK(result.num_cells() == 8);
    CHECK(result.num_nonzeros() == 8);
}

void test_mbr_with_gap() {
    // Two diagonally-placed blocks: bounding rect has a real gap.
    const std::vector<Block> blocks = {
        make_block(0, 0, 2, 2), // rows[0,1] cols[0,1]
        make_block(2, 2, 2, 2), // rows[2,3] cols[2,3]
    };

    Block result = benchmark_core::mbr(blocks);
    CHECK(result.r == 0 && result.c == 0 && result.h == 4 && result.w == 4);
    CHECK(result.num_cells() == 16);
    CHECK(result.imperfections == 8); // 16 - (4 + 4) covered cells
    CHECK(result.num_nonzeros() == 8);
}

void test_block_fusion_end_to_end() {
    // One A block overlapping two B blocks whose output tiles overlap in
    // output space, and must be fused into a single block.
    const std::vector<Block> a_blocks = {make_block(0, 0, 2, 2)}; // rows[0,1] cols[0,1]
    const std::vector<Block> b_blocks = {
        make_block(0, 0, 2, 3), // rows[0,1] cols[0,2]
        make_block(0, 1, 2, 3), // rows[0,1] cols[1,3]
    };

    auto found = benchmark_core::find_intersecting_pairs(a_blocks, b_blocks);
    CHECK(found.contributions.size() == 2);
    CHECK(found.output_blocks.size() == 2);

    auto pairs = benchmark_core::find_overlapping_output_blocks(found.output_blocks);
    CHECK(pairs.size() == 1);

    auto groups = benchmark_core::merge_groups(found.output_blocks, pairs);
    CHECK(groups.size() == 1);
    CHECK(groups[0].members.size() == 2);

    auto fused = benchmark_core::block_fusion(found.output_blocks, found.contributions, groups);
    CHECK(fused.fused_blocks.size() == 1);
    CHECK(fused.fused_contributions.size() == 1);
    CHECK(fused.fused_contributions[0].size() == 2);

    const Block& out = fused.fused_blocks[0];
    CHECK(out.r == 0 && out.c == 0 && out.h == 2 && out.w == 4);
    CHECK(out.imperfections == 0); // the two output tiles exactly tile the MBR
}

// ── Parallel-correctness tests ────────────────────────────────────────────────
// These tests use the same functions as the sequential tests above. When run
// under OMP_NUM_THREADS=1 they exercise the single-threaded path; under
// OMP_NUM_THREADS>1 they exercise the parallel path. The expected values are
// hand-derived from the block geometry and must match in both cases.

void test_find_intersecting_pairs_parallel() {
    // 3 A-blocks and 2 B-blocks; all A columns = [0,1], B rows overlap them.
    // A0: rows[0,1]  cols[0,1]
    // A1: rows[5,6]  cols[0,1]
    // A2: rows[10,11] cols[0,1]
    // B0: rows[0,1]  cols[10,11]  → k-range [0,1] with every A block
    // B1: rows[1,2]  cols[20,21]  → k-range [1,1] with every A block
    const std::vector<Block> a_blocks = {
        make_block(0,  0, 2, 2),
        make_block(5,  0, 2, 2),
        make_block(10, 0, 2, 2),
    };
    const std::vector<Block> b_blocks = {
        make_block(0, 10, 2, 2),
        make_block(1, 20, 2, 2),
    };

    auto result = benchmark_core::find_intersecting_pairs(a_blocks, b_blocks);

    // 3 A × 2 B = 6 contributions (all pairs intersect).
    CHECK(result.contributions.size() == 6);
    CHECK(result.output_blocks.size() == 6);

    // Sort for deterministic comparison regardless of thread scheduling.
    auto cmp = [](const benchmark_core::Contribution &x,
                  const benchmark_core::Contribution &y) {
        if (x.a_index != y.a_index) return x.a_index < y.a_index;
        if (x.b_index != y.b_index) return x.b_index < y.b_index;
        return x.k.k0 < y.k.k0;
    };
    std::sort(result.contributions.begin(), result.contributions.end(), cmp);

    // Expected: (A0,B0,k=[0,1]), (A0,B1,k=[1,1]),
    //           (A1,B0,k=[0,1]), (A1,B1,k=[1,1]),
    //           (A2,B0,k=[0,1]), (A2,B1,k=[1,1])
    CHECK(result.contributions[0].a_index==0 && result.contributions[0].b_index==0 && result.contributions[0].k.k0==0 && result.contributions[0].k.k1==1);
    CHECK(result.contributions[1].a_index==0 && result.contributions[1].b_index==1 && result.contributions[1].k.k0==1 && result.contributions[1].k.k1==1);
    CHECK(result.contributions[2].a_index==1 && result.contributions[2].b_index==0 && result.contributions[2].k.k0==0 && result.contributions[2].k.k1==1);
    CHECK(result.contributions[3].a_index==1 && result.contributions[3].b_index==1 && result.contributions[3].k.k0==1 && result.contributions[3].k.k1==1);
    CHECK(result.contributions[4].a_index==2 && result.contributions[4].b_index==0 && result.contributions[4].k.k0==0 && result.contributions[4].k.k1==1);
    CHECK(result.contributions[5].a_index==2 && result.contributions[5].b_index==1 && result.contributions[5].k.k0==1 && result.contributions[5].k.k1==1);
}

void test_merge_overlapping_output_blocks_parallel() {
    // 5 output blocks: two overlapping pairs plus one isolated block.
    // Pair A: blk0 and blk1 share a row+col region.
    // Pair B: blk2 and blk3 share a row+col region.
    // Singleton: blk4, far from all others.
    const std::vector<Block> blocks = {
        make_block(0,  0, 4, 4),   // blk0: rows[0,3]  cols[0,3]
        make_block(2,  2, 4, 4),   // blk1: rows[2,5]  cols[2,5]  → overlaps blk0
        make_block(10, 0, 4, 4),   // blk2: rows[10,13] cols[0,3]
        make_block(12, 2, 4, 4),   // blk3: rows[12,15] cols[2,5] → overlaps blk2
        make_block(30, 30, 2, 2),  // blk4: isolated
    };

    auto groups = benchmark_core::merge_overlapping_output_blocks(blocks);

    CHECK(groups.size() == 3);

    // Normalise: sort groups by min member, sort members within each group.
    for (auto &g : groups)
        std::sort(g.members.begin(), g.members.end());
    std::sort(groups.begin(), groups.end(),
              [](const Group &a, const Group &b) {
                  return a.members.front() < b.members.front();
              });

    CHECK(groups[0].members.size() == 2);
    CHECK(groups[0].members[0] == 0 && groups[0].members[1] == 1);

    CHECK(groups[1].members.size() == 2);
    CHECK(groups[1].members[0] == 2 && groups[1].members[1] == 3);

    CHECK(groups[2].members.size() == 1);
    CHECK(groups[2].members[0] == 4);
}

void test_block_fusion_parallel() {
    // 4 independent groups (no spatial overlap between groups), each with 2
    // members that produce the same MBR geometry at different row positions.
    //
    // Each pair: blk_i at (r, 0, 4, 4) and blk_i+1 at (r+2, 2, 4, 4).
    // MBR: r=r, c=0, h=6, w=6.
    // Union area = 4*4 + 4*4 - 2*2 = 28; imperfections = 36-28 = 8.
    const std::vector<Block> output_blocks = {
        make_block(0,  0, 4, 4),  // group 0
        make_block(2,  2, 4, 4),
        make_block(20, 0, 4, 4),  // group 1
        make_block(22, 2, 4, 4),
        make_block(40, 0, 4, 4),  // group 2
        make_block(42, 2, 4, 4),
        make_block(60, 0, 4, 4),  // group 3
        make_block(62, 2, 4, 4),
    };
    // Dummy contributions — only a_index/b_index used for correctness, not
    // checked in this test (block_fusion only reads them through to the group).
    const std::vector<Contribution> contributions = {
        {0,0,{0,0}}, {0,1,{0,0}},
        {1,0,{0,0}}, {1,1,{0,0}},
        {2,0,{0,0}}, {2,1,{0,0}},
        {3,0,{0,0}}, {3,1,{0,0}},
    };
    const std::vector<Group> groups = {
        {0, {0,1}}, {2, {2,3}}, {4, {4,5}}, {6, {6,7}},
    };

    auto fused = benchmark_core::block_fusion(output_blocks, contributions, groups);

    CHECK(fused.fused_blocks.size() == 4);
    CHECK(fused.fused_contributions.size() == 4);

    const int expected_rows[] = {0, 20, 40, 60};
    for (int gi = 0; gi < 4; ++gi) {
        CHECK(fused.fused_contributions[gi].size() == 2);
        const Block &b = fused.fused_blocks[gi];
        CHECK(b.r == expected_rows[gi]);
        CHECK(b.c == 0);
        CHECK(b.h == 6 && b.w == 6);
        CHECK(b.imperfections == 8);
    }
}

} // namespace

int main() {
    test_intersect_overlap();
    test_intersect_no_overlap();
    test_find_intersecting_pairs();
    test_find_overlapping_output_blocks();
    test_merge_groups();
    test_mbr_exact_union();
    test_mbr_with_gap();
    test_block_fusion_end_to_end();
    test_find_intersecting_pairs_parallel();
    test_merge_overlapping_output_blocks_parallel();
    test_block_fusion_parallel();

    std::printf("All tests passed.\n");
    return 0;
}
