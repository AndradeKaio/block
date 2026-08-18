import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from intervaltree import IntervalTree
from segmenttree import SegmentTree


@dataclass
class Block:
    r: int
    c: int
    h: int
    w: int
    imperfections: int = 0
    offset: int = 0

    def __repr__(self):
        return f"Block({self.h}x{self.w}, start=({self.r}, {self.c}))"

    def num_cells(self):
        return self.h * self.w

    def num_nonzeros(self):
        return self.num_cells() - self.imperfections

    @property
    def r_start(self) -> int:
        return self.r

    @property
    def r_end(self) -> int:
        return self.r + self.h - 1

    @property
    def c_start(self) -> int:
        return self.c

    @property
    def c_end(self) -> int:
        return self.c + self.w - 1

    @property
    def end(self) -> Tuple[int, int]:
        return (self.r + self.w - 1, self.c + self.h - 1)

    def intersect(self, other: "Block") -> Optional[Tuple[Tuple[int, int], "Block"]]:
        if self.c_end < other.r_start or self.c_start > other.r_end:
            return None
        start = self.c_start if self.c_start > other.r_start else other.r_start
        end = self.c_end if self.c_end < other.r_end else other.r_end
        return (
            (start, end),
            Block(r=self.r_start, h=self.h, c=other.c_start, w=other.w),
        )


def find_intersecting_pairs(
    A_blocks: List[Block], B_blocks: List[Block]
) -> Tuple[Tuple[int, int, Tuple[int, int], List[Block]]]:
    b_tree = IntervalTree()
    for bi, b in enumerate(B_blocks):
        b_tree[b.r_start : b.r_end + 1] = bi  # +1: inclusive -> half-open

    print(b_tree)
    contributions = []  # (ai, bi, k_range, output_block)
    output_blocks = []
    for ai, a in enumerate(A_blocks):
        for iv in b_tree[a.c_start : a.c_end + 1]:
            bi = iv.data
            result = a.intersect(B_blocks[bi])
            if result is not None:
                k_range, out_block = result
                contributions.append((ai, bi, k_range))
                output_blocks.append(out_block)
    return contributions, output_blocks


def find_overlapping_output_blocks(output_blocks: List[Block]):
    END, START = 0, 1
    events = []
    for idx, b in enumerate(output_blocks):
        events.append((b.r_start, START, idx))
        events.append((b.r_end + 1, END, idx))  # +1: inclusive -> half-open
    events.sort(key=lambda e: (e[0], e[1]))

    active = IntervalTree()
    pairs = set()

    for coord, kind, idx in events:
        b = output_blocks[idx]
        if kind == START:
            for iv in active[b.c_start : b.c_end + 1]:
                pairs.add(frozenset((idx, iv.data)))
            active[b.c_start : b.c_end + 1] = idx
        else:  # END
            active.removei(b.c_start, b.c_end + 1, idx)

    return pairs


def merge_groups(output_blocks: List[Block], overlap_pairs) -> Dict[int, List[int]]:
    parent = {i: i for i in range(len(output_blocks))}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for pair in overlap_pairs:
        x, y = tuple(pair)
        union(x, y)

    groups = defaultdict(list)
    for i in range(len(output_blocks)):
        groups[find(i)].append(i)
    return groups


def mbr(blocks: List[Block]) -> Block:
    ys = sorted(set([b.r_start for b in blocks] + [b.r_end + 1 for b in blocks]))
    tree = SegmentTree(ys)

    events = []
    min_row = min_col = math.inf
    max_row = max_col = -math.inf
    for block in blocks:
        if block.r_start < min_row:
            min_row = block.r_start
        if block.c_start < min_col:
            min_col = block.c_start
        if block.r_end > max_row:
            max_row = block.r_end
        if block.c_end > max_col:
            max_col = block.c_end
        events.append((block.c_start, 1, block.r_start, block.r_end + 1))
        events.append((block.c_end + 1, -1, block.r_start, block.r_end + 1))

    events.sort(key=lambda e: e[0])

    union_area = 0
    prev_x = 0

    for x, type_, y1, y2 in events:
        union_area += (x - prev_x) * tree.total_covered()
        tree.add_interval(y1, y2, type_)
        prev_x = x

    block = Block(
        r=min_row,
        c=min_col,
        w=(max_col - min_col + 1),
        h=(max_row - min_row + 1),
    )

    block.imperfections = block.num_cells() - union_area

    return block


def row_split(
    output_blocks: List[Block], group_indices: Dict[int : Tuple[int, int]]
) -> List[Block]:

    blocks = []
    for group, bids in group_indices.items():
        row_tree = IntervalTree()
        intervals = []
        for bid in bids:
            block = output_blocks[bid]
            row_tree[block.r_start : block.r_end + 1] = bid
            intervals.extend([block.r_start, block.r_end + 1])
        intervals = sorted(set(intervals))
        for lower, upper in zip(intervals, intervals[1:]):
            min_cols = []
            max_cols = []
            for interval in row_tree[lower:upper]:
                block = output_blocks[interval.data]
                min_cols.append(block.c_start)
                max_cols.append(block.c_end)
            min_col = min(min_cols)
            max_col = max(max_cols)
            blocks.append(
                Block(r=lower, c=min_col, h=(upper - lower), w=(max_col - min_col + 1))
            )
    return blocks


def block_fusion(output_blocks, contributions, group_indices):
    merged_blocks = []
    merged_contributions = defaultdict(list)
    for group_id, bids in group_indices.items():
        blocks = []
        for block_idx in bids:
            merged_contributions[group_id].append(contributions[block_idx])
            blocks.append(output_blocks[block_idx])
        merged_blocks.append(mbr(blocks))
    return merged_contributions, merged_blocks


if __name__ == "__main__":
    lhs_blocks = [
        Block(r=0, c=0, w=2, h=2),
        Block(r=6, c=0, w=3, h=2),
        Block(r=2, c=2, w=3, h=3),
        Block(r=0, c=6, w=2, h=2),
        Block(r=3, c=7, w=1, h=2),
        Block(r=6, c=6, w=2, h=2),
    ]
    rhs_blocks = [
        Block(r=0, c=0, w=4, h=2),
        Block(r=2, c=0, w=2, h=3),
        Block(r=3, c=3, w=2, h=2),
        Block(r=6, c=1, w=2, h=2),
        Block(r=5, c=5, w=3, h=3),
    ]
    contributions, output_blocks = find_intersecting_pairs(lhs_blocks, rhs_blocks)

    lhs_blocks = [
        Block(r=1, c=1, w=3, h=2, offset=0),
        Block(r=4, c=2, w=2, h=3, offset=6),
        Block(r=8, c=0, w=2, h=1, offset=12),
        Block(r=0, c=7, w=2, h=2, offset=14),
        Block(r=6, c=5, w=4, h=2, offset=18),
    ]

    contributions, output_blocks = find_intersecting_pairs(lhs_blocks, lhs_blocks)
    groups = find_overlapping_output_blocks(output_blocks)
    group_indices = merge_groups(output_blocks, groups)

    # contributions ordered with the fused blocks
    # each group_id has a list of contributions (lhs_block_idx, rhs_block_idx, (contraction range))
    contributions, new_output_blocks = block_fusion(
        output_blocks, contributions, group_indices
    )
    # for (i = 0; i < lhs.h; i++)
    #     for (j = pair[2][0]; j <= pair[2][1]; j++)       // global contraction index, inclusive
    #         for (k = 0; k < rhs.w; k++)
    #             C[out.offset + (lhs.r_start + i - out.r_start) * out.w + (rhs.c_start + k - out.c_start)]
    #                 += A[lhs.offset + i * lhs.w + (j - lhs.c_start)]
    #                  * B[rhs.offset + (j - rhs.r_start) * rhs.w + k]
    A = [1] * sum(b.num_cells() for b in lhs_blocks)
    B = [1] * sum(b.num_cells() for b in lhs_blocks)
    C = [0] * sum(b.num_cells() for b in new_output_blocks)
    # print(new_output_blocks)
    C_out = new_output_blocks.copy()
    C_out.sort(key=lambda e: (e.r_start, e.c_start))
    offset_dict = defaultdict(int)
    offset_count = 0
    for block in C_out:
        offset_dict[id(block)] = offset_count
        offset_count += block.num_cells()
    for block in new_output_blocks:
        block.offset = offset_dict[id(block)]
    for idx, contribution in enumerate(contributions.values()):
        # print(new_output_blocks[idx], "=", sep="")
        out_block = new_output_blocks[idx]
        for pair in contribution:
            lhs, rhs = lhs_blocks[pair[0]], lhs_blocks[pair[1]]
            crange = pair[2]
            for i in range(lhs.h):
                for j in range(crange[0], crange[1] + 1):  # inclusive on both ends
                    for k in range(rhs.w):
                        C[
                            out_block.offset
                            + (lhs.r_start + i - out_block.r_start)
                            * out_block.w  # lhs, not rhs
                            + (rhs.c_start + k - out_block.c_start)
                        ] += (
                            A[lhs.offset + i * lhs.w + (j - lhs.c_start)]
                            * B[rhs.offset + (j - rhs.r_start) * rhs.w + k]
                        )
    print(C)
