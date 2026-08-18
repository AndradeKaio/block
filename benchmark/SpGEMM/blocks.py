from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from intervaltree import IntervalTree


class SegmentTree:
    def __init__(self, coords: List[int]):
        self.coords = sorted(set(coords))
        self.n = len(self.coords) - 1
        size = 4 * max(self.n, 1)
        self.count = [0] * size
        self.covered = [0] * size
        self._idx = {c: i for i, c in enumerate(self.coords)}

    def add_interval(self, y1: int, y2: int, val: int):
        if self.n > 0:
            self._update(1, 0, self.n, self._idx[y1], self._idx[y2], val)

    def total_covered(self) -> int:
        return self.covered[1] if self.n > 0 else 0

    def _update(self, node, lo, hi, l, r, val):
        if l >= hi or r <= lo:
            return
        if l <= lo and hi <= r:
            self.count[node] += val
        else:
            mid = (lo + hi) // 2
            self._update(2 * node, lo, mid, l, r, val)
            self._update(2 * node + 1, mid, hi, l, r, val)
        if self.count[node] > 0:
            self.covered[node] = self.coords[hi] - self.coords[lo]
        elif lo + 1 == hi:
            self.covered[node] = 0
        else:
            self.covered[node] = self.covered[2 * node] + self.covered[2 * node + 1]


# ── Block + pipeline ──────────────────────────────────────────────────────────


@dataclass
class Block:
    r: int
    c: int
    h: int
    w: int
    imperfections: int = 0
    offset: int = 0

    def __repr__(self):
        return f"Block({self.h}x{self.w}, start=({self.r},{self.c}))"

    def num_cells(self) -> int:
        return self.h * self.w

    def num_nonzeros(self) -> int:
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

    def intersect(self, other: "Block") -> Optional[Tuple[Tuple[int, int], "Block"]]:
        """
        Check if self.cols overlaps other.rows (shared k-dimension).
        Returns (k_range_inclusive, output_block) or None.
        """
        if self.c_end < other.r_start or self.c_start > other.r_end:
            return None
        k0 = max(self.c_start, other.r_start)
        k1 = min(self.c_end, other.r_end)
        return (k0, k1), Block(r=self.r_start, h=self.h, c=other.c_start, w=other.w)


def find_intersecting_pairs(
    A_blocks: List[Block], B_blocks: List[Block]
) -> Tuple[List[Tuple], List[Block]]:
    """
    For SpGEMM C = A @ B: find all (A_block, B_block) pairs whose shared
    k-dimension ranges overlap, via an interval tree on B block row ranges.
    Returns contributions [(ai, bi, k_range)] and the corresponding output blocks.
    """
    b_tree = IntervalTree()
    for bi, b in enumerate(B_blocks):
        b_tree[b.r_start : b.r_end + 1] = bi  # half-open for IntervalTree

    contributions = []
    output_blocks = []
    for ai, a in enumerate(A_blocks):
        for iv in sorted(b_tree[a.c_start : a.c_end + 1], key=lambda x: x.data):
            bi = iv.data
            result = a.intersect(B_blocks[bi])
            if result is not None:
                k_range, out_block = result
                contributions.append((ai, bi, k_range))
                output_blocks.append(out_block)
    return contributions, output_blocks


def find_overlapping_output_blocks(output_blocks: List[Block]) -> Set[frozenset]:
    """
    Sweep-line over output blocks to find which pairs overlap in 2D.
    Returns a set of frozenset({i, j}) for each overlapping pair.
    """
    END, START = 0, 1
    events = []
    for idx, b in enumerate(output_blocks):
        events.append((b.r_start, START, idx))
        events.append((b.r_end + 1, END, idx))
    events.sort(key=lambda e: (e[0], e[1]))

    active = IntervalTree()
    pairs: Set[frozenset] = set()

    for _, kind, idx in events:
        b = output_blocks[idx]
        if kind == START:
            for iv in active[b.c_start : b.c_end + 1]:
                pairs.add(frozenset((idx, iv.data)))
            active[b.c_start : b.c_end + 1] = idx
        else:
            active.removei(b.c_start, b.c_end + 1, idx)
    return pairs


def merge_groups(output_blocks: List[Block], overlap_pairs) -> Dict[int, List[int]]:
    """Union-Find: merge output blocks that overlap into groups."""
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

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(output_blocks)):
        groups[find(i)].append(i)
    return groups


def mbr(blocks: List[Block]) -> Block:
    """
    Minimum bounding rectangle of a set of blocks, with imperfection count
    computed via Klee's algorithm (coordinate-compressed segment tree area sweep).
    """
    ys = sorted({b.r_start for b in blocks} | {b.r_end + 1 for b in blocks})
    tree = SegmentTree(ys)

    min_row = min(b.r_start for b in blocks)
    max_row = max(b.r_end for b in blocks)
    min_col = min(b.c_start for b in blocks)
    max_col = max(b.c_end for b in blocks)

    events = []
    for b in blocks:
        events.append((b.c_start, +1, b.r_start, b.r_end + 1))
        events.append((b.c_end + 1, -1, b.r_start, b.r_end + 1))
    events.sort(key=lambda e: e[0])

    union_area = 0
    prev_x = 0
    for x, t, y1, y2 in events:
        union_area += (x - prev_x) * tree.total_covered()
        tree.add_interval(y1, y2, t)
        prev_x = x

    result = Block(
        r=min_row, c=min_col, h=max_row - min_row + 1, w=max_col - min_col + 1
    )
    result.imperfections = result.num_cells() - union_area
    return result


def block_fusion(
    output_blocks: List[Block],
    contributions: List[Tuple],
    group_indices: Dict[int, List[int]],
) -> Tuple[Dict[int, List], List[Block]]:
    """
    Fuse overlapping output blocks per group:
      - MBR of the group's output blocks → one fused output block
      - All contributions that land in the group are accumulated together
    """
    fused_contributions: Dict[int, List] = defaultdict(list)
    fused_blocks: List[Block] = []
    for group_id, bids in group_indices.items():
        for bid in bids:
            fused_contributions[group_id].append(contributions[bid])
        fused_blocks.append(mbr([output_blocks[bid] for bid in bids]))
    return fused_contributions, fused_blocks
