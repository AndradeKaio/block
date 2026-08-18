#pragma once

#include <vector>

namespace benchmark_core {

// Coordinate-compressed segment tree over a 1D axis, tracking how much of
// the axis is currently covered by an active multiset of intervals. Used by
// `mbr()` to compute rectangle union area via Klee's algorithm (a sweep over
// one axis while this tree tracks coverage on the other).
//
// Mirrors blocks.py's `SegmentTree`.
class SegmentTree {
public:
    // `coords` need not be sorted or unique; the tree compresses them itself.
    // Every interval passed to add_interval() must use endpoints from this set.
    explicit SegmentTree(std::vector<int> coords);

    // Add (val > 0) or remove (val < 0) coverage over the half-open range
    // [y1, y2). Both endpoints must be members of the coordinate set.
    void add_interval(int y1, int y2, int val);

    // Total length of the axis currently covered by at least one interval.
    long long total_covered() const;

private:
    void update(int node, int lo, int hi, int l, int r, int val);
    int index_of(int coord) const;

    std::vector<int> coords_;
    int n_ = 0;
    std::vector<int> count_;
    std::vector<long long> covered_;
};

} // namespace benchmark_core
