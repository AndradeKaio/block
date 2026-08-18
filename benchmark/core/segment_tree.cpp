#include "segment_tree.hpp"

#include <algorithm>

namespace benchmark_core {

SegmentTree::SegmentTree(std::vector<int> coords) {
    std::sort(coords.begin(), coords.end());
    coords.erase(std::unique(coords.begin(), coords.end()), coords.end());
    coords_ = std::move(coords);
    n_ = static_cast<int>(coords_.size()) - 1;
    const int size = 4 * std::max(n_, 1);
    count_.assign(size, 0);
    covered_.assign(size, 0);
}

int SegmentTree::index_of(int coord) const {
    return static_cast<int>(
        std::lower_bound(coords_.begin(), coords_.end(), coord) - coords_.begin());
}

void SegmentTree::add_interval(int y1, int y2, int val) {
    if (n_ > 0) {
        update(1, 0, n_, index_of(y1), index_of(y2), val);
    }
}

long long SegmentTree::total_covered() const {
    return n_ > 0 ? covered_[1] : 0;
}

void SegmentTree::update(int node, int lo, int hi, int l, int r, int val) {
    if (l >= hi || r <= lo) {
        return;
    }
    if (l <= lo && hi <= r) {
        count_[node] += val;
    } else {
        const int mid = (lo + hi) / 2;
        update(2 * node, lo, mid, l, r, val);
        update(2 * node + 1, mid, hi, l, r, val);
    }
    if (count_[node] > 0) {
        covered_[node] = coords_[hi] - coords_[lo];
    } else if (lo + 1 == hi) {
        covered_[node] = 0;
    } else {
        covered_[node] = covered_[2 * node] + covered_[2 * node + 1];
    }
}

} // namespace benchmark_core
