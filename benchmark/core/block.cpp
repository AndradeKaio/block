#include "block.hpp"

#include <algorithm>

namespace benchmark_core {

long long Block::num_cells() const {
    return static_cast<long long>(h) * static_cast<long long>(w);
}

long long Block::num_nonzeros() const {
    return num_cells() - imperfections;
}

std::optional<Intersection> intersect(const Block& a, const Block& b) {
    if (a.c_end() < b.r_start() || a.c_start() > b.r_end()) {
        return std::nullopt;
    }
    KRange k{std::max(a.c_start(), b.r_start()), std::min(a.c_end(), b.r_end())};
    Block output;
    output.r = a.r_start();
    output.h = a.h;
    output.c = b.c_start();
    output.w = b.w;
    return Intersection{k, output};
}

long long assign_offsets(std::vector<Block>& blocks) {
    long long offset = 0;
    for (auto& b : blocks) {
        b.offset = offset;
        offset += b.num_cells();
    }
    return offset;
}

} // namespace benchmark_core
