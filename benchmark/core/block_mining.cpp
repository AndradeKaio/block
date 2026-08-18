#include "block_mining.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace benchmark_core {
namespace {

struct CSRMiner {
    int M, N;
    const int* row_ptr;       // [M+1]  — borrowed
    const int* col_idx;       // [nnz]  — borrowed
    std::vector<bool> alive;  // one entry per NNZ, initially all true
    int first_alive_row = 0;  // cursor to amortise get_start_point

    // Returns {row, col} of the next alive NNZ in row-major order,
    // or {-1, -1} when all NNZs have been consumed.
    std::pair<int, int> get_start_point() {
        while (first_alive_row < M) {
            int lo = row_ptr[first_alive_row];
            int hi = row_ptr[first_alive_row + 1];
            for (int i = lo; i < hi; ++i) {
                if (alive[i])
                    return {first_alive_row, col_idx[i]};
            }
            ++first_alive_row;
        }
        return {-1, -1};
    }

    // Count alive NNZs in row `row` whose column index is in [c0, c1).
    int query_row(int row, int c0, int c1) const {
        const int* beg = col_idx + row_ptr[row];
        const int* end = col_idx + row_ptr[row + 1];
        const int* lo  = std::lower_bound(beg, end, c0);
        const int* hi  = std::lower_bound(lo,  end, c1);
        int count = 0;
        for (const int* p = lo; p < hi; ++p)
            if (alive[p - col_idx])
                ++count;
        return count;
    }

    // Count alive NNZs in column `col` whose row index is in [r0, r1).
    int query_col(int col, int r0, int r1) const {
        int count = 0;
        for (int r = r0; r < r1; ++r) {
            const int* beg = col_idx + row_ptr[r];
            const int* end = col_idx + row_ptr[r + 1];
            const int* p   = std::lower_bound(beg, end, col);
            if (p < end && *p == col && alive[p - col_idx])
                ++count;
        }
        return count;
    }

    // Mark all alive NNZs inside the rectangle [r, r+h) x [c, c+w) as dead.
    void delete_block(int r, int c, int h, int w) {
        for (int row = r; row < r + h; ++row) {
            const int* beg = col_idx + row_ptr[row];
            const int* end = col_idx + row_ptr[row + 1];
            const int* lo  = std::lower_bound(beg, end, c);
            for (const int* p = lo; p < end && *p < c + w; ++p)
                alive[p - col_idx] = false;
        }
    }
};

// retry_expand=true: both directions re-evaluated after every growth step.
// Produces fewer, larger blocks with more padding zeros.
static Block expand_block_retry(int seed_r, int seed_c,
                                 CSRMiner& csr, const MineParams& p) {
    int r = seed_r, c = seed_c;
    int h = 1, w = 1;
    long long imp = 1 - csr.query_row(r, c, c + 1);

    for (;;) {
        bool grew = false;

        if (c + w < csr.N) {
            int fill          = csr.query_col(c + w, r, r + h);
            long long new_imp = imp + (h - fill);
            float new_area    = float(h) * float(w + 1);
            float aspect      = float(std::max(h, w + 1)) / float(std::min(h, w + 1));
            if (float(fill) >= float(h) * (1.0f - p.Twf) &&
                float(new_imp) / new_area <= p.To &&
                aspect <= p.Thslim) {
                imp = new_imp; ++w; grew = true;
            }
        }

        if (r + h < csr.M) {
            int fill          = csr.query_row(r + h, c, c + w);
            long long new_imp = imp + (w - fill);
            float new_area    = float(h + 1) * float(w);
            float aspect      = float(std::max(h + 1, w)) / float(std::min(h + 1, w));
            if (float(fill) >= float(w) * (1.0f - p.Twf) &&
                float(new_imp) / new_area <= p.To &&
                aspect <= p.Thslim) {
                imp = new_imp; ++h; grew = true;
            }
        }

        if (!grew) break;
    }

    return Block{r, c, h, w, imp, /*offset=*/0};
}

// retry_expand=false: mirrors Python mine_matrices.py — a direction is
// permanently disabled the first time it fails, even if the block later grows
// in the other dimension and the condition might now be satisfiable.
// Produces more, smaller blocks with less padding.
static Block expand_block_perm_disable(int seed_r, int seed_c,
                                        CSRMiner& csr, const MineParams& p) {
    int r = seed_r, c = seed_c;
    int h = 1, w = 1;
    long long imp = 1 - csr.query_row(r, c, c + 1);

    bool expand_right = true;
    bool expand_down  = true;

    while (expand_right || expand_down) {
        bool moved = false;

        if (expand_right) {
            int nw = w + 1;
            if (float(std::max(h, nw)) / float(std::min(h, nw)) > p.Thslim && nw > h) {
                expand_right = false;
            } else {
                int fill = csr.query_col(c + w, r, r + h);
                if (float(fill) < float(h) * (1.0f - p.Twf)) {
                    expand_right = false;
                } else {
                    long long new_imp = imp + (h - fill);
                    if (float(new_imp) / (float(h) * float(nw)) > p.To) {
                        expand_right = false;
                    } else {
                        imp = new_imp; w = nw; moved = true;
                    }
                }
            }
        }

        if (expand_down) {
            int nh = h + 1;
            if (r + h >= csr.M) {
                expand_down = false;
            } else if (float(std::max(nh, w)) / float(std::min(nh, w)) > p.Thslim && nh > w) {
                expand_down = false;
            } else {
                int fill = csr.query_row(r + h, c, c + w);
                if (float(fill) < float(w) * (1.0f - p.Twf)) {
                    expand_down = false;
                } else {
                    long long new_imp = imp + (w - fill);
                    if (float(new_imp) / (float(nh) * float(w)) > p.To) {
                        expand_down = false;
                    } else {
                        imp = new_imp; h = nh; moved = true;
                    }
                }
            }
        }

        if (!moved) break;
    }

    return Block{r, c, h, w, imp, /*offset=*/0};
}

} // anonymous namespace

std::vector<Block> mine_blocks(int M, int N,
                                const int* row_ptr,
                                const int* col_idx,
                                MineParams params) {
    const int nnz = row_ptr[M];
    CSRMiner csr{M, N, row_ptr, col_idx,
                 std::vector<bool>(nnz, true), 0};

    auto expand = params.retry_expand ? expand_block_retry
                                      : expand_block_perm_disable;

    std::vector<Block> blocks;
    for (;;) {
        auto [sr, sc] = csr.get_start_point();
        if (sr < 0) break;
        Block b = expand(sr, sc, csr, params);
        csr.delete_block(b.r, b.c, b.h, b.w);
        blocks.push_back(b);
    }
    return blocks;
}

} // namespace benchmark_core
