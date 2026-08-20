#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "block.hpp"
#include "block_mining.hpp"
#include "matrix.hpp"
#include "matrix_io.hpp"

using namespace benchmark_core;

// ── Minimal MTX reader (COO → CSR, 0-based, pattern only) ────────────────────

struct COO { int r, c; double v; };

static std::vector<COO> read_mtx(const char* path, int& M, int& N) {
    FILE* f = fopen(path, "r");
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);

    char line[1024];

    // Parse banner: %%MatrixMarket matrix coordinate <type> <symmetry>.
    // Only the FIRST line is the actual banner -- fixed by the MatrixMarket
    // spec. Free-text %-comment lines that follow are documentation, not
    // format, and must NOT be scanned for these keywords: real matrices
    // routinely use words like "pattern"/"symmetric" in prose (e.g. "arises
    // from a symmetric physical system") without being symmetric-format,
    // which previously false-positived is_symmetric/is_pattern here and
    // silently corrupted the mined .bsp for those matrices (extra mirrored
    // entries, or every value replaced by 1.0) -- the exact same bug already
    // found and fixed in SpMM/bench_taco_spmm.cpp's read_mtx; this mirrors
    // that fix.
    if (!fgets(line, sizeof(line), f))
        throw std::runtime_error(std::string("empty file: ") + path);
    char banner[1024];
    std::strncpy(banner, line, sizeof(banner));
    banner[sizeof(banner) - 1] = '\0';
    for (char* p = banner; *p; ++p) *p = (char)tolower((unsigned char)*p);
    bool is_pattern   = strstr(banner, "pattern") != nullptr;
    bool is_symmetric = strstr(banner, "symmetric") || strstr(banner, "hermitian");

    // Skip the remaining %-comment lines without inspecting their content.
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    // `line` now holds the first non-comment line: M N NNZ
    int declared_nnz = 0;
    if (sscanf(line, "%d %d %d", &M, &N, &declared_nnz) < 2)
        throw std::runtime_error("bad MTX header");

    std::vector<COO> coo;
    coo.reserve(is_symmetric ? declared_nnz * 2 : declared_nnz);

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%' || line[0] == '\n') continue;
        int r, c; double v = 1.0;
        if (is_pattern) {
            if (sscanf(line, "%d %d", &r, &c) < 2) continue;
        } else {
            if (sscanf(line, "%d %d %lf", &r, &c, &v) < 2) continue;
        }
        --r; --c;  // MTX is 1-based
        coo.push_back({r, c, v});
        if (is_symmetric && r != c)
            coo.push_back({c, r, v});
    }
    fclose(f);
    return coo;
}

static void coo_to_csr(std::vector<COO>& coo, int M,
                        std::vector<int>& row_ptr,
                        std::vector<int>& col_idx,
                        std::vector<double>& val_csr) {
    // Sort by (row, col), then merge-SUM any duplicate (row, col) pairs --
    // NOT arbitrarily keep-one-discard-the-rest. MatrixMarket's coordinate
    // format convention (and what scipy.io.mmread + .tocsr() actually does)
    // is that duplicate listed entries at the same position are summed --
    // FEM assembly commonly writes per-element contributions to a shared
    // DOF pair as separate lines rather than pre-summed. The previous
    // std::unique-based dedup silently dropped every duplicate but the
    // first, corrupting the value at that position -- found by comparing
    // against scipy's reference (validate_bsp.py --deep) on real
    // SuiteSparse matrices: pkustk01/07/08, thread, tsyl201, k3plates,
    // msc10848, opt1, cegb3024, bundle1 all had a handful of positions
    // silently wrong, every one a stiffness/FEM-assembled matrix.
    std::sort(coo.begin(), coo.end(), [](const COO& a, const COO& b) {
        return a.r != b.r ? a.r < b.r : a.c < b.c;
    });
    std::vector<COO> merged;
    merged.reserve(coo.size());
    for (const auto& e : coo) {
        if (!merged.empty() && merged.back().r == e.r && merged.back().c == e.c)
            merged.back().v += e.v;
        else
            merged.push_back(e);
    }
    coo = std::move(merged);

    const int nnz = (int)coo.size();
    row_ptr.assign(M + 1, 0);
    col_idx.resize(nnz);
    val_csr.resize(nnz);
    for (auto& e : coo) row_ptr[e.r + 1]++;
    for (int i = 0; i < M; ++i) row_ptr[i + 1] += row_ptr[i];
    for (int k = 0; k < nnz; ++k) {
        col_idx[k] = coo[k].c;
        val_csr[k] = coo[k].v;
    }
}

// ── Stats ─────────────────────────────────────────────────────────────────────

static void print_stats_json(const std::vector<Block>& blocks,
                              int small_threshold) {
    // Partition into large (num_nonzeros >= threshold) and small
    std::vector<const Block*> large;
    for (auto& b : blocks)
        if (b.num_nonzeros() >= small_threshold)
            large.push_back(&b);

    // Dominant shape
    std::map<std::pair<int,int>, int> shape_count;
    for (auto* b : large)
        shape_count[{b->h, b->w}]++;

    std::pair<int,int> dom_shape{0,0};
    int dom_count = 0;
    for (auto& [shape, cnt] : shape_count)
        if (cnt > dom_count) { dom_count = cnt; dom_shape = shape; }

    double dom_share = large.empty() ? 0.0 : (double)dom_count / (double)large.size();

    long long max_nnz = 0;
    double    mean_nnz = 0.0;
    for (auto* b : large) {
        long long n = b->num_nonzeros();
        if (n > max_nnz) max_nnz = n;
        mean_nnz += (double)n;
    }
    if (!large.empty()) mean_nnz /= (double)large.size();

    // padding_zeros / covered_nnz: over dominant-shape blocks only
    long long padding_zeros = 0, covered_nnz = 0;
    for (auto* b : large) {
        if (b->h == dom_shape.first && b->w == dom_shape.second) {
            padding_zeros += b->imperfections;
            covered_nnz   += b->num_nonzeros();
        }
    }

    // total_padding: imperfections summed over ALL blocks
    long long total_padding = 0;
    for (auto& b : blocks) total_padding += b.imperfections;

    // n_singles: 1×1 blocks (isolated NNZ that couldn't join a larger block)
    long long n_singles = 0;
    for (auto& b : blocks)
        if (b.h == 1 && b.w == 1) ++n_singles;

    char dom_str[64];
    std::snprintf(dom_str, sizeof(dom_str), "%dx%d", dom_shape.first, dom_shape.second);

    std::printf(
        "{\"n_patterns\":%zu,\"n_large\":%zu,"
        "\"dominant_shape\":\"%s\",\"dominant_count\":%d,\"dominant_share\":%.4f,"
        "\"max_nnz\":%lld,\"mean_nnz\":%.4f,"
        "\"padding_zeros\":%lld,\"covered_nnz\":%lld,\"total_padding\":%lld,"
        "\"n_singles\":%lld}\n",
        blocks.size(), large.size(),
        dom_str, dom_count, dom_share,
        max_nnz, mean_nnz,
        padding_zeros, covered_nnz, total_padding, n_singles
    );
}

// ── CLI ───────────────────────────────────────────────────────────────────────

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <input.mtx> <output.bsp> [options]\n"
        "Options:\n"
        "  --retry-expand          retry both directions after each growth step\n"
        "                          (default: off — matches Python miner)\n"
        "  --twf F                 wavefront fill threshold (default: 0.5)\n"
        "  --to F                  max imperfection ratio   (default: 0.3)\n"
        "  --thslim N              max aspect ratio         (default: 50)\n"
        "  --small-threshold N     min NNZ for a large block (default: 10)\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const char* input_path  = argv[1];
    const char* output_path = argv[2];

    MineParams params;
    params.retry_expand = false;
    int small_threshold = 10;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--retry-expand") == 0) {
            params.retry_expand = true;
        } else if (std::strcmp(argv[i], "--twf") == 0 && i + 1 < argc) {
            params.Twf = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--to") == 0 && i + 1 < argc) {
            params.To = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--thslim") == 0 && i + 1 < argc) {
            params.Thslim = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--small-threshold") == 0 && i + 1 < argc) {
            small_threshold = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    try {
        // 1. Read MTX → CSR
        int M = 0, N = 0;
        auto coo = read_mtx(input_path, M, N);

        std::vector<int>    row_ptr, col_idx;
        std::vector<double> val_csr;
        coo_to_csr(coo, M, row_ptr, col_idx, val_csr);
        coo.clear();
        coo.shrink_to_fit();

        // 2. Mine blocks
        auto blocks = mine_blocks(M, N, row_ptr.data(), col_idx.data(), params);

        // 3. Assign offsets
        long long n_values = assign_offsets(blocks);

        // 4. Build Matrix<double>: zero-initialise, then fill actual non-zeros.
        // Blocks may have "imperfection" cells (positions within the h×w
        // rectangle where the original matrix has no non-zero).  Those must
        // remain 0.0; only the cells that correspond to actual matrix entries
        // get 1.0.  Using std::fill with 1.0 here would corrupt the GEMM by
        // making every imperfection cell look like a real non-zero.
        Matrix<double> mat;
        mat.M        = M;
        mat.N        = N;
        mat.blocks   = blocks;
        mat.n_values = (std::size_t)n_values;
        mat.values   = new double[mat.n_values](); // zero-initialised
        for (const Block& b : mat.blocks) {
            for (int ri = 0; ri < b.h; ++ri) {
                const int row = b.r + ri;
                for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
                    const int col = col_idx[k];
                    if (col >= b.c && col < b.c + b.w)
                        mat.values[b.offset + (long long)ri * b.w + (col - b.c)] = val_csr[k];
                }
            }
        }

        // 5. Write .bsp
#ifdef HAVE_HDF5
        write_matrix_binsparse<double>(mat, output_path);
#else
        std::fprintf(stderr, "HDF5 not available — cannot write %s\n", output_path);
        return 1;
#endif

        // 6. Print JSON stats to stdout
        print_stats_json(blocks, small_threshold);

    } catch (const std::exception& e) {
        std::fprintf(stderr, "mine_matrix: %s\n", e.what());
        return 1;
    }

    return 0;
}
