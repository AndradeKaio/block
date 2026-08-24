#include "matrix.hpp"

#include <algorithm>
#include <random>
#include <tuple>

#include "block_generator.hpp"

namespace benchmark_core {

template <typename T>
Matrix<T>::~Matrix() {
    delete[] values;
}

template <typename T>
Matrix<T>::Matrix(const Matrix& other)
    : M(other.M), N(other.N), blocks(other.blocks), n_values(other.n_values) {
    values = n_values > 0 ? new T[n_values] : nullptr;
    std::copy_n(other.values, other.n_values, values);
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) {
    if (this == &other) {
        return *this;
    }
    T* new_values = other.n_values > 0 ? new T[other.n_values] : nullptr;
    std::copy_n(other.values, other.n_values, new_values);

    delete[] values;
    M = other.M;
    N = other.N;
    blocks = other.blocks;
    values = new_values;
    n_values = other.n_values;
    return *this;
}

template <typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept
    : M(other.M),
      N(other.N),
      blocks(std::move(other.blocks)),
      values(other.values),
      n_values(other.n_values) {
    other.values = nullptr;
    other.n_values = 0;
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    delete[] values;
    M = other.M;
    N = other.N;
    blocks = std::move(other.blocks);
    values = other.values;
    n_values = other.n_values;
    other.values = nullptr;
    other.n_values = 0;
    return *this;
}

template <typename T>
std::span<T> Matrix<T>::block_data(const Block& b) {
    return std::span<T>(values + b.offset, static_cast<std::size_t>(b.num_cells()));
}

template <typename T>
std::span<const T> Matrix<T>::block_data(const Block& b) const {
    return std::span<const T>(values + b.offset, static_cast<std::size_t>(b.num_cells()));
}

template <typename T>
CSR<T> Matrix<T>::to_csr() const {
    // (row, col, value) for every cell whose stored value is a genuine
    // nonzero -- blocks can overlap in row range (two different blocks
    // covering the same rows at different columns), so entries can't be
    // read off in row order just by walking `blocks`; collect then sort.
    std::vector<std::tuple<int, int, T>> entries;
    for (const Block& b : blocks) {
        for (int ri = 0; ri < b.h; ++ri) {
            for (int ci = 0; ci < b.w; ++ci) {
                const T v = values[b.offset + static_cast<long long>(ri) * b.w + ci];
                if (v != T(0)) {
                    entries.emplace_back(b.r + ri, b.c + ci, v);
                }
            }
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& x, const auto& y) {
        return std::tie(std::get<0>(x), std::get<1>(x)) < std::tie(std::get<0>(y), std::get<1>(y));
    });

    CSR<T> csr;
    csr.row_ptr.assign(static_cast<std::size_t>(M) + 1, 0);
    csr.col_idx.resize(entries.size());
    csr.values.resize(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& [r, c, v] = entries[i];
        ++csr.row_ptr[static_cast<std::size_t>(r) + 1];
        csr.col_idx[i] = c;
        csr.values[i] = v;
    }
    for (int r = 0; r < M; ++r) {
        csr.row_ptr[static_cast<std::size_t>(r) + 1] += csr.row_ptr[static_cast<std::size_t>(r)];
    }
    return csr;
}

template <typename T>
Matrix<T> generate_random_matrix(int rows, int cols, int n_blocks, std::pair<int, int> h_range,
                                  std::pair<int, int> w_range, std::uint64_t seed,
                                  bool snap_to_tc, double block_density) {
    Matrix<T> mat;
    mat.M = rows;
    mat.N = cols;
    mat.blocks = generate_random_blocks(rows, cols, n_blocks, h_range, w_range, seed, snap_to_tc);

    const long long total = assign_offsets(mat.blocks);
    mat.n_values = static_cast<std::size_t>(total);
    mat.values = mat.n_values > 0 ? new T[mat.n_values] : nullptr;

    // A distinctly-seeded stream from the one generate_random_blocks() uses
    // internally, so this doesn't collide with the `seed + 1` convention
    // callers use for a second matrix's geometry (see main.cpp).
    std::seed_seq value_seed{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
                              0x76616c75u /* 'valu' */};
    std::mt19937_64 rng(value_seed);
    std::normal_distribution<T> dist(T{0}, T{1});
    std::bernoulli_distribution keep(block_density);

    for (const auto& b : mat.blocks) {
        const long long n = b.num_cells();
        for (long long i = 0; i < n; ++i) {
            mat.values[static_cast<std::size_t>(b.offset + i)] = keep(rng) ? dist(rng) : T{0};
        }
    }
    return mat;
}

// Explicit instantiations: only float and double are supported today.
template struct Matrix<float>;
template struct Matrix<double>;

template Matrix<float> generate_random_matrix<float>(int, int, int, std::pair<int, int>,
                                                       std::pair<int, int>, std::uint64_t, bool, double);
template Matrix<double> generate_random_matrix<double>(int, int, int, std::pair<int, int>,
                                                         std::pair<int, int>, std::uint64_t, bool, double);

} // namespace benchmark_core
