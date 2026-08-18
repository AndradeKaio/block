#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "block.hpp"

namespace benchmark_core {

// Scalar type tag for Matrix<T>: lets code identify a matrix's element type
// at runtime without templating on it. Only F32/F64 exist today since those
// are the only types this codebase natively supports; add a new enumerator
// + scalar_traits<T> specialization here (and a matching explicit
// instantiation in matrix.cpp) before instantiating Matrix<T> /
// generate_random_matrix<T> for any additional T.
enum class DataType : std::uint8_t { F32, F64 };

template <typename T>
struct scalar_traits; // no generic definition: an unsupported T is a
                       // compile error at the point of use.

template <>
struct scalar_traits<float> {
    static constexpr DataType dtype = DataType::F32;
};

template <>
struct scalar_traits<double> {
    static constexpr DataType dtype = DataType::F64;
};

// An owning, block-sparse matrix: block geometry (`blocks`) plus a single
// flat, homogeneous-dtype value buffer (`values`) holding every block's
// cells concatenated in `blocks` order, row-major within each block.
// Mirrors shared.py's `generate_matrices` return value (minus the COO
// row/col/val lists, which exist there only for .mtx file writing).
//
// Each block b's data occupies values[b.offset, b.offset + b.num_cells()).
// Call assign_offsets(blocks) (block.hpp) after any change to `blocks` that
// should be reflected in `values`' layout.
//
// `values` is an owning raw buffer (new T[]/delete[]), not std::vector<T>.
// Matrix<T> implements the Rule of 5 so it stays safe to copy, move, and
// return by value despite owning a raw pointer: copying deep-copies the
// buffer; moving steals the pointer and leaves the source empty
// (values == nullptr, n_values == 0).
//
// Definitions live in matrix.cpp; only float and double are instantiated
// there.
template <typename T>
struct Matrix {
    static constexpr DataType dtype = scalar_traits<T>::dtype;

    int M = 0;
    int N = 0;
    std::vector<Block> blocks;
    T* values = nullptr;
    std::size_t n_values = 0;

    Matrix() = default;
    ~Matrix();

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    // View onto block b's data within `values`: b.h rows x b.w cols,
    // row-major. `b` must be offset-assigned against this matrix's buffer
    // (e.g. one of `blocks`, after assign_offsets()).
    std::span<T> block_data(const Block& b);
    std::span<const T> block_data(const Block& b) const;
};

// Random matrix generator mirroring shared.py's `generate_matrices`
// end-to-end: places non-overlapping blocks via generate_random_blocks()
// (block_generator.hpp), assigns their offsets via assign_offsets(), and
// fills `values` with standard-normal random values, block by block
// (row-major within each block).
//
// Note: unlike shared.py (which draws block geometry and cell values from
// one interleaved numpy Generator), this draws them from two independent
// std::mt19937_64 streams, so results will not bit-for-bit match the Python
// reference for the same seed.
template <typename T>
Matrix<T> generate_random_matrix(int rows, int cols, int n_blocks, std::pair<int, int> h_range,
                                  std::pair<int, int> w_range, std::uint64_t seed,
                                  bool snap_to_tc = false, double block_density = 1.0);

} // namespace benchmark_core
