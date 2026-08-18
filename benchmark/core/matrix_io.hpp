#pragma once
#include <string>
#include "matrix.hpp"

namespace benchmark_core {

// Binsparse / HDF5 binary format (.bsp)
//
// The file is an HDF5 container holding:
//   - attribute "binsparse"   : JSON descriptor (format="CSP", version="0.1")
//   - attribute "matrix_rows" : int32  M
//   - attribute "matrix_cols" : int32  N
//   - dataset  "values"       : float32 or float64 [n_values]
//   - dataset  "block_r/c/h/w": int32  [n_blocks]
//   - dataset  "block_offsets": int64  [n_blocks]
//   - dataset  "block_imps"   : int64  [n_blocks]
//
// Requires HDF5 (HAVE_HDF5 compile definition set by CMake when found).
// Throws std::runtime_error on I/O failure or missing datasets.
#ifdef HAVE_HDF5
template <typename T>
void write_matrix_binsparse(const Matrix<T>& mat, const std::string& path);

template <typename T>
Matrix<T> read_matrix_binsparse(const std::string& path);

extern template void write_matrix_binsparse<float> (const Matrix<float>&,  const std::string&);
extern template void write_matrix_binsparse<double>(const Matrix<double>&, const std::string&);
extern template Matrix<float>  read_matrix_binsparse<float> (const std::string&);
extern template Matrix<double> read_matrix_binsparse<double>(const std::string&);
#endif // HAVE_HDF5

} // namespace benchmark_core
