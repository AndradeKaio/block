#include "matrix_io.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

namespace benchmark_core {

// ── Binsparse / HDF5 ─────────────────────────────────────────────────────────

#ifdef HAVE_HDF5

namespace {

template <typename C>
void write_dataset(hid_t file, const char* name, hid_t h5type,
                   const C* data, hsize_t count) {
    hid_t space = H5Screate_simple(1, &count, nullptr);
    hid_t ds    = H5Dcreate2(file, name, h5type, space,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (ds < 0) throw std::runtime_error(std::string("HDF5: cannot create dataset ") + name);
    H5Dwrite(ds, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Sclose(space);
}

template <typename C>
void read_dataset(hid_t file, const char* name, hid_t h5type, std::vector<C>& out) {
    hid_t ds    = H5Dopen2(file, name, H5P_DEFAULT);
    if (ds < 0) throw std::runtime_error(std::string("HDF5: cannot open dataset ") + name);
    hid_t space = H5Dget_space(ds);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    out.resize(dims[0]);
    H5Dread(ds, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data());
    H5Sclose(space);
    H5Dclose(ds);
}

void write_int_attr(hid_t file, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(file, name, H5T_NATIVE_INT32, space,
                              H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT32, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

int read_int_attr(hid_t file, const char* name) {
    hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
    if (attr < 0) throw std::runtime_error(std::string("HDF5: missing attribute ") + name);
    int value = 0;
    H5Aread(attr, H5T_NATIVE_INT32, &value);
    H5Aclose(attr);
    return value;
}

void write_binsparse_attr(hid_t file, int M, int N, std::size_t n_values,
                          const char* val_type) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "{"
        "\"binsparse\":{"
        "\"version\":\"0.1\","
        "\"format\":\"CSP\","
        "\"shape\":[%d,%d],"
        "\"number_of_stored_values\":%zu,"
        "\"data_types\":{"
        "\"values\":\"%s\","
        "\"block_r\":\"int32\","
        "\"block_c\":\"int32\","
        "\"block_h\":\"int32\","
        "\"block_w\":\"int32\","
        "\"block_offsets\":\"int64\","
        "\"block_imps\":\"int64\""
        "}"
        "}"
        "}",
        M, N, n_values, val_type);

    hid_t strtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype, H5T_VARIABLE);
    H5Tset_cset(strtype, H5T_CSET_UTF8);

    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr  = H5Acreate2(file, "binsparse", strtype, space,
                              H5P_DEFAULT, H5P_DEFAULT);
    const char* ptr = buf;
    H5Awrite(attr, strtype, &ptr);
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(strtype);
}

} // anonymous namespace

template <typename T>
void write_matrix_binsparse(const Matrix<T>& mat, const std::string& path) {
    const hsize_t nb = mat.blocks.size();
    const hsize_t nv = mat.n_values;

    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
        throw std::runtime_error("write_matrix_binsparse: cannot create '" + path + "'");

    const char* val_type = (sizeof(T) == 4) ? "float32" : "float64";
    hid_t h5val = (sizeof(T) == 4) ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;

    write_binsparse_attr(file, mat.M, mat.N, nv, val_type);
    write_int_attr(file, "matrix_rows", mat.M);
    write_int_attr(file, "matrix_cols", mat.N);

    write_dataset(file, "values", h5val, mat.values, nv);

    std::vector<int32_t> br(nb), bc(nb), bh(nb), bw(nb);
    std::vector<int64_t> bo(nb), bi(nb);
    for (std::size_t i = 0; i < nb; ++i) {
        br[i] = mat.blocks[i].r;
        bc[i] = mat.blocks[i].c;
        bh[i] = mat.blocks[i].h;
        bw[i] = mat.blocks[i].w;
        bo[i] = mat.blocks[i].offset;
        bi[i] = mat.blocks[i].imperfections;
    }
    write_dataset(file, "block_r",       H5T_NATIVE_INT32, br.data(), nb);
    write_dataset(file, "block_c",       H5T_NATIVE_INT32, bc.data(), nb);
    write_dataset(file, "block_h",       H5T_NATIVE_INT32, bh.data(), nb);
    write_dataset(file, "block_w",       H5T_NATIVE_INT32, bw.data(), nb);
    write_dataset(file, "block_offsets", H5T_NATIVE_INT64, bo.data(), nb);
    write_dataset(file, "block_imps",    H5T_NATIVE_INT64, bi.data(), nb);

    H5Fclose(file);
}

template <typename T>
Matrix<T> read_matrix_binsparse(const std::string& path) {
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
        throw std::runtime_error("read_matrix_binsparse: cannot open '" + path + "'");

    Matrix<T> mat;
    mat.M = read_int_attr(file, "matrix_rows");
    mat.N = read_int_attr(file, "matrix_cols");

    hid_t h5val = (sizeof(T) == 4) ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;

    std::vector<T>       vals;
    std::vector<int32_t> br, bc, bh, bw;
    std::vector<int64_t> bo, bi;

    read_dataset(file, "values",        h5val,            vals);
    read_dataset(file, "block_r",       H5T_NATIVE_INT32, br);
    read_dataset(file, "block_c",       H5T_NATIVE_INT32, bc);
    read_dataset(file, "block_h",       H5T_NATIVE_INT32, bh);
    read_dataset(file, "block_w",       H5T_NATIVE_INT32, bw);
    read_dataset(file, "block_offsets", H5T_NATIVE_INT64, bo);
    read_dataset(file, "block_imps",    H5T_NATIVE_INT64, bi);

    H5Fclose(file);

    const std::size_t nb = br.size();
    mat.blocks.resize(nb);
    for (std::size_t i = 0; i < nb; ++i) {
        mat.blocks[i].r             = br[i];
        mat.blocks[i].c             = bc[i];
        mat.blocks[i].h             = bh[i];
        mat.blocks[i].w             = bw[i];
        mat.blocks[i].offset        = bo[i];
        mat.blocks[i].imperfections = bi[i];
    }

    mat.n_values = vals.size();
    mat.values   = new T[mat.n_values];
    std::memcpy(mat.values, vals.data(), mat.n_values * sizeof(T));

    return mat;
}

template void write_matrix_binsparse<float> (const Matrix<float>&,  const std::string&);
template void write_matrix_binsparse<double>(const Matrix<double>&, const std::string&);
template Matrix<float>  read_matrix_binsparse<float> (const std::string&);
template Matrix<double> read_matrix_binsparse<double>(const std::string&);

#endif // HAVE_HDF5

} // namespace benchmark_core
