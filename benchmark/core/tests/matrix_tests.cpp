// Small assertion-based test suite for Matrix<T> (no external test
// framework, to stay consistent with tests/pipeline_tests.cpp).

#include <cstdio>
#include <cstdlib>

#include "block.hpp"
#include "matrix.hpp"

using benchmark_core::assign_offsets;
using benchmark_core::Block;
using benchmark_core::DataType;
using benchmark_core::generate_random_matrix;
using benchmark_core::Matrix;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "CHECK FAILED: %s at %s:%d\n", #cond, __FILE__, \
                          __LINE__);                                             \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace {

Block make_block(int r, int c, int h, int w) {
    Block b;
    b.r = r;
    b.c = c;
    b.h = h;
    b.w = w;
    return b;
}

void test_assign_offsets_contiguous() {
    std::vector<Block> blocks = {
        make_block(0, 0, 2, 3),  // 6 cells
        make_block(5, 5, 4, 4),  // 16 cells
        make_block(9, 9, 1, 5),  // 5 cells
    };

    const long long total = assign_offsets(blocks);
    CHECK(blocks[0].offset == 0);
    CHECK(blocks[1].offset == 6);
    CHECK(blocks[2].offset == 22);
    CHECK(total == 27);
}

void test_generate_random_matrix_shape() {
    auto mat = generate_random_matrix<float>(200, 200, 4, {10, 20}, {10, 20}, 123);

    CHECK(mat.M == 200);
    CHECK(mat.N == 200);
    CHECK(!mat.blocks.empty());

    long long expected = 0;
    for (const auto& b : mat.blocks) {
        expected += b.num_cells();
    }
    CHECK(static_cast<long long>(mat.n_values) == expected);
}

void test_generate_random_matrix_offsets_contiguous() {
    auto mat = generate_random_matrix<float>(200, 200, 4, {10, 20}, {10, 20}, 123);

    long long expected_offset = 0;
    for (const auto& b : mat.blocks) {
        CHECK(b.offset == expected_offset);
        expected_offset += b.num_cells();
    }
}

void test_block_data_slice() {
    auto mat = generate_random_matrix<float>(200, 200, 4, {10, 20}, {10, 20}, 123);
    CHECK(mat.blocks.size() >= 2); // need a non-first block (offset > 0)

    const Block& b = mat.blocks[1];
    CHECK(b.offset > 0);

    auto view = mat.block_data(b);
    CHECK(view.size() == static_cast<std::size_t>(b.num_cells()));
    CHECK(view.data() == mat.values + b.offset);
}

void test_block_data_const_overload() {
    const auto mat = generate_random_matrix<float>(200, 200, 4, {10, 20}, {10, 20}, 123);
    const Block& b = mat.blocks[0];

    std::span<const float> view = mat.block_data(b);
    CHECK(view.size() == static_cast<std::size_t>(b.num_cells()));
    CHECK(view.data() == mat.values + b.offset);
}

void test_matrix_double_instantiation() {
    auto mat = generate_random_matrix<double>(200, 200, 4, {10, 20}, {10, 20}, 123);

    CHECK(mat.M == 200);
    CHECK(mat.N == 200);
    long long expected = 0;
    for (const auto& b : mat.blocks) {
        expected += b.num_cells();
    }
    CHECK(static_cast<long long>(mat.n_values) == expected);
}

void test_dtype_tag() {
    static_assert(Matrix<float>::dtype == DataType::F32);
    static_assert(Matrix<double>::dtype == DataType::F64);
}

void test_copy_semantics() {
    auto original = generate_random_matrix<float>(200, 200, 4, {10, 20}, {10, 20}, 123);
    CHECK(original.n_values > 0);
    const float original_first_value = original.values[0];

    Matrix<float> copy = original; // copy constructor
    CHECK(copy.values != original.values);
    CHECK(copy.n_values == original.n_values);

    copy.values[0] += 1.0f;
    CHECK(original.values[0] == original_first_value); // original untouched
    CHECK(copy.values[0] != original.values[0]);

    Matrix<float> assigned;
    assigned = original; // copy assignment
    CHECK(assigned.values != original.values);
    CHECK(assigned.n_values == original.n_values);
    CHECK(assigned.values[0] == original.values[0]);
}

void test_move_semantics() {
    auto src = generate_random_matrix<float>(200, 200, 4, {10, 20}, {10, 20}, 123);
    float* original_ptr = src.values;
    const std::size_t original_n = src.n_values;
    CHECK(original_n > 0);

    Matrix<float> dst = std::move(src); // move constructor
    CHECK(src.values == nullptr);
    CHECK(src.n_values == 0);
    CHECK(dst.values == original_ptr); // buffer was stolen, not reallocated
    CHECK(dst.n_values == original_n);

    Matrix<float> assigned;
    assigned = std::move(dst); // move assignment
    CHECK(dst.values == nullptr);
    CHECK(dst.n_values == 0);
    CHECK(assigned.values == original_ptr);
    CHECK(assigned.n_values == original_n);
}

} // namespace

int main() {
    test_assign_offsets_contiguous();
    test_generate_random_matrix_shape();
    test_generate_random_matrix_offsets_contiguous();
    test_block_data_slice();
    test_block_data_const_overload();
    test_matrix_double_instantiation();
    test_dtype_tag();
    test_copy_semantics();
    test_move_semantics();

    std::printf("All tests passed.\n");
    return 0;
}
