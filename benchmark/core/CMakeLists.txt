cmake_minimum_required(VERSION 3.18) # 3.18+ needed for CMAKE_CUDA_ARCHITECTURES
project(block_pipeline_core LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

find_package(OpenMP)
find_package(HDF5 COMPONENTS C)
if(HDF5_FOUND)
    message(STATUS "HDF5 found — Binsparse I/O enabled")
else()
    message(STATUS "HDF5 not found — write_matrix_binsparse / read_matrix_binsparse unavailable")
endif()

add_library(block_pipeline STATIC
    block.cpp
    segment_tree.cpp
    interval_tree.cpp
    pipeline.cpp
    block_generator.cpp
    block_mining.cpp
    matrix.cpp
    matrix_io.cpp
)
target_include_directories(block_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(OpenMP_CXX_FOUND)
    target_link_libraries(block_pipeline PUBLIC OpenMP::OpenMP_CXX)
endif()

if(HDF5_FOUND)
    target_include_directories(block_pipeline PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(block_pipeline PUBLIC ${HDF5_C_LIBRARIES})
    target_compile_definitions(block_pipeline PUBLIC HAVE_HDF5)
endif()

add_executable(block_pipeline_demo main.cpp)
target_link_libraries(block_pipeline_demo PRIVATE block_pipeline)

enable_testing()
add_executable(block_pipeline_tests tests/pipeline_tests.cpp)
target_link_libraries(block_pipeline_tests PRIVATE block_pipeline)
add_test(NAME pipeline_tests COMMAND block_pipeline_tests)

add_executable(block_matrix_tests tests/matrix_tests.cpp)
target_link_libraries(block_matrix_tests PRIVATE block_pipeline)
add_test(NAME matrix_tests COMMAND block_matrix_tests)

# ── Optional GPU dispatch demo (phase 5) ─────────────────────────────────────
# Only builds when a CUDA toolchain is available, so plain-C++ builds (e.g.
# CI or a CPU-only dev box) stay unaffected.
#
# tc_hybrid_kernel uses nvcuda::wmma precision::tf32 (16×16×8 tile), which is
# only defined in CUDA headers when compiling for sm_80–sm_89 (Ampere / Ada).
# Newer architectures (Hopper sm_90, Blackwell sm_120) use a different tensor-
# core API (wgmma / tcgen05) and the old fragment specialisations are absent
# from CUDA 12.3+ headers for those targets. So we pin to sm_80 unconditionally:
# the resulting PTX binary is JIT-compiled and runs correctly on any newer GPU
# (including Blackwell / RTX 50xx) via CUDA's forward-compat path, at the cost
# of not using native scheduling for those architectures.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 80) # pinned: see comment above

    add_executable(gpu_dispatch_demo gpu_dispatch_demo.cu)
    target_link_libraries(gpu_dispatch_demo PRIVATE block_pipeline)
else()
    message(STATUS "No CUDA compiler found - skipping gpu_dispatch_demo target")
endif()
