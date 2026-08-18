#pragma once
// spmm_gpu_dispatch.cuh — stub redirect.
// Dispatch is implemented in spmm_gpu_kernels.cuh; this file exists only
// because benchmark_spmm_gpu.py copies it into each per-matrix build directory
// as one of _MATRIX_SPECIFIC_FILES.
#include "spmm_gpu_kernels.cuh"
