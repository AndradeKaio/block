#!/bin/bash
# run_gpu_validation.sh — first real-hardware validation pass for Prisma SpMM
# GPU. Nothing in this GPU codebase has ever executed on an actual device
# before this script exists to run it -- it has only ever been compiled
# (nvcc accepted it) and code-reviewed, never run. Run this on a machine
# with a real NVIDIA GPU + working nvcc.
#
# Reuses the ALREADY-BUILT, already-bug-fixed pipeline
# (suite-sparse/benchmark_spmm_gpu.py + suite-sparse/validate_spmm.py) --
# this script is just a thin, auto-detecting driver around them, not new
# logic. Two phases:
#   1. Smoke test: 5 hand-picked matrices spanning the structural extremes
#      this codebase has had to handle (tiny/single-row-group-collapse,
#      small-with-some-TC-eligible-blocks, and bundle1 specifically --
#      6449 blocks, block widths up to 855, mixed TC+CUDA-fallback in one
#      matrix -- the case that drove the row/k-tiling corrections). Fast
#      to iterate on if this surfaces a real bug, which is a real
#      possibility given zero kernels have ever executed before now.
#   2. Full sweep: only run this AFTER the smoke test passes -- the 28 or
#      48-matrix lists this project has already validated on CPU.
#
# Usage:
#   ./run_gpu_validation.sh [path-to-repo-root]
# Defaults to /workspace if no argument given.

set -uo pipefail  # NOT -e: a real correctness/runtime FAILURE from
                  # validate_spmm.py or benchmark_spmm_gpu.py is exactly
                  # the diagnostic information this first run exists to
                  # surface -- the script should keep going and write out
                  # everything it collected, not die on the first one.

REPO_ROOT="${1:-/workspace}"
SUITE_SPARSE_DIR="$REPO_ROOT/suite-sparse"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$SUITE_SPARSE_DIR/gpu_validation_$TS"

echo "=================================================================="
echo "Prisma SpMM GPU -- first real-hardware validation"
echo "Repo root : $REPO_ROOT"
echo "=================================================================="
echo

# --- Prerequisite check (before creating any output dir) ---------------
echo "=== Prerequisite check ==="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "FATAL: nvcc not found on PATH. Install the CUDA toolkit first"
  echo "  (e.g. 'apt-get install nvidia-cuda-toolkit' or the NVIDIA-provided installer)."
  exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi failed -- no GPU detected, or no device passthrough"
  echo "  into this environment (check container --gpus flag / driver install)."
  exit 1
fi
mkdir -p "$OUT_DIR"
echo "Output dir: $OUT_DIR"
nvcc --version
echo
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
echo

# Auto-detect -arch and --cuda-home instead of trusting the scripts' own
# defaults (benchmark_spmm_gpu.py defaults to -arch=sm_120 and
# --cuda-home=/usr/local/cuda, which assume a specific machine -- neither
# is safe to assume here, and getting -arch wrong produces a binary that
# either fails to run on this GPU or silently targets the wrong SM
# version).
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
ARCH="sm_$(echo "$COMPUTE_CAP" | tr -d '.')"
NVCC_PATH="$(command -v nvcc)"
CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
echo "Detected: compute_cap=$COMPUTE_CAP -> -arch=$ARCH"
echo "Detected: --cuda-home=$CUDA_HOME"
echo

cd "$SUITE_SPARSE_DIR"

run_phase() {
  local label="$1"; shift
  echo "=== $label ==="
  echo "+ $*"
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo ">>> $label exited $rc -- see log above/in $OUT_DIR for details. Continuing."
  fi
  echo
  return $rc
}

# --- Phase 1: smoke test -------------------------------------------------
SMOKE_CSV="$SCRIPT_DIR/gpu_smoke_test.csv"

run_phase "Smoke test: compile + timing (benchmark_spmm_gpu.py)" \
  python3 benchmark_spmm_gpu.py "$SMOKE_CSV" \
    --runs 5 --arch "$ARCH" --cuda-home "$CUDA_HOME" \
    --out "$OUT_DIR/smoke_gpu_timing.csv" \
    2>&1 | tee "$OUT_DIR/smoke_benchmark.log"

run_phase "Smoke test: correctness (validate_spmm.py, vs scipy reference)" \
  python3 validate_spmm.py "$SMOKE_CSV" \
    --kernels prisma_cpu,prisma_gpu_cuda_fp64,prisma_gpu_cuda_fp32 \
    --timeout 120 \
    2>&1 | tee "$OUT_DIR/smoke_validate.log"

SMOKE_FAIL=$(grep -c "FAILED\|FAIL " "$OUT_DIR/smoke_validate.log" || true)
echo "=== Smoke test summary: $SMOKE_FAIL FAIL/FAILED lines in smoke_validate.log ==="
echo

if [ "$SMOKE_FAIL" -gt 0 ]; then
  echo "Smoke test found failures -- STOPPING before the full sweep."
  echo "Inspect $OUT_DIR/smoke_validate.log and $OUT_DIR/smoke_benchmark.log first."
  echo "(Re-run this script after a fix; it's safe to re-run, each run gets its own timestamped dir.)"
  exit 1
fi

echo "Smoke test clean. Proceeding to the full sweep."
echo

# --- Phase 2: full sweep (only reached if smoke test was clean) ---------
FULL_CSV="matrices_valid_small.csv"   # the 28-matrix list already used for
                                      # the CPU artifact; swap for
                                      # matrices_no_singles_32k_32k.csv
                                      # (48 matrices) if you want the
                                      # larger set.

run_phase "Full sweep: compile + timing (benchmark_spmm_gpu.py)" \
  python3 benchmark_spmm_gpu.py "$FULL_CSV" \
    --runs 5 --arch "$ARCH" --cuda-home "$CUDA_HOME" \
    --out "$OUT_DIR/full_gpu_timing.csv" \
    2>&1 | tee "$OUT_DIR/full_benchmark.log"

run_phase "Full sweep: correctness (validate_spmm.py, vs scipy reference)" \
  python3 validate_spmm.py "$FULL_CSV" \
    --kernels prisma_cpu,prisma_gpu_cuda_fp64,prisma_gpu_cuda_fp32 \
    --timeout 300 \
    2>&1 | tee "$OUT_DIR/full_validate.log"

echo "=================================================================="
echo "Done. Results in: $OUT_DIR"
echo "  smoke_benchmark.log / smoke_gpu_timing.csv   -- smoke test timing"
echo "  smoke_validate.log                           -- smoke test correctness"
echo "  full_benchmark.log  / full_gpu_timing.csv    -- full sweep timing"
echo "  full_validate.log                            -- full sweep correctness"
echo "=================================================================="

