#!/bin/bash
# docker-entrypoint.sh — mine the matrices named in $MATRICES_CSV, then run all
# three CPU benchmarks (SpGEMM/SpMM/SpMV) against that same matrix list, each
# swept from nproc down to 1 threads by halving.
#
# Env vars (all optional, defaults shown):
#   MATRICES_CSV=/input/matrices.csv   bind-mount the user's CSV here (needs
#                                       'name' + 'group' columns)
#   WORKERS=4                          mine_matrices.py download/mine parallelism
#   RUNS=5                             timed repetitions per matrix per thread count
#   ENABLE_PERF=0                      set to 1 to also collect --perf hardware
#                                       counters (needs --cap-add=SYS_ADMIN or a
#                                       relaxed perf_event_paranoid on the host)
set -euo pipefail

: "${MATRICES_CSV:=/input/matrices.csv}"
: "${WORKERS:=4}"
: "${RUNS:=5}"
: "${ENABLE_PERF:=0}"

DATA_ROOT="/home/kaio/datasets/suite-sparse"
RESULTS_DIR="/results"
WORK_DIR="/work"

if [ ! -f "$MATRICES_CSV" ]; then
  echo "ERROR: $MATRICES_CSV not found." >&2
  echo "Mount your matrices.csv there, e.g.:" >&2
  echo "  docker run -v \$(pwd)/matrices.csv:/input/matrices.csv:ro ..." >&2
  exit 1
fi

mkdir -p "$DATA_ROOT" "$RESULTS_DIR" "$WORK_DIR"

# mine_matrices.py writes resolved SuiteSparse ids back into the CSV it's
# given -- copy to a writable path so a read-only bind mount of the user's
# original file still works, and so both the mining step and all three
# benchmark scripts share one consistent, already-resolved matrix list.
cp "$MATRICES_CSV" "$WORK_DIR/matrices.csv"

echo "== Mining matrices (workers=$WORKERS) =="
python mining/mine_matrices.py \
  --matrices-csv "$WORK_DIR/matrices.csv" \
  --output-dir "$DATA_ROOT" \
  --workers "$WORKERS"

PERF_FLAG=()
if [ "$ENABLE_PERF" = "1" ]; then
  PERF_FLAG=(--perf)
fi

THREADS="$(nproc)"
echo "== Running benchmarks (threads-sweep=$THREADS, runs=$RUNS, perf=$ENABLE_PERF) =="

python benchmark_spgemm_cpu.py "$WORK_DIR/matrices.csv" \
  --out "$RESULTS_DIR/spgemm_results.csv" \
  --runs "$RUNS" --threads-sweep "$THREADS" "${PERF_FLAG[@]}"

python benchmark_spmm_cpu.py "$WORK_DIR/matrices.csv" \
  --out "$RESULTS_DIR/spmm_results.csv" \
  --runs "$RUNS" --threads-sweep "$THREADS" "${PERF_FLAG[@]}"

python benchmark_spmv_cpu.py "$WORK_DIR/matrices.csv" \
  --out "$RESULTS_DIR/spmv_results.csv" \
  --runs "$RUNS" --threads-sweep "$THREADS" "${PERF_FLAG[@]}"

echo "== Done. Results in $RESULTS_DIR =="
