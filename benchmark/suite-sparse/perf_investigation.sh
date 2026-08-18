#!/bin/bash
cd "$(dirname "$0")/.."

# 1. n-tile sweep (no code change needed, flag already exists in the binary)
BSP=/home/kaio/datasets/suite-sparse/DNVS/thread/thread.bsp
for nt in 256 512 1024 2048 4096 8192; do
  echo "--n-tile $nt:"
  /tmp/_prismac/prisma_gpu_spmm_bench_thread "$BSP" --runs 3 --seed 42 --n-tile "$nt" --precision fp64 2>&1 | grep -E "plan:|runs="
done

echo
echo "=== block fill / overlap ==="
for m in pkustk01 pkustk07 pkustk08 raefsky4 thread; do
  bsp=$(find /home/kaio/datasets/suite-sparse -iname "${m}.bsp" | head -1)
  python3 suite-sparse/check_block_fill.py "$m" "$bsp"
done
