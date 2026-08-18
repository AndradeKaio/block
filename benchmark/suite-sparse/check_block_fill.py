#!/usr/bin/env python3
"""Check how much of Prisma's block-decomposition FLOPs are spent on
literal zero cells (block fill) vs true nonzeros, for a given .bsp.

Block-sparse mining intentionally snaps irregular sparsity into regular
dense rectangles to get vectorizable/tensor-core-friendly compute -- the
tradeoff is computing on some zero cells that fall inside a block's
rectangle but aren't real nonzeros (tracked per-block as `imperfections`,
see core/block.hpp). A high fill ratio here directly explains a slowdown
vs a canonical CSR SpMM (cuSPARSE), which touches only true nonzeros and
never wastes FLOPs on zeros -- independent of any GPU-kernel-level
tiling/launch-overhead question.

Usage: python3 check_block_fill.py <name> <bsp_path>
"""
import sys
import h5py
import numpy as np

name, bsp_path = sys.argv[1], sys.argv[2]

with h5py.File(bsp_path, "r") as f:
    M = int(f.attrs["matrix_rows"])
    N = int(f.attrs["matrix_cols"])
    br = f["block_r"][:]
    bh = f["block_h"][:]
    bw = f["block_w"][:]
    bimp = f["block_imps"][:]

total_cells = int(np.sum(bh.astype(np.int64) * bw.astype(np.int64)))
total_imperfections = int(np.sum(bimp))
total_nonzeros = total_cells - total_imperfections
n_blocks = len(bh)

fill_ratio = total_cells / total_nonzeros if total_nonzeros else float("inf")

print(f"[{name}] M={M} N={N} n_blocks={n_blocks}")
print(f"  total block cells (Prisma's nominal FLOP-proportional work): {total_cells:,}")
print(f"  true nonzeros (imperfections subtracted):                    {total_nonzeros:,}")
print(f"  fill ratio (cells / true nonzeros):                          {fill_ratio:.2f}x")
print(f"  block h range: {bh.min()}..{bh.max()}   block w range: {bw.min()}..{bw.max()}")

# Row-overlap depth: how many blocks cover a given row, on average/at max
# -- high overlap means MULTIPLE blocks independently atomicAdd into the
# same C cells, each redoing a full K-contraction over their own width,
# which is additional real (not wasted-on-zeros) work a canonical CSR
# SpMM structurally can't have (each row visited once).
overlap = np.zeros(M, dtype=np.int32)
for r0, h in zip(br, bh):
    overlap[r0:r0 + h] += 1
print(f"  row overlap depth: mean={overlap.mean():.2f}  max={overlap.max()}  "
      f"rows with overlap>1: {(overlap > 1).sum()}/{M} ({100*(overlap>1).mean():.1f}%)")
