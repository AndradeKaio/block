import h5py
import numpy as np
import scipy.io
import scipy.sparse as sp
import os
import sys
import warnings

bsp_path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                              else "~/experiments/pattern-mining/data/Norris/lung1/lung1.bsp")
mtx_path = os.path.splitext(bsp_path)[0] + ".mtx"

# ── Load BSP ──────────────────────────────────────────────────────────────────
with h5py.File(bsp_path, "r") as f:
    block_r  = f["block_r"][:]
    block_c  = f["block_c"][:]
    block_h  = f["block_h"][:]
    block_w  = f["block_w"][:]
    offsets  = f["block_offsets"][:]
    values   = f["values"][:]

n_blocks = len(block_r)
print(f"BSP blocks : {n_blocks}")
print(f"BSP values : {len(values):,}  total dense cells")
print(f"BSP nnz    : {np.count_nonzero(values):,}  (stored as non-zero)")

# ── Load original matrix ──────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    A_raw = scipy.io.mmread(mtx_path)
A = sp.csr_matrix(A_raw, dtype=np.float32)
A.eliminate_zeros()
print(f"\nMatrix     : {A.shape}  nnz={A.nnz:,}")

# ── For each non-zero in BSP, check if it exists in the matrix ───────────────
spurious = 0
MAX_REPORT = 20
reported = 0

for bi in range(n_blocks):
    r0  = int(block_r[bi])
    c0  = int(block_c[bi])
    h   = int(block_h[bi])
    w   = int(block_w[bi])
    off = int(offsets[bi])
    blk = values[off:off + h * w].reshape(h, w)

    nz_rows, nz_cols = np.nonzero(blk)
    for ri, ci in zip(nz_rows.tolist(), nz_cols.tolist()):
        row = r0 + ri
        col = c0 + ci
        if row >= A.shape[0] or col >= A.shape[1]:
            val_in_A = 0.0
        else:
            val_in_A = A[row, col]
        if val_in_A == 0.0:
            spurious += 1
            if reported < MAX_REPORT:
                print(f"  spurious non-zero: block {bi} at ({row},{col})"
                      f"  bsp_val={blk[ri,ci]:.4g}  matrix_val={val_in_A:.4g}")
                reported += 1

print(f"\nBSP non-zeros at actual-zero positions: {spurious:,}")
if spurious == 0:
    print("BSP values are consistent with the matrix.")
else:
    expected_spurious = np.count_nonzero(values) - A.nnz
    print(f"Expected from nnz difference: {expected_spurious:,}")
    print("These spurious values feed into the GEMM and produce false C entries.")
