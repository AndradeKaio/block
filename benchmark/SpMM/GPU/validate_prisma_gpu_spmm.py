#!/usr/bin/env python3
"""
validate_prisma_gpu_spmm.py — Standalone correctness check for prisma_gpu_spmm_bench.

Runs the binary with --dump-c and --dump-d, then compares the GPU output against
scipy's C_ref = S @ D computed in float64.

Usage:
  python validate_prisma_gpu_spmm.py /tmp/prisma_gpu_spmm_bench matrix.bsp
  python validate_prisma_gpu_spmm.py /tmp/prisma_gpu_spmm_bench matrix.bsp --precision fp32
  python validate_prisma_gpu_spmm.py /tmp/prisma_gpu_spmm_bench matrix.bsp --force-cuda-fallback
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse


def load_bsp(bsp: Path) -> scipy.sparse.csr_matrix:
    with h5py.File(str(bsp), "r") as f:
        M    = int(f.attrs["matrix_rows"])
        N    = int(f.attrs["matrix_cols"])
        br   = f["block_r"][:]
        bc   = f["block_c"][:]
        bh   = f["block_h"][:]
        bw   = f["block_w"][:]
        bo   = f["block_offsets"][:]
        vals = f["values"][:].astype(np.float64)

    rows, cols, data = [], [], []
    for k in range(len(br)):
        r, c, h, w, off = int(br[k]), int(bc[k]), int(bh[k]), int(bw[k]), int(bo[k])
        blk = vals[off: off + h * w].reshape(h, w)
        ri, ci = np.nonzero(blk)
        rows.append(ri + r)
        cols.append(ci + c)
        data.append(blk[ri, ci])

    if rows:
        r_cat = np.concatenate(rows)
        c_cat = np.concatenate(cols)
        d_cat = np.concatenate(data)
    else:
        r_cat = c_cat = d_cat = np.array([], dtype=np.float64)

    return scipy.sparse.csr_matrix((d_cat, (r_cat, c_cat)), shape=(M, N))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary",    help="path to prisma_gpu_spmm_bench binary")
    ap.add_argument("bsp",       help="path to .bsp sparse matrix file")
    ap.add_argument("--precision", default="fp64", choices=["fp32", "fp64"])
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--runs",    type=int, default=1)
    ap.add_argument("--force-cuda-fallback", action="store_true")
    ap.add_argument("--rtol",    type=float, default=1e-4,
                    help="relative tolerance (default 1e-4; fp32 or TC may need 1e-2)")
    ap.add_argument("--atol",    type=float, default=1e-6)
    args = ap.parse_args()

    binary = Path(args.binary)
    bsp    = Path(args.bsp)

    if not binary.exists():
        sys.exit(f"binary not found: {binary}")
    if not bsp.exists():
        sys.exit(f".bsp not found: {bsp}")

    print(f"Loading S from {bsp} …")
    S = load_bsp(bsp)
    M, N = S.shape
    print(f"  S: {M}×{N}  nnz={S.nnz}")

    with tempfile.TemporaryDirectory(prefix="validate_prisma_gpu_") as tmp_str:
        tmp      = Path(tmp_str)
        d_path   = tmp / "D.bin"
        c_path   = tmp / "C.bin"

        cmd = [
            str(binary), str(bsp),
            "--runs",      str(args.runs),
            "--seed",      str(args.seed),
            "--precision", args.precision,
            "--dump-d",    str(d_path),
            "--dump-c",    str(c_path),
        ]
        if args.force_cuda_fallback:
            cmd.append("--force-cuda-fallback")

        print(f"Running: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=False)
        if r.returncode != 0:
            sys.exit(f"binary exited {r.returncode}")

        if not d_path.exists():
            sys.exit("binary did not write --dump-d file")
        if not c_path.exists():
            sys.exit("binary did not write --dump-c file")

        D_raw = np.fromfile(str(d_path), dtype=np.float64)
        if D_raw.size != N * N:
            sys.exit(f"D has {D_raw.size} elements, expected {N}×{N}={N*N}")
        D = D_raw.reshape(N, N)

        C_raw = np.fromfile(str(c_path), dtype=np.float64)
        if C_raw.size != M * N:
            sys.exit(f"C has {C_raw.size} elements, expected {M}×{N}={M*N}")
        C_gpu = C_raw.reshape(M, N)

    print("Computing scipy reference C_ref = S @ D …")
    C_ref = (S @ D)   # scipy returns np.matrix or ndarray; keep as-is
    C_ref = np.asarray(C_ref)

    # For fp32 precision or TC paths, anchor tolerance to the global magnitude
    # rather than per-cell (cancelled-to-near-zero cells inflate relative error).
    use_global_scale = (args.precision == "fp32") or (not args.force_cuda_fallback)
    rtol = args.rtol
    atol = args.atol

    diff  = np.abs(C_gpu - C_ref)
    if use_global_scale:
        scale = atol + rtol * float(np.abs(C_ref).max())
    else:
        scale = atol + rtol * np.abs(C_ref)

    fail_mask    = diff > scale
    n_fail       = int(fail_mask.sum())
    max_abs_err  = float(diff.max())
    max_rel_err  = float((diff / (np.abs(C_ref) + 1e-300)).max())

    print()
    print(f"  precision       : {args.precision}")
    print(f"  force_cuda      : {args.force_cuda_fallback}")
    print(f"  tolerance       : rtol={rtol}  atol={atol}  "
          f"{'global-scale' if use_global_scale else 'per-cell'}")
    print(f"  max abs error   : {max_abs_err:.3e}")
    print(f"  max rel error   : {max_rel_err:.3e}")
    print(f"  failures        : {n_fail} / {C_gpu.size}")
    print()

    if n_fail == 0:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        # Print a few of the worst offenders to help diagnose.
        flat_idx = np.argsort(diff.ravel())[::-1][:5]
        for idx in flat_idx:
            row, col = divmod(int(idx), N)
            print(f"  worst [{row},{col}]: gpu={C_gpu[row,col]:.6g}  "
                  f"ref={C_ref[row,col]:.6g}  diff={diff[row,col]:.3e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
