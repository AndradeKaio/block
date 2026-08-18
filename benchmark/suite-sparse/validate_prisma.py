#!/usr/bin/env python3
"""
Validate PRISMA SpGEMM result against a scipy reference.

Usage:
  python validate_prisma.py <A.bsp> [B.bsp]
      [--prisma-bin PATH] [--runs N] [--tol FLOAT] [--tc-kernel tile|block]

For A×A squaring, omit B.bsp (or pass the same path twice).
The .mtx file must exist alongside the .bsp file with the same stem.

Examples:
  python validate_prisma.py data/pkustk06/pkustk06.bsp
  python validate_prisma.py data/crankseg_2/crankseg_2.bsp --tol 1e-2
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


# ── helpers ──────────────────────────────────────────────────────────────────

def find_prisma_bin(hint: str | None) -> Path:
    candidates = []
    if hint:
        candidates.append(Path(hint))
    # common build locations relative to this script
    root = Path(__file__).resolve().parent.parent.parent
    candidates += [
        root / "SpGEMM" / "GPU" / "prisma_bench",
        root / "build" / "prisma_bench",
        Path("prisma_bench"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "prisma_bench binary not found. "
        "Build it or pass --prisma-bin PATH."
    )


def load_matrix(mtx_path: Path) -> sp.csr_matrix:
    """Read a Matrix Market file as a real CSR matrix."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = scipy.io.mmread(str(mtx_path))
    A = sp.csr_matrix(A, dtype=np.float32)
    A.eliminate_zeros()
    return A


def load_coo_file(path: Path):
    """Load a prisma_bench --validate output file → (rows, cols, vals)."""
    rows, cols, vals = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            rows.append(int(parts[0]))
            cols.append(int(parts[1]))
            vals.append(float(parts[2]))
    return (np.array(rows, dtype=np.int32),
            np.array(cols, dtype=np.int32),
            np.array(vals, dtype=np.float32))


def run_prisma(prisma_bin: Path, a_bsp: Path, b_bsp: Path,
               validate_path: Path, runs: int, tc_kernel: str) -> None:
    cmd = [str(prisma_bin), str(a_bsp), str(b_bsp),
           "--runs", str(runs),
           "--validate", str(validate_path)]
    if tc_kernel:
        cmd += ["--tc-kernel", tc_kernel]
    result = subprocess.run(cmd, capture_output=True)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        print("prisma_bench stderr:", stderr[:2000], file=sys.stderr)
        raise RuntimeError("prisma_bench exited with non-zero status")
    # Echo prisma output so the user can see timing/progress
    for line in stdout.splitlines():
        print(" ", line)


def compare(prisma_coo: Path, ref: sp.csr_matrix, tol: float) -> bool:
    rows_p, cols_p, vals_p = load_coo_file(prisma_coo)

    # Reference: float32 COO, duplicates summed, zeros removed
    ref32 = ref.astype(np.float32)
    ref32.sum_duplicates()
    ref32.eliminate_zeros()
    ref_coo = ref32.tocoo()

    # Sort both by (row, col) for aligned comparison
    order_p = np.lexsort((cols_p, rows_p))
    rows_p = rows_p[order_p]; cols_p = cols_p[order_p]; vals_p = vals_p[order_p]

    order_r = np.lexsort((ref_coo.col, ref_coo.row))
    rows_r = ref_coo.row[order_r]
    cols_r = ref_coo.col[order_r]
    vals_r = ref_coo.data[order_r]

    nnz_p = len(rows_p)
    nnz_r = len(rows_r)
    print(f"  PRISMA nnz : {nnz_p:,}")
    print(f"  scipy  nnz : {nnz_r:,}")

    # Check sparsity pattern
    pat_p = set(zip(rows_p.tolist(), cols_p.tolist()))
    pat_r = set(zip(rows_r.tolist(), cols_r.tolist()))

    only_prisma = pat_p - pat_r
    only_ref    = pat_r - pat_p
    ok = True

    if only_prisma:
        print(f"  FAIL: {len(only_prisma):,} (row,col) entries in PRISMA not in scipy:")
        for r, c in sorted(only_prisma)[:5]:
            idx = np.searchsorted(rows_p * (cols_p.max() + 1) + cols_p,
                                  r * (cols_p.max() + 1) + c)
            print(f"    ({r},{c}) = {vals_p[idx]:.6e}")
        ok = False

    if only_ref:
        print(f"  FAIL: {len(only_ref):,} (row,col) entries in scipy not in PRISMA:")
        for r, c in sorted(only_ref)[:5]:
            idx = np.searchsorted(rows_r * (cols_r.max() + 1) + cols_r,
                                  r * (cols_r.max() + 1) + c)
            print(f"    ({r},{c}) = {vals_r[idx]:.6e}")
        ok = False

    if not ok:
        return False

    # Same pattern — compare values element-wise (both sorted identically)
    denom = np.maximum(np.abs(vals_r), 1e-6)
    rel_err = np.abs(vals_p - vals_r) / denom
    abs_err = np.abs(vals_p - vals_r)

    max_rel = float(rel_err.max())
    max_abs = float(abs_err.max())
    mean_rel = float(rel_err.mean())
    print(f"  max_rel_err  : {max_rel:.3e}")
    print(f"  mean_rel_err : {mean_rel:.3e}")
    print(f"  max_abs_err  : {max_abs:.3e}")

    if max_rel > tol:
        worst = np.argsort(rel_err)[-5:][::-1]
        print(f"  FAIL: max relative error {max_rel:.3e} exceeds tolerance {tol:.1e}")
        print("  Worst mismatches:")
        for i in worst:
            print(f"    ({rows_p[i]},{cols_p[i]})"
                  f"  prisma={vals_p[i]:.6e}  scipy={vals_r[i]:.6e}"
                  f"  rel={rel_err[i]:.3e}")
        return False

    print(f"  PASS  (tol={tol:.1e})")
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Validate PRISMA SpGEMM result against scipy reference.")
    ap.add_argument("a_bsp", type=Path, help="Path to A.bsp")
    ap.add_argument("b_bsp", type=Path, nargs="?", default=None,
                    help="Path to B.bsp (default: same as A.bsp for A×A)")
    ap.add_argument("--prisma-bin", default=None,
                    help="Path to prisma_bench binary")
    ap.add_argument("--runs", type=int, default=1,
                    help="Number of timed runs passed to prisma_bench (default 1)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="Max relative error tolerance (default 1e-3)")
    ap.add_argument("--tc-kernel", default="",
                    help="TC kernel variant: tile, block, or empty (CUDA-only)")
    args = ap.parse_args()

    a_bsp: Path = args.a_bsp.resolve()
    b_bsp: Path = (args.b_bsp.resolve() if args.b_bsp else a_bsp)
    square = (a_bsp == b_bsp)

    for p in (a_bsp, b_bsp):
        if not p.is_file():
            sys.exit(f"Error: .bsp not found: {p}")

    # Locate .mtx files (same directory, same stem)
    a_mtx = a_bsp.with_suffix(".mtx")
    b_mtx = b_bsp.with_suffix(".mtx")
    for p in ({a_mtx} | ({b_mtx} if not square else set())):
        if not p.is_file():
            sys.exit(f"Error: .mtx not found alongside .bsp: {p}")

    prisma_bin = find_prisma_bin(args.prisma_bin)
    print(f"prisma_bench : {prisma_bin}")
    print(f"A.bsp        : {a_bsp}")
    print(f"B.bsp        : {b_bsp}")
    print(f"mode         : {'A×A squaring' if square else 'A×B'}")
    print(f"tolerance    : {args.tol:.1e}")

    # Compute scipy reference
    print("\nLoading matrices …")
    A_ref = load_matrix(a_mtx)
    B_ref = load_matrix(b_mtx) if not square else A_ref
    print(f"  A: {A_ref.shape}  nnz={A_ref.nnz:,}")
    if not square:
        print(f"  B: {B_ref.shape}  nnz={B_ref.nnz:,}")

    # PRISMA's BSP mining ignores actual matrix values and stores 1.0 for every
    # non-zero (mine_matrix.cpp is pattern-only).  Normalise the reference to
    # all-1.0 so the comparison is apples-to-apples.
    A_ref.data[:] = 1.0
    if not square:
        B_ref.data[:] = 1.0

    print("Computing scipy reference C = A @ B (float32, all values = 1.0) …")
    C_ref = (A_ref @ B_ref).astype(np.float32)
    C_ref.eliminate_zeros()
    print(f"  C: {C_ref.shape}  nnz={C_ref.nnz:,}")

    # Run prisma_bench --validate
    with tempfile.NamedTemporaryFile(suffix=".coo", delete=False) as tf:
        coo_path = Path(tf.name)

    print(f"\nRunning prisma_bench --validate {coo_path} …")
    run_prisma(prisma_bin, a_bsp, b_bsp, coo_path,
               args.runs, args.tc_kernel)

    if not coo_path.is_file() or coo_path.stat().st_size == 0:
        sys.exit("Error: prisma_bench produced no validate output")

    print("\nComparing …")
    passed = compare(coo_path, C_ref, args.tol)
    coo_path.unlink(missing_ok=True)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
