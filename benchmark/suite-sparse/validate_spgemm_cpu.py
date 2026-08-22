#!/usr/bin/env python3
"""
suite-sparse/validate_spgemm_cpu.py — Correctness validation for CPU SpGEMM
contenders (TACO + Prisma).

For each matrix, C = A @ A (square self-product) is computed by every
contender via its own dump flag (--dump-c for bench_taco.c, --validate for
prisma_cpu_bench.cpp -- both write bare "row col val" COO text, no header)
and compared against a scipy reference.

Both contenders are full double precision now: taco_cpu/taco_cpu_opt
(bench_taco.c's Bv/Cv/A_t->vals are all `double`) and prisma_generic/
prisma_top10 (prisma_cpu_bench.cpp's Scalar -- was float, switched to
double; see that file's comment and gen_spgemm_kernels.py's matching
_pd-intrinsics rewrite). mine_matrix.cpp also writes .bsp values as
double (Matrix<double> -> write_matrix_binsparse<double>, see
core/matrix_io.cpp), so there's no meaningful precision gap left between
TACO's C_ref_mtx = S_mtx @ S_mtx and Prisma's C_ref_bsp = S_bsp @ S_bsp
either -- a single tight tolerance (default 1e-6/1e-6) applies everywhere,
same as validate_spmm_cpu.py.

A direct S_bsp-vs-S_mtx check (independent of any compute kernel) runs
first, same rationale as validate_spmm_cpu.py's: it has exactly one
correct answer, catches a corrupted .bsp directly instead of only
inferring it from downstream C differences.

bench_taco.c's --dump-c flag was added specifically to close this
validator's original TACO gap -- see its own comment for the exact format
contract with prisma_cpu_bench's --validate output.

Usage:
  python validate_spgemm_cpu.py MATRICES.csv
  python validate_spgemm_cpu.py MATRICES.csv --bin-dir /tmp/spgemm_bins --no-compile
  python validate_spgemm_cpu.py MATRICES.csv --kernels prisma_generic
"""

import argparse
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

try:
    import h5py
    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from benchmark_spgemm_cpu import (  # noqa: E402
    _TMP_DIR,
    compile_prisma_spgemm,
    compile_prisma_spgemm_per_matrix,
    compile_taco_cpu,
    analyze_spgemm_shapes,
    ensure_real_general,
    gen_spgemm_kernels_files,
    find_mtx,
    load_matrix_list,
)

# ---------------------------------------------------------------------------
# Contenders — CPU SpGEMM: TACO (.mtx) + Prisma (.bsp), both double precision
# ---------------------------------------------------------------------------

TACO_CONTENDERS = ["taco_cpu", "taco_cpu_opt"]
CONTENDERS = ["prisma_generic", "prisma_top10"]
_ALL_LABELS = TACO_CONTENDERS + CONTENDERS

_DEFAULT_MATRICES = [
    {"name": "bundle1",   "group": "Janna", "rows": "10294", "cols": "10294", "nnz": "1000000"},
    {"name": "bcsstk27",  "group": "HB",    "rows": "1224",  "cols": "1224",  "nnz": "56126"},
    {"name": "linverse",  "group": "Bova",  "rows": "11999", "cols": "11999", "nnz": "9921"},
]


def _load_bsp_as_csr(bsp: Path) -> scipy.sparse.csr_matrix | None:
    """Load a .bsp (HDF5 block-sparse) file into a scipy CSR matrix, upcast
    to float64. Same schema/logic as validate_spmm_cpu.py's _load_bsp_as_csr
    -- duplicated (not imported) to keep the SpMM/SpGEMM domains independent,
    matching benchmark_spgemm_cpu.py's own existing self-containment."""
    if not _HAVE_H5PY:
        return None
    with h5py.File(str(bsp), "r") as f:
        M   = int(f.attrs["matrix_rows"])
        N   = int(f.attrs["matrix_cols"])
        br  = f["block_r"][:]
        bc  = f["block_c"][:]
        bh  = f["block_h"][:]
        bw  = f["block_w"][:]
        bo  = f["block_offsets"][:]
        vals = f["values"][:].astype(np.float64)

    rows_list, cols_list, data_list = [], [], []
    for k in range(len(br)):
        r, c, h, w, off = int(br[k]), int(bc[k]), int(bh[k]), int(bw[k]), int(bo[k])
        block = vals[off: off + h * w].reshape(h, w)
        ri, ci = np.nonzero(block)
        rows_list.append(ri + r)
        cols_list.append(ci + c)
        data_list.append(block[ri, ci])

    if rows_list:
        all_r = np.concatenate(rows_list)
        all_c = np.concatenate(cols_list)
        all_d = np.concatenate(data_list)
    else:
        all_r = all_c = all_d = np.array([], dtype=np.float64)

    return scipy.sparse.csr_matrix((all_d, (all_r, all_c)), shape=(M, N), dtype=np.float64)


def _load_coo(path: Path, M: int, N: int) -> np.ndarray:
    """Read prisma_cpu_bench's --validate / bench_taco.c's --dump-c output:
    bare 'row col val' lines (COO, no header) → dense float64 array."""
    arr = np.zeros((M, N), dtype=np.float64)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                arr[int(parts[0]), int(parts[1])] = float(parts[2])
    return arr


def _compare(C: np.ndarray, C_ref: np.ndarray,
             rtol: float, atol: float) -> tuple[bool, float, float, int]:
    diff  = np.abs(C - C_ref)
    scale = atol + rtol * np.abs(C_ref)
    mask  = diff > scale
    return (
        not mask.any(),
        float(diff.max()),
        float((diff / (np.abs(C_ref) + 1e-300)).max()),
        int(mask.sum()),
    )


def _top_failures(C: np.ndarray, C_ref: np.ndarray, rtol: float, atol: float,
                  limit: int = 5) -> list[tuple[int, int, float, float]]:
    """Return up to `limit` (row, col, got, ref) tuples for the worst-by-
    absolute-diff failing cells -- lets a FAIL be traced to specific output
    coordinates instead of only a count, e.g. to cross-reference against
    prisma_bench.cu's --dump-plan output and find which compute region
    produced a given wrong cell."""
    diff  = np.abs(C - C_ref)
    scale = atol + rtol * np.abs(C_ref)
    rows, cols = np.nonzero(diff > scale)
    if rows.size == 0:
        return []
    order = np.argsort(-diff[rows, cols])[:limit]
    return [(int(rows[i]), int(cols[i]), float(C[rows[i], cols[i]]),
             float(C_ref[rows[i], cols[i]])) for i in order]


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------


def validate_matrix(
    row: dict, mtx: Path,
    taco_bin: Path | None, taco_opt_bin: Path | None, prisma_bin: Path | None,
    active: list, top_n: int, blas: bool, rtol: float, atol: float,
    timeout: int, tmp: Path, work_dir: Path, mtx_cache: Path,
    merge_strategy: str = "sequential",
) -> bool:
    name = row["name"]
    bsp  = mtx.with_suffix(".bsp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_mtx = scipy.io.mmread(str(mtx)).tocsr().astype(np.float64)
    M, N = S_mtx.shape
    C_ref_mtx = np.asarray((S_mtx @ S_mtx).toarray())

    all_pass = True
    results: dict[str, np.ndarray] = {}

    # --- S_bsp vs S_mtx: are they the SAME matrix? --------------------------
    # Checked directly, independent of any compute kernel -- see
    # validate_spmm_cpu.py's identical check for the full rationale (catches
    # a corrupted .bsp directly instead of only inferring it from downstream
    # C differences). Tight tolerance throughout: mine_matrix.cpp writes
    # .bsp values as double (Matrix<double>), so there's no float32-
    # truncation gap to allow for here the way SpMM's check needs.
    if bsp.exists():
        S_bsp = _load_bsp_as_csr(bsp)
        if S_bsp is not None:
            S_diff = np.abs((S_bsp - S_mtx).toarray())
            S_scale = atol + rtol * np.abs(S_mtx.toarray())
            S_mask = S_diff > S_scale
            if S_mask.any():
                n_bad = int(S_mask.sum())
                print(f"  [S_bsp vs S_mtx       ] FAIL  {n_bad} entries differ beyond "
                      f"tolerance -- .bsp does not represent the same matrix as .mtx "
                      f"(max_diff={float(S_diff.max()):.3g}). Every downstream Prisma "
                      f"check below is validating against this same wrong S, so their "
                      f"PASS does not mean Prisma is correct on the real matrix.")
                all_pass = False
            else:
                print(f"  [S_bsp vs S_mtx       ] PASS  ({S_bsp.nnz} vs {S_mtx.nnz} nnz)")

    # --- TACO (reads .mtx) ----------------------------------------------------
    for label, binary in [("taco_cpu", taco_bin), ("taco_cpu_opt", taco_opt_bin)]:
        if label not in active:
            continue
        if binary is None:
            print(f"  [{label:<22}] SKIP (no {label} binary)")
            continue
        print(f"  [{label:<22}] ", end="", flush=True)
        vf = tmp / f"{name}_C_{label}.coo"
        try:
            taco_mtx = ensure_real_general(mtx, mtx_cache)
            cmd = [str(binary), str(taco_mtx), str(taco_mtx), "1", "--dump-c", str(vf)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            all_pass = False
            continue
        if r.returncode != 0:
            print(f"FAILED (exit {r.returncode})")
            all_pass = False
            continue
        if not vf.exists():
            print("FAILED (no output file written)")
            all_pass = False
            continue
        C = _load_coo(vf, M, N)
        results[label] = C
        ok, max_err, max_rel, failures = _compare(C, C_ref_mtx, rtol, atol)
        if ok:
            print("PASS")
        else:
            print(f"FAIL  max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                  f"failures={failures}/{C.size}")
            all_pass = False

    # --- Prisma (reads .bsp) ---------------------------------------------------
    want_prisma = any(l in active for l in CONTENDERS)
    if want_prisma and prisma_bin is None:
        print("  [prisma_*             ] SKIP (no prisma_cpu_bench binary)")
    elif want_prisma and not bsp.exists():
        print(f"  [prisma_*             ] SKIP (no BSP: {bsp.name})")
    elif want_prisma:
        S_bsp = _load_bsp_as_csr(bsp)
        if S_bsp is None:
            print("  [prisma_*             ] SKIP (h5py not available)")
        else:
            C_ref_bsp = np.asarray((S_bsp @ S_bsp).toarray())

            if "prisma_generic" in active:
                label = "prisma_generic"
                print(f"  [{label:<22}] ", end="", flush=True)
                vf = tmp / f"{name}_C_{label}.coo"
                cmd = [str(prisma_bin), str(bsp), str(bsp), "--runs", "1", "--validate", str(vf),
                       "--merge-strategy", merge_strategy]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                except subprocess.TimeoutExpired:
                    print("TIMEOUT")
                    all_pass = False
                    r = None
                if r is not None:
                    if r.returncode != 0:
                        print(f"FAILED (exit {r.returncode})")
                        all_pass = False
                    elif not vf.exists():
                        print("FAILED (no output file written)")
                        all_pass = False
                    else:
                        C = _load_coo(vf, M, N)
                        results[label] = C
                        ok, max_err, max_rel, failures = _compare(C, C_ref_bsp, rtol, atol)
                        if ok:
                            print("PASS")
                        else:
                            print(f"FAIL  max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                                  f"failures={failures}/{C.size}")
                            all_pass = False

            if "prisma_top10" in active:
                label = "prisma_top10"
                print(f"  [{label:<22}] ", end="", flush=True)
                try:
                    shapes, _ = analyze_spgemm_shapes(prisma_bin, bsp, top_n, timeout)
                    if not shapes:
                        print("SKIP (no shapes returned)")
                    else:
                        mat_work = work_dir / name
                        mat_work.mkdir(parents=True, exist_ok=True)
                        kernels_hpp, dispatch_hpp = gen_spgemm_kernels_files(shapes, mat_work)
                        top10_bin = compile_prisma_spgemm_per_matrix(
                            dispatch_hpp, mat_work / "prisma_cpu_bench_top10", blas,
                            kernels_hpp=kernels_hpp,
                        )
                        vf = tmp / f"{name}_C_{label}.coo"
                        cmd = [str(top10_bin), str(bsp), str(bsp), "--runs", "1",
                               "--specialized-kernels", "--validate", str(vf),
                               "--merge-strategy", merge_strategy]
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                        if r.returncode != 0:
                            print(f"FAILED (exit {r.returncode})")
                            all_pass = False
                        elif not vf.exists():
                            print("FAILED (no output file written)")
                            all_pass = False
                        else:
                            C = _load_coo(vf, M, N)
                            results[label] = C
                            ok, max_err, max_rel, failures = _compare(C, C_ref_bsp, rtol, atol)
                            if ok:
                                print("PASS")
                            else:
                                print(f"FAIL  max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                                      f"failures={failures}/{C.size}")
                                all_pass = False
                except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                    print(f"FAILED ({e})")
                    all_pass = False

    # --- Cross-compare all captured outputs -----------------------------------
    # Single uniform tolerance for every pair now -- taco and prisma are both
    # double precision reading double-precision-stored input, so there's no
    # "expected precision gap" to special-case anymore; any mismatch here is
    # a real bug.
    labels = list(results.keys())
    if len(labels) > 1:
        cross_ok = True
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                ok, max_err, max_rel, failures = _compare(
                    results[la], results[lb], rtol, atol
                )
                if not ok:
                    print(f"  CROSS-MISMATCH: {la} vs {lb}  "
                          f"max_diff={max_err:.3g}  max_rel={max_rel:.3g}  "
                          f"failures={failures}/{results[la].size}")
                    cross_ok = False
                    all_pass = False
        if cross_ok:
            print(f"  Cross-check: all {len(labels)} contenders agree ✓")

    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="SpGEMM CPU correctness validation (TACO + Prisma vs scipy reference; "
                     "both double precision)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", metavar="MATRICES.csv", nargs="?", default=None,
                   help="input CSV with at least a 'name' column")
    p.add_argument("--bin-dir", default="", dest="bin_dir",
                   help=f"directory for compiled binaries (default: {_TMP_DIR})")
    p.add_argument("--no-compile", action="store_true",
                   help="skip compilation; binaries must already exist in --bin-dir")
    p.add_argument("--prisma-bin", default="", dest="prisma_bin",
                   help="pre-built prisma_cpu_bench binary (skips compilation)")
    p.add_argument("--taco-bin", default="", dest="taco_bin",
                   help="pre-built bench_taco_cpu binary (skips compilation)")
    p.add_argument("--taco-opt-bin", default="", dest="taco_opt_bin",
                   help="pre-built bench_taco_cpu_opt binary (skips compilation)")
    p.add_argument("--threads", type=int, default=1,
                   help="NUM_THREADS baked into the TACO CPU binaries at compile time (default: 1, "
                        "since correctness doesn't depend on thread count)")
    p.add_argument("--kernels", default="",
                   help="comma-separated list of kernel labels to run "
                        f"(from: {', '.join(_ALL_LABELS)})")
    p.add_argument("--top-n", type=int, default=10, dest="top_n",
                   help="number of top shapes to specialise for prisma_top10 (default: 10)")
    p.add_argument("--blas", action="store_true",
                   help="link BLAS when compiling prisma (enables BLAS tile path)")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance (default 1e-6 -- both TACO and Prisma SpGEMM "
                        "CPU compute in double now, same as validate_spmm_cpu.py's default)")
    p.add_argument("--atol", type=float, default=1e-6,
                   help="absolute tolerance (default 1e-6, see --rtol)")
    p.add_argument("--timeout", type=int, default=300,
                   help="per-contender timeout in seconds (default 300)")
    p.add_argument("--merge-strategy", default="sequential", dest="merge_strategy",
                   choices=["sequential", "panels"],
                   help="prisma_cpu_bench's merge_overlapping_output_blocks "
                        "implementation to validate: 'sequential' (default, "
                        "single global sweep) or 'panels' (row-panel decomposition, "
                        "parallel via OpenMP -- see core/pipeline.hpp, provably same "
                        "grouping as 'sequential')")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bin_dir = Path(args.bin_dir) if args.bin_dir else _TMP_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    active = [l for l in _ALL_LABELS if l in args.kernels.split(",")] if args.kernels else list(_ALL_LABELS)

    if args.csv:
        matrices = load_matrix_list(Path(args.csv))
    else:
        print("No MATRICES.csv given — using built-in smoke-test list")
        matrices = _DEFAULT_MATRICES

    prisma_bin: Path | None = None
    want_prisma = any(l in active for l in CONTENDERS)
    if want_prisma:
        if args.prisma_bin:
            prisma_bin = Path(args.prisma_bin)
        elif args.no_compile:
            prisma_bin = bin_dir / "prisma_cpu_bench"
            if not prisma_bin.exists():
                prisma_bin = None
        else:
            try:
                prisma_bin = compile_prisma_spgemm(bin_dir / "prisma_cpu_bench", args.blas)
            except RuntimeError as e:
                print(f"prisma_cpu_bench compilation failed: {e}")
                prisma_bin = None

    taco_bin: Path | None = None
    taco_opt_bin: Path | None = None
    if "taco_cpu" in active:
        if args.taco_bin:
            taco_bin = Path(args.taco_bin)
        elif args.no_compile:
            taco_bin = bin_dir / "bench_taco_cpu"
            if not taco_bin.exists():
                taco_bin = None
        else:
            try:
                taco_bin = compile_taco_cpu("taco_kernel.h", bin_dir / "bench_taco_cpu", args.threads)
            except RuntimeError as e:
                print(f"bench_taco_cpu compilation failed: {e}")
                taco_bin = None
    if "taco_cpu_opt" in active:
        if args.taco_opt_bin:
            taco_opt_bin = Path(args.taco_opt_bin)
        elif args.no_compile:
            taco_opt_bin = bin_dir / "bench_taco_cpu_opt"
            if not taco_opt_bin.exists():
                taco_opt_bin = None
        else:
            try:
                taco_opt_bin = compile_taco_cpu("taco_kernel_opt.h", bin_dir / "bench_taco_cpu_opt", args.threads)
            except RuntimeError as e:
                print(f"bench_taco_cpu_opt compilation failed: {e}")
                taco_opt_bin = None

    print(f"Matrices : {len(matrices)}")
    print(f"Kernels  : {active}")
    print(f"Tolerances: rtol={args.rtol}  atol={args.atol}")
    print(f"Merge strategy: {args.merge_strategy}")
    print()

    n_pass = n_fail = n_skip = 0
    mtx_cache = bin_dir / "mtx_cache"

    with tempfile.TemporaryDirectory(prefix="validate_spgemm_cpu_") as tmp_str:
        tmp = Path(tmp_str)
        for i, row in enumerate(matrices, 1):
            name  = row["name"]
            group = row.get("group", "")
            print(f"[{i}/{len(matrices)}] {name}")

            mtx = find_mtx(name, group)
            if mtx is None:
                print("  MTX not found — skipping")
                n_skip += 1
                continue

            t0 = time.time()
            ok = validate_matrix(
                row, mtx, taco_bin, taco_opt_bin, prisma_bin, active, args.top_n, args.blas,
                args.rtol, args.atol, args.timeout, tmp, bin_dir, mtx_cache,
                merge_strategy=args.merge_strategy,
            )
            elapsed = time.time() - t0
            print(f"  ({elapsed:.1f}s)")
            if ok:
                n_pass += 1
            else:
                n_fail += 1

    print()
    print(f"Summary: {n_pass} PASS  {n_fail} FAIL  {n_skip} SKIP")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
