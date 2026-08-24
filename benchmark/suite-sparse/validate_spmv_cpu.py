#!/usr/bin/env python3
"""
suite-sparse/validate_spmv_cpu.py — Correctness validation for CPU SpMV
contenders (TACO + Prisma CPU).

For each matrix, every contender is run with --seed S and --dump-c to write its
output y vector.  prisma_cpu additionally dumps x via --dump-x.  The scipy
reference y_ref = S @ x is computed in Python and compared against each binary's
output.  Since all contenders share the same seed and the same mt19937_64 RNG,
they produce identical x, so all y outputs can also be cross-compared directly.

Usage:
  python validate_spmv_cpu.py MATRICES.csv
  python validate_spmv_cpu.py MATRICES.csv --bin-dir ../SpMV/CPU/ --no-compile
  python validate_spmv_cpu.py MATRICES.csv --kernels prisma_cpu,prisma_static
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

try:
    import h5py
    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False

# ---------------------------------------------------------------------------
# Paths (mirrors benchmark_spmv_cpu.py)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_SPMV_DIR   = _SCRIPT_DIR.parent / "SpMV" / "CPU"
_DATA_ROOT  = Path("/home/kaio/datasets/suite-sparse")

# ---------------------------------------------------------------------------
# Contenders — CPU only. One shared binary per stem (no per-matrix
# specialization, unlike SpMM's Prisma contender -- see
# prisma_cpu_spmv_bench.cpp / benchmark_spmv_cpu.py for why).
# ---------------------------------------------------------------------------

# (label, binary_stem, input_ext, extra_flags)
# prisma_specialized has no fixed stem -- it's compiled per-matrix (own top-N
# shapes, exact-match dispatch, mirroring SpGEMM's prisma_top10), handled as
# a special case in validate_matrix() below rather than a fixed bin_dir/stem
# lookup.
CONTENDERS = [
    ("taco",               "bench_taco_spmv_taco",     ".mtx", []),
    ("taco_opt",           "bench_taco_spmv_taco_opt", ".mtx", []),
    ("prisma_cpu",         "prisma_cpu_spmv_bench",    ".bsp", []),
    ("prisma_static",      "prisma_cpu_spmv_bench",    ".bsp", ["--static"]),
    ("prisma_specialized", "",                         ".bsp", ["--specialized-kernels"]),
]

# ---------------------------------------------------------------------------
# Matrix location (same as benchmark_spmv_cpu.py)
# ---------------------------------------------------------------------------


def _readable(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0 and os.access(p, os.R_OK)
    except OSError:
        return False


def find_mtx(name: str, group: str) -> Path | None:
    mat_dir = _DATA_ROOT / group / name if group else _DATA_ROOT / name
    mtx = mat_dir / f"{name}.mtx"
    if _readable(mtx):
        return mtx
    candidates = [p for p in _DATA_ROOT.rglob(f"{name}.mtx") if _readable(p)]
    return candidates[0] if candidates else None


def load_matrix_list(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        rows = list(reader)
    if not rows:
        sys.exit(f"No rows found in {csv_path}")
    if "name" not in rows[0]:
        sys.exit(f"Input CSV must have a 'name' column; got: {list(rows[0].keys())}")
    return rows

# ---------------------------------------------------------------------------
# Compilation (delegates to benchmark_spmv_cpu.py helpers)
# ---------------------------------------------------------------------------


def _load_bsp_as_csr(bsp: Path) -> scipy.sparse.csr_matrix | None:
    """Load a .bsp (HDF5 block-sparse) file into a scipy CSR matrix.

    Values are read at whatever precision is stored in the file (float32 or
    float64) and upcast to float64.  This matches exactly what
    read_matrix_binsparse<double> does inside prisma_cpu_spmv_bench.
    """
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

    rows_list = []
    cols_list = []
    data_list = []
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


def compile_all(bin_dir: Path) -> None:
    sys.path.insert(0, str(_SCRIPT_DIR))
    from benchmark_spmv_cpu import compile_binary, compile_prisma_cpu_spmv, _KERNELS
    print("Compiling TACO kernels:")
    for k, d in _KERNELS:
        compile_binary(bin_dir, k, d)
    print()
    print("Compiling Prisma CPU SpMV:")
    compile_prisma_cpu_spmv(bin_dir)
    print()


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------


def _fwrite_check(path: Path, label: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        print(f"  [{label:<22}] FAILED (no output file written)")
        return False
    return True


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


def validate_matrix(row: dict, mtx: Path, bin_dir: Path,
                    active: list, seed: int, rtol: float, atol: float,
                    timeout: int, tmp: Path) -> bool:
    name = row["name"]
    bsp  = mtx.with_suffix(".bsp")

    S_mtx = scipy.io.mmread(str(mtx)).tocsr().astype(np.float64)
    M, N  = S_mtx.shape  # M and N need not be equal -- x has length N, y has length M

    # Load S from BSP so Prisma reference uses the same precision as the binary.
    S_bsp = _load_bsp_as_csr(bsp) if bsp.exists() else None

    all_pass = True

    # --- S_bsp vs S_mtx: are they the SAME matrix? --------------------------
    if S_bsp is not None:
        S_diff = np.abs((S_bsp - S_mtx).toarray())
        S_scale = 1e-6 + 1e-5 * np.abs(S_mtx.toarray())
        S_mask = S_diff > S_scale
        if S_mask.any():
            n_bad = int(S_mask.sum())
            print(f"  [S_bsp vs S_mtx       ] FAIL  {n_bad} entries differ beyond "
                  f"float32-truncation tolerance -- .bsp does not represent the "
                  f"same matrix as .mtx (max_diff={float(S_diff.max()):.3g}). "
                  f"Every downstream Prisma check below is validating against "
                  f"this same wrong S, so their PASS does not mean Prisma is "
                  f"correct on the real matrix.")
            all_pass = False
        else:
            print(f"  [S_bsp vs S_mtx       ] PASS  (nonzero-valued entries: "
                  f"{S_bsp.nnz} vs {S_mtx.nnz} -- a difference here alone is not "
                  f"a failure signal, see comment above)")

    # --- Get x from prisma_cpu (same RNG as all Prisma variants) -----------
    # TACO also uses mt19937_64 with the same seed.
    x_path  = tmp / f"{name}_x.bin"
    cp_path = tmp / f"{name}_y_prisma_cpu.bin"
    x       = None

    prisma_bin = bin_dir / "prisma_cpu_spmv_bench"
    have_bsp   = bsp.exists() and prisma_bin.exists()
    if have_bsp:
        cmd = [str(prisma_bin), str(bsp), "--runs", "1", "--seed", str(seed),
               "--dump-x", str(x_path), "--dump-y", str(cp_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0 and x_path.exists():
            raw = np.fromfile(str(x_path), dtype=np.float64)
            if raw.size == N:
                x = raw
            else:
                # A malformed dump (wrong size, e.g. truncated or empty) is
                # treated the same as "no dump produced" -- everything
                # downstream that depends on x degrades to "no reference"
                # instead of crashing the whole validation run on a later
                # S_bsp @ x dimension-mismatch (see validate_spmm_cpu.py's
                # identical guard -- same failure class hit there first).
                print(f"  [x dump               ] WARNING: expected {N} "
                      f"elements, got {raw.size} -- treating as no reference "
                      f"for this matrix")

    # Two references depending on which S precision the contender uses:
    #   y_ref_bsp — for Prisma variants (S from BSP, same float precision as binary)
    #   y_ref_mtx — for TACO variants (S from MTX, full float64)
    y_ref_bsp = (S_bsp @ x) if (x is not None and S_bsp is not None) else None
    y_ref_mtx = (S_mtx @ x) if x is not None else None

    results: dict[str, np.ndarray] = {}
    label_ext: dict[str, str] = {}

    for label, stem, ext, extra_flags in active:
        if ext == ".bsp":
            if not have_bsp:
                print(f"  [{label:<22}] SKIP (no BSP)")
                continue
            inp = bsp
        else:
            inp = mtx

        if label == "prisma_specialized":
            from benchmark_spmv_cpu import compile_prisma_spmv_for_matrix
            _, binary, msg = compile_prisma_spmv_for_matrix(bin_dir, bin_dir, name, bsp)
            if binary is None:
                print(f"  [{label:<22}] SKIP ({msg})")
                continue
        else:
            binary = bin_dir / stem
            if not binary.exists():
                print(f"  [{label:<22}] SKIP (binary not found)")
                continue

        ck_path = tmp / f"{name}_y_{label}.bin"
        # TACO's binary keeps the original --dump-c flag name (output
        # dump); Prisma's uses --dump-y (see prisma_cpu_spmv_bench.cpp's
        # deliberate --dump-d/--dump-c -> --dump-x/--dump-y rename).
        dump_flag = "--dump-c" if ext == ".mtx" else "--dump-y"
        cmd = ([str(binary), str(inp), "--runs", "1", "--seed", str(seed),
                dump_flag, str(ck_path)] + extra_flags)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"  [{label:<22}] TIMEOUT")
            all_pass = False
            continue

        if r.returncode != 0:
            print(f"  [{label:<22}] FAILED (exit {r.returncode})")
            all_pass = False
            continue

        if not _fwrite_check(ck_path, label):
            all_pass = False
            continue

        y = np.fromfile(str(ck_path), dtype=np.float64)
        if y.size != M:
            print(f"  [{label:<22}] FAIL  output has {y.size} elements, "
                  f"expected M={M} (likely a .bsp/.mtx shape mismatch — "
                  f"mismatched companion file, not a compute bug)")
            all_pass = False
            continue
        results[label] = y
        label_ext[label] = ext

        # Choose reference by which INPUT FORMAT the contender reads, not
        # by label naming: .bsp readers (Prisma) get the truncated-
        # precision-matching y_ref_bsp; .mtx readers (TACO) get the
        # full-float64 y_ref_mtx.
        y_ref = y_ref_bsp if ext == ".bsp" else y_ref_mtx
        if y_ref is not None:
            ok, max_err, max_rel, failures = _compare(y, y_ref, rtol, atol)
            if ok:
                print(f"  [{label:<22}] PASS")
            else:
                print(f"  [{label:<22}] FAIL  "
                      f"max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                      f"failures={failures}/{y.size}")
                all_pass = False
        else:
            print(f"  [{label:<22}] (no reference — output captured)")

    # --- Cross-compare all captured outputs --------------------------------
    _F32_CROSS_RTOL = 1e-4
    _F32_CROSS_ATOL = 1e-6

    labels = list(results.keys())
    if len(labels) > 1:
        cross_ok = True
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                same_domain = label_ext[la] == label_ext[lb]
                cmp_rtol, cmp_atol = (rtol, atol) if same_domain else \
                    (max(rtol, _F32_CROSS_RTOL), max(atol, _F32_CROSS_ATOL))
                ok, max_err, max_rel, failures = _compare(
                    results[la], results[lb], cmp_rtol, cmp_atol
                )
                if not ok:
                    tag = "CROSS-MISMATCH" if same_domain else \
                        "CROSS-DIFF (expected .bsp-vs-.mtx storage precision gap)"
                    print(f"  {tag}: {la} vs {lb}  "
                          f"max_diff={max_err:.3g}  max_rel={max_rel:.3g}  "
                          f"failures={failures}/{results[la].size}")
                    cross_ok = False
                    if same_domain:
                        all_pass = False
        if cross_ok:
            print(f"  Cross-check: all {len(labels)} contenders agree ✓")

    return all_pass

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="SpMV CPU correctness validation (TACO + Prisma CPU vs scipy reference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", metavar="MATRICES.csv", nargs="?", default=None,
                   help="input CSV with at least a 'name' column")
    p.add_argument("--bin-dir", default="", dest="bin_dir",
                   help="directory with compiled binaries (default: ../SpMV/CPU/)")
    p.add_argument("--no-compile", action="store_true",
                   help="skip compilation; binaries must already exist")
    p.add_argument("--no-taco", action="store_true",
                   help="skip all TACO variants")
    p.add_argument("--no-prisma", action="store_true",
                   help="skip all Prisma variants")
    p.add_argument("--kernels", default="",
                   help="comma-separated list of kernel labels to run")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed passed to all binaries (default 42)")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance for comparisons (default 1e-6). "
                        "1e-10 is tighter than double-precision reduction can "
                        "guarantee once the summation order differs (scipy's "
                        "serial reference vs. any parallel/blocked kernel); "
                        "1e-6 leaves comfortable margin while still catching "
                        "real bugs (which show as >1e-2).")
    p.add_argument("--atol", type=float, default=1e-6,
                   help="absolute tolerance for comparisons (default 1e-6, see --rtol)")
    p.add_argument("--timeout", type=int, default=300,
                   help="per-contender timeout in seconds (default 300)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DEFAULT_MATRICES = [
    {"name": "bundle1",   "group": "Janna",      "rows": "10294", "cols": "10294", "nnz": "1000000"},
    {"name": "bcsstk27",  "group": "HB",         "rows": "1224",  "cols": "1224",  "nnz": "56126"},
    {"name": "linverse",  "group": "Bova",        "rows": "11999", "cols": "11999", "nnz": "9921"},
]


def main() -> None:
    args = parse_args()

    bin_dir = Path(args.bin_dir) if args.bin_dir else _SPMV_DIR

    # Select active contenders
    if args.kernels:
        want = set(args.kernels.split(","))
        active = [(l, s, e, f) for l, s, e, f in CONTENDERS if l in want]
        if not active:
            sys.exit(f"No contenders matched --kernels {args.kernels!r}")
    else:
        active = []
        for l, s, e, f in CONTENDERS:
            if args.no_taco   and l.startswith("taco"):   continue
            if args.no_prisma and l.startswith("prisma"): continue
            active.append((l, s, e, f))

    # Load matrix list
    if args.csv:
        matrices = load_matrix_list(Path(args.csv))
    else:
        print("No MATRICES.csv given — using built-in smoke-test list")
        matrices = _DEFAULT_MATRICES

    # Compile if needed
    if not args.no_compile:
        try:
            compile_all(bin_dir)
        except Exception as e:
            sys.exit(f"Compilation failed: {e}")

    print(f"Matrices : {len(matrices)}")
    print(f"Kernels  : {[l for l, *_ in active]}")
    print(f"Seed     : {args.seed}")
    print(f"Tolerances: rtol={args.rtol}  atol={args.atol}")
    print()

    n_pass = n_fail = n_skip = 0

    with tempfile.TemporaryDirectory(prefix="validate_spmv_cpu_") as tmp_str:
        tmp = Path(tmp_str)
        for i, row in enumerate(matrices, 1):
            name  = row["name"]
            group = row.get("group", "")
            print(f"[{i}/{len(matrices)}] {name}")

            mtx = find_mtx(name, group)
            if mtx is None:
                print(f"  MTX not found — skipping")
                n_skip += 1
                continue

            t0 = time.time()
            ok = validate_matrix(row, mtx, bin_dir, active,
                                 args.seed, args.rtol, args.atol,
                                 args.timeout, tmp)
            elapsed = time.time() - t0
            print(f"  ({elapsed:.1f}s)")
            if ok:
                n_pass += 1
            else:
                n_fail += 1

            # tmp is shared across the whole matrix list -- see
            # validate_spmm_cpu.py's identical fix for the full rationale
            # (dumps here are dense M/N-length vectors, smaller than SpMM's
            # dense matrices, but still unbounded over a long matrix list).
            for f in tmp.glob(f"{name}_*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    print()
    print(f"Summary: {n_pass} PASS  {n_fail} FAIL  {n_skip} SKIP")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
