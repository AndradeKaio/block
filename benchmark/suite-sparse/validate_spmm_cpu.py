#!/usr/bin/env python3
"""
suite-sparse/validate_spmm_cpu.py — Correctness validation for CPU SpMM
contenders (TACO + Prisma CPU).

For each matrix, every contender is run with --seed S and --dump-c to write its
output C matrix.  prisma_cpu additionally dumps D via --dump-d.  The scipy
reference C_ref = S @ D is computed in Python and compared against each binary's
output.  Since all contenders share the same seed and the same mt19937_64 RNG,
they produce identical D, so all C outputs can also be cross-compared directly.

GPU contenders (prisma_gpu_cuda_*, cusparse_*) are validated separately by
validate_spmm_gpu.py — kept apart so this script only needs the CPU/TACO
toolchain (g++), mirroring the benchmark_spmm_cpu.py / benchmark_spmm_gpu.py
split.

Usage:
  python validate_spmm_cpu.py MATRICES.csv
  python validate_spmm_cpu.py MATRICES.csv --bin-dir ../SpMM/ --no-compile
  python validate_spmm_cpu.py MATRICES.csv --kernels prisma_tiled,prisma_static
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
# Paths (mirrors benchmark_spmm_cpu.py)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_SPMM_DIR   = _SCRIPT_DIR.parent / "SpMM"
_DATA_ROOT  = Path("/home/kaio/datasets/suite-sparse")

# ---------------------------------------------------------------------------
# Contenders — CPU only
# ---------------------------------------------------------------------------

# (label, binary_stem, input_ext, extra_flags)
CONTENDERS = [
    ("taco",               "bench_taco_spmm_taco",      ".mtx", []),
    ("taco_opt0",          "bench_taco_spmm_taco_opt0", ".mtx", []),
    ("taco_opt1",          "bench_taco_spmm_taco_opt1", ".mtx", []),
    ("prisma_cpu",         "prisma_cpu_spmm_bench",     ".bsp", []),
    ("prisma_specialized", "prisma_cpu_spmm_bench",     ".bsp", ["--specialized-kernels"]),
    ("prisma_static",      "prisma_cpu_spmm_bench",     ".bsp", ["--specialized-kernels", "--static"]),
    ("prisma_tiled",       "prisma_cpu_spmm_bench",     ".bsp", ["--specialized-kernels", "--tile-n", "512"]),
    ("prisma_auto",        "prisma_cpu_spmm_bench",     ".bsp", ["--specialized-kernels", "--auto"]),
]

# ---------------------------------------------------------------------------
# Matrix location (same as benchmark_spmm_cpu.py)
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
# Compilation (delegates to benchmark_spmm_cpu.py helpers)
# ---------------------------------------------------------------------------


def _load_bsp_as_csr(bsp: Path) -> scipy.sparse.csr_matrix | None:
    """Load a .bsp (HDF5 block-sparse) file into a scipy CSR matrix.

    Values are read at whatever precision is stored in the file (float32 or
    float64) and upcast to float64.  This matches exactly what
    read_matrix_binsparse<double> does inside prisma_cpu_spmm_bench.
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


def compile_all(bin_dir: Path, matrices: list[dict]) -> None:
    sys.path.insert(0, str(_SCRIPT_DIR))
    from benchmark_spmm_cpu import compile_binary, compile_prisma_per_matrix, _KERNELS
    print("Compiling TACO kernels:")
    for k, d in _KERNELS:
        compile_binary(bin_dir, k, d)
    print()
    print("Compiling Prisma CPU SpMM (per-matrix kernels):")
    compile_prisma_per_matrix(bin_dir, matrices)
    print()


def _prisma_binary_path(bin_dir: Path, name: str, label: str = "prisma_cpu",
                        stem: str = "") -> Path:
    """Resolve a contender's binary path for one matrix.

    Two binary-layout families:
      1. Prisma CPU (label startswith "prisma") -- per-matrix specialized,
         under --bin-dir. See compile_prisma_per_matrix in
         benchmark_spmm_cpu.py.
      2. Pooled TACO binaries (`stem` used as-is, no per-matrix compilation)
         -- live under --bin-dir (CPU toolchain, see benchmark_spmm_cpu.py's
         compile_binary).
    """
    if label.startswith("prisma"):
        return bin_dir / f"prisma_cpu_spmm_bench_{name}"
    return bin_dir / stem

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


# Kept here (rather than dropped) because validate_spmm_gpu.py imports these
# generic comparison helpers from this module -- prisma_gpu_cuda_fp64's tf32
# tensor-core path needs a global-scale (not per-cell) tolerance; no CPU
# contender needs it, but this stays the shared home for both validators'
# comparison utilities.
_TC_RTOL = 1e-2
_TC_ATOL = 1e-6


def _compare_global_scale(C: np.ndarray, C_ref: np.ndarray,
                          rtol: float, atol: float) -> tuple[bool, float, float, int]:
    diff = np.abs(C - C_ref)
    global_scale = float(np.nanmax(np.abs(C_ref))) if C_ref.size else 0.0
    scale = atol + rtol * global_scale
    mask = diff > scale
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
    M, N  = S_mtx.shape

    # Load S from BSP so Prisma reference uses the same precision as the binary.
    # Existing BSPs may store float32 values; read_matrix_binsparse<double> does
    # an HDF5 float32→float64 upcast (no extra loss), so _load_bsp_as_csr matches.
    S_bsp = _load_bsp_as_csr(bsp) if bsp.exists() else None

    all_pass = True

    # --- S_bsp vs S_mtx: are they the SAME matrix? --------------------------
    # This is checked directly, independent of D and of any compute kernel,
    # because it has exactly one correct answer (mod. float32 storage
    # truncation) — unlike the C-output cross-checks below, which legitimately
    # need a looser tolerance since Prisma and TACO are handed different-
    # precision S by design. Comparing S directly here, rather than only
    # inferring it from downstream C differences, is what actually catches
    # a corrupted .bsp (e.g. mine_matrix.cpp's is_symmetric/is_pattern
    # false-positive from scanning free-text comment lines, not just the MTX
    # banner) instead of that corruption hiding behind a "different domain,
    # expect some gap" cross-check tolerance loose enough to swallow it too.
    if S_bsp is not None:
        # A numeric diff (not a structural/position-set comparison) is used
        # deliberately: _load_bsp_as_csr drops stored zeros within a block
        # via np.nonzero() (matching read_matrix_binsparse<double>'s own
        # semantics -- a block CAN legitimately store an explicit zero at a
        # real, non-implicit position), so S_bsp.nnz and S_mtx.nnz can differ
        # even for a fully correct .bsp. Diffing values directly sidesteps
        # that: a dropped explicit zero still comes out as 0 - 0 = 0 here,
        # not a spurious mismatch.
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

    # --- Get D from prisma_cpu (same RNG as all Prisma variants) -----------
    # TACO also uses mt19937_64 with the same seed after the .cpp rename.
    d_path  = tmp / f"{name}_D.bin"
    cp_path = tmp / f"{name}_C_prisma_cpu.bin"
    D       = None

    prisma_bin = _prisma_binary_path(bin_dir, name)
    have_bsp   = bsp.exists() and prisma_bin.exists()
    if have_bsp:
        cmd = [str(prisma_bin), str(bsp), "--runs", "1", "--seed", str(seed),
               "--dump-d", str(d_path), "--dump-c", str(cp_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0 and d_path.exists():
            D = np.fromfile(str(d_path), dtype=np.float64).reshape(N, N)

    # Two references depending on which S precision the contender uses:
    #   C_ref_bsp — for Prisma variants (S from BSP, same float precision as binary)
    #   C_ref_mtx — for TACO variants (S from MTX, full float64)
    C_ref_bsp = (S_bsp @ D) if (D is not None and S_bsp is not None) else None
    C_ref_mtx = (S_mtx @ D) if D is not None else None

    results: dict[str, np.ndarray] = {}
    label_ext: dict[str, str] = {}  # for _domain()'s ext-based classification

    for label, stem, ext, extra_flags in active:
        if ext == ".bsp":
            if not have_bsp:
                print(f"  [{label:<22}] SKIP (no BSP)")
                continue
            inp = bsp
        else:
            inp = mtx

        binary = _prisma_binary_path(bin_dir, name, label, stem)
        if not binary.exists():
            print(f"  [{label:<22}] SKIP (binary not found)")
            continue

        ck_path = tmp / f"{name}_C_{label}.bin"
        cmd = ([str(binary), str(inp), "--runs", "1", "--seed", str(seed),
                "--dump-c", str(ck_path)] + extra_flags)
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

        raw = np.fromfile(str(ck_path), dtype=np.float64)
        if raw.size != M * N:
            print(f"  [{label:<22}] FAIL  output has {raw.size} elements, "
                  f"expected {M}x{N}={M * N} (likely a .bsp/.mtx shape "
                  f"mismatch — mismatched companion file, not a compute bug)")
            all_pass = False
            continue
        C = raw.reshape(M, N)
        results[label] = C
        label_ext[label] = ext

        # Choose reference by which INPUT FORMAT the contender reads, not
        # by label naming: .bsp readers (Prisma) get the truncated-
        # precision-matching C_ref_bsp; .mtx readers (TACO) get the
        # full-float64 C_ref_mtx.
        C_ref = C_ref_bsp if ext == ".bsp" else C_ref_mtx
        if C_ref is not None:
            ok, max_err, max_rel, failures = _compare(C, C_ref, rtol, atol)
            if ok:
                print(f"  [{label:<22}] PASS")
            else:
                print(f"  [{label:<22}] FAIL  "
                      f"max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                      f"failures={failures}/{C.size}")
                all_pass = False
        else:
            print(f"  [{label:<22}] (no reference — output captured)")

    # --- Cross-compare all captured outputs --------------------------------
    # S storage precision is the only real precision axis left once GPU
    # contenders are out of scope: Prisma reads S from .bsp (float32-stored
    # values, upcast to double); TACO reads S from .mtx (native float64).
    # Verified: bsp values are bit-exact float32 truncations of the mtx
    # values, so a pair differing on this axis needs a relaxed tolerance;
    # only a same-ext pair shares its reference's precision exactly and
    # keeps the caller's strict tolerance, gating all_pass.
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
        description="SpMM CPU correctness validation (TACO + Prisma CPU vs scipy reference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", metavar="MATRICES.csv", nargs="?", default=None,
                   help="input CSV with at least a 'name' column")
    p.add_argument("--bin-dir", default="", dest="bin_dir",
                   help="directory with compiled binaries (default: ../SpMM/)")
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
                        "1e-10 (the old default) is tighter than double-precision "
                        "reduction can guarantee once the summation order differs "
                        "(scipy's serial reference vs. any parallel/blocked GEMM); "
                        "observed order-of-summation noise on this suite tops out "
                        "around 1e-7 relative, so 1e-6 leaves comfortable margin "
                        "while still catching real bugs (which show as >1e-2).")
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

    bin_dir = Path(args.bin_dir) if args.bin_dir else _SPMM_DIR

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
            compile_all(bin_dir, matrices)
        except Exception as e:
            sys.exit(f"Compilation failed: {e}")

    print(f"Matrices : {len(matrices)}")
    print(f"Kernels  : {[l for l, *_ in active]}")
    print(f"Seed     : {args.seed}")
    print(f"Tolerances: rtol={args.rtol}  atol={args.atol}")
    print()

    n_pass = n_fail = n_skip = 0

    with tempfile.TemporaryDirectory(prefix="validate_spmm_cpu_") as tmp_str:
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

    print()
    print(f"Summary: {n_pass} PASS  {n_fail} FAIL  {n_skip} SKIP")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
