#!/usr/bin/env python3
"""
suite-sparse/validate_spmm_gpu.py — Correctness validation for GPU SpMM
contenders only (prisma_gpu_cuda_fp64[/_row_group/_nofallback],
prisma_gpu_cuda_fp32, cusparse_fp64/fp32).

A GPU-only counterpart to validate_spmm_cpu.py (TACO + prisma_cpu), which
needs the CPU/TACO toolchain (g++) compiled before running anything. This
script only needs nvcc — mirroring exactly why benchmark_spmm_gpu.py is a
separate file from benchmark_spmm_cpu.py (see that module's docstring).

For each matrix, every active GPU contender is run with --seed S and
--dump-c to write its output C matrix; the first contender that runs also
dumps D via --dump-d. The scipy reference C_ref = S_bsp @ D is computed in
Python (S read from .bsp, matching every GPU contender's own truncated
storage precision) and compared against each binary's output. Since every
contender shares the same seed and the same mt19937_64 RNG, they also all
produce identical D, so outputs are cross-compared directly against each
other too.

Usage:
  python validate_spmm_gpu.py MATRICES.csv
  python validate_spmm_gpu.py MATRICES.csv --work-dir /tmp/_prismac --no-compile
  python validate_spmm_gpu.py MATRICES.csv --kernels prisma_gpu_cuda_fp64,cusparse_fp64
  python validate_spmm_gpu.py MATRICES.csv --no-cusparse --row-group
"""

import argparse
import sys
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from benchmark_spmm_cpu import find_mtx, load_matrix_list  # noqa: E402
from benchmark_spmm_gpu import (  # noqa: E402
    _TMP_DIR,
    compile_cusparse_bench,
    compile_prisma_gpu_per_matrix,
)
from validate_spmm_cpu import (  # noqa: E402
    _compare,
    _compare_global_scale,
    _load_bsp_as_csr,
    _TC_ATOL,
    _TC_RTOL,
    _DEFAULT_MATRICES,
)

# ---------------------------------------------------------------------------
# Contenders — GPU only
# ---------------------------------------------------------------------------

# (label, precision, extra_flags)
PRISMA_GPU_CONTENDERS = [
    ("prisma_gpu_cuda_fp64", "fp64", []),
    ("prisma_gpu_cuda_fp32", "fp32", []),
    ("prisma_gpu_cuda_fp64_nofallback", "fp64", ["--force-cuda-fallback"]),
    ("prisma_gpu_cuda_fp64_row_group", "fp64", ["--row-group"]),
]

# (label, precision)
CUSPARSE_CONTENDERS = [
    ("cusparse_fp64", "fp64"),
    ("cusparse_fp32", "fp32"),
]

_F32_CROSS_RTOL = 1e-4
_F32_CROSS_ATOL = 1e-6


def _is_tc_lane(label: str) -> bool:
    """Same rule as validate_spmm_cpu.py's _is_tc_lane: every fp64 prisma_gpu
    label still routes TC-eligible blocks through tf32 EXCEPT _nofallback
    (--force-cuda-fallback disables TC entirely)."""
    return label.startswith("prisma_gpu_cuda_fp64") and not label.endswith("_nofallback")


def _needs_global_scale(label: str) -> bool:
    """True for any lane where a per-cell relative tolerance is meaningless,
    not just TC lanes: on the ill-conditioned matrices in this suite
    (bcsstk27/bundle1 -- dynamic range up to 1e8+, see _compare_global_scale's
    docstring), individual C cells are often near-zero results of
    catastrophic cancellation between much larger terms, regardless of
    whether tf32 tensor cores are involved. A plain fp32 lane (no TC) hits
    the exact same problem -- reduced precision on a cancellation-heavy
    cell inflates an ordinary, bounded absolute error into an astronomical
    "relative" one. Previously only _is_tc_lane triggered global-scale
    comparison, so prisma_gpu_cuda_fp32/cusparse_fp32 were checked with the
    same strict per-cell rtol as the fp64 lanes and failed by 6 orders of
    magnitude on bundle1/bcsstk27 for a reason that had nothing to do with
    an actual kernel bug."""
    return _is_tc_lane(label) or label.endswith("fp32")


def _domain(label: str) -> tuple[bool, bool]:
    """Precision domain for cross-check tolerance selection: (is_fp32,
    is_tc_lane). Unlike validate_spmm_cpu.py's version there's no S-storage-
    format axis to track — every contender here reads S from .bsp."""
    return (label.endswith("fp32"), _is_tc_lane(label))


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------


def _fwrite_check(path: Path, label: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        print(f"  [{label:<22}] FAILED (no output file written)")
        return False
    return True


def validate_matrix(
    row: dict,
    mtx: Path,
    bsp: Path,
    active: list,
    prisma_binary: Path | None,
    cusparse_binary: Path | None,
    seed: int,
    rtol: float,
    atol: float,
    timeout: int,
    tmp: Path,
) -> bool:
    name = row["name"]
    S_bsp = _load_bsp_as_csr(bsp)
    if S_bsp is None:
        print("  SKIP (h5py not available or .bsp unreadable)")
        return False
    M, N = S_bsp.shape

    all_pass = True
    results: dict[str, np.ndarray] = {}
    C_ref: np.ndarray | None = None
    d_path = tmp / f"{name}_D.bin"

    for label, precision, extra_flags in active:
        if label.startswith("prisma_gpu"):
            if prisma_binary is None:
                print(f"  [{label:<22}] SKIP (binary not found)")
                continue
            binary = prisma_binary
        else:
            if cusparse_binary is None:
                print(f"  [{label:<22}] SKIP (binary not found)")
                continue
            binary = cusparse_binary

        ck_path = tmp / f"{name}_C_{label}.bin"
        cmd = [
            str(binary), str(bsp),
            "--runs", "1", "--seed", str(seed),
            "--precision", precision,
            "--dump-c", str(ck_path),
        ] + extra_flags
        if C_ref is None:
            # First contender to run also dumps D, from which C_ref is built.
            cmd += ["--dump-d", str(d_path)]

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

        if C_ref is None:
            if not d_path.exists():
                print(f"  [{label:<22}] FAILED (no --dump-d output written)")
                all_pass = False
                continue
            D = np.fromfile(str(d_path), dtype=np.float64).reshape(N, N)
            C_ref = np.asarray(S_bsp @ D)

        raw = np.fromfile(str(ck_path), dtype=np.float64)
        if raw.size != M * N:
            print(f"  [{label:<22}] FAIL  output has {raw.size} elements, "
                  f"expected {M}x{N}={M * N}")
            all_pass = False
            continue
        C = raw.reshape(M, N)
        results[label] = C

        if _needs_global_scale(label):
            ok, max_err, max_rel, failures = _compare_global_scale(
                C, C_ref, _TC_RTOL, _TC_ATOL)
        else:
            ok, max_err, max_rel, failures = _compare(C, C_ref, rtol, atol)

        if ok:
            print(f"  [{label:<22}] PASS")
        else:
            print(f"  [{label:<22}] FAIL  "
                  f"max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                  f"failures={failures}/{C.size}")
            all_pass = False

    # --- Cross-compare all captured outputs --------------------------------
    labels = list(results.keys())
    if len(labels) > 1:
        cross_ok = True
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                same_domain = _domain(la) == _domain(lb)
                if _needs_global_scale(la) or _needs_global_scale(lb):
                    cmp_fn = _compare_global_scale
                    cmp_rtol, cmp_atol = _TC_RTOL, _TC_ATOL
                    gap_desc = "reduced-precision (tf32/fp32) catastrophic-cancellation gap"
                else:
                    cmp_fn = _compare
                    cmp_rtol, cmp_atol = (rtol, atol) if same_domain else \
                        (max(rtol, _F32_CROSS_RTOL), max(atol, _F32_CROSS_ATOL))
                    gap_desc = "float32-vs-float64 precision gap"
                ok, max_err, max_rel, failures = cmp_fn(
                    results[la], results[lb], cmp_rtol, cmp_atol
                )
                if not ok:
                    tag = "CROSS-MISMATCH" if same_domain else \
                        f"CROSS-DIFF (expected {gap_desc})"
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
        description="GPU SpMM correctness validation (prisma_gpu / cusparse vs scipy reference)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", metavar="MATRICES.csv", nargs="?", default=None,
                   help="input CSV with at least a 'name' column")
    p.add_argument("--work-dir", default="", dest="work_dir",
                   help="directory for compiled binaries (default: /tmp/_prismac/, "
                        "same as benchmark_spmm_gpu.py)")
    p.add_argument("--no-compile", action="store_true",
                   help="skip compilation; binaries must already exist under --work-dir")
    p.add_argument("--no-fp64", action="store_true", help="skip the prisma_gpu_cuda_fp64 lane")
    p.add_argument("--no-fp32", action="store_true", help="skip the prisma_gpu_cuda_fp32 lane")
    p.add_argument("--no-cusparse", action="store_true", help="skip both cuSPARSE contenders")
    p.add_argument("--nofallback", action="store_true",
                   help="also validate prisma_gpu_cuda_fp64_nofallback (--force-cuda-fallback)")
    p.add_argument("--row-group", action="store_true", dest="row_group",
                   help="also validate prisma_gpu_cuda_fp64_row_group (--row-group)")
    p.add_argument("--kernels", default="",
                   help="comma-separated list of kernel labels to run (overrides --no-* flags)")
    p.add_argument("--cuda-home", default="/usr/local/cuda", dest="cuda_home")
    p.add_argument("--arch", default="sm_120")
    p.add_argument("--top-n", type=int, default=10, dest="top_n",
                   help="tile shapes to specialise per matrix when compiling prisma_gpu (default: 10)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed passed to all binaries (default 42)")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance for comparisons (default 1e-6)")
    p.add_argument("--atol", type=float, default=1e-6,
                   help="absolute tolerance for comparisons (default 1e-6)")
    p.add_argument("--timeout", type=int, default=300,
                   help="per-contender timeout in seconds (default 300)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bin_dir = Path(args.work_dir) if args.work_dir else _TMP_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    # Select active contenders
    if args.kernels:
        want = set(args.kernels.split(","))
        active_prisma = [c for c in PRISMA_GPU_CONTENDERS if c[0] in want]
        active_cusparse = [c for c in CUSPARSE_CONTENDERS if c[0] in want]
        if not active_prisma and not active_cusparse:
            sys.exit(f"No contenders matched --kernels {args.kernels!r}")
    else:
        active_prisma = [
            c for c in PRISMA_GPU_CONTENDERS
            if c[0] != "prisma_gpu_cuda_fp64_nofallback" or args.nofallback
            if c[0] != "prisma_gpu_cuda_fp64_row_group" or args.row_group
            if c[0] != "prisma_gpu_cuda_fp64" or not args.no_fp64
            if c[0] != "prisma_gpu_cuda_fp32" or not args.no_fp32
        ]
        active_cusparse = [] if args.no_cusparse else list(CUSPARSE_CONTENDERS)

    active = active_prisma + [(l, p, []) for l, p in active_cusparse]

    # Load matrix list
    if args.csv:
        matrices = load_matrix_list(Path(args.csv))
    else:
        print("No MATRICES.csv given — using built-in smoke-test list")
        matrices = _DEFAULT_MATRICES

    # Compile
    prisma_binaries: dict[str, Path] = {}
    cusparse_binary: Path | None = None
    if not args.no_compile:
        if active_prisma:
            print("Compiling prisma_gpu_spmm_bench (per-matrix kernels):")
            prisma_binaries = compile_prisma_gpu_per_matrix(
                bin_dir, matrices, args.cuda_home, args.arch, top_n=args.top_n)
            print()
        if active_cusparse:
            out = compile_cusparse_bench(bin_dir, args.cuda_home, args.arch)
            if out is not None:
                cusparse_binary = out
            print()
    else:
        for row in matrices:
            p = bin_dir / f"prisma_gpu_spmm_bench_{row['name']}"
            if p.exists():
                prisma_binaries[row["name"]] = p
        p = bin_dir / "cusparse_spmm_bench"
        if p.exists():
            cusparse_binary = p

    print(f"Matrices : {len(matrices)}")
    print(f"Kernels  : {[l for l, *_ in active]}")
    print(f"Seed     : {args.seed}")
    print(f"Tolerances: rtol={args.rtol}  atol={args.atol}")
    print()

    n_pass = n_fail = n_skip = 0

    with tempfile.TemporaryDirectory(prefix="validate_spmm_gpu_") as tmp_str:
        tmp = Path(tmp_str)
        for i, row in enumerate(matrices, 1):
            name = row["name"]
            group = row.get("group", "")
            print(f"[{i}/{len(matrices)}] {name}")

            mtx = find_mtx(name, group)
            if mtx is None:
                print("  MTX not found — skipping")
                n_skip += 1
                continue
            bsp = mtx.with_suffix(".bsp")
            if not bsp.exists():
                print("  BSP not found — skipping")
                n_skip += 1
                continue

            t0 = time.time()
            ok = validate_matrix(
                row, mtx, bsp, active,
                prisma_binaries.get(name), cusparse_binary,
                args.seed, args.rtol, args.atol, args.timeout, tmp,
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
