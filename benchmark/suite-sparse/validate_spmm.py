#!/usr/bin/env python3
"""
suite-sparse/validate_spmm.py — Correctness validation for all SpMM contenders.

For each matrix, every contender is run with --seed S and --dump-c to write its
output C matrix.  prisma_cpu additionally dumps D via --dump-d.  The scipy
reference C_ref = S @ D is computed in Python and compared against each binary's
output.  Since all contenders share the same seed and the same mt19937_64 RNG,
they produce identical D, so all C outputs can also be cross-compared directly.

Usage:
  python validate_spmm.py MATRICES.csv
  python validate_spmm.py MATRICES.csv --bin-dir ../SpMM/ --no-compile
  python validate_spmm.py MATRICES.csv --kernels prisma_tiled,prisma_static
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
# Paths (mirrors benchmark_spmm.py)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_SPMM_DIR   = _SCRIPT_DIR.parent / "SpMM"
_DATA_ROOT  = Path("/home/kaio/datasets/suite-sparse")

# ---------------------------------------------------------------------------
# Contenders
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
    # These are now the BASE GPU lane (2026-08-15): prisma_gpu_spmm_bench's
    # --specialized-kernels defaults to false (mirrors prisma_cpu_spmm_bench
    # exactly -- see that binary's own --specialized-kernels flag), so the
    # CUDA-fallback path always runs the generic kernel here, never the
    # generated per-shape specialized dispatch -- that codegen path is
    # newer/less exercised and was producing a base-vs-optimized comparison
    # gap (no unoptimized GPU baseline existed) until this default flipped.
    # TC (tensor-core) path is unaffected -- still active for TC-eligible
    # blocks either way, see spmm_tc_tile_kernel.cuh.
    ("prisma_gpu_cuda_fp64", "prisma_gpu_spmm_bench",   ".bsp", ["--precision", "fp64"]),
    ("prisma_gpu_cuda_fp32", "prisma_gpu_spmm_bench",   ".bsp", ["--precision", "fp32"]),
    # Debug/bisection row, NOT for normal use -- same binary as
    # prisma_gpu_cuda_fp64 (label prefix "prisma_gpu" resolves to the same
    # per-matrix binary in _prisma_binary_path regardless of suffix, so no
    # recompile is needed), but forces every block through the CUDA-fallback
    # path via prisma_gpu_spmm_bench.cu's --force-cuda-fallback flag. Added
    # to definitively confirm/deny whether a correctness bug is confined to
    # the TC (tensor-core) path: run with
    #   --kernels prisma_cpu,prisma_gpu_cuda_fp64,prisma_gpu_cuda_fp64_nofallback
    # If this row PASSes while prisma_gpu_cuda_fp64 FAILs on the same
    # matrix, the bug is TC-specific.
    ("prisma_gpu_cuda_fp64_nofallback", "prisma_gpu_spmm_bench", ".bsp",
     ["--precision", "fp64", "--force-cuda-fallback"]),
    # --row-group alternative CUDA-fallback dispatch (2026-08-15, brand new
    # -- see spmm_gpu_plan.hpp's RowGroupTask/RowGroupItem and
    # spmm_cuda_tile_kernel.cuh's spmm_row_group_kernel for the design).
    # Same binary/precision domain as prisma_gpu_cuda_fp64 (full double,
    # no tf32) -- expected to PASS at the caller's strict tolerance and
    # agree with prisma_cpu/prisma_gpu_cuda_fp64 bit-for-bit modulo normal
    # floating-point summation-order noise, since it computes the exact
    # same sum, just grouped differently before each atomicAdd. This is
    # the FIRST real-hardware test of this code path -- run it before
    # trusting any --row-group timing number.
    ("prisma_gpu_cuda_fp64_row_group", "prisma_gpu_spmm_bench", ".bsp",
     ["--precision", "fp64", "--row-group"]),
    # cuSPARSE vendor-library baseline (SpMM/GPU/cusparse_spmm_bench.cu) --
    # a single POOLED binary (no per-matrix specialization, unlike Prisma),
    # compiled by benchmark_spmm_gpu.py's compile_cusparse_bench into
    # _GPU_BIN_DIR, same as prisma_gpu_* -- see _prisma_binary_path's
    # "cusparse" branch. Reads S from .bsp (same truncated-then-upcast
    # precision every other .bsp contender reads), so it's compared
    # against C_ref_bsp like Prisma, not C_ref_mtx like TACO -- see the
    # ext-based (not label-based) reference selection below.
    # cusparse_fp64 is full native double, no tf32 shortcuts -- expected to
    # PASS at the caller's strict tolerance, same as prisma_cpu.
    # cusparse_fp32 is expected to fail its OWN strict individual check the
    # same way prisma_gpu_cuda_fp32 already does (full float32 compute,
    # not just S storage) -- only cross-checks between same-fp32-domain
    # contenders get _F32_CROSS_RTOL/_F32_CROSS_ATOL relaxed treatment.
    ("cusparse_fp64", "cusparse_spmm_bench", ".bsp", ["--precision", "fp64"]),
    ("cusparse_fp32", "cusparse_spmm_bench", ".bsp", ["--precision", "fp32"]),
]

# ---------------------------------------------------------------------------
# Matrix location (same as benchmark_spmm.py)
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
# Compilation (delegates to benchmark_spmm.py helpers)
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
    from benchmark_spmm import compile_binary, compile_prisma_per_matrix, _KERNELS
    print("Compiling TACO kernels:")
    for k, d in _KERNELS:
        compile_binary(bin_dir, k, d)
    print()
    print("Compiling Prisma CPU SpMM (per-matrix kernels):")
    compile_prisma_per_matrix(bin_dir, matrices)
    print()


_GPU_BIN_DIR = Path("/tmp/_prismac/")  # benchmark_spmm_gpu.py's own default
# --work-dir (matches benchmark_spmm.py's own _TMP_DIR default for the same
# reason -- keep compiled per-matrix binaries out of the source tree). Found
# stale (still pointing at the old SpMM/GPU/ location) while actually
# running benchmark_spmm_gpu.py's real compile path for the first time --
# without this fix, validate_spmm.py would silently SKIP every GPU
# contender ("binary not found") even right after a successful compile.


def _prisma_binary_path(bin_dir: Path, name: str, label: str = "prisma_cpu",
                        stem: str = "") -> Path:
    """Resolve a contender's binary path for one matrix. Three binary-layout
    families exist:
      1. Prisma CPU (label startswith "prisma", not "prisma_gpu") --
         per-matrix specialized, under --bin-dir. See
         compile_prisma_per_matrix in benchmark_spmm.py.
      2. Prisma GPU (prisma_gpu_*) -- per-matrix specialized, always under
         _GPU_BIN_DIR regardless of the caller's --bin-dir: compiled by a
         completely separate script (benchmark_spmm_gpu.py, nvcc-only)
         with its own directory convention, so following the CPU-oriented
         --bin-dir here would silently look in the wrong place. See
         compile_prisma_gpu_per_matrix in benchmark_spmm_gpu.py.
      3. Pooled binaries (`stem` used as-is, no per-matrix compilation) --
         TACO's pool lives under --bin-dir (CPU toolchain, see
         benchmark_spmm.py's compile_binary); cuSPARSE's pool lives under
         _GPU_BIN_DIR (compiled by benchmark_spmm_gpu.py's nvcc toolchain,
         same reason prisma_gpu_* does -- see compile_cusparse_bench).
         Distinguished by label prefix rather than input ext, since a
         pooled *CPU* contender reading .bsp is conceivable in principle
         even though none exists today.
    """
    if label.startswith("prisma_gpu"):
        return _GPU_BIN_DIR / f"prisma_gpu_spmm_bench_{name}"
    if label.startswith("prisma"):
        return bin_dir / f"prisma_cpu_spmm_bench_{name}"
    if label.startswith("cusparse"):
        return _GPU_BIN_DIR / stem
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


# prisma_gpu_cuda_fp64's TC (tensor-core) path computes tf32 for any
# TC-eligible S-block (see spmm_tc_tile_kernel.cuh) -- ~10 mantissa bits,
# not fp64's 52. On a well-conditioned matrix that's just a small uniform
# relative error, but on the ill-conditioned matrices in this suite
# (bcsstk27/bundle1/msc10848 -- stiffness/bundle-adjustment matrices with
# per-row dynamic range up to 1e8+, verified directly against real .bsp
# data during the 2026-08-14 investigation) individual C cells are often
# near-zero results of catastrophic cancellation between much larger
# terms. A PER-CELL relative tolerance is meaningless there: a
# numerically unremarkable tf32-scale absolute error on a cancelled-to-
# near-zero cell reads as a huge "relative" error despite nothing being
# wrong. _compare_global_scale anchors the tolerance to the reference
# matrix's own overall magnitude instead of each cell's own (possibly
# ~0) value -- this still catches real corruption (the actual bug found
# during that investigation put cells at ~1e34-1e38, dwarfing any
# reasonable global-scale bound) while tolerating expected tf32 rounding
# noise on cancellation-heavy cells. Not used for
# prisma_gpu_cuda_fp64_nofallback, which computes fully in T (double) via
# the CUDA-fallback path with no tf32 involved at all, so the standard
# per-cell _compare is the correct, tighter check there.
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

    def _is_tc_lane(label: str) -> bool:
        """True for any prisma_gpu_cuda_fp64* variant that still routes
        TC-eligible blocks through the (unchanged) tf32 tensor-core path
        -- i.e. every fp64 Prisma-GPU label EXCEPT _nofallback, which
        explicitly disables TC via --force-cuda-fallback. --row-group and
        --specialized-kernels (and any future *_<suffix> variant) only
        change the CUDA-fallback dispatch strategy, never the TC path
        itself, so they all still need the SAME relaxed tf32 tolerance
        prisma_gpu_cuda_fp64 does -- see _compare_global_scale's
        docstring. Originally an exact-string match
        (label == "prisma_gpu_cuda_fp64"), which silently stopped
        applying the relaxed check to prisma_gpu_cuda_fp64_row_group and
        made a numerically-correct result (verified independently against
        the base lane to ~1e-9, pure summation-order noise) look like a
        FAIL under the strict tolerance -- found the hard way validating
        --row-group's first real-hardware run (2026-08-15).
        """
        return (label.startswith("prisma_gpu_cuda_fp64")
                and not label.endswith("_nofallback"))

    all_pass = True
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
        # by label naming: .bsp readers (Prisma, cuSPARSE) get the
        # truncated-precision-matching C_ref_bsp; .mtx readers (TACO) get
        # the full-float64 C_ref_mtx.
        C_ref = C_ref_bsp if ext == ".bsp" else C_ref_mtx
        if C_ref is not None:
            # Any TC lane (see _is_tc_lane) computes tf32 for TC-eligible
            # blocks -- see _compare_global_scale's docstring for why a
            # per-cell relative tolerance doesn't fit it.
            if _is_tc_lane(label):
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
        else:
            print(f"  [{label:<22}] (no reference — output captured)")

    # --- Cross-compare all captured outputs --------------------------------
    # Three independent precision axes distinguish contenders, each its own
    # real, expected gap rather than a compute bug:
    #   (a) S storage precision — Prisma reads S from .bsp (float32-stored
    #       values, upcast to double); TACO reads S from .mtx (native
    #       float64). Verified: bsp values are bit-exact float32
    #       truncations of the mtx values.
    #   (b) C compute precision — prisma_gpu_cuda_fp32 computes the whole
    #       GEMM (D generation downcast, every FMA, atomicAdd accumulation)
    #       in float32; every other contender computes in float64. This is
    #       a coarser, different-natured gap than (a) — not just S's
    #       stored constants losing precision, but the accumulation itself.
    #   (c) tf32 tensor-core acceleration — prisma_gpu_cuda_fp64 computes
    #       TC-eligible S-blocks in tf32 (see _compare_global_scale's
    #       docstring); prisma_gpu_cuda_fp64_nofallback does not (forces
    #       every block through the CUDA-fallback path, fully double), so
    #       despite the similar label it belongs in the full-precision
    #       domain along with prisma_cpu, not this one.
    # A pair differing on ANY axis needs a relaxed tolerance; only a pair
    # matching on all three shares its reference's precision exactly and
    # keeps the caller's strict tolerance, gating all_pass — a mismatch
    # there would indicate a real kernel bug. The individual PASS checks
    # above already validate each contender against the reference matching
    # its own S precision, which is the real correctness gate; cross-domain
    # mismatches here are reported for visibility only.
    _F32_CROSS_RTOL = 1e-4
    _F32_CROSS_ATOL = 1e-6

    def _domain(label: str, ext: str) -> tuple[str, bool, bool]:
        """Classify a contender by precision domain for cross-check
        tolerance selection. First component used to be
        label.startswith("prisma") -- a naming-based proxy for "reads S
        from .bsp (block-truncated storage precision) vs .mtx (full
        float64)", which is what actually determines whether two
        contenders' S data is close enough to expect strict agreement.
        Using ext directly instead is backward-compatible (every
        pre-cuSPARSE contender's ext and the old naming proxy always
        coincided: .bsp <=> prisma-family, .mtx <=> TACO) and correctly
        buckets a new non-"prisma"-named .bsp contender (cuSPARSE) into
        the same strict-agreement domain as prisma_cpu/
        prisma_gpu_cuda_fp64_nofallback instead of treating it as an
        always-expected cross-domain mismatch against them.
        """
        return (ext, label.endswith("_fp32"), _is_tc_lane(label))

    labels = list(results.keys())
    if len(labels) > 1:
        cross_ok = True
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                same_domain = _domain(la, label_ext[la]) == _domain(lb, label_ext[lb])
                if _is_tc_lane(la) or _is_tc_lane(lb):
                    cmp_fn = _compare_global_scale
                    cmp_rtol, cmp_atol = _TC_RTOL, _TC_ATOL
                    gap_desc = "tf32 tensor-core precision gap"
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
        description="SpMM correctness validation (all contenders vs scipy reference)",
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

    with tempfile.TemporaryDirectory(prefix="validate_spmm_") as tmp_str:
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
