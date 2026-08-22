#!/usr/bin/env python3
"""
suite-sparse/validate_spgemm_gpu.py — Correctness validation for GPU SpGEMM
contenders (Prisma tc_tile/tc_block/cuda, TileSpGEMM, cuSPARSE, TACO GPU,
TC_SpGEMM), matching validate_spgemm_cpu.py's interface and structure.

For each matrix, C = A @ A (square self-product) is computed by every
contender via its own dump flag (--validate for prisma_bench.cu, --output
for bench_taco_gpu.cu/bench_tc_spgemm.cu/TileSpGEMM's own -o) and compared
against a scipy reference C_ref = S_mtx @ S_mtx. A direct S_bsp-vs-S_mtx
check (same rationale as validate_spmm_cpu.py's) runs first, since Prisma
reads .bsp while every other contender reads .mtx.

Known gap, unchanged from before this rewrite: cuSPARSE isn't independently
validated -- it reads TileSpGEMM's own printed [PASSED]/[NOT PASSED] verdict
rather than comparing its output to the scipy reference directly (TileSpGEMM
doesn't dump cuSPARSE's C separately), so it's excluded from the cross-check
below (no captured output array to compare).

Tolerance is a single uniform value here (default 2e-2/5e-3, loose enough
for the tf32 tensor-core lanes) rather than per-contender precision domains
like validate_spmm_gpu.py has -- this file hasn't had that same close
precision audit yet (prisma_tc_tile/tc_block use tf32, prisma_cuda/
tilespgemm/taco_gpu/tc_spgemm precision hasn't been individually confirmed
the way SpMM's GPU lanes were).

Usage:
  python validate_spgemm_gpu.py MATRICES.csv
  python validate_spgemm_gpu.py MATRICES.csv --bin-dir /tmp/spgemm_gpu_bins --no-compile
  python validate_spgemm_gpu.py MATRICES.csv --kernels prisma_cuda,taco_gpu
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

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from benchmark_spgemm_cpu import (  # noqa: E402
    ensure_real_general,
    find_mtx,
    load_matrix_list,
)
from benchmark_spgemm_gpu import (  # noqa: E402
    _TMP_DIR,
    compile_prisma_bench,
    compile_taco_gpu,
    compile_tc_spgemm,
)
from validate_spgemm_cpu import (  # noqa: E402
    _compare,
    _load_bsp_as_csr,
    _load_coo,
    _top_failures,
    _DEFAULT_MATRICES,
)

# ---------------------------------------------------------------------------
# Contenders — GPU SpGEMM
# ---------------------------------------------------------------------------

# Contenders that produce a captured C array (usable for cross-check).
_ARRAY_CONTENDERS = [
    "prisma_tc_tile", "prisma_tc_block", "prisma_cuda",
    "tilespgemm", "taco_gpu", "tc_spgemm",
]
# cusparse trusts TileSpGEMM's own verdict instead (see module docstring) --
# no captured array, excluded from cross-checking.
_ALL_LABELS = _ARRAY_CONTENDERS + ["cusparse"]

_DEFAULT_TILESPGEMM_DIR = "/home/kaio/artifacts/TileSpGEMM/src"
_DEFAULT_CUDA_HOME = "/usr/local/cuda"
_DEFAULT_ARCH = "sm_120"


def _parse_dump_plan(path: Path) -> list[dict]:
    """Parse prisma_bench.cu's --dump-plan output: one region per line
    (type, fused_id, row_start, row_end, col_start, col_end, k_count)."""
    regions = []
    if not path.exists():
        return regions
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) != 7:
                continue
            kind, fid, r0, r1, c0, c1, kc = parts
            regions.append({
                "type": kind, "fused_id": int(fid),
                "row_start": int(r0), "row_end": int(r1),
                "col_start": int(c0), "col_end": int(c1),
                "k_count": int(kc),
            })
    return regions


def _regions_covering(regions: list[dict], row: int, col: int) -> list[dict]:
    return [r for r in regions
            if r["row_start"] <= row < r["row_end"]
            and r["col_start"] <= col < r["col_end"]]


def _print_top_failures(C: np.ndarray, C_ref: np.ndarray, rtol: float,
                        atol: float, regions: list[dict] | None = None) -> None:
    """Print a few worst-offending (row, col) cells for a FAIL. If `regions`
    (from prisma_bench.cu's --dump-plan) is given, also print which compute
    region(s) cover each failing cell -- traces a wrong cell back to the
    exact region that produced it, to tell apart a bug in SubRegion
    construction (shared by all --tc-kernel variants) from a specific
    compute kernel."""
    for r, c, got, ref in _top_failures(C, C_ref, rtol, atol):
        print(f"    ({r}, {c}): got={got:.6g}  ref={ref:.6g}  "
              f"diff={abs(got - ref):.6g}")
        if regions:
            hits = _regions_covering(regions, r, c)
            if not hits:
                print("      (no covering region found in --dump-plan output)")
            for h in hits:
                print(f"      region: type={h['type']}  fused_id={h['fused_id']}  "
                      f"rows=[{h['row_start']},{h['row_end']})  "
                      f"cols=[{h['col_start']},{h['col_end']})  "
                      f"k_count={h['k_count']}")


def _load_mtx(path: Path, M: int, N: int) -> tuple[np.ndarray | None, str | None]:
    """Load a TileSpGEMM/bench_taco_gpu/bench_tc_spgemm --output .mtx dump
    (a real MatrixMarket file, dense or sparse) into a dense float64 array,
    padded/cropped to (M, N) in case the tool wrote a differently-shaped
    result (a shape mismatch is itself useful signal, not something to hide).

    Returns (array, None) on success or (None, error_message) on a malformed
    file -- external tools (TileSpGEMM in particular) can write a corrupt
    .mtx (e.g. a column index past the declared N), and that must degrade to
    a FAIL for that one contender, not an uncaught exception that kills the
    whole run partway through 41 matrices."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mtx = scipy.io.mmread(str(path))
        if scipy.sparse.issparse(mtx):
            got = mtx.toarray().astype(np.float64)
        else:
            got = np.asarray(mtx, dtype=np.float64)
    except Exception as e:
        return None, str(e)
    if got.shape != (M, N):
        pad = np.zeros((M, N), dtype=np.float64)
        r, c = min(got.shape[0], M), min(got.shape[1], N)
        pad[:r, :c] = got[:r, :c]
        got = pad
    return got, None


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------


def validate_matrix(
    row: dict, mtx: Path,
    prisma_bin: Path | None, tilespgemm_bin: Path | None,
    taco_bin: Path | None, tc_spgemm_bin: Path | None,
    active: list, rtol: float, atol: float, device: int, timeout: int,
    tmp: Path,
) -> bool:
    name = row["name"]
    bsp  = mtx.with_suffix(".bsp")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_mtx = scipy.io.mmread(str(mtx)).tocsr().astype(np.float64)
    M, N = S_mtx.shape
    C_ref = np.asarray((S_mtx @ S_mtx).toarray())

    all_pass = True
    results: dict[str, np.ndarray] = {}

    # --- S_bsp vs S_mtx: are they the SAME matrix? --------------------------
    # Same rationale as validate_spmm_cpu.py's identical check: Prisma reads
    # S from .bsp while every other contender here reads .mtx, so this has
    # exactly one correct answer, checked directly rather than only inferred
    # from downstream C differences.
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

    def _run(cmd, timeout_s=timeout):
        try:
            return subprocess.run([str(c) for c in cmd], capture_output=True,
                                  text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None

    # --- Prisma (reads .bsp): tc_tile / tc_block / cuda -----------------------
    prisma_lanes = [
        ("prisma_tc_tile",  ["--tc-kernel", "tile"]),
        ("prisma_tc_block", ["--tc-kernel", "block"]),
        ("prisma_cuda",     []),
    ]
    for label, extra_flags in prisma_lanes:
        if label not in active:
            continue
        if prisma_bin is None:
            print(f"  [{label:<22}] SKIP (no prisma_bench binary)")
            continue
        if not bsp.exists():
            print(f"  [{label:<22}] SKIP (no BSP: {bsp.name})")
            continue
        print(f"  [{label:<22}] ", end="", flush=True)
        vf = tmp / f"{name}_C_{label}.coo"
        pf = tmp / f"{name}_plan_{label}.txt"
        r = _run([prisma_bin, bsp, bsp, *extra_flags,
                  "--validate", vf, "--dump-plan", pf])
        if r is None:
            print("TIMEOUT")
            all_pass = False
            continue
        if r.returncode != 0:
            print(f"FAILED (exit {r.returncode})")
            if r.stderr.strip():
                print(f"    stderr: {r.stderr.strip()[-800:]}")
            all_pass = False
            continue
        if not vf.exists():
            print("FAILED (no output file written)")
            all_pass = False
            continue
        regions = _parse_dump_plan(pf)
        C = _load_coo(vf, M, N)
        results[label] = C
        ok, max_err, max_rel, failures = _compare(C, C_ref, rtol, atol)
        if ok:
            print("PASS")
        else:
            print(f"FAIL  max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                  f"failures={failures}/{C.size}")
            _print_top_failures(C, C_ref, rtol, atol, regions=regions)
            all_pass = False

    # --- TileSpGEMM (+ inline cuSPARSE verdict) -------------------------------
    tile_r = None
    if "tilespgemm" in active or "cusparse" in active:
        if tilespgemm_bin is None:
            if "tilespgemm" in active:
                print("  [tilespgemm           ] SKIP (no TileSpGEMM binary)")
            if "cusparse" in active:
                print("  [cusparse             ] SKIP (no TileSpGEMM binary)")
        else:
            # TileSpGEMM is external/third-party code with no symmetric- or
            # pattern-format handling of its own (unlike bench_taco_gpu.cu/
            # bench_tc_spgemm.cu, which both have their own is_pattern/
            # is_symmetric expansion) -- fed the raw suite-sparse .mtx
            # directly it fails on every symmetric/pattern matrix. Convert
            # first, same as benchmark_spgemm_gpu.py now does.
            tile_mtx = ensure_real_general(mtx, tmp)
            c_tile = tmp / f"{name}_C_tilespgemm.mtx"
            tile_r = _run([tilespgemm_bin, "-d", device, tile_mtx, tile_mtx, "-o", c_tile])
            if "tilespgemm" in active:
                print("  [tilespgemm           ] ", end="", flush=True)
                if tile_r is None:
                    print("TIMEOUT")
                    all_pass = False
                elif tile_r.returncode != 0:
                    print(f"FAILED (exit {tile_r.returncode})")
                    if tile_r.stderr.strip():
                        print(f"    stderr: {tile_r.stderr.strip()[-800:]}")
                    all_pass = False
                elif not c_tile.exists():
                    print("FAILED (no output file written)")
                    all_pass = False
                else:
                    C, err = _load_mtx(c_tile, M, N)
                    if err is not None:
                        print(f"FAILED (couldn't parse output: {err})")
                        all_pass = False
                    else:
                        results["tilespgemm"] = C
                        ok, max_err, max_rel, failures = _compare(C, C_ref, rtol, atol)
                        if ok:
                            print("PASS")
                        else:
                            print(f"FAIL  max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                                  f"failures={failures}/{C.size}")
                            _print_top_failures(C, C_ref, rtol, atol)
                            all_pass = False
            if "cusparse" in active:
                print("  [cusparse             ] ", end="", flush=True)
                if tile_r is None or tile_r.returncode != 0:
                    print("SKIP (TileSpGEMM didn't run -- cuSPARSE's verdict "
                          "is read from its stdout, see module docstring)")
                else:
                    passed = "[PASSED]" in tile_r.stdout and "[NOT PASSED]" not in tile_r.stdout
                    if passed:
                        print("PASS (per TileSpGEMM's own verdict)")
                    else:
                        print("FAIL (TileSpGEMM reported [NOT PASSED])")
                        all_pass = False

    # --- TACO GPU (reads .mtx) ------------------------------------------------
    if "taco_gpu" in active:
        if taco_bin is None:
            print("  [taco_gpu             ] SKIP (no bench_taco_gpu binary)")
        else:
            print("  [taco_gpu             ] ", end="", flush=True)
            # bench_taco_gpu.cu's read_mtx() has no symmetric/pattern
            # expansion (same limitation as SpGEMM CPU's bench_taco.c) --
            # convert first, same as TileSpGEMM above.
            taco_mtx = ensure_real_general(mtx, tmp)
            c_taco = tmp / f"{name}_C_taco_gpu.mtx"
            r = _run([taco_bin, taco_mtx, taco_mtx, "--output", c_taco])
            if r is None:
                print("TIMEOUT")
                all_pass = False
            elif r.returncode != 0:
                print(f"FAILED (exit {r.returncode})")
                if r.stderr.strip():
                    print(f"    stderr: {r.stderr.strip()[-800:]}")
                all_pass = False
            elif not c_taco.exists():
                print("FAILED (no output file written)")
                all_pass = False
            else:
                C, err = _load_mtx(c_taco, M, N)
                if err is not None:
                    print(f"FAILED (couldn't parse output: {err})")
                    all_pass = False
                else:
                    results["taco_gpu"] = C
                    ok, max_err, max_rel, failures = _compare(C, C_ref, rtol, atol)
                    if ok:
                        print("PASS")
                    else:
                        print(f"FAIL  max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                              f"failures={failures}/{C.size}")
                        _print_top_failures(C, C_ref, rtol, atol)
                        all_pass = False

    # --- TC_SpGEMM (reads .mtx) -----------------------------------------------
    if "tc_spgemm" in active:
        if tc_spgemm_bin is None:
            print("  [tc_spgemm            ] SKIP (no bench_tc_spgemm binary)")
        else:
            print("  [tc_spgemm            ] ", end="", flush=True)
            # Same reader limitation as bench_taco_gpu.cu -- see comment above.
            tc_mtx = ensure_real_general(mtx, tmp)
            c_tc = tmp / f"{name}_C_tc_spgemm.mtx"
            r = _run([tc_spgemm_bin, tc_mtx, tc_mtx, "--output", c_tc])
            if r is None:
                print("TIMEOUT")
                all_pass = False
            elif r.returncode != 0:
                print(f"FAILED (exit {r.returncode})")
                if r.stderr.strip():
                    print(f"    stderr: {r.stderr.strip()[-800:]}")
                all_pass = False
            elif not c_tc.exists():
                print("FAILED (no output file written)")
                all_pass = False
            else:
                C, err = _load_mtx(c_tc, M, N)
                if err is not None:
                    print(f"FAILED (couldn't parse output: {err})")
                    all_pass = False
                else:
                    results["tc_spgemm"] = C
                    ok, max_err, max_rel, failures = _compare(C, C_ref, rtol, atol)
                    if ok:
                        print("PASS")
                    else:
                        print(f"FAIL  max_err={max_err:.3g}  max_rel={max_rel:.3g}  "
                              f"failures={failures}/{C.size}")
                        _print_top_failures(C, C_ref, rtol, atol)
                        all_pass = False

    # --- Cross-compare all captured outputs -----------------------------------
    # Single uniform tolerance for every pair (see module docstring's caveat
    # about not having done a per-contender precision-domain split here yet).
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
        description="SpGEMM GPU correctness validation (Prisma/TileSpGEMM/cuSPARSE/"
                     "TACO GPU/TC_SpGEMM vs scipy reference)",
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
                   help="pre-built prisma_bench binary (skips compilation)")
    p.add_argument("--taco-gpu-bin", default="", dest="taco_gpu_bin",
                   help="pre-built bench_taco_gpu binary (skips compilation)")
    p.add_argument("--tc-spgemm-bin", default="", dest="tc_spgemm_bin",
                   help="pre-built bench_tc_spgemm binary (skips compilation)")
    p.add_argument("--tilespgemm-dir", default=_DEFAULT_TILESPGEMM_DIR, metavar="DIR",
                   help="directory containing TileSpGEMM's own 'test' binary "
                        "(external, never compiled by this script)")
    p.add_argument("--kernels", default="",
                   help="comma-separated list of kernel labels to run "
                        f"(from: {', '.join(_ALL_LABELS)})")
    p.add_argument("--cuda-home", default=_DEFAULT_CUDA_HOME)
    p.add_argument("--arch", default=_DEFAULT_ARCH)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rtol", type=float, default=2e-2,
                   help="relative tolerance (default 2e-2 -- loose enough for the "
                        "tf32 tensor-core lanes; see module docstring)")
    p.add_argument("--atol", type=float, default=5e-3,
                   help="absolute tolerance (default 5e-3, see --rtol)")
    p.add_argument("--timeout", type=int, default=300,
                   help="per-contender timeout in seconds (default 300)")
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

    want_prisma = any(l in active for l in ("prisma_tc_tile", "prisma_tc_block", "prisma_cuda"))
    prisma_bin: Path | None = None
    if want_prisma:
        if args.prisma_bin:
            prisma_bin = Path(args.prisma_bin)
        elif args.no_compile:
            prisma_bin = bin_dir / "prisma_bench"
            if not prisma_bin.exists():
                prisma_bin = None
        else:
            try:
                prisma_bin = compile_prisma_bench(args.cuda_home, args.arch, bin_dir)
            except RuntimeError as e:
                print(f"prisma_bench compilation failed: {e}")
                prisma_bin = None

    tilespgemm_bin: Path | None = None
    if "tilespgemm" in active or "cusparse" in active:
        candidate = Path(args.tilespgemm_dir) / "test"
        tilespgemm_bin = candidate if candidate.exists() else None

    taco_bin: Path | None = None
    if "taco_gpu" in active:
        if args.taco_gpu_bin:
            taco_bin = Path(args.taco_gpu_bin)
        elif args.no_compile:
            taco_bin = bin_dir / "bench_taco_gpu"
            if not taco_bin.exists():
                taco_bin = None
        else:
            try:
                taco_bin = compile_taco_gpu(args.cuda_home, args.arch, bin_dir)
            except RuntimeError as e:
                print(f"bench_taco_gpu compilation failed: {e}")
                taco_bin = None

    tc_spgemm_bin: Path | None = None
    if "tc_spgemm" in active:
        if args.tc_spgemm_bin:
            tc_spgemm_bin = Path(args.tc_spgemm_bin)
        elif args.no_compile:
            tc_spgemm_bin = bin_dir / "bench_tc_spgemm"
            if not tc_spgemm_bin.exists():
                tc_spgemm_bin = None
        else:
            try:
                tc_spgemm_bin = compile_tc_spgemm(args.cuda_home, args.arch, bin_dir)
            except RuntimeError as e:
                print(f"bench_tc_spgemm compilation failed: {e}")
                tc_spgemm_bin = None

    print(f"Matrices : {len(matrices)}")
    print(f"Kernels  : {active}")
    print(f"Tolerances: rtol={args.rtol}  atol={args.atol}")
    print()

    n_pass = n_fail = n_skip = 0

    with tempfile.TemporaryDirectory(prefix="validate_spgemm_gpu_") as tmp_str:
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
                row, mtx, prisma_bin, tilespgemm_bin, taco_bin, tc_spgemm_bin,
                active, args.rtol, args.atol, args.device, args.timeout, tmp,
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
