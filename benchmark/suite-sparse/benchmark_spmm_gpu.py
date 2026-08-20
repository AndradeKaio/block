#!/usr/bin/env python3
"""
suite-sparse/benchmark_spmm_gpu.py — GPU Prisma SpMM benchmark on real
SuiteSparse matrices.

Computes C = S * D on GPU (SpMM/GPU/prisma_gpu_spmm_bench.cu) where S is
loaded from a .bsp and D is an in-memory N×N random dense matrix. Two
precision lanes, both from the same binary via --precision:
  prisma_gpu_cuda_fp64  (primary — matches the CPU kernels' double precision)
  prisma_gpu_cuda_fp32  (secondary — never blended into the fp64 comparison;
                         validate_spmm_gpu.py's existing relaxed cross-precision
                         tolerance applies)

Also compiles/runs the cuSPARSE SpMM baseline (SpMM/GPU/cusparse_spmm_bench.cu)
via the same --seed/D-generation contract, on the SAME .bsp S every Prisma GPU
lane reads -- a single pooled binary (no per-matrix specialization, unlike
Prisma), compiled once via compile_cusparse_bench and reused across every
matrix, mirroring TACO's own single-pooled-binary pattern. See --no-cusparse.

A separate file from benchmark_spmm_cpu.py (which reuses `find_mtx`,
`load_matrix_list`, `analyze_bsp_shapes` from it) rather than folded in, the
same way SpGEMM's GPU benchmark (benchmark_gpu.py) is separate from its CPU
one: the compile path here is inherently nvcc-only (cuda-home/arch flags,
.cu sources), and benchmark_spmm_cpu.py must keep working on a GPU-less box —
this file simply isn't runnable there, which is made explicit through
--no-compile's clean-skip behavior rather than silently broken imports.

Mirrors benchmark_spmm_cpu.py's per-matrix, per-matrix-specialized-binary
structure (see compile_prisma_gpu_for_matrix below, itself mirroring
compile_prisma_for_matrix) — one prisma_gpu_spmm_bench_<name> binary per
matrix, specialized to that matrix's own top-N block shapes via
gen_spmm_gpu_kernels.py, exactly the same FLOP-coverage rationale as the
CPU side (pooling across a matrix list starves matrices whose shapes don't
dominate the aggregate).

Usage:
  python benchmark_spmm_gpu.py matrices.csv
  python benchmark_spmm_gpu.py matrices.csv --out spmm_gpu_results.csv --runs 5
  python benchmark_spmm_gpu.py matrices.csv --no-compile --work-dir /path/to/bins
"""

import argparse
import concurrent.futures
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_SPMM_DIR = _SCRIPT_DIR.parent / "SpMM"
_GPU_DIR = _SPMM_DIR / "GPU"
_SPGEMM_GPU_DIR = _SCRIPT_DIR.parent / "SpGEMM" / "GPU"
_CORE_DIR = _SCRIPT_DIR.parent / "core"
_TMP_DIR = Path("/tmp/_prismac/")

sys.path.insert(0, str(_SCRIPT_DIR))
from benchmark_spmm_cpu import (  # noqa: E402  (path must be set up first)
    _fmt,
    _hdf5_paths,
    _needs_header,
    analyze_bsp_shapes,
    find_mtx,
    load_matrix_list,
    _parse_json_block,
)

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

# Field names match prisma_gpu_spmm_bench.cu's actual JSON output (tile
# counts, since this plan is flat over post-tiling (row,col,k) chunks --
# see spmm_gpu_plan.hpp -- not block counts as an earlier design sketch
# assumed before implementation settled on tiling).
_CSV_FIELDS = [
    "matrix_name",
    "group",
    "rows",
    "cols",
    "nnz",
    "kernel",
    "run_id",
    "precision",
    "symbolic_ms",
    "tc_ms",
    "cuda_ms",
    "compute_ms",
    "total_ms",
    "n_tc_tiles",
    "n_cuda_tiles",
    "n_specialized_shapes",
]

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

# Fewer core/*.cpp sources than SpGEMM's GPU bench needs -- no
# intersection/fusion sources, matching prisma_gpu_spmm_bench.cu's own
# documented compile line (see that file's header comment for why: D is
# fully dense, so SpMM has no fusion pass to begin with).
# block_generator.cpp is required at link time even though nothing here
# calls generate_random_matrix/generate_random_blocks directly: matrix.cpp
# explicitly instantiates generate_random_matrix<float>/<double>, and those
# definitions reference generate_random_blocks -- the linker needs it
# resolved regardless of whether this binary's own code path ever calls it.
# Found by actually linking with nvcc (undefined reference to
# generate_random_blocks), not caught by code review -- the "fewer sources
# than SpGEMM needs" claim in prisma_gpu_spmm_bench.cu's header comment was
# reasoned from which PASSES are needed (no fusion), not verified by a real
# compile, and missed this transitive dependency.
_CORE_SRCS = ["block.cpp", "block_generator.cpp", "matrix.cpp", "matrix_io.cpp"]

# Files that must be copied into each matrix's own build directory because
# they (transitively) quote-include the three matrix-specific generated
# headers (spmm_gpu_shape_table.hpp, spmm_kernels_generated.cuh,
# spmm_gpu_dispatch_table.cuh) as BARE filenames, which resolve to whatever
# co-located copy sits in the including file's own directory first -- see
# spmm_gpu_dispatch.cuh's header comment. spmm_tc_tile_kernel.cuh /
# spmm_cuda_tile_kernel.cuh are NOT copied: they don't reference any
# generated file, so the shared originals (resolved via -I) are fine.
_MATRIX_SPECIFIC_FILES = [
    "prisma_gpu_spmm_bench.cu",
    "spmm_gpu_dispatch.cuh",
    "spmm_gpu_plan.hpp",
]


def _needs_compile(out: Path, *srcs: Path) -> bool:
    if not out.exists():
        return True
    t = out.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in srcs)


def compile_prisma_gpu_for_matrix(
    build_root: Path,
    bin_dir: Path,
    name: str,
    bsp: Path,
    cuda_home: str,
    arch: str,
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
) -> tuple[str, Path | None, str]:
    """Compile a prisma_gpu_spmm_bench specialized to ONE matrix's own top-N
    block shapes. Mirrors benchmark_spmm_cpu.py's compile_prisma_for_matrix —
    see that function's docstring for the FLOP-coverage rationale, which
    applies identically here.

    Returns (name, binary_path_or_None, status_message) rather than
    raising, so a parallel driver can report all results without one
    failure aborting the batch.
    """
    work = build_root / name
    work.mkdir(parents=True, exist_ok=True)
    for fname in _MATRIX_SPECIFIC_FILES:
        shutil.copy(_GPU_DIR / fname, work / fname)

    shapes = analyze_bsp_shapes(
        bsp, top_n=top_n, min_area=min_area, rank_by_flops=rank_by_flops
    )
    shapes_arg = ",".join(f"{h}x{w}" for h, w in sorted(shapes))
    r = subprocess.run(
        [
            sys.executable,
            str(_GPU_DIR / "gen_spmm_gpu_kernels.py"),
            "--shapes",
            shapes_arg,
            "--min-area",
            str(min_area),
            "--out-dir",
            str(work),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return name, None, f"kernel generation failed: {r.stderr[-500:]}"

    hdf5_inc, hdf5_lib = _hdf5_paths()
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / f"prisma_gpu_spmm_bench_{name}"
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    srcs.append(str(work / "prisma_gpu_spmm_bench.cu"))
    cmd = [
        nvcc,
        "-O3",
        "--expt-relaxed-constexpr",
        "-std=c++20",
        f"-arch={arch}",
        "-DHAVE_HDF5",
        f"-I{_CORE_DIR}",
        f"-I{_SPGEMM_GPU_DIR}",
        f"-I{_GPU_DIR}",  # resolves spmm_tc_tile_kernel.cuh/spmm_cuda_tile_kernel.cuh,
                          # deliberately NOT copied into work/ (see
                          # _MATRIX_SPECIFIC_FILES's comment) -- found missing by
                          # actually running this compile path with nvcc for the
                          # first time (fatal error: spmm_tc_tile_kernel.cuh: No
                          # such file or directory), not caught by review.
        f"-I{hdf5_inc}",
        *srcs,
        str(Path(hdf5_lib) / "libhdf5.so"),
        "-o",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return name, None, f"compile failed: {r.stderr[-500:]}"
    return name, out, f"{len(shapes)} shapes"


def compile_prisma_gpu_per_matrix(
    bin_dir: Path,
    matrices: list[dict],
    cuda_home: str,
    arch: str,
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
    max_workers: int = 12,
) -> dict[str, Path]:
    """Compile one matrix-specialized prisma_gpu binary per matrix, in
    parallel. Mirrors benchmark_spmm_cpu.py's compile_prisma_per_matrix."""
    build_root = bin_dir / "_prisma_gpu_build"
    build_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for row in matrices:
        mtx = find_mtx(row["name"], row.get("group", ""))
        if mtx is None:
            continue
        bsp = mtx.with_suffix(".bsp")
        if not bsp.exists():
            continue
        jobs.append((row["name"], bsp))

    print(
        f"  Compiling {len(jobs)} matrix-specialized prisma_gpu binaries "
        f"(up to {max_workers} in parallel) …"
    )
    t0 = time.time()
    results: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                compile_prisma_gpu_for_matrix,
                build_root,
                bin_dir,
                name,
                bsp,
                cuda_home,
                arch,
                top_n,
                min_area,
                rank_by_flops,
            ): name
            for name, bsp in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            name, path, msg = fut.result()
            if path is None:
                print(f"    [{name:<20s}] FAILED — {msg}")
            else:
                results[name] = path
                print(f"    [{name:<20s}] ok — {msg}")
    print(f"  {len(results)}/{len(jobs)} compiled ({time.time() - t0:.1f}s total)")
    return results


def compile_cusparse_bench(bin_dir: Path, cuda_home: str, arch: str) -> Path | None:
    """Compile SpMM/GPU/cusparse_spmm_bench.cu ONCE -- unlike Prisma GPU,
    cuSPARSE has no per-matrix kernel-specialization step (no
    gen_spmm_gpu_kernels.py analog: it's a vendor library call, not
    generated code), so this is a single pooled binary reused across every
    matrix, the same pattern benchmark_spmm_cpu.py's compile_binary uses for
    TACO -- just via nvcc/-lcusparse instead of g++.

    Returns None (not raise) on failure so the caller can report and
    continue rather than aborting the whole run.
    """
    hdf5_inc, hdf5_lib = _hdf5_paths()
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "cusparse_spmm_bench"
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    srcs.append(str(_GPU_DIR / "cusparse_spmm_bench.cu"))
    cmd = [
        nvcc,
        "-O3",
        "--expt-relaxed-constexpr",
        "-std=c++20",
        f"-arch={arch}",
        "-DHAVE_HDF5",
        f"-I{_CORE_DIR}",
        f"-I{hdf5_inc}",
        *srcs,
        str(Path(hdf5_lib) / "libhdf5.so"),
        "-lcusparse",
        "-o",
        str(out),
    ]
    print("  Compiling cuSPARSE SpMM bench … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(r.stderr[-1500:])
        return None
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Per-matrix run
# ---------------------------------------------------------------------------


def run_prisma_gpu(
    binary: Path,
    bsp: Path,
    runs: int,
    timeout: int,
    precision: str,
    seed: int = 42,
    specialized: bool = False,
    row_group: bool = False,
) -> dict:
    cmd = [
        str(binary),
        str(bsp),
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--precision",
        precision,
    ]
    if specialized:
        cmd.append("--specialized-kernels")
    if row_group:
        cmd.append("--row-group")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"prisma_gpu_spmm_bench exited {r.returncode}:\n{r.stderr[-800:]}"
        )
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError(f"prisma_gpu_spmm_bench: no JSON output:\n{r.stdout[-400:]}")
    return d


def benchmark_matrix_prisma_gpu(
    row: dict,
    mtx: Path,
    binary: Path,
    runs: int,
    timeout: int,
    writer,
    f_csv,
    precision: str,
    specialized: bool = False,
    row_group: bool = False,
) -> None:
    """specialized=False (default) is now the BASE GPU lane -- the
    CUDA-fallback path's generated per-shape specialized kernels
    (gen_spmm_gpu_kernels.py) are opt-in only, mirroring
    prisma_cpu_spmm_bench's own --specialized-kernels flag/default (see
    prisma_gpu_spmm_bench.cu's Args::specialized_kernels comment for why:
    that codegen path is newer and far less exercised than the generic
    kernel). Not wired into main()'s default loop with specialized=True --
    call this directly with specialized=True if/when that lane is worth
    benchmarking again.

    row_group=True selects the --row-group alternative CUDA-fallback
    dispatch (see prisma_gpu_spmm_bench.cu's Args::row_group and
    spmm_gpu_plan.hpp's RowGroupTask/RowGroupItem). Brand-new, untested on
    real hardware as of 2026-08-15 -- deliberately NOT wired into main()'s
    default loop either, until it's been validated
    (suite-sparse/validate_spmm_gpu.py's prisma_gpu_cuda_fp64_row_group row)
    on real hardware. Call this directly with row_group=True once that's
    clean."""
    bsp = mtx.with_suffix(".bsp")
    if not bsp.exists():
        print(f"  [prisma_gpu_{precision}] BSP not found — skipping")
        return
    base = {
        "matrix_name": row["name"],
        "group": row.get("group", ""),
        "rows": row.get("rows", ""),
        "cols": row.get("cols", ""),
        "nnz": row.get("nnz", ""),
    }
    suffix = "_specialized" if specialized else ("_row_group" if row_group else "")
    kernel = f"prisma_gpu_cuda_{precision}{suffix}"
    print(f"  [{kernel}] {row['name']} … ", end="", flush=True)
    try:
        d = run_prisma_gpu(binary, bsp, runs, timeout, precision,
                           specialized=specialized, row_group=row_group)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"FAILED ({e})")
        for run_id in range(runs + 1):
            writer.writerow(
                {
                    **base,
                    "kernel": kernel,
                    "run_id": run_id,
                    "precision": precision,
                    "symbolic_ms": "nan",
                    "tc_ms": "nan",
                    "cuda_ms": "nan",
                    "compute_ms": "nan",
                    "total_ms": "nan",
                    "n_tc_tiles": "",
                    "n_cuda_tiles": "",
                    "n_specialized_shapes": "",
                }
            )
        f_csv.flush()
        return

    symbolic_ms = d.get("symbolic_ms", [])
    tc_ms = d.get("tc_ms", [])
    cuda_ms = d.get("cuda_ms", [])
    n_tc_tiles = d.get("n_tc_tiles", "")
    n_cuda_tiles = d.get("n_cuda_tiles", "")
    n_specialized = d.get("n_specialized_shapes", "")
    if not tc_ms or not cuda_ms:
        print("FAILED (empty tc_ms/cuda_ms in JSON)")
        for run_id in range(runs + 1):
            writer.writerow(
                {
                    **base,
                    "kernel": kernel,
                    "run_id": run_id,
                    "precision": precision,
                    "symbolic_ms": "nan",
                    "tc_ms": "nan",
                    "cuda_ms": "nan",
                    "compute_ms": "nan",
                    "total_ms": "nan",
                    "n_tc_tiles": "",
                    "n_cuda_tiles": "",
                    "n_specialized_shapes": "",
                }
            )
        f_csv.flush()
        return

    for run_id, (sym, tc, cu) in enumerate(zip(symbolic_ms, tc_ms, cuda_ms)):
        compute = tc + cu
        writer.writerow(
            {
                **base,
                "kernel": kernel,
                "run_id": run_id,
                "precision": precision,
                "symbolic_ms": _fmt(sym),
                "tc_ms": _fmt(tc),
                "cuda_ms": _fmt(cu),
                "compute_ms": _fmt(compute),
                "total_ms": _fmt(sym + compute),
                "n_tc_tiles": n_tc_tiles,
                "n_cuda_tiles": n_cuda_tiles,
                "n_specialized_shapes": n_specialized,
            }
        )
    f_csv.flush()

    timed = [
        sym + tc + cu for sym, tc, cu in zip(symbolic_ms[1:], tc_ms[1:], cuda_ms[1:])
    ] or [sym + tc + cu for sym, tc, cu in zip(symbolic_ms, tc_ms, cuda_ms)]
    avg = sum(timed) / len(timed)
    print(f"avg {avg:.3f} ms  ({len(tc_ms)} runs incl. warmup)")


def run_cusparse(
    binary: Path,
    bsp: Path,
    runs: int,
    timeout: int,
    precision: str,
    algo: str = "default",
    seed: int = 42,
) -> dict:
    cmd = [
        str(binary),
        str(bsp),
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--precision",
        precision,
        "--algo",
        algo,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"cusparse_spmm_bench exited {r.returncode}:\n{r.stderr[-800:]}"
        )
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError(f"cusparse_spmm_bench: no JSON output:\n{r.stdout[-400:]}")
    return d


def benchmark_matrix_cusparse(
    row: dict,
    mtx: Path,
    binary: Path,
    runs: int,
    timeout: int,
    writer,
    f_csv,
    precision: str,
    algo: str = "default",
) -> None:
    """Mirrors benchmark_matrix_prisma_gpu's structure and CSV-writing
    conventions. cuSPARSE has no TC/CUDA-fallback split and no
    per-matrix-specialized tile counts, so those columns are written blank
    for every cusparse_* row -- consistent with how Prisma's own one-time
    structural costs (bsp_read_ms/plan_build_ms/device_upload_ms/
    pipe_total_ms) are JSON-only, never CSV columns, today."""
    bsp = mtx.with_suffix(".bsp")
    if not bsp.exists():
        print(f"  [cusparse_{precision}] BSP not found — skipping")
        return
    base = {
        "matrix_name": row["name"],
        "group": row.get("group", ""),
        "rows": row.get("rows", ""),
        "cols": row.get("cols", ""),
        "nnz": row.get("nnz", ""),
    }
    kernel = f"cusparse_{precision}"
    print(f"  [{kernel}] {row['name']} … ", end="", flush=True)
    try:
        d = run_cusparse(binary, bsp, runs, timeout, precision, algo)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"FAILED ({e})")
        for run_id in range(runs + 1):
            writer.writerow(
                {
                    **base,
                    "kernel": kernel,
                    "run_id": run_id,
                    "precision": precision,
                    "symbolic_ms": "nan",
                    "tc_ms": "",
                    "cuda_ms": "",
                    "compute_ms": "nan",
                    "total_ms": "nan",
                    "n_tc_tiles": "",
                    "n_cuda_tiles": "",
                    "n_specialized_shapes": "",
                }
            )
        f_csv.flush()
        return

    symbolic_ms = d.get("symbolic_ms", [])
    compute_ms = d.get("compute_ms", [])
    if not compute_ms:
        print("FAILED (empty compute_ms in JSON)")
        for run_id in range(runs + 1):
            writer.writerow(
                {
                    **base,
                    "kernel": kernel,
                    "run_id": run_id,
                    "precision": precision,
                    "symbolic_ms": "nan",
                    "tc_ms": "",
                    "cuda_ms": "",
                    "compute_ms": "nan",
                    "total_ms": "nan",
                    "n_tc_tiles": "",
                    "n_cuda_tiles": "",
                    "n_specialized_shapes": "",
                }
            )
        f_csv.flush()
        return

    for run_id, (sym, comp) in enumerate(zip(symbolic_ms, compute_ms)):
        writer.writerow(
            {
                **base,
                "kernel": kernel,
                "run_id": run_id,
                "precision": precision,
                "symbolic_ms": _fmt(sym),
                "tc_ms": "",
                "cuda_ms": "",
                "compute_ms": _fmt(comp),
                "total_ms": _fmt(sym + comp),
                "n_tc_tiles": "",
                "n_cuda_tiles": "",
                "n_specialized_shapes": "",
            }
        )
    f_csv.flush()

    timed = compute_ms[1:] or compute_ms
    avg = sum(timed) / len(timed)
    print(f"avg {avg:.3f} ms  ({len(compute_ms)} runs incl. warmup)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="GPU Prisma SpMM benchmark on SuiteSparse matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "csv", metavar="MATRICES.csv", help="input CSV with at least a 'name' column"
    )

    g = p.add_argument_group("Run control")
    g.add_argument(
        "--runs",
        type=int,
        default=5,
        help="timed repetitions per matrix; run_id 0 = warmup (default: 5)",
    )
    g.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for D generation (default 42) -- must match "
        "validate_spmm_gpu.py / other contenders for cross-comparison",
    )
    g.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-matrix timeout in seconds (default: 300)",
    )

    g = p.add_argument_group("Paths")
    g.add_argument(
        "--out",
        default="spmm_gpu_results.csv",
        help="output CSV, append mode (default: spmm_gpu_results.csv)",
    )
    g.add_argument(
        "--work-dir",
        default="",
        help="directory for compiled binaries (default: SpMM/GPU/)",
    )
    g.add_argument(
        "--prisma-gpu-bin",
        default="",
        dest="prisma_gpu_bin",
        help="pre-built prisma_gpu_spmm_bench binary (skips "
        "compilation, used for every matrix — overrides "
        "per-matrix compilation)",
    )
    g.add_argument(
        "--cusparse-bin",
        default="",
        dest="cusparse_bin",
        help="pre-built cusparse_spmm_bench binary (skips compilation)",
    )

    g = p.add_argument_group("Kernel generation")
    g.add_argument(
        "--top-n",
        type=int,
        default=10,
        dest="top_n",
        help="number of tile shapes to specialise per matrix "
        "(default: 10) -- nvcc compiles markedly slower than "
        "g++ per-shape, so this starts smaller than the CPU "
        "generator's default of 30; a cost/benefit knob to "
        "sweep on real hardware, not a correctness one.",
    )
    g.add_argument(
        "--min-block-area",
        type=int,
        default=4,
        dest="min_block_area",
        help="minimum mined H*W for a shape to be eligible for "
        "tile-shape decomposition (default: 4), same "
        "semantics as the CPU generator's flag",
    )
    g.add_argument(
        "--count-ranking",
        action="store_true",
        dest="count_ranking",
        help="rank candidate shapes by block count instead of FLOPs",
    )

    g = p.add_argument_group("Build / skip")
    g.add_argument(
        "--no-compile",
        action="store_true",
        help="skip compilation; binaries must already exist",
    )
    g.add_argument(
        "--no-fp64", action="store_true", help="skip the prisma_gpu_cuda_fp64 lane"
    )
    g.add_argument(
        "--no-fp32", action="store_true", help="skip the prisma_gpu_cuda_fp32 lane"
    )
    g.add_argument(
        "--no-cusparse", action="store_true", help="skip the cuSPARSE contender entirely"
    )
    g.add_argument(
        "--row-group",
        action="store_true",
        dest="row_group",
        help="ALSO benchmark the --row-group alternative CUDA-fallback "
        "dispatch (prisma_gpu_cuda_fp64_row_group / _fp32_row_group), "
        "alongside the base lane -- brand-new, validate correctness via "
        "validate_spmm_gpu.py's prisma_gpu_cuda_fp64_row_group row before "
        "trusting these timings",
    )
    g.add_argument("--cuda-home", default="/usr/local/cuda")
    g.add_argument("--arch", default="sm_120")
    g.add_argument(
        "--prisma-compile-workers",
        type=int,
        default=12,
        dest="prisma_compile_workers",
        help="parallel nvcc compiles when building per-matrix "
        "prisma_gpu binaries (default: 12)",
    )
    g.add_argument(
        "--cusparse-algo",
        default="default",
        dest="cusparse_algo",
        choices=["default", "alg1", "alg2", "alg3"],
        help="cusparseSpMMAlg_t to use (default: cuSPARSE's own internal "
        "heuristic) -- a performance sweep/ablation knob, analogous to "
        "--top-n for Prisma's specialized kernels; not something to "
        "cherry-pick per-matrix for a favorable head-to-head number",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    matrices = load_matrix_list(Path(args.csv))

    prisma_binary: Path | None = None  # single-binary mode (--prisma-gpu-bin)
    prisma_binaries: dict[str, Path] = {}  # per-matrix mode (default)

    bin_dir = Path(args.work_dir) if args.work_dir else _TMP_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    if args.prisma_gpu_bin:
        prisma_binary = Path(args.prisma_gpu_bin)
        if not prisma_binary.exists():
            sys.exit(f"prisma_gpu binary not found: {prisma_binary}")
    elif args.no_compile:
        for row in matrices:
            b = bin_dir / f"prisma_gpu_spmm_bench_{row['name']}"
            if b.exists():
                prisma_binaries[row["name"]] = b
        if not prisma_binaries:
            sys.exit(
                f"no prisma_gpu_spmm_bench_<matrix> binaries found in "
                f"{bin_dir} — build first (drop --no-compile, or point "
                f"--work-dir at a directory with pre-built binaries)"
            )
    else:
        print("Compiling Prisma GPU SpMM (per-matrix kernels):")
        prisma_binaries = compile_prisma_gpu_per_matrix(
            bin_dir,
            matrices,
            args.cuda_home,
            args.arch,
            top_n=args.top_n,
            min_area=args.min_block_area,
            rank_by_flops=not args.count_ranking,
            max_workers=args.prisma_compile_workers,
        )
        print()

    cusparse_binary: Path | None = None
    if not args.no_cusparse:
        if args.cusparse_bin:
            cusparse_binary = Path(args.cusparse_bin)
            if not cusparse_binary.exists():
                sys.exit(f"cusparse binary not found: {cusparse_binary}")
        elif args.no_compile:
            b = bin_dir / "cusparse_spmm_bench"
            if b.exists():
                cusparse_binary = b
            else:
                print(
                    f"  cusparse_spmm_bench not found in {bin_dir} — "
                    "skipping cuSPARSE lane"
                )
        else:
            print("Compiling cuSPARSE SpMM bench:")
            cusparse_binary = compile_cusparse_bench(
                bin_dir, args.cuda_home, args.arch
            )
            print()

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output  : {csv_path}")
    print(f"Matrices: {len(matrices)}")
    print(f"Runs    : {args.runs}")
    print()

    write_header = _needs_header(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f_csv.write(f"# {ts}  input={args.csv}  runs={args.runs}\n")
        writer = csv.DictWriter(
            f_csv, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()

        for i, row in enumerate(matrices, 1):
            name = row["name"]
            group = row.get("group", "")
            print(f"[{i}/{len(matrices)}] {name}")

            mtx = find_mtx(name, group)
            if mtx is None:
                print("  MTX not found — skipping")
                continue

            active_binary = (
                prisma_binary
                if prisma_binary is not None
                else prisma_binaries.get(name)
            )
            if active_binary is None:
                print(f"  [prisma_gpu_*] no compiled binary for {name} — skipping")
            else:
                if not args.no_fp64:
                    benchmark_matrix_prisma_gpu(
                        row,
                        mtx,
                        active_binary,
                        args.runs,
                        args.timeout,
                        writer,
                        f_csv,
                        "fp64",
                    )
                    if args.row_group:
                        benchmark_matrix_prisma_gpu(
                            row,
                            mtx,
                            active_binary,
                            args.runs,
                            args.timeout,
                            writer,
                            f_csv,
                            "fp64",
                            row_group=True,
                        )
                if not args.no_fp32:
                    benchmark_matrix_prisma_gpu(
                        row,
                        mtx,
                        active_binary,
                        args.runs,
                        args.timeout,
                        writer,
                        f_csv,
                        "fp32",
                    )
                    if args.row_group:
                        benchmark_matrix_prisma_gpu(
                            row,
                            mtx,
                            active_binary,
                            args.runs,
                            args.timeout,
                            writer,
                            f_csv,
                            "fp32",
                            row_group=True,
                        )

            # cuSPARSE is a pooled binary, independent of whether Prisma's
            # per-matrix compile succeeded for THIS matrix -- don't couple
            # the two lanes' availability.
            if cusparse_binary is not None:
                if not args.no_fp64:
                    benchmark_matrix_cusparse(
                        row,
                        mtx,
                        cusparse_binary,
                        args.runs,
                        args.timeout,
                        writer,
                        f_csv,
                        "fp64",
                        args.cusparse_algo,
                    )
                if not args.no_fp32:
                    benchmark_matrix_cusparse(
                        row,
                        mtx,
                        cusparse_binary,
                        args.runs,
                        args.timeout,
                        writer,
                        f_csv,
                        "fp32",
                        args.cusparse_algo,
                    )
    print(f"\nDone. Results in {csv_path}")


if __name__ == "__main__":
    main()
