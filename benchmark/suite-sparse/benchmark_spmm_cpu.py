#!/usr/bin/env python3
"""
suite-sparse/benchmark_spmm_cpu.py — SpMM benchmark on real SuiteSparse matrices.

Computes C = S * D where S is loaded from MTX and D is an in-memory N×N
random dense matrix (N = number of columns of S; matrices are always square).
Timing is split into symbolic and compute phases, one row per run.
run_id=0 is the warmup run; timed runs are run_id=1..R.

The symbolic phase (TACO's pack_B(); Prisma's row-group build + D-locality
resort + task-list build) is redone from scratch on every timed run, not
built once and amortized -- a one-off caller pays the full symbolic cost on
every call, so symbolic_ms is a mean of R real per-run measurements.

Usage:
  python benchmark_spmm_cpu.py matrices.csv
  python benchmark_spmm_cpu.py matrices.csv --out spmm_results.csv --runs 5
  python benchmark_spmm_cpu.py matrices.csv --no-compile --bin /path/to/bench_taco_spmm
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import perf_wrap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_SPMM_DIR = _SCRIPT_DIR.parent / "SpMM"
_TMP_DIR = Path("/tmp/_prismac/")


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "matrix_name",
    "group",
    "rows",
    "cols",
    "nnz",
    "kernel",
    "run_id",
    "threads",
    "symbolic_ms",
    "compute_ms",
    "total_ms",
] + perf_wrap.PERF_CSV_FIELDS

_NAN = float("nan")


def _parse_thread_sweep(spec: str) -> list[int]:
    """'32,16,8,4,2,1' -> that exact list; '32' -> halved down to 1
    (32,16,8,4,2,1) since integer division naturally terminates at 1."""
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    n = int(spec)
    vals = []
    while n >= 1:
        vals.append(n)
        n //= 2
    return vals


def _fmt(v: float) -> str:
    return "nan" if v != v else f"{v:.4f}"


def _needs_header(csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True
    expected = ",".join(_CSV_FIELDS)
    with open(csv_path, newline="") as f:
        for line in f:
            if not line.startswith("#"):
                return line.rstrip("\r\n") != expected
    return True


# ---------------------------------------------------------------------------
# Matrix list and MTX location
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("/home/kaio/datasets/suite-sparse")


def _readable(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0 and os.access(p, os.R_OK)
    except OSError:
        return False


def find_mtx(name: str, group: str) -> Path | None:
    """Return the .mtx path from the local dataset, or None if not found."""
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
# Compilation
# ---------------------------------------------------------------------------


def compile_binary(work_dir: Path, kernel: str, define: str | None) -> Path:
    out = work_dir / f"bench_taco_spmm_{kernel}"
    src = str(_SPMM_DIR / "bench_taco_spmm.cpp")
    # -march=native matches prisma_cpu_spmm's compile flags (see
    # compile_prisma_cpu_spmm below). Without it, GCC only auto-vectorizes
    # TACO's compute() loop to baseline SSE2 (2-wide, no FMA) while prisma's
    # hand-written kernels get AVX-512+FMA (8-wide) — a compiler-flag
    # handicap unrelated to the algorithms being compared, and a violation
    # of apples-to-apples timing (same class of unfairness as comparing
    # fp32 vs fp64).
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-fopenmp",
        "-march=native",
        "-Drestrict=__restrict__",
        f"-I{_SPMM_DIR}",
        src,
        "-lm",
        "-o",
        str(out),
    ]
    if define:
        cmd.insert(1, f"-D{define}")
    label = f"bench_taco_spmm ({kernel})"
    print(f"  Compiling {label} … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-3000:]}")
        raise RuntimeError(f"{label} compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Per-matrix run — TACO
# ---------------------------------------------------------------------------

_ASM_RE = re.compile(r"run_(\d+)_assemble_ns=(\d+)")
_COMP_RE = re.compile(r"run_(\d+)_compute_ns=(\d+)")


def _parse_json_block(stdout: str) -> dict:
    if "JSON_BEGIN" not in stdout or "JSON_END" not in stdout:
        return {}
    s = stdout.index("JSON_BEGIN") + len("JSON_BEGIN")
    e = stdout.index("JSON_END")
    try:
        return json.loads(stdout[s:e].strip())
    except json.JSONDecodeError:
        return {}


def run_spmm(
    binary: Path, mtx: Path, runs: int, timeout: int, perf: bool = False
) -> tuple[list[tuple[float, float, float]], dict]:
    """Run bench_taco_spmm and return ([(symbolic_ms, compute_ms, total_ms)],
    perf_metrics) for run_id 0..runs (inclusive).

    perf_metrics is one aggregate perf-stat/RSS reading for the entire
    process lifetime, from a second, separate invocation wrapped in
    `perf stat` -- see perf_wrap.py. NaN in every field if perf=False or perf
    itself failed (never affects the real timing/success of this run)."""
    cmd = [str(binary), str(mtx), "--runs", str(runs)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"bench_taco_spmm exited {r.returncode}:\n{r.stderr[-800:]}")
    asm_ns: dict[int, int] = {}
    comp_ns: dict[int, int] = {}
    for line in r.stdout.splitlines():
        m = _ASM_RE.match(line)
        if m:
            asm_ns[int(m.group(1))] = int(m.group(2))
        m = _COMP_RE.match(line)
        if m:
            comp_ns[int(m.group(1))] = int(m.group(2))
    if not asm_ns or not comp_ns:
        raise RuntimeError(
            f"bench_taco_spmm: could not parse timing from stdout:\n{r.stdout[-400:]}"
        )
    result = []
    for run_id in range(runs + 1):
        sym = asm_ns.get(run_id, 0) / 1e6
        comp = comp_ns.get(run_id, 0) / 1e6
        result.append((sym, comp, sym + comp))
    perf_metrics = perf_wrap.measure(cmd, timeout) if perf else perf_wrap.empty_metrics()
    return result, perf_metrics


def benchmark_matrix(
    row: dict,
    mtx: Path,
    binary: Path,
    kernel: str,
    runs: int,
    timeout: int,
    writer,
    f_csv,
    threads: int = 0,
    perf: bool = False,
) -> None:
    base = {
        "matrix_name": row["name"],
        "group": row.get("group", ""),
        "rows": row.get("rows", ""),
        "cols": row.get("cols", ""),
        "nnz": row.get("nnz", ""),
        "threads": threads,
    }
    print(f"  [{kernel}] {row['name']} … ", end="", flush=True)
    try:
        triples, perf_metrics = run_spmm(binary, mtx, runs, timeout, perf)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"FAILED ({e})")
        for run_id in range(runs + 1):
            writer.writerow(
                {
                    **base,
                    "kernel": kernel,
                    "run_id": run_id,
                    "symbolic_ms": "nan",
                    "compute_ms": "nan",
                    "total_ms": "nan",
                    **perf_wrap.empty_metrics(),
                }
            )
        f_csv.flush()
        return

    for run_id, (s, c, t) in enumerate(triples):
        writer.writerow(
            {
                **base,
                "kernel": kernel,
                "run_id": run_id,
                "symbolic_ms": _fmt(s),
                "compute_ms": _fmt(c),
                "total_ms": _fmt(t),
                **perf_metrics,
            }
        )
    f_csv.flush()

    timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
    avg = sum(timed) / len(timed)
    print(f"avg {avg:.3f} ms  ({len(triples)} runs incl. warmup)")


# ---------------------------------------------------------------------------
# Prisma CPU SpMM
# ---------------------------------------------------------------------------

_CPU_DIR = _SCRIPT_DIR.parent / "SpGEMM" / "CPU"  # for cpu_dispatch.hpp
_CORE_DIR = _SCRIPT_DIR.parent / "core"
_GEN_KERNEL = _CORE_DIR / "gen_kernel.py"


def _nr_for_h(h: int) -> int:
    """NR (column chunk width) for a given block height H.

    One value per H bucket -- unlike the old dual nr512/nr256 menu (from the
    now-deleted gen_spmm_kernels.py), a single NR works for every ISA now:
    core/gen_kernel.py's row-tiling automatically keeps AVX2's smaller
    register budget safe at the SAME N, the same way SpGEMM already relies
    on for its own mined N. Buckets match the old menu's nr512 column
    (AVX-512: keep H*(NR/8) <= 28, leaving 4 for B-vectors + scratch).
    """
    if h <= 6:
        return 32
    elif h <= 9:
        return 24
    elif h <= 13:
        return 16
    else:
        return 8


def _gen_spmm_dispatch_table(shapes: list[tuple[int, int, int]]) -> str:
    """Generate spmm_dispatch_table.hpp's spmm_dispatch() body: a switch on
    a packed (H,W) key selecting spmm_kernel<H,W,NR>, falling back to
    gemm_fallback for shapes not in the table.

    Ported from the deleted gen_spmm_kernels.py's gen_dispatch_table(),
    simplified to one NR per shape (see _nr_for_h) now that gemm_fixed's
    row-tiling makes a separate AVX2-specific NR unnecessary. Switch-based
    (not an if-chain) for the same reason as before: block counts run into
    the tens of thousands on some matrices, and this is called once per
    block, so a jump table over a packed (H<<16)|W key avoids up to
    len(shapes) sequential compares per call.
    """
    lines = [
        "// AUTO-GENERATED by benchmark_spmm_cpu.py — do not edit.",
        "// spmm_dispatch — runtime (H,W) -> spmm_kernel<H,W,NR> dispatch via a",
        "// switch on a packed (H,W) key (jump table, not a linear if-chain).",
        "// Falls back to gemm_fallback for shapes not in the table.",
        "inline void spmm_dispatch(int H, int W, int N,",
        "                          const double* A, int lda,",
        "                          const double* B, int ldb,",
        "                          double*       C, int ldc) {",
        "  switch ((H << 16) | W) {",
    ]
    for H, W, NR in shapes:
        lines.append(f"    case ({H} << 16) | {W}:")
        lines.append(
            f"      spmm_detail::spmm_kernel<{H}, {W}, {NR}>(N, A, lda, B, ldb, C, ldc);"
        )
        lines.append("      return;")
    lines += [
        "    default: break;",
        "  }",
        "  benchmark_core::cpu_detail::gemm_fallback(H, W, N, A, lda, B, ldb, C, ldc);",
        "}",
    ]
    return "\n".join(lines) + "\n"

_CORE_SRCS = [
    "block.cpp",
    "block_generator.cpp",
    "block_mining.cpp",
    "interval_tree.cpp",
    "matrix.cpp",
    "matrix_io.cpp",
    "pipeline.cpp",
    "segment_tree.cpp",
]


def _hdf5_paths() -> tuple[str, str]:
    import platform

    for inc in ["/usr/include/hdf5/serial", "/usr/local/include", "/usr/include"]:
        if Path(inc).joinpath("hdf5.h").exists():
            break
    else:
        inc = "/usr/include/hdf5/serial"
    machine = platform.machine()
    triplet = {"x86_64": "x86_64-linux-gnu", "aarch64": "aarch64-linux-gnu"}.get(
        machine, machine + "-linux-gnu"
    )
    for lib in [
        f"/usr/lib/{triplet}/hdf5/serial",
        "/usr/lib/hdf5/serial",
        "/usr/local/lib",
        f"/usr/lib/{triplet}",
    ]:
        if (
            Path(lib).joinpath("libhdf5.so").exists()
            or Path(lib).joinpath("libhdf5.a").exists()
        ):
            break
    else:
        lib = f"/usr/lib/{triplet}/hdf5/serial"
    return inc, lib


def analyze_bsp_shapes(
    bsp_path: Path,
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
) -> list[tuple[int, int]]:
    """Return the top-N (h, w) block shapes from a .bsp file.

    Args:
        top_n:         How many shapes to return.
        min_area:      Minimum H*W to consider.  Shapes smaller than this are
                       skipped entirely — they do not benefit from a specialized
                       kernel over gemm_fallback's #pragma omp simd AXPY path.
                       Default 4 excludes all 1×K and K×1 (singleton-row/col)
                       shapes, which are pure AXPY and auto-vectorise equally well.
        rank_by_flops: If True (default), rank shapes by total FLOPs they drive
                       (count × H × W × 2) rather than by raw block count.
                       FLOPs ranking picks the shapes that actually matter for
                       runtime, not just the most numerous small blocks.
    """
    try:
        import h5py
        from collections import Counter
    except ImportError:
        return []
    try:
        with h5py.File(bsp_path, "r") as f:
            hs = f["block_h"][:]
            ws = f["block_w"][:]
    except Exception:
        return []

    # Accumulate counts and FLOPs per (h, w) shape.
    counts: Counter[tuple[int, int]] = Counter()
    flops: dict[tuple[int, int], int] = {}
    for h, w in zip(hs.tolist(), ws.tolist()):
        if h * w < min_area:
            continue
        shape = (h, w)
        counts[shape] += 1
        flops[shape] = flops.get(shape, 0) + h * w * 2

    if rank_by_flops:
        ranked = sorted(flops.items(), key=lambda kv: kv[1], reverse=True)
        return [shape for shape, _ in ranked[:top_n]]
    else:
        return [shape for shape, _ in counts.most_common(top_n)]


def _regenerate_kernels(
    matrices: list[dict],
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
) -> bool:
    """Analyze BSPs for all matrices, collect top-N shapes, regenerate kernel headers.

    Shapes with H*W < min_area are excluded even if they dominate block counts.
    Ranking by FLOPs (default) ensures the shapes that drive the most arithmetic
    get specializations, rather than the most numerous (potentially tiny) shapes.

    Returns True if regeneration succeeded, False if skipped (h5py unavailable or no BSPs).
    """
    # Accumulate per-shape counts and FLOPs across every BSP in the matrix list.
    try:
        import h5py
    except ImportError:
        return False

    from collections import Counter

    global_counts: Counter[tuple[int, int]] = Counter()
    global_flops: dict[tuple[int, int], int] = {}
    analyzed = 0

    for row in matrices:
        mtx = find_mtx(row["name"], row.get("group", ""))
        if mtx is None:
            continue
        bsp = mtx.with_suffix(".bsp")
        if not bsp.exists():
            continue
        try:
            with h5py.File(bsp, "r") as f:
                hs = f["block_h"][:]
                ws = f["block_w"][:]
            for h, w in zip(hs.tolist(), ws.tolist()):
                if h * w < min_area:
                    continue
                shape = (h, w)
                global_counts[shape] += 1
                global_flops[shape] = global_flops.get(shape, 0) + h * w * 2
        except Exception:
            continue
        analyzed += 1

    if not global_flops:
        return False

    # Pick top_n shapes by FLOPs (or count) across all matrices.
    if rank_by_flops:
        ranked = sorted(global_flops.items(), key=lambda kv: kv[1], reverse=True)
    else:
        ranked = [
            (s, global_counts[s]) for s in (sh for sh, _ in global_counts.most_common())
        ]
    top_shapes = [shape for shape, _ in ranked[:top_n]]

    triples = [(h, w, _nr_for_h(h)) for h, w in top_shapes]
    shapes_arg = ",".join(f"{h}x{w}x{nr}" for h, w, nr in sorted(triples))
    rank_label = "FLOPs" if rank_by_flops else "count"
    print(
        f"  Analyzed {analyzed} BSPs (min_area={min_area}, rank_by={rank_label}) "
        f"→ {len(top_shapes)} shapes: {shapes_arg}"
    )
    r = subprocess.run(
        [
            sys.executable,
            str(_GEN_KERNEL),
            "--shapes",
            shapes_arg,
            "--out-dir",
            str(_SPMM_DIR),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"  WARNING: kernel regeneration failed:\n{r.stderr[-1000:]}")
        return False
    (_SPMM_DIR / "spmm_dispatch_table.hpp").write_text(_gen_spmm_dispatch_table(triples))
    print(f"  {r.stdout.strip()}")
    return True


def compile_prisma_cpu_spmm(
    bin_dir: Path,
    matrices: list[dict] | None = None,
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
) -> Path:
    out = bin_dir / "prisma_cpu_spmm_bench"
    hdf5_inc, hdf5_lib = _hdf5_paths()

    # Regenerate kernel headers from per-matrix shape analysis before compiling.
    if matrices:
        _regenerate_kernels(
            matrices, top_n=top_n, min_area=min_area, rank_by_flops=rank_by_flops
        )

    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    srcs.append(str(_SPMM_DIR / "prisma_cpu_spmm_bench.cpp"))
    cmd = [
        "g++",
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=native",
        "-DHAVE_HDF5",
        f'-DGEMM_KERNELS_H="{(_SPMM_DIR / "kernels_generated.hpp").resolve()}"',
        f"-I{_CORE_DIR}",
        f"-I{_CPU_DIR}",
        f"-I{hdf5_inc}",
        *srcs,
        str(Path(hdf5_lib) / "libhdf5.so"),
        "-o",
        str(out),
    ]
    print("  Compiling prisma_cpu_spmm … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-3000:]}")
        raise RuntimeError("prisma_cpu_spmm compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def compile_prisma_for_matrix(
    build_root: Path,
    bin_dir: Path,
    name: str,
    bsp: Path,
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
) -> tuple[str, Path | None, str]:
    """Compile a prisma_cpu_spmm_bench specialized to ONE matrix's own top-N
    block shapes, rather than shapes pooled across a whole benchmark list.

    Pooling starves any matrix whose dominant shapes don't also dominate the
    aggregate (measured: 39% mean FLOP coverage pooled vs. 83% generating
    per-matrix, with several matrices at 0% pooled vs. 100% per-matrix) — the
    10 register-blocked kernel slots are a real, scarce resource, and mining
    blocks for one matrix and then specializing for it is also how Prisma
    would actually be deployed (mine once, specialize once, run repeatedly),
    not an artifact of sharing one binary across an arbitrary matrix list.

    Builds in an isolated `build_root/<name>/` copy of prisma_cpu_spmm_bench.cpp
    and spmm_dispatch.hpp -- spmm_dispatch.hpp's own quote-include of
    spmm_dispatch_table.hpp resolves relative to its own directory, so a
    copy with a fresh matrix-specific spmm_dispatch_table.hpp next to it
    doesn't touch the shared SpMM/ originals, letting many matrices compile
    concurrently without racing on a shared file. (The specialized kernel
    bodies themselves, from core/gen_kernel.py, are pulled in via a
    -DGEMM_KERNELS_H=<absolute path> compiler define instead of a
    quote-include -- same mechanism SpGEMM/SpMV use -- so they don't need to
    live in this directory at all, only spmm_dispatch_table.hpp does.)
    Returns (name, binary_path_or_None, status_message) rather than raising,
    so a parallel driver can report all results without one failure
    aborting the batch.
    """
    work = build_root / name
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        _SPMM_DIR / "prisma_cpu_spmm_bench.cpp", work / "prisma_cpu_spmm_bench.cpp"
    )
    shutil.copy(_SPMM_DIR / "spmm_dispatch.hpp", work / "spmm_dispatch.hpp")

    shapes = analyze_bsp_shapes(
        bsp, top_n=top_n, min_area=min_area, rank_by_flops=rank_by_flops
    )
    triples = [(h, w, _nr_for_h(h)) for h, w in shapes]
    shapes_arg = ",".join(f"{h}x{w}x{nr}" for h, w, nr in sorted(triples))
    r = subprocess.run(
        [
            sys.executable,
            str(_GEN_KERNEL),
            "--shapes",
            shapes_arg,
            "--out-dir",
            str(work),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return name, None, f"kernel generation failed: {r.stderr[-500:]}"
    (work / "spmm_dispatch_table.hpp").write_text(_gen_spmm_dispatch_table(triples))

    hdf5_inc, hdf5_lib = _hdf5_paths()
    out = bin_dir / f"prisma_cpu_spmm_bench_{name}"
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    srcs.append(str(work / "prisma_cpu_spmm_bench.cpp"))
    cmd = [
        "g++",
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=native",
        "-DHAVE_HDF5",
        f'-DGEMM_KERNELS_H="{(work / "kernels_generated.hpp").resolve()}"',
        f"-I{_CORE_DIR}",
        f"-I{_CPU_DIR}",
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


def compile_prisma_per_matrix(
    bin_dir: Path,
    matrices: list[dict],
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
    max_workers: int = 12,
) -> dict[str, Path]:
    """Compile one matrix-specialized prisma binary per matrix, in parallel.

    g++ -O3 -march=native on the heavily-unrolled AVX-512 kernel
    specializations is slow (~1-2 min per binary even with only ~10 shapes),
    and that cost is identical whether shapes are pooled or per-matrix — so
    compiling 28 matrices sequentially would take 30-60 minutes. Each
    compile is an independent subprocess in its own build directory (see
    compile_prisma_for_matrix), so they're safe to run concurrently; capped
    at max_workers to avoid oversubscribing memory/CPU on very large
    matrix lists.
    """
    build_root = bin_dir / "_prisma_build"
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
        f"  Compiling {len(jobs)} matrix-specialized prisma binaries "
        f"(up to {max_workers} in parallel) …"
    )
    t0 = time.time()
    results: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                compile_prisma_for_matrix,
                build_root,
                bin_dir,
                name,
                bsp,
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


def run_prisma_cpu_spmm(
    binary: Path,
    bsp: Path,
    runs: int,
    timeout: int,
    specialized: bool = False,
    tile_n: int = 0,
    threads: int = 0,
    perf: bool = False,
) -> tuple[list[tuple[float, float, float]], dict]:
    cmd = [str(binary), str(bsp), "--runs", str(runs)]
    if specialized:
        cmd.append("--specialized-kernels")
    if tile_n > 0:
        cmd.extend(["--tile-n", str(tile_n)])
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"prisma_cpu_spmm exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError(f"prisma_cpu_spmm: no JSON output:\n{r.stdout[-400:]}")
    compute_ms = d.get("compute_ms", [])
    if not compute_ms:
        raise RuntimeError("prisma_cpu_spmm: empty compute_ms in JSON")
    # symbolic_ms is the block-grouping/dispatch-prep cost (row-group build +
    # D-locality resort + task-list build) -- redone from scratch on every
    # timed run inside prisma_cpu_spmm_bench.cpp's main loop, so this is a
    # real per-run measurement, not a single one-shot cost re-reported. A
    # real, one-off caller pays this cost every call; TACO's own pack_B()
    # (bench_taco_spmm.cpp) is redone every run the same way, folded into
    # run_%d_assemble_ns.
    symbolic_ms = d.get("symbolic_ms", [0.0] * len(compute_ms))
    triples = [(sym, comp, sym + comp) for sym, comp in zip(symbolic_ms, compute_ms)]
    perf_metrics = perf_wrap.measure(cmd, timeout) if perf else perf_wrap.empty_metrics()
    return triples, perf_metrics


def benchmark_matrix_prisma(
    row: dict,
    mtx: Path,
    binary: Path,
    runs: int,
    timeout: int,
    writer,
    f_csv,
    kernel: str = "prisma_cpu",
    specialized: bool = False,
    tile_n: int = 0,
    threads: int = 0,
    perf: bool = False,
) -> None:
    bsp = mtx.with_suffix(".bsp")
    if not bsp.exists():
        print(f"  [{kernel}] BSP not found — skipping")
        return
    base = {
        "matrix_name": row["name"],
        "group": row.get("group", ""),
        "rows": row.get("rows", ""),
        "cols": row.get("cols", ""),
        "nnz": row.get("nnz", ""),
        "threads": threads,
    }
    print(f"  [{kernel}] {row['name']} … ", end="", flush=True)
    try:
        triples, perf_metrics = run_prisma_cpu_spmm(
            binary, bsp, runs, timeout, specialized, tile_n, threads, perf,
        )
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"FAILED ({e})")
        for run_id in range(runs + 1):
            writer.writerow(
                {
                    **base,
                    "kernel": kernel,
                    "run_id": run_id,
                    "symbolic_ms": "nan",
                    "compute_ms": "nan",
                    "total_ms": "nan",
                    **perf_wrap.empty_metrics(),
                }
            )
        f_csv.flush()
        return

    for run_id, (s, c, t) in enumerate(triples):
        writer.writerow(
            {
                **base,
                "kernel": kernel,
                "run_id": run_id,
                "symbolic_ms": _fmt(s),
                "compute_ms": _fmt(c),
                "total_ms": _fmt(t),
                **perf_metrics,
            }
        )
    f_csv.flush()
    timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
    print(f"avg {sum(timed) / len(timed):.3f} ms  ({len(triples)} runs incl. warmup)")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="SpMM benchmark (TACO + Prisma) on SuiteSparse matrices",
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
        "--timeout",
        type=int,
        default=300,
        help="per-matrix timeout in seconds (default: 300)",
    )
    g.add_argument(
        "--threads",
        type=int,
        default=0,
        dest="threads",
        help="OpenMP threads for prisma_cpu_spmm_bench and OMP_NUM_THREADS "
        "for TACO (default: 0, meaning min(16, all available cores) -- see "
        "prisma_cpu_spmm_bench.cpp's own default-cap rationale). Ignored "
        "if --threads-sweep is given.",
    )
    g.add_argument(
        "--threads-sweep",
        default=None,
        dest="threads_sweep",
        help="Sweep multiple thread counts instead of one fixed value. "
        "Either a comma-separated list (e.g. '32,16,8,4,2,1') or a single "
        "max value that gets auto-expanded by halving down to 1 (e.g. "
        "'32' -> 32,16,8,4,2,1). The full matrix list is benchmarked once "
        "per thread count, all appended to the same --out CSV, "
        "distinguished by the 'threads' column.",
    )
    g.add_argument(
        "--perf",
        action="store_true",
        help="Also collect hardware perf counters (cycles, instructions, "
        "cache/branch/TLB misses) and peak RSS via `perf stat` + `time -v`, "
        "one extra wrapped re-run per kernel call (needs perf_event_open "
        "access; see perf_wrap.py). Roughly doubles wall time per call.",
    )

    g = p.add_argument_group("Paths")
    g.add_argument(
        "--out",
        default="spmm_results.csv",
        help="output CSV, append mode (default: spmm_results.csv)",
    )
    g.add_argument(
        "--bin", default="", help="pre-built bench_taco_spmm binary (skips compilation)"
    )
    g.add_argument(
        "--work-dir",
        default="",
        help="directory for compiled binaries (default: SpMM/)",
    )

    g = p.add_argument_group("Kernel generation")
    g.add_argument(
        "--top-n",
        type=int,
        default=30,
        dest="top_n",
        help="number of shapes to specialise per matrix (default: 30). "
        "Register budget isn't the constraint (each shape compiles "
        "to its own independent function) — compile time is: ~10.7s "
        "for 30 shapes on the most shape-diverse matrix in-suite, "
        "run in parallel across matrices. Coverage has a long tail "
        "(some matrices have 700+ distinct shapes), so this is a "
        "cost/benefit knob, not a correctness one — shapes beyond "
        "what a matrix actually has are simply unused, so raising "
        "this only costs compile time on matrices that need it.",
    )
    g.add_argument(
        "--min-block-area",
        type=int,
        default=4,
        dest="min_block_area",
        help="minimum H*W for a shape to be eligible for a specialised "
        "kernel (default: 4).  Shapes below this threshold fall back "
        "to gemm_fallback, which is already as fast for thin/singleton "
        "blocks (pure DAXPY, auto-vectorised).  Set to 1 to disable.",
    )
    g.add_argument(
        "--count-ranking",
        action="store_true",
        dest="count_ranking",
        help="rank candidate shapes by block count instead of FLOPs. "
        "FLOPs ranking (default) selects shapes that drive the most "
        "arithmetic; count ranking picks the most numerous blocks, "
        "which may be dominated by cheap small shapes.",
    )

    g = p.add_argument_group("Build / skip")
    g.add_argument(
        "--no-compile",
        action="store_true",
        help="skip compilation; binaries must already exist",
    )
    g.add_argument("--no-taco", action="store_true", help="skip all TACO kernels")
    g.add_argument(
        "--no-prisma",
        action="store_true",
        help="skip both Prisma CPU kernels (fallback + specialized)",
    )
    g.add_argument(
        "--no-prisma-specialized",
        action="store_true",
        dest="no_prisma_specialized",
        help="skip the prisma_specialized kernel (keep prisma_cpu)",
    )
    g.add_argument(
        "--no-prisma-tiled",
        action="store_true",
        dest="no_prisma_tiled",
        help="skip the prisma_specialized_tiled kernel (column tiling)",
    )
    g.add_argument(
        "--tile-n",
        type=int,
        default=512,
        dest="tile_n",
        help="column tile size for prisma_specialized_tiled (default: 512)",
    )
    g.add_argument(
        "--prisma-bin",
        default="",
        dest="prisma_bin",
        help="pre-built prisma_cpu_spmm_bench binary (skips compilation, "
        "used for every matrix — overrides per-matrix compilation)",
    )
    g.add_argument(
        "--pooled-kernels",
        action="store_true",
        dest="pooled_kernels",
        help="compile ONE prisma binary with kernels pooled across the "
        "whole matrix list (the old behavior), instead of the default "
        "of one binary per matrix specialized to its own top-N "
        "shapes. Pooling starves matrices whose shapes don't also "
        "dominate the aggregate — kept only for comparison.",
    )
    g.add_argument(
        "--prisma-compile-workers",
        type=int,
        default=12,
        dest="prisma_compile_workers",
        help="parallel g++ compiles when building per-matrix prisma "
        "binaries (default: 12) — each is independent, unlike the "
        "pooled-binary compile which is a single g++ invocation",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# (kernel_label, gcc_define_or_None)
_KERNELS = [
    ("taco", None),
    ("taco_opt0", "KERNEL_OPT_0"),
    ("taco_opt1", "KERNEL_OPT_1"),
]


def main() -> None:
    args = parse_args()

    # Load matrix list early so BSP paths are available for kernel generation.
    matrices = load_matrix_list(Path(args.csv))

    # ── TACO binaries ─────────────────────────────────────────────────────────
    taco_binaries: list[tuple[Path, str]] = []
    if not args.no_taco:
        bin_dir = Path(args.work_dir) if args.work_dir else _TMP_DIR
        bin_dir.mkdir(parents=True, exist_ok=True)

        if args.bin:
            binary = Path(args.bin)
            if not binary.exists():
                sys.exit(f"Binary not found: {binary}")
            taco_binaries = [(binary, binary.stem)]
        elif args.no_compile:
            for k, _ in _KERNELS:
                b = bin_dir / f"bench_taco_spmm_{k}"
                if not b.exists():
                    sys.exit(
                        f"bench_taco_spmm ({k}) not found in {bin_dir} — build first"
                    )
                taco_binaries.append((b, k))
        else:
            print("Compiling TACO kernels:")
            taco_binaries = [(compile_binary(bin_dir, k, d), k) for k, d in _KERNELS]
            print()

    # ── Prisma CPU SpMM binaries ───────────────────────────────────────────────
    # One binary per matrix by default (see compile_prisma_per_matrix): each
    # gets its own top-N shapes specialized rather than sharing 10 slots
    # pooled across the whole list. --prisma-bin / --pooled-kernels opt back
    # into a single shared binary for comparison.
    prisma_binary: Path | None = None  # single-binary mode
    prisma_binaries: dict[str, Path] = {}  # per-matrix mode
    if not args.no_prisma:
        bin_dir = Path(args.work_dir) if args.work_dir else _TMP_DIR
        bin_dir.mkdir(parents=True, exist_ok=True)
        if args.prisma_bin:
            prisma_binary = Path(args.prisma_bin)
            if not prisma_binary.exists():
                sys.exit(f"Prisma binary not found: {prisma_binary}")
        elif args.pooled_kernels:
            if args.no_compile:
                prisma_binary = bin_dir / "prisma_cpu_spmm_bench"
                if not prisma_binary.exists():
                    sys.exit(
                        f"prisma_cpu_spmm_bench not found in {bin_dir} — build first"
                    )
            else:
                print("Compiling Prisma CPU SpMM (pooled kernels):")
                try:
                    prisma_binary = compile_prisma_cpu_spmm(
                        bin_dir,
                        matrices,
                        top_n=args.top_n,
                        min_area=args.min_block_area,
                        rank_by_flops=not args.count_ranking,
                    )
                except RuntimeError as e:
                    sys.exit(str(e))
                print()
        elif args.no_compile:
            for row in matrices:
                b = bin_dir / f"prisma_cpu_spmm_bench_{row['name']}"
                if b.exists():
                    prisma_binaries[row["name"]] = b
            if not prisma_binaries:
                sys.exit(
                    f"no prisma_cpu_spmm_bench_<matrix> binaries found in {bin_dir} — build first"
                )
        else:
            print("Compiling Prisma CPU SpMM (per-matrix kernels):")
            prisma_binaries = compile_prisma_per_matrix(
                bin_dir,
                matrices,
                top_n=args.top_n,
                min_area=args.min_block_area,
                rank_by_flops=not args.count_ranking,
                max_workers=args.prisma_compile_workers,
            )
            print()

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Thread count(s) applied to BOTH contenders identically: TACO's
    # binaries have no --threads flag of their own and only ever read
    # OMP_NUM_THREADS from the environment, while prisma_cpu_spmm_bench
    # gets it explicitly via --threads. Without this, TACO would run
    # uncapped (all cores) by default while prisma defaults to
    # min(16, nproc) internally -- given this codebase's own measured
    # finding that oversubscription can be up to 15x slower, that asymmetry
    # would bias every comparison in prisma's favor for reasons unrelated
    # to either algorithm. Computed once so both contenders always see the
    # exact same value, whether --threads was passed or not.
    thread_values = (
        _parse_thread_sweep(args.threads_sweep) if args.threads_sweep
        else [args.threads if args.threads > 0 else min(16, os.cpu_count() or 16)]
    )

    print(f"Output  : {csv_path}")
    print(f"Matrices: {len(matrices)}")
    print(f"Runs    : {args.runs}")
    print(f"Threads : {thread_values}")
    print()

    write_header = _needs_header(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()

        # prisma_binaries is compiled once, upfront (above) -- independent
        # of thread count, so sweeping threads here costs nothing extra in
        # compile time, unlike SpGEMM's inline-per-matrix compile.
        for threads in thread_values:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f_csv.write(
                f"# {ts}  input={args.csv}  runs={args.runs}  threads={threads}\n"
            )
            env = {**os.environ, "OMP_NUM_THREADS": str(threads)}

            old_env = os.environ.copy()
            os.environ.update(env)
            try:
                for i, row in enumerate(matrices, 1):
                    name = row["name"]
                    group = row.get("group", "")
                    print(f"[threads={threads}] [{i}/{len(matrices)}] {name}")

                    mtx = find_mtx(name, group)
                    if mtx is None:
                        print(f"  MTX not found — skipping")
                        continue

                    for binary, kernel in taco_binaries:
                        benchmark_matrix(
                            row, mtx, binary, kernel, args.runs, args.timeout,
                            writer, f_csv, threads=threads, perf=args.perf,
                        )

                    active_prisma_binary = (
                        prisma_binary
                        if prisma_binary is not None
                        else prisma_binaries.get(name)
                    )
                    if active_prisma_binary is None and not args.no_prisma:
                        print(f"  [prisma_*] no compiled binary for {name} — skipping")
                    if active_prisma_binary is not None:
                        benchmark_matrix_prisma(
                            row,
                            mtx,
                            active_prisma_binary,
                            args.runs,
                            args.timeout,
                            writer,
                            f_csv,
                            kernel="prisma_cpu",
                            specialized=False,
                            threads=threads,
                            perf=args.perf,
                        )
                        if not args.no_prisma_specialized:
                            benchmark_matrix_prisma(
                                row,
                                mtx,
                                active_prisma_binary,
                                args.runs,
                                args.timeout,
                                writer,
                                f_csv,
                                kernel="prisma_specialized",
                                specialized=True,
                                threads=threads,
                                perf=args.perf,
                            )
                        if not args.no_prisma_tiled:
                            benchmark_matrix_prisma(
                                row,
                                mtx,
                                active_prisma_binary,
                                args.runs,
                                args.timeout,
                                writer,
                                f_csv,
                                kernel="prisma_tiled",
                                specialized=True,
                                tile_n=args.tile_n,
                                threads=threads,
                                perf=args.perf,
                            )
            finally:
                os.environ.clear()
                os.environ.update(old_env)
    print(f"\nDone. Results in {csv_path}")


if __name__ == "__main__":
    main()
