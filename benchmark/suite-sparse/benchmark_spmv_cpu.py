#!/usr/bin/env python3
"""
suite-sparse/benchmark_spmv_cpu.py — SpMV benchmark on real SuiteSparse matrices.

Computes y = S * x where S is loaded from MTX and x is an in-memory
N-length random dense vector (N = number of columns of S; S need not be
square -- x has length N, y has length M). Timing is split into symbolic
and compute phases, one row per run. run_id=0 is the warmup run; timed
runs are run_id=1..R.

The symbolic phase (TACO's pack_A(); Prisma's row-group build + x-locality
resort) is redone from scratch on every timed run, not built once and
amortized -- a one-off caller pays the full symbolic cost on every call, so
symbolic_ms is a mean of R real per-run measurements.

Kernels: taco, taco_opt (TACO-generated), prisma_cpu, prisma_specialized
(no tiled/auto variants: SpMV's dense operand is a single vector, so
there's no column-tiling axis to specialize or tile across, unlike SpMM).

Usage:
  python benchmark_spmv_cpu.py matrices.csv
  python benchmark_spmv_cpu.py matrices.csv --out spmv_results.csv --runs 5
  python benchmark_spmv_cpu.py matrices.csv --no-compile --bin /path/to/bench_taco_spmv
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import perf_wrap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_SPMV_DIR = _SCRIPT_DIR.parent / "SpMV" / "CPU"
_CORE_DIR = _SCRIPT_DIR.parent / "core"
_CPU_DIR = _SCRIPT_DIR.parent / "SpGEMM" / "CPU"  # for cpu_dispatch.hpp (gemm_fixed, gemm_fallback)
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
    out = work_dir / f"bench_taco_spmv_{kernel}"
    src = str(_SPMV_DIR / "bench_taco_spmv.cpp")
    # -march=native matches prisma_cpu_spmv's compile flags (see
    # compile_prisma_cpu_spmv below) -- without it, GCC only auto-vectorizes
    # TACO's compute() loop to baseline SSE2 while prisma's kernel gets
    # whatever the host ISA offers, a compiler-flag handicap unrelated to
    # the algorithms being compared.
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        "-fopenmp",
        "-march=native",
        "-Drestrict=__restrict__",
        f"-I{_SPMV_DIR}",
        src,
        "-lm",
        "-o",
        str(out),
    ]
    if define:
        cmd.insert(1, f"-D{define}")
    label = f"bench_taco_spmv ({kernel})"
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


def run_spmv(
    binary: Path, mtx: Path, runs: int, timeout: int, perf: bool = False
) -> tuple[list[tuple[float, float, float]], dict]:
    """Run bench_taco_spmv and return ([(symbolic_ms, compute_ms, total_ms)],
    perf_metrics) for run_id 0..runs (inclusive).

    perf_metrics is one aggregate perf-stat/RSS reading for the entire
    process lifetime, from a second, separate invocation wrapped in
    `perf stat` -- see perf_wrap.py. NaN in every field if perf=False or perf
    itself failed (never affects the real timing/success of this run)."""
    cmd = [str(binary), str(mtx), "--runs", str(runs)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"bench_taco_spmv exited {r.returncode}:\n{r.stderr[-800:]}")
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
            f"bench_taco_spmv: could not parse timing from stdout:\n{r.stdout[-400:]}"
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
        triples, perf_metrics = run_spmv(binary, mtx, runs, timeout, perf)
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
# Prisma CPU SpMV
# ---------------------------------------------------------------------------

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


_GEN_KERNEL = _CORE_DIR / "gen_kernel.py"


def analyze_bsp_shapes_spmv(
    bsp_path: Path,
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
) -> list[tuple[int, int]]:
    """Return the top-N (h, w) block shapes from a .bsp file.

    Same h5py-based approach as SpMM's analyze_bsp_shapes -- duplicated
    (not imported) per this codebase's convention of keeping the SpGEMM/
    SpMM/SpMV domains independent (see _load_bsp_as_csr, similarly
    duplicated across the validate_sp*_cpu.py scripts).
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

    counts: Counter[tuple[int, int]] = Counter()
    flops: dict[tuple[int, int], int] = {}
    for h, w in zip(hs.tolist(), ws.tolist()):
        if h * w < min_area:
            continue
        shape = (h, w)
        counts[shape] += 1
        # SpMV's FLOPs per block are h*w*2 (one multiply-add per element,
        # N=1 -- no column-width factor the way SpMM's h*w*2 has an
        # implicit "per output column" scaling that matters when N is wide).
        flops[shape] = flops.get(shape, 0) + h * w * 2

    if rank_by_flops:
        ranked = sorted(flops.items(), key=lambda kv: kv[1], reverse=True)
        return [shape for shape, _ in ranked[:top_n]]
    else:
        return [shape for shape, _ in counts.most_common(top_n)]


def compile_prisma_spmv_for_matrix(
    bin_dir: Path,
    work_dir: Path,
    name: str,
    bsp: Path,
    top_n: int = 10,
    min_area: int = 4,
    rank_by_flops: bool = True,
) -> tuple[str, Path | None, str]:
    """Compile a prisma_cpu_spmv_bench specialized to ONE matrix's own top-N
    (H,W) block shapes (N is always 1 for SpMV -- see prisma_cpu_spmv_bench.cpp).

    Mirrors SpGEMM's per-matrix top10 compile exactly (compile_prisma_spgemm_per_matrix
    in benchmark_spgemm_cpu.py): exact-shape dispatch via -DGEMM_KERNELS_H/
    -DGEMM_DISPATCH_H compiler defines pointing at absolute paths, so unlike
    SpMM's copy-into-isolated-directory dance (needed only because
    spmm_dispatch.hpp uses quote-includes), no source files need copying
    here -- the generated files can live anywhere.
    """
    work = work_dir / name
    work.mkdir(parents=True, exist_ok=True)

    shapes = analyze_bsp_shapes_spmv(
        bsp, top_n=top_n, min_area=min_area, rank_by_flops=rank_by_flops
    )
    if not shapes:
        return name, None, "no shapes returned"
    shapes_arg = ",".join(f"{h}x{w}x1" for h, w in sorted(shapes))
    r = subprocess.run(
        [sys.executable, str(_GEN_KERNEL), "--shapes", shapes_arg, "--out-dir", str(work)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return name, None, f"kernel generation failed: {r.stderr[-500:]}"

    hdf5_inc, hdf5_lib = _hdf5_paths()
    out = bin_dir / f"prisma_cpu_spmv_bench_{name}"
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    srcs.append(str(_SPMV_DIR / "prisma_cpu_spmv_bench.cpp"))
    cmd = [
        "g++",
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=native",
        "-DHAVE_HDF5",
        f'-DGEMM_KERNELS_H="{(work / "kernels_generated.hpp").resolve()}"',
        f'-DGEMM_DISPATCH_H="{(work / "dispatch_generated.hpp").resolve()}"',
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


def compile_prisma_cpu_spmv(bin_dir: Path) -> Path:
    """Compile the single, shared prisma_cpu_spmv_bench binary.

    Unlike SpMM's Prisma contender, there's no per-matrix shape
    specialization here (see module docstring / the design rationale in
    SpMV/CPU/prisma_cpu_spmv_bench.cpp) -- one binary covers every matrix,
    same as the TACO contenders.
    """
    out = bin_dir / "prisma_cpu_spmv_bench"
    hdf5_inc, hdf5_lib = _hdf5_paths()
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    srcs.append(str(_SPMV_DIR / "prisma_cpu_spmv_bench.cpp"))
    cmd = [
        "g++",
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=native",
        "-DHAVE_HDF5",
        f"-I{_CORE_DIR}",
        f"-I{_CPU_DIR}",
        f"-I{hdf5_inc}",
        *srcs,
        str(Path(hdf5_lib) / "libhdf5.so"),
        "-o",
        str(out),
    ]
    print("  Compiling prisma_cpu_spmv … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-3000:]}")
        raise RuntimeError("prisma_cpu_spmv compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def run_prisma_cpu_spmv(
    binary: Path,
    bsp: Path,
    runs: int,
    timeout: int,
    specialized: bool = False,
    threads: int = 0,
    perf: bool = False,
) -> tuple[list[tuple[float, float, float]], dict]:
    cmd = [str(binary), str(bsp), "--runs", str(runs)]
    if specialized:
        cmd.append("--specialized-kernels")
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"prisma_cpu_spmv exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError(f"prisma_cpu_spmv: no JSON output:\n{r.stdout[-400:]}")
    compute_ms = d.get("compute_ms", [])
    if not compute_ms:
        raise RuntimeError("prisma_cpu_spmv: empty compute_ms in JSON")
    # symbolic_ms is the row-group build + x-locality resort cost -- redone
    # from scratch on every timed run inside prisma_cpu_spmv_bench.cpp's
    # main loop, so this is a real per-run measurement, not a single
    # one-shot cost re-reported. TACO's own pack_A() (bench_taco_spmv.cpp)
    # is redone every run the same way, folded into run_%d_assemble_ns.
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
        triples, perf_metrics = run_prisma_cpu_spmv(
            binary, bsp, runs, timeout, specialized, threads, perf
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
        description="SpMV benchmark (TACO + Prisma) on SuiteSparse matrices",
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
        help="OpenMP threads for prisma_cpu_spmv_bench and OMP_NUM_THREADS "
        "for TACO (default: 0, meaning min(16, all available cores) -- see "
        "prisma_cpu_spmv_bench.cpp's own default-cap rationale). Ignored "
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
        default="spmv_results.csv",
        help="output CSV, append mode (default: spmv_results.csv)",
    )
    g.add_argument(
        "--bin", default="", help="pre-built bench_taco_spmv binary (skips compilation)"
    )
    g.add_argument(
        "--work-dir",
        default="",
        help="directory for compiled binaries (default: /tmp/_prismac/)",
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
        help="skip all Prisma CPU kernels (prisma_cpu + prisma_specialized)",
    )
    g.add_argument(
        "--no-prisma-specialized",
        action="store_true",
        dest="no_prisma_specialized",
        help="skip prisma_specialized (per-matrix named-register gemm_fixed<H,W,1> "
             "kernels, core/gen_kernel.py -- exact-shape dispatch, same mechanism "
             "SpGEMM's prisma_top10 uses)",
    )
    g.add_argument(
        "--prisma-bin",
        default="",
        dest="prisma_bin",
        help="pre-built prisma_cpu_spmv_bench binary (skips compilation)",
    )

    g = p.add_argument_group("Kernel generation (prisma_specialized only)")
    g.add_argument(
        "--top-n",
        type=int,
        default=10,
        dest="top_n",
        help="number of (H,W) shapes to specialize per matrix (default: 10)",
    )
    g.add_argument(
        "--min-block-area",
        type=int,
        default=4,
        dest="min_block_area",
        help="minimum H*W for a shape to be eligible for specialization (default: 4)",
    )
    g.add_argument(
        "--count-ranking",
        action="store_true",
        dest="count_ranking",
        help="rank candidate shapes by block count instead of FLOPs",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# (kernel_label, gcc_define_or_None)
_KERNELS = [
    ("taco", None),
    ("taco_opt", "KERNEL_OPT"),
]


def main() -> None:
    args = parse_args()

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
                b = bin_dir / f"bench_taco_spmv_{k}"
                if not b.exists():
                    sys.exit(
                        f"bench_taco_spmv ({k}) not found in {bin_dir} — build first"
                    )
                taco_binaries.append((b, k))
        else:
            print("Compiling TACO kernels:")
            taco_binaries = [(compile_binary(bin_dir, k, d), k) for k, d in _KERNELS]
            print()

    # ── Prisma CPU SpMV binary — one shared binary, no per-matrix specialization ──
    prisma_binary: Path | None = None
    if not args.no_prisma:
        bin_dir = Path(args.work_dir) if args.work_dir else _TMP_DIR
        bin_dir.mkdir(parents=True, exist_ok=True)
        if args.prisma_bin:
            prisma_binary = Path(args.prisma_bin)
            if not prisma_binary.exists():
                sys.exit(f"Prisma binary not found: {prisma_binary}")
        elif args.no_compile:
            prisma_binary = bin_dir / "prisma_cpu_spmv_bench"
            if not prisma_binary.exists():
                sys.exit(f"prisma_cpu_spmv_bench not found in {bin_dir} — build first")
        else:
            print("Compiling Prisma CPU SpMV:")
            try:
                prisma_binary = compile_prisma_cpu_spmv(bin_dir)
            except RuntimeError as e:
                sys.exit(str(e))
            print()

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Thread count(s) applied to BOTH contenders identically: TACO's
    # binaries have no --threads flag of their own and only ever read
    # OMP_NUM_THREADS from the environment, while prisma_cpu_spmv_bench
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

        # NOTE: prisma_specialized's per-matrix kernel is compiled inline
        # per matrix (not cached upfront, unlike SpMM), so sweeping threads
        # here recompiles it once per thread_values entry per matrix -- a
        # real, known cost of this loop ordering. Fine for a handful of
        # sweep points; expensive for a long sweep across many matrices.
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

                    if prisma_binary is None and not args.no_prisma:
                        print(f"  [prisma_*] no compiled binary — skipping")
                    if prisma_binary is not None:
                        benchmark_matrix_prisma(
                            row,
                            mtx,
                            prisma_binary,
                            args.runs,
                            args.timeout,
                            writer,
                            f_csv,
                            kernel="prisma_cpu",
                            threads=threads,
                            perf=args.perf,
                        )
                        if not args.no_prisma_specialized:
                            # Per-matrix specialized binary (own top-N (H,W) shapes,
                            # N=1 always) -- mirrors SpGEMM's prisma_top10 exactly,
                            # not SpMM's pooled/shared-binary model, since SpMV's
                            # dispatch is exact-match like SpGEMM's, not an
                            # N-chunking wrapper like SpMM's.
                            print(f"  [{'prisma_specialized':<20}] ", end="", flush=True)
                            work_dir = Path(args.work_dir) if args.work_dir else _TMP_DIR
                            bsp = mtx.with_suffix(".bsp")
                            if not bsp.exists():
                                print("SKIP (no BSP)")
                            else:
                                _, spec_bin, msg = compile_prisma_spmv_for_matrix(
                                    work_dir, work_dir, name, bsp,
                                    top_n=args.top_n, min_area=args.min_block_area,
                                    rank_by_flops=not args.count_ranking,
                                )
                                if spec_bin is None:
                                    print(f"SKIPPED ({msg})")
                                else:
                                    print(msg)
                                    benchmark_matrix_prisma(
                                        row,
                                        mtx,
                                        spec_bin,
                                        args.runs,
                                        args.timeout,
                                        writer,
                                        f_csv,
                                        kernel="prisma_specialized",
                                        specialized=True,
                                        threads=threads,
                                        perf=args.perf,
                                    )
            finally:
                os.environ.clear()
                os.environ.update(old_env)
    print(f"\nDone. Results in {csv_path}")


if __name__ == "__main__":
    main()
