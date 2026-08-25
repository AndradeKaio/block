#!/usr/bin/env python3
"""
suite-sparse/benchmark_spgemm_cpu.py — CPU SpGEMM benchmark on SuiteSparse matrices.

Computes C = A × A (square self-product) and compares:
  taco_cpu       — TACO-generated SpGEMM kernel (unoptimized)
  taco_cpu_opt   — TACO-generated SpGEMM kernel (parallelized + reordered)
  prisma_generic      — Prisma, gemm_fallback (#pragma omp simd, auto-vectorised)
  prisma_top10        — Prisma, per-matrix compiled with top-N (M,K,N) dispatch table

The symbolic pipeline (TACO's assemble(); Prisma's intersect pairs → merge groups
→ block fusion → plan build) is redone from scratch on every timed run, not run
once and amortized -- a cold, one-off caller pays the full symbolic cost each
call, so the reported symbolic_ms is a mean of N real measurements. The one
exception is prisma_top10's one-time top-shapes analysis (needed before it can
even compile its specialized kernel): that setup cost genuinely happens once
regardless of how many runs follow, so it alone is amortized (divided by the
number of timed runs) on top of each run's own real symbolic measurement.

Usage:
  python benchmark_spgemm_cpu.py matrices.csv
  python benchmark_spgemm_cpu.py matrices.csv --out spgemm_cpu_results.csv --runs 5
  python benchmark_spgemm_cpu.py matrices.csv --no-compile --prisma-bin /path/to/bin
  python benchmark_spgemm_cpu.py matrices.csv --top-n 20
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
_CPU_DIR = _SCRIPT_DIR.parent / "SpGEMM" / "CPU"
_CORE_DIR = _SCRIPT_DIR.parent / "core"
_GEN_KERNEL = _SCRIPT_DIR.parent / "core" / "gen_kernel.py"
# Shared compile output dir for every SpGEMM kernel (CPU/GPU, benchmark/
# validate) -- same /tmp/_prismac/ root SpMM uses, nested under spgemm/ so
# the two don't collide, and reused across runs instead of a fresh mkdtemp
# each time so repeated invocations don't need to recompile from scratch.
_TMP_DIR = Path("/tmp/_prismac/spgemm/")

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
    "n_pairs",
    "n_groups",
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
# Matrix list and MTX/BSP location
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("/home/kaio/datasets/suite-sparse")


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


def ensure_real_general(mtx_path: Path, cache_dir: Path) -> Path:
    """Convert MTX to real-general format if needed (required by TACO kernels)."""
    import scipy.io as sio
    import scipy.sparse as sp

    with open(mtx_path) as f:
        header = f.readline().lower()
    if ("real" in header or "integer" in header) and "general" in header:
        return mtx_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / (mtx_path.stem + "_real.mtx")
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path
    A = sio.mmread(str(mtx_path))
    A = sp.csr_matrix(A, dtype=float)
    A.eliminate_zeros()
    sio.mmwrite(str(out_path), A, field="real", symmetry="general")
    return out_path


# ---------------------------------------------------------------------------
# JSON / text output parsing
# ---------------------------------------------------------------------------


def _parse_json_block(stdout: str) -> dict:
    if "JSON_BEGIN" not in stdout or "JSON_END" not in stdout:
        return {}
    s = stdout.index("JSON_BEGIN") + len("JSON_BEGIN")
    e = stdout.index("JSON_END")
    try:
        return json.loads(stdout[s:e].strip())
    except json.JSONDecodeError:
        return {}


_TACO_ASM_RE = re.compile(r"run_(\d+)_assemble_ns=(\d+)")
_TACO_COMP_RE = re.compile(r"run_(\d+)_compute_ns=(\d+)")


# ---------------------------------------------------------------------------
# Compilation
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


def _find_hdf5() -> tuple[str, str]:
    import platform

    for candidate in [
        "/usr/include/hdf5/serial",
        "/usr/local/include",
        "/usr/include",
    ]:
        if Path(candidate).joinpath("hdf5.h").exists():
            inc = candidate
            break
    else:
        inc = "/usr/include/hdf5/serial"

    machine = platform.machine()
    triplet = {
        "x86_64": "x86_64-linux-gnu",
        "aarch64": "aarch64-linux-gnu",
        "arm": "arm-linux-gnueabihf",
    }.get(machine, machine + "-linux-gnu")

    for candidate in [
        f"/usr/lib/{triplet}/hdf5/serial",
        "/usr/lib/hdf5/serial",
        "/usr/local/lib",
        f"/usr/lib/{triplet}",
    ]:
        if (Path(candidate) / "libhdf5.so").exists() or (
            Path(candidate) / "libhdf5.a"
        ).exists():
            lib = candidate
            break
    else:
        lib = f"/usr/lib/{triplet}/hdf5/serial"

    return inc, lib


_HDF5_INC, _HDF5_LIB = _find_hdf5()


def compile_taco_cpu(kernel_h: str, out: Path, n_threads: int) -> Path:
    """Compile bench_taco.c against the given TACO kernel header."""
    src = str(_CPU_DIR / "bench_taco.c")
    cmd = [
        "gcc",
        "-O3",
        "-march=native",
        f'-DTACO_KERNEL_H="{kernel_h}"',
        f"-DNUM_THREADS={n_threads}",
        "-fopenmp",
        f"-I{_CPU_DIR}",
        src,
        "-lm",
        "-o",
        str(out),
    ]
    label = out.name
    print(f"  Compiling {label} … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-2000:]}")
        raise RuntimeError(f"{label} compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def compile_prisma_spgemm(out: Path, blas: bool = False) -> Path:
    """Compile prisma_cpu_bench (SpGEMM) with core sources and HDF5."""
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    srcs.append(str(_CPU_DIR / "prisma_cpu_bench.cpp"))
    cmd = [
        "g++",
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=native",
        "-DHAVE_HDF5",
        f"-I{_CORE_DIR}",
        f"-I{_CPU_DIR}",
        f"-I{_HDF5_INC}",
        *srcs,
        str(Path(_HDF5_LIB) / "libhdf5.so"),
        "-o",
        str(out),
    ]
    if blas:
        cmd[-1:-1] = ["-DHAVE_BLAS", "-lblas"]
    print("  Compiling prisma_cpu_bench (SpGEMM) … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-3000:]}")
        raise RuntimeError("prisma_cpu_bench compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_taco_cpu(
    binary: Path,
    mtx: Path,
    runs: int,
    timeout: int,
    perf: bool = False,
) -> tuple[list[tuple[float, float, float]], dict]:
    """Run bench_taco (A×A) and return ([(symbolic_ms, compute_ms, total_ms)], perf_metrics).

    Every run redoes TACO's assemble() from scratch (see bench_taco.c) instead
    of assembling once and reusing the pattern, so symbolic_ms below is a real
    per-run measurement -- a cold, one-off caller pays the full assemble()
    cost, not a single measurement divided by n_timed. Averaging N real
    measurements (which plot_spgemm_cpu.py's groupby(...).mean() already does)
    reflects that honestly.

    perf_metrics is one aggregate perf-stat/RSS reading for the *entire*
    process lifetime (all runs+1 iterations combined, plus startup), from a
    second, separate invocation of the same command wrapped in `perf stat` --
    see perf_wrap.py. NaN in every field if perf=False or perf itself failed
    (never affects the real timing/success of this run).
    """
    cmd = [str(binary), str(mtx), str(mtx), str(runs + 1)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{binary.name} exited {r.returncode}:\n{r.stderr[-800:]}")

    asm_ns: dict[int, int] = {}
    comp_ns: dict[int, int] = {}
    for line in r.stdout.splitlines():
        m = _TACO_ASM_RE.match(line)
        if m:
            asm_ns[int(m.group(1))] = int(m.group(2))
        m = _TACO_COMP_RE.match(line)
        if m:
            comp_ns[int(m.group(1))] = int(m.group(2))

    if not comp_ns or not asm_ns:
        raise RuntimeError(f"{binary.name}: could not parse timing from stdout")

    result = []
    for run_id in range(runs + 1):
        sym = asm_ns.get(run_id, 0) / 1e6
        comp = comp_ns.get(run_id, 0) / 1e6
        result.append((sym, comp, sym + comp))

    perf_metrics = perf_wrap.measure(cmd, timeout) if perf else perf_wrap.empty_metrics()
    return result, perf_metrics


def run_prisma_spgemm(
    binary: Path,
    bsp: Path,
    runs: int,
    timeout: int,
    specialized: bool = False,
    extra_sym_ms: float = 0.0,
    perf: bool = False,
    threads: int = 0,
) -> tuple[list[tuple[float, float, float]], int, int, dict]:
    """Run prisma_cpu_bench with A=B=bsp (A×A) and return timing triples.

    Returns (triples, n_pairs, n_groups, perf_metrics).  triples is
    [(sym_ms, compute_ms, total_ms)] for run_id 0..runs.  Every run redoes the
    full symbolic pipeline (intersect -> merge -> fuse -> plan_build) from
    scratch inside prisma_cpu_bench (see its main loop), so sym_ms is a real
    per-run measurement, not one measurement divided by n_timed -- a cold,
    one-off caller pays the full symbolic cost, and averaging N real
    measurements reflects that honestly.

    extra_sym_ms is a genuinely one-time cost from a prior --print-shapes
    invocation (only nonzero for prisma_top10, which must analyse shapes once
    before it can even compile its specialized kernel) -- unlike the per-run
    symbolic cost above, that setup truly happens once regardless of n_timed,
    so it alone is amortised (divided by n_timed) on top of each run's real
    symbolic measurement.

    perf_metrics is one aggregate perf-stat/RSS reading for the entire
    process lifetime (all runs, plus startup), from a second, separate
    invocation of the same command wrapped in `perf stat` -- see
    perf_wrap.py. NaN in every field if perf=False or perf itself failed
    (never affects the real timing/success of this run).
    """
    if not bsp.exists():
        raise FileNotFoundError(f"BSP not found: {bsp}")
    cmd = [str(binary), str(bsp), str(bsp), "--runs", str(runs)]
    if specialized:
        cmd.append("--specialized-kernels")
    if threads > 0:
        cmd.extend(["--threads", str(threads)])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"prisma_cpu_bench exited {r.returncode}:\n{r.stderr[-800:]}"
        )
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError(f"prisma_cpu_bench: no JSON output:\n{r.stdout[-400:]}")
    compute_ms = d.get("compute_ms")
    symbolic_ms = d.get("symbolic_ms")
    if not compute_ms or not symbolic_ms or len(compute_ms) != len(symbolic_ms):
        raise RuntimeError(
            "prisma_cpu_bench: empty/mismatched compute_ms or symbolic_ms in JSON "
            f"(n_pairs={d.get('n_pairs', '?')} -- 0 means the matrix genuinely "
            f"has no intersecting blocks for A×A, not a parse bug): {d}"
        )

    n_pairs = int(d.get("n_pairs", 0))
    n_groups = int(d.get("n_groups", 0))
    n_timed = max(len(compute_ms) - 1, 1)
    amortised_extra = extra_sym_ms / n_timed

    triples = []
    for s, c in zip(symbolic_ms, compute_ms):
        sym = float(s) + amortised_extra
        comp = float(c)
        triples.append((sym, comp, sym + comp))

    perf_metrics = perf_wrap.measure(cmd, timeout) if perf else perf_wrap.empty_metrics()
    return triples, n_pairs, n_groups, perf_metrics


def analyze_spgemm_shapes(
    binary: Path,
    bsp: Path,
    top_n: int,
    timeout: int,
) -> tuple[list[tuple[int, int, int]], float]:
    """Run binary --print-shapes to get the top-N (M,K,N) shapes for this matrix.

    Returns (shapes, pipe_total_ms) where pipe_total_ms includes the symbolic
    pipeline cost plus the top-shapes counting time.
    """
    if not bsp.exists():
        raise FileNotFoundError(f"BSP not found: {bsp}")
    cmd = [str(binary), str(bsp), str(bsp), "--print-shapes", str(top_n)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"prisma_cpu_bench (--print-shapes) exited {r.returncode}:\n{r.stderr[-800:]}"
        )
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError(
            f"prisma_cpu_bench: no JSON from --print-shapes:\n{r.stdout[-400:]}"
        )
    shapes = [tuple(s) for s in d.get("top_shapes", [])]
    pipe_total_ms = float(d.get("pipe_total_ms", 0.0))
    return shapes, pipe_total_ms


def gen_spgemm_kernels_files(
    shapes: list[tuple[int, int, int]], out_dir: Path
) -> tuple[Path, Path]:
    """Call the shared core/gen_kernel.py to produce named-register SIMD
    specializations for SpGEMM's mined (M,K,N) shapes.

    Returns (kernels_hpp, dispatch_hpp) paths written inside out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes_arg = ",".join(f"{m}x{k}x{n}" for m, k, n in shapes)
    cmd = [
        sys.executable,
        str(_GEN_KERNEL),
        "--shapes",
        shapes_arg,
        "--out-dir",
        str(out_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"gen_kernel.py failed:\n{r.stderr[-1000:]}\n{r.stdout[-500:]}"
        )
    return (
        out_dir / "kernels_generated.hpp",
        out_dir / "dispatch_generated.hpp",
    )


def compile_prisma_spgemm_per_matrix(
    dispatch_hpp: Path,
    out: Path,
    blas: bool = False,
    kernels_hpp: Path | None = None,
) -> Path:
    """Compile prisma_cpu_bench with a per-matrix generated dispatch (and kernels) table."""
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    srcs.append(str(_CPU_DIR / "prisma_cpu_bench.cpp"))
    cmd = [
        "g++",
        "-O3",
        "-std=c++20",
        "-fopenmp",
        "-march=native",
        "-DHAVE_HDF5",
        f'-DGEMM_DISPATCH_H="{dispatch_hpp.resolve()}"',
        f"-I{_CORE_DIR}",
        f"-I{_CPU_DIR}",
        f"-I{_HDF5_INC}",
        *srcs,
        str(Path(_HDF5_LIB) / "libhdf5.so"),
        "-o",
        str(out),
    ]
    if kernels_hpp is not None:
        cmd.insert(
            cmd.index("-DHAVE_HDF5") + 1,
            f'-DGEMM_KERNELS_H="{kernels_hpp.resolve()}"',
        )
    if blas:
        cmd[-1:-1] = ["-DHAVE_BLAS", "-lblas"]
    label = out.name
    print(f"    Compiling {label} … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-3000:]}")
        raise RuntimeError(f"{label} compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Per-matrix benchmark
# ---------------------------------------------------------------------------


def _emit(
    writer,
    f_csv,
    base: dict,
    kernel: str,
    run_id: int,
    sym: float,
    comp: float,
    total: float,
    n_pairs: int | str = "",
    n_groups: int | str = "",
    perf_metrics: dict | None = None,
) -> None:
    writer.writerow(
        {
            **base,
            "kernel": kernel,
            "run_id": run_id,
            "symbolic_ms": _fmt(sym),
            "compute_ms": _fmt(comp),
            "total_ms": _fmt(total),
            "n_pairs": n_pairs,
            "n_groups": n_groups,
            **(perf_metrics or perf_wrap.empty_metrics()),
        }
    )
    f_csv.flush()


def benchmark_matrix(
    row: dict,
    mtx: Path,
    taco_bin: Path | None,
    taco_opt_bin: Path | None,
    prisma_bin: Path | None,
    runs: int,
    timeout: int,
    writer,
    f_csv,
    run_generic: bool = True,
    run_top10: bool = True,
    top_n: int = 10,
    blas: bool = False,
    work_dir: Path | None = None,
    mtx_cache: Path = Path("/tmp/mtx_cache"),
    threads: int = 0,
    perf: bool = False,
) -> None:
    name = row["name"]
    base = {
        "matrix_name": name,
        "group": row.get("group", ""),
        "rows": row.get("rows", ""),
        "cols": row.get("cols", ""),
        "nnz": row.get("nnz", ""),
        "threads": threads,
    }

    # ── TACO (needs real-general MTX) ─────────────────────────────────────────
    for label, binary in [("taco_cpu", taco_bin), ("taco_cpu_opt", taco_opt_bin)]:
        if binary is None:
            continue
        print(f"  [{label:<28}] ", end="", flush=True)
        try:
            taco_mtx = ensure_real_general(mtx, mtx_cache)
            triples, perf_metrics = run_taco_cpu(binary, taco_mtx, runs, timeout, perf)
        except (RuntimeError, subprocess.TimeoutExpired, Exception) as e:
            print(f"FAILED ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, label, run_id, _NAN, _NAN, _NAN)
            continue
        for run_id, (s, c, t) in enumerate(triples):
            _emit(writer, f_csv, base, label, run_id, s, c, t, perf_metrics=perf_metrics)
        timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
        print(
            f"avg {sum(timed) / len(timed):.3f} ms  ({len(triples)} runs incl. warmup)"
        )

    # ── Prisma (needs .bsp) ───────────────────────────────────────────────────
    if prisma_bin is None:
        return

    bsp = mtx.with_suffix(".bsp")
    if not bsp.exists():
        print(f"  [prisma_*] BSP not found ({bsp.name}) — skipping Prisma variants")
        return

    def _run_prisma(
        label: str, binary: Path, specialized: bool, extra_sym_ms: float = 0.0,
    ) -> None:
        print(f"  [{label:<28}] ", end="", flush=True)
        try:
            triples, n_pairs, n_groups, perf_metrics = run_prisma_spgemm(
                binary, bsp, runs, timeout, specialized, extra_sym_ms, perf, threads
            )
        except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"FAILED ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, label, run_id, _NAN, _NAN, _NAN)
            return
        for run_id, (s, c, t) in enumerate(triples):
            _emit(
                writer, f_csv, base, label, run_id, s, c, t,
                n_pairs, n_groups, perf_metrics=perf_metrics,
            )
        timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
        sym_timed = [s for s, _, _ in triples[1:]] or [s for s, _, _ in triples]
        print(
            f"avg {sum(timed) / len(timed):.3f} ms  "
            f"(sym avg={sum(sym_timed) / len(sym_timed):.2f} ms, "
            f"pairs={n_pairs}, groups={n_groups})"
        )

    if run_generic:
        _run_prisma("prisma_generic", prisma_bin, specialized=False)

    if run_top10:
        print(f"  [{'prisma_top10':<28}] ", end="", flush=True)
        mat_work = (work_dir / name) if work_dir else (_TMP_DIR / name)
        mat_work.mkdir(parents=True, exist_ok=True)
        try:
            shapes, shapes_sym_ms = analyze_spgemm_shapes(
                prisma_bin, bsp, top_n, timeout
            )
            print(shapes, shapes_sym_ms)
            if not shapes:
                print("SKIPPED (no shapes returned)")
            else:
                kernels_hpp, dispatch_hpp = gen_spgemm_kernels_files(shapes, mat_work)
                top10_bin = compile_prisma_spgemm_per_matrix(
                    dispatch_hpp,
                    mat_work / "prisma_cpu_bench_top10",
                    blas,
                    kernels_hpp=kernels_hpp,
                )
                _run_prisma(
                    "prisma_top10",
                    top10_bin,
                    specialized=True,
                    extra_sym_ms=shapes_sym_ms,
                )
        except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"FAILED ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, "prisma_top10", run_id, _NAN, _NAN, _NAN)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="CPU SpGEMM benchmark (C = A×A) on SuiteSparse matrices",
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
        default=min(16, os.cpu_count() or 16),
        help="threads for all runs, passed to prisma_cpu_bench via --threads "
        "and to TACO via OMP_NUM_THREADS (default: min(16, nproc) -- matches "
        "SpMM/SpMV's safe-cap convention; this codebase's own measured "
        "finding is that oversubscription past ~16 threads can be up to "
        "~15x slower). Ignored if --threads-sweep is given.",
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
        default="spgemm_cpu_results.csv",
        help="output CSV, append mode (default: spgemm_cpu_results.csv)",
    )
    g.add_argument(
        "--work-dir",
        default="",
        help=f"directory for compiled binaries (default: {_TMP_DIR})",
    )
    g.add_argument(
        "--prisma-bin",
        default="",
        dest="prisma_bin",
        help="pre-built prisma_cpu_bench binary (skips compilation)",
    )
    g.add_argument(
        "--taco-bin",
        default="",
        dest="taco_bin",
        help="pre-built bench_taco_cpu binary (skips compilation)",
    )
    g.add_argument(
        "--taco-opt-bin",
        default="",
        dest="taco_opt_bin",
        help="pre-built bench_taco_cpu_opt binary (skips compilation)",
    )

    g = p.add_argument_group("Build / skip")
    g.add_argument(
        "--no-compile",
        action="store_true",
        help="skip compilation; binaries must already exist in work-dir",
    )
    g.add_argument(
        "--blas",
        action="store_true",
        help="link BLAS when compiling prisma (enables BLAS tile path)",
    )
    g.add_argument(
        "--no-taco", action="store_true", dest="no_taco", help="skip both TACO kernels"
    )
    g.add_argument(
        "--no-prisma",
        action="store_true",
        dest="no_prisma",
        help="skip all Prisma kernels",
    )
    g.add_argument(
        "--no-generic",
        action="store_true",
        dest="no_generic",
        help="skip prisma_generic kernel",
    )
    g.add_argument(
        "--no-top10",
        action="store_true",
        dest="no_top10",
        help="skip prisma_top10 per-matrix specialized kernel",
    )
    g.add_argument(
        "--top-n",
        type=int,
        default=10,
        dest="top_n",
        help="number of top shapes to specialise per matrix (default: 10)",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else _TMP_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    taco_bin: Path | None = None
    taco_opt_bin: Path | None = None
    prisma_bin: Path | None = None

    if args.no_compile:
        if not args.no_taco:
            taco_bin = (
                Path(args.taco_bin) if args.taco_bin else work_dir / "bench_taco_cpu"
            )
            taco_opt_bin = (
                Path(args.taco_opt_bin)
                if args.taco_opt_bin
                else work_dir / "bench_taco_cpu_opt"
            )
            if not taco_bin.exists():
                taco_bin = None
            if not taco_opt_bin.exists():
                taco_opt_bin = None
        if not args.no_prisma:
            prisma_bin = (
                Path(args.prisma_bin)
                if args.prisma_bin
                else work_dir / "prisma_cpu_bench"
            )
            if not prisma_bin.exists():
                prisma_bin = None
    else:
        print("Compiling:")
        if not args.no_taco:
            if args.taco_bin:
                taco_bin = Path(args.taco_bin)
            else:
                try:
                    taco_bin = compile_taco_cpu(
                        "taco_kernel.h", work_dir / "bench_taco_cpu", args.threads
                    )
                except RuntimeError as e:
                    print(f"  bench_taco_cpu failed — skipping: {e}")
            if args.taco_opt_bin:
                taco_opt_bin = Path(args.taco_opt_bin)
            else:
                try:
                    taco_opt_bin = compile_taco_cpu(
                        "taco_kernel_opt.h",
                        work_dir / "bench_taco_cpu_opt",
                        args.threads,
                    )
                except RuntimeError as e:
                    print(f"  bench_taco_cpu_opt failed — skipping: {e}")
        if not args.no_prisma:
            if args.prisma_bin:
                prisma_bin = Path(args.prisma_bin)
            else:
                try:
                    prisma_bin = compile_prisma_spgemm(
                        work_dir / "prisma_cpu_bench", args.blas
                    )
                except RuntimeError as e:
                    print(f"  prisma_cpu_bench failed — skipping: {e}")
        print()

    matrices = load_matrix_list(Path(args.csv))

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    thread_values = (
        _parse_thread_sweep(args.threads_sweep) if args.threads_sweep else [args.threads]
    )

    print(f"TACO CPU bin   : {taco_bin or '(disabled)'}")
    print(f"TACO opt bin   : {taco_opt_bin or '(disabled)'}")
    print(f"Prisma bin     : {prisma_bin or '(disabled)'}")
    print(f"Top-N shapes   : {args.top_n}  (per-matrix specialized)")
    print(f"Output CSV     : {csv_path}")
    print(f"Matrices       : {len(matrices)}")
    print(f"Runs/matrix    : {args.runs}  (run_id 0 = warmup)")
    print(f"Threads        : {thread_values}")
    print(f"Timeout (s)    : {args.timeout}")
    print()

    write_header = _needs_header(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()

        # NOTE: prisma_top10's per-matrix specialized kernel is compiled
        # inline inside benchmark_matrix (not cached upfront), so sweeping
        # threads here recompiles it once per thread_values entry per
        # matrix -- a real, known cost of this loop ordering, traded for
        # not having to split benchmark_matrix into separate compile/run
        # phases. Fine for a handful of sweep points; expensive for a long
        # sweep across many matrices.
        for threads in thread_values:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f_csv.write(
                f"# {ts}  input={args.csv}  runs={args.runs}  threads={threads}\n"
            )
            env = {**os.environ, "OMP_NUM_THREADS": str(threads)}

            for i, row in enumerate(matrices, 1):
                name = row["name"]
                print(f"\n[threads={threads}] [{i}/{len(matrices)}] {name}")

                mtx = find_mtx(name, row.get("group", ""))
                if mtx is None:
                    print(f"  MTX not found in {_DATA_ROOT} — skipping")
                    continue
                print(f"  MTX → {mtx}")

                old_env = os.environ.copy()
                os.environ.update(env)
                try:
                    benchmark_matrix(
                        row=row,
                        mtx=mtx,
                        taco_bin=taco_bin,
                        taco_opt_bin=taco_opt_bin,
                        prisma_bin=prisma_bin,
                        runs=args.runs,
                        timeout=args.timeout,
                        writer=writer,
                        f_csv=f_csv,
                        run_generic=not args.no_generic and not args.no_prisma,
                        run_top10=not args.no_top10 and not args.no_prisma,
                        top_n=args.top_n,
                        blas=args.blas,
                        work_dir=work_dir,
                        threads=threads,
                        perf=args.perf,
                    )
                finally:
                    os.environ.clear()
                    os.environ.update(old_env)

    print(f"\nDone. Results appended to {csv_path}")


if __name__ == "__main__":
    main()
