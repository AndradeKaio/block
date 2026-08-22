#!/usr/bin/env python3
"""
suite-sparse/benchmark_spgemm_gpu.py — GPU SpGEMM benchmark on real
SuiteSparse matrices.

Computes C = A @ A (square self-product) and compares:
  prisma_tc_tile        prisma_bench --tc-kernel tile   (SpGEMM/GPU/prisma_bench.cu)
  prisma_tc_block       prisma_bench --tc-kernel block  (ablation row, off by default)
  prisma_cuda            prisma_bench (no --tc-kernel)
  tilespgemm              external TileSpGEMM 'test' binary
  cusparse_tilespgemm     cuSPARSE timing parsed from TileSpGEMM's own stdout
                          (TileSpGEMM doesn't dump cuSPARSE's C separately, so
                          there is nothing to validate independently here — see
                          validate_spgemm_gpu.py's docstring)
  taco_gpu                 bench_taco_gpu.cu
  tc_spgemm                bench_tc_spgemm.cu

All contenders read the SAME real .mtx/.bsp pair (A×A squaring), unlike
SpGEMM/GPU/sweep.py's competitor set which benchmarks synthetic block-sparse
matrices generated on the fly (different CSV schema: M/K/N/blocks_A/blocks_B/...
instead of matrix_name/nnz/...) and therefore can't feed plot_spgemm_gpu.py.
Runner logic for taco_gpu/tc_spgemm/tilespgemm+cusparse is otherwise identical
to sweep.py's (those binaries already take real mtx paths); prisma_bench is
driven directly here (sweep.py instead uses gpu_dispatch_demo, which only
ever generates synthetic matrices and has no real-matrix input mode).

prisma_bench.cu's symbolic pipeline (intersect pairs → merge groups → block
fusion → classify → build plan) runs once per matrix; its cost (pipe_total_ms)
is amortized over the timed compute runs, the same convention
benchmark_spgemm_cpu.py's run_prisma_spgemm already uses for the CPU side —
prisma_bench's own per-run "plan_ms" is always 0 (plan is cached across runs,
see prisma_bench.cu's comment), so blindly trusting it would silently zero out
symbolic cost the way sweep.py's CSV output currently does.

Usage:
  python benchmark_spgemm_gpu.py matrices.csv
  python benchmark_spgemm_gpu.py matrices.csv --out spgemm_gpu_results.csv --runs 5
  python benchmark_spgemm_gpu.py matrices.csv --no-compile --work-dir /path/to/bins
  python benchmark_spgemm_gpu.py matrices.csv --tc-block --no-tc-spgemm
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_SPGEMM_GPU_DIR = _SCRIPT_DIR.parent / "SpGEMM" / "GPU"
_CORE_DIR = _SCRIPT_DIR.parent / "core"
_TMP_DIR = Path("/tmp/_prismac/spgemm/")

sys.path.insert(0, str(_SCRIPT_DIR))
from benchmark_spmm_cpu import (  # noqa: E402  (path must be set up first)
    _fmt,
    _hdf5_paths,
    _needs_header,
    find_mtx,
    load_matrix_list,
    _parse_json_block,
)
from benchmark_spgemm_cpu import ensure_real_general  # noqa: E402

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
    "symbolic_ms",
    "compute_ms",
    "total_ms",
    "n_pairs",
    "n_groups",
]

_NAN = float("nan")

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

_CORE_SRCS = [
    "block.cpp", "block_generator.cpp", "interval_tree.cpp",
    "matrix.cpp", "matrix_io.cpp", "pipeline.cpp", "segment_tree.cpp",
]


def _needs_compile(out: Path, *srcs: Path) -> bool:
    if not out.exists():
        return True
    t = out.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in srcs)


def compile_prisma_bench(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    hdf5_inc, hdf5_lib = _hdf5_paths()
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "prisma_bench"
    src_paths = [_CORE_DIR / s for s in _CORE_SRCS] + [_SPGEMM_GPU_DIR / "prisma_bench.cu"]
    if not _needs_compile(out, *src_paths):
        print("  prisma_bench … up-to-date")
        return out
    srcs = [str(p) for p in src_paths]
    cmd = [
        nvcc, "-O3", "--expt-relaxed-constexpr", "-std=c++20", f"-arch={arch}",
        "-DHAVE_HDF5",
        f"-I{_CORE_DIR}", f"-I{_SPGEMM_GPU_DIR}", f"-I{hdf5_inc}",
        *srcs,
        str(Path(hdf5_lib) / "libhdf5.so"),
        "-o", str(out),
    ]
    print("  Compiling prisma_bench … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(r.stderr[-2000:])
        raise RuntimeError("prisma_bench compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def compile_taco_gpu(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "bench_taco_gpu"
    if not _needs_compile(out, _SPGEMM_GPU_DIR / "bench_taco_gpu.cu"):
        print("  bench_taco_gpu … up-to-date")
        return out
    src = str(_SPGEMM_GPU_DIR / "bench_taco_gpu.cu")
    cmd = [nvcc, "-O3", f"-arch={arch}", f"-I{_SPGEMM_GPU_DIR}", src, "-lcudart", "-o", str(out)]
    print("  Compiling bench_taco_gpu … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(r.stderr[-2000:])
        raise RuntimeError("bench_taco_gpu compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def compile_tc_spgemm(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "bench_tc_spgemm"
    if not _needs_compile(out, _SPGEMM_GPU_DIR / "bench_tc_spgemm.cu"):
        print("  bench_tc_spgemm … up-to-date")
        return out
    src = str(_SPGEMM_GPU_DIR / "bench_tc_spgemm.cu")
    cmd = [nvcc, "-O3", f"-arch={arch}", "-std=c++17", "-DTC_SPGEMM_NO_MAIN",
           f"-I{_SPGEMM_GPU_DIR}", src, "-o", str(out)]
    print("  Compiling bench_tc_spgemm … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(r.stderr[-2000:])
        raise RuntimeError("bench_tc_spgemm compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

_TILE_TOT_RE  = re.compile(r"CUDA TileSpGEMM run \d+ time is\s+([\d.]+)\s+ms")
_TILE_SYM_RE  = re.compile(r"CUDA TileSpGEMM run \d+ symbolic is\s+([\d.]+)\s+ms")
_TILE_CMP_RE  = re.compile(r"CUDA TileSpGEMM run \d+ compute is\s+([\d.]+)\s+ms")
_CSPARSE_TOT_RE = re.compile(r"CUDA cuSPARSE SpGEMM run \d+ time is\s+([\d.]+)\s+ms")
_CSPARSE_SYM_RE = re.compile(r"CUDA cuSPARSE SpGEMM run \d+ symbolic is\s+([\d.]+)\s+ms")
_CSPARSE_CMP_RE = re.compile(r"CUDA cuSPARSE SpGEMM run \d+ compute is\s+([\d.]+)\s+ms")


def run_prisma_bench_gpu(
    binary: Path, bsp: Path, runs: int, timeout: int, tc_kernel: str,
) -> tuple[list[tuple[float, float, float]], int, int]:
    """Run prisma_bench with A=B=bsp (A×A). Returns (triples, n_pairs, n_groups).

    triples is [(symbolic_ms, compute_ms, total_ms)] for run_id 0..runs.
    symbolic_ms is 0.0 for run_id 0 (warmup); for timed runs it's
    pipe_total_ms amortised over the timed runs — prisma_bench's own
    per-run "plan_ms" field is always 0 (plan is cached, see the .cu file's
    comment), so it can't be used directly the way tc_ms/cuda_ms can.
    """
    cmd = [str(binary), str(bsp), str(bsp), "--runs", str(runs)]
    if tc_kernel:
        cmd.extend(["--tc-kernel", tc_kernel])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"prisma_bench exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError(f"prisma_bench: no JSON output:\n{r.stdout[-400:]}")
    tc_ms   = d.get("tc_ms", [])
    cuda_ms = d.get("cuda_ms", [])
    if not tc_ms or not cuda_ms:
        raise RuntimeError("prisma_bench: empty tc_ms/cuda_ms in JSON")

    pipe_total = float(d.get("pipe_total_ms", 0.0))
    n_pairs    = int(d.get("n_pairs", 0))
    n_groups   = int(d.get("n_groups", 0))
    n_timed    = max(len(tc_ms) - 1, 1)
    amortised_sym = pipe_total / n_timed

    triples = []
    for run_id, (tc, cu) in enumerate(zip(tc_ms, cuda_ms)):
        comp = float(tc) + float(cu)
        sym  = 0.0 if run_id == 0 else amortised_sym
        triples.append((sym, comp, sym + comp))
    return triples, n_pairs, n_groups


def run_taco_gpu(binary: Path, mtx_a: Path, mtx_b: Path, runs: int,
                  timeout: int) -> list[tuple[float, float, float]]:
    """Run bench_taco_gpu once with --runs N. taco has no symbolic phase:
    symbolic_ms=0, compute_ms=total_ms=taco_ms."""
    cmd = [str(binary), str(mtx_a), str(mtx_b), "--runs", str(runs)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"bench_taco_gpu exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    times = d.get("taco_ms")
    if not times:
        raise RuntimeError("bench_taco_gpu: could not parse taco_ms from JSON output")
    return [(0.0, float(t), float(t)) for t in times]


def run_tc_spgemm(binary: Path, mtx_a: Path, mtx_b: Path, runs: int,
                   timeout: int) -> list[tuple[float, float, float]]:
    """Run bench_tc_spgemm once with --runs N."""
    cmd = [str(binary), str(mtx_a), str(mtx_b), "--runs", str(runs)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"bench_tc_spgemm exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    syms = d.get("tc_spgemm_symbolic_ms")
    cmps = d.get("tc_spgemm_compute_ms")
    tots = d.get("tc_spgemm_ms")
    if not syms or not cmps or not tots:
        raise RuntimeError("bench_tc_spgemm: could not parse phase timings from JSON output")
    return [(float(s), float(c), float(t)) for s, c, t in zip(syms, cmps, tots)]


def run_tilespgemm_and_cusparse(
    binary: Path, mtx_a: Path, mtx_b: Path, device: int, runs: int, timeout: int,
) -> tuple[list, list, list, list, list, list]:
    """Run the TileSpGEMM 'test' binary once with --runs N. Returns
    (tile_sym, tile_cmp, tile_tot, csparse_sym, csparse_cmp, csparse_tot),
    each a list of runs+1 floats (index 0 = warmup, 1..N = timed)."""
    cmd = [str(binary), "-d", str(device), "--runs", str(runs), str(mtx_a), str(mtx_b)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"TileSpGEMM exited {r.returncode}:\n{r.stderr[-800:]}")
    tile_sym = [float(x) for x in _TILE_SYM_RE.findall(r.stdout)]
    tile_cmp = [float(x) for x in _TILE_CMP_RE.findall(r.stdout)]
    tile_tot = [float(x) for x in _TILE_TOT_RE.findall(r.stdout)]
    cs_sym   = [float(x) for x in _CSPARSE_SYM_RE.findall(r.stdout)]
    cs_cmp   = [float(x) for x in _CSPARSE_CMP_RE.findall(r.stdout)]
    cs_tot   = [float(x) for x in _CSPARSE_TOT_RE.findall(r.stdout)]
    if not tile_tot:
        raise RuntimeError("TileSpGEMM: could not parse any TileSpGEMM run timings from stdout")
    if not cs_tot:
        raise RuntimeError("TileSpGEMM: could not parse any cuSPARSE run timings from stdout")
    return tile_sym, tile_cmp, tile_tot, cs_sym, cs_cmp, cs_tot


# ---------------------------------------------------------------------------
# Per-matrix benchmark
# ---------------------------------------------------------------------------


def _emit(writer, f_csv, base: dict, kernel: str, run_id: int,
          sym: float, comp: float, total: float,
          n_pairs: int | str = "", n_groups: int | str = "") -> None:
    writer.writerow({
        **base,
        "kernel":      kernel,
        "run_id":      run_id,
        "symbolic_ms": _fmt(sym),
        "compute_ms":  _fmt(comp),
        "total_ms":    _fmt(total),
        "n_pairs":     n_pairs,
        "n_groups":    n_groups,
    })
    f_csv.flush()


def benchmark_matrix(
    row: dict,
    mtx: Path,
    prisma_bin: Path | None,
    tilespgemm_bin: Path | None,
    taco_bin: Path | None,
    tc_spgemm_bin: Path | None,
    runs: int,
    timeout: int,
    device: int,
    writer,
    f_csv,
    run_tc_block: bool,
    mtx_cache: Path = Path("/tmp/mtx_cache"),
) -> None:
    name = row["name"]
    base = {
        "matrix_name": name,
        "group": row.get("group", ""),
        "rows":  row.get("rows",  ""),
        "cols":  row.get("cols",  ""),
        "nnz":   row.get("nnz",   ""),
    }
    bsp = mtx.with_suffix(".bsp")

    def _fail_all(label: str, e: Exception) -> None:
        print(f"FAILED ({e})")
        for run_id in range(runs + 1):
            _emit(writer, f_csv, base, label, run_id, _NAN, _NAN, _NAN)

    # ── Prisma (needs .bsp) ───────────────────────────────────────────────────
    if prisma_bin is not None:
        if not bsp.exists():
            print(f"  [prisma_*] BSP not found ({bsp.name}) — skipping Prisma variants")
        else:
            prisma_lanes = [("prisma_tc_tile", "tile"), ("prisma_cuda", "")]
            if run_tc_block:
                prisma_lanes.insert(1, ("prisma_tc_block", "block"))
            for label, tc_kernel in prisma_lanes:
                print(f"  [{label:<28}] ", end="", flush=True)
                try:
                    triples, n_pairs, n_groups = run_prisma_bench_gpu(
                        prisma_bin, bsp, runs, timeout, tc_kernel
                    )
                except (RuntimeError, subprocess.TimeoutExpired) as e:
                    _fail_all(label, e)
                    continue
                for run_id, (s, c, t) in enumerate(triples):
                    _emit(writer, f_csv, base, label, run_id, s, c, t, n_pairs, n_groups)
                timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
                print(f"avg {sum(timed) / len(timed):.3f} ms  "
                      f"(pairs={n_pairs}, groups={n_groups})")

    # ── TileSpGEMM + cuSPARSE (needs .mtx) ────────────────────────────────────
    if tilespgemm_bin is not None:
        print(f"  [{'tilespgemm+cusparse':<28}] ", end="", flush=True)
        try:
            # TileSpGEMM is external/third-party code with no symmetric- or
            # pattern-format handling of its own (unlike bench_taco_gpu.cu and
            # bench_tc_spgemm.cu, which both have their own is_pattern/
            # is_symmetric expansion, see their read_mtx()) -- fed the raw
            # suite-sparse .mtx directly, it fails on every symmetric/pattern
            # matrix, which for typical SuiteSparse A×A test sets is most or
            # all of them. Convert first, exactly like TACO CPU already does
            # (benchmark_spgemm_cpu.py's ensure_real_general), instead of
            # passing the raw file through unconverted.
            tile_mtx = ensure_real_general(mtx, mtx_cache)
            tile_sym, tile_cmp, tile_tot, cs_sym, cs_cmp, cs_tot = \
                run_tilespgemm_and_cusparse(tilespgemm_bin, tile_mtx, tile_mtx, device, runs, timeout)
            for run_id, (s, c, t) in enumerate(zip(tile_sym, tile_cmp, tile_tot)):
                _emit(writer, f_csv, base, "tilespgemm", run_id, s, c, t)
            for run_id, (s, c, t) in enumerate(zip(cs_sym, cs_cmp, cs_tot)):
                _emit(writer, f_csv, base, "cusparse_tilespgemm", run_id, s, c, t)
            timed = tile_tot[1:] or tile_tot
            print(f"avg {sum(timed) / len(timed):.3f} ms  ({len(tile_tot)} runs incl. warmup)")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"FAILED ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, "tilespgemm",          run_id, _NAN, _NAN, _NAN)
                _emit(writer, f_csv, base, "cusparse_tilespgemm", run_id, _NAN, _NAN, _NAN)

    # ── TACO GPU (needs .mtx) ──────────────────────────────────────────────────
    if taco_bin is not None:
        print(f"  [{'taco_gpu':<28}] ", end="", flush=True)
        try:
            # bench_taco_gpu.cu's read_mtx() has no symmetric/pattern
            # expansion (same limitation as SpGEMM CPU's bench_taco.c,
            # which is why benchmark_spgemm_cpu.py already converts before
            # calling it) -- convert here too instead of feeding it a raw
            # symmetric/pattern matrix it can't parse correctly.
            taco_mtx = ensure_real_general(mtx, mtx_cache)
            triples = run_taco_gpu(taco_bin, taco_mtx, taco_mtx, runs, timeout)
            for run_id, (s, c, t) in enumerate(triples):
                _emit(writer, f_csv, base, "taco_gpu", run_id, s, c, t)
            timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
            print(f"avg {sum(timed) / len(timed):.3f} ms  ({len(triples)} runs incl. warmup)")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            _fail_all("taco_gpu", e)

    # ── TC_SpGEMM (needs .mtx) ─────────────────────────────────────────────────
    if tc_spgemm_bin is not None:
        print(f"  [{'tc_spgemm':<28}] ", end="", flush=True)
        try:
            # Same reader limitation as bench_taco_gpu.cu -- see comment above.
            tc_mtx = ensure_real_general(mtx, mtx_cache)
            triples = run_tc_spgemm(tc_spgemm_bin, tc_mtx, tc_mtx, runs, timeout)
            for run_id, (s, c, t) in enumerate(triples):
                _emit(writer, f_csv, base, "tc_spgemm", run_id, s, c, t)
            timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
            print(f"avg {sum(timed) / len(timed):.3f} ms  ({len(triples)} runs incl. warmup)")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            _fail_all("tc_spgemm", e)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="GPU SpGEMM benchmark (C = A×A) on SuiteSparse matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", metavar="MATRICES.csv",
                   help="input CSV with at least a 'name' column")

    g = p.add_argument_group("Run control")
    g.add_argument("--runs", type=int, default=5,
                   help="timed repetitions per matrix; run_id 0 = warmup (default: 5)")
    g.add_argument("--timeout", type=int, default=300,
                   help="per-contender timeout in seconds (default: 300)")
    g.add_argument("--device", type=int, default=0,
                   help="GPU device index passed to TileSpGEMM (default: 0)")

    g = p.add_argument_group("Paths")
    g.add_argument("--out", default="spgemm_gpu_results.csv",
                   help="output CSV, append mode (default: spgemm_gpu_results.csv)")
    g.add_argument("--work-dir", default="",
                   help=f"directory for compiled binaries (default: {_TMP_DIR})")
    g.add_argument("--tilespgemm-dir", default="/home/kaio/artifacts/TileSpGEMM/src",
                   metavar="DIR", help="directory containing TileSpGEMM 'test' binary")
    g.add_argument("--prisma-bin", default="", dest="prisma_bin",
                   help="pre-built prisma_bench binary (skips compilation)")
    g.add_argument("--taco-gpu-bin", default="", dest="taco_gpu_bin",
                   help="pre-built bench_taco_gpu binary (skips compilation)")
    g.add_argument("--tc-spgemm-bin", default="", dest="tc_spgemm_bin",
                   help="pre-built bench_tc_spgemm binary (skips compilation)")

    g = p.add_argument_group("Build / skip")
    g.add_argument("--no-compile", action="store_true",
                   help="skip compilation; binaries must already exist under --work-dir")
    g.add_argument("--no-prisma", action="store_true", help="skip all Prisma GPU lanes")
    g.add_argument("--tc-block", action="store_true",
                   help="also benchmark prisma_tc_block (ablation row, off by default)")
    g.add_argument("--no-tilespgemm", action="store_true",
                   help="skip TileSpGEMM AND cuSPARSE (cuSPARSE timing is parsed "
                        "from TileSpGEMM's own stdout, so they can't run independently)")
    g.add_argument("--no-taco-gpu", action="store_true", help="skip the TACO GPU contender")
    g.add_argument("--no-tc-spgemm", action="store_true", help="skip the TC_SpGEMM contender")
    g.add_argument("--cuda-home", default="/usr/local/cuda")
    g.add_argument("--arch", default="sm_120")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    bin_dir = Path(args.work_dir) if args.work_dir else _TMP_DIR
    bin_dir.mkdir(parents=True, exist_ok=True)

    tilespgemm_bin: Path | None = None
    if not args.no_tilespgemm:
        tilespgemm_bin = Path(args.tilespgemm_dir) / "test"
        if not tilespgemm_bin.exists():
            sys.exit(
                f"TileSpGEMM binary not found: {tilespgemm_bin}\n"
                f"Build it first with 'make' inside {args.tilespgemm_dir}, "
                f"or pass --no-tilespgemm to skip it."
            )

    prisma_bin: Path | None = None
    taco_bin: Path | None = None
    tc_spgemm_bin: Path | None = None

    if not args.no_compile:
        print("Compiling:")
        if not args.no_prisma:
            if args.prisma_bin:
                prisma_bin = Path(args.prisma_bin)
            else:
                try:
                    prisma_bin = compile_prisma_bench(args.cuda_home, args.arch, bin_dir)
                except RuntimeError as e:
                    print(f"  prisma_bench failed — skipping: {e}")
        if not args.no_taco_gpu:
            if args.taco_gpu_bin:
                taco_bin = Path(args.taco_gpu_bin)
            else:
                try:
                    taco_bin = compile_taco_gpu(args.cuda_home, args.arch, bin_dir)
                except RuntimeError as e:
                    print(f"  bench_taco_gpu failed — skipping: {e}")
        if not args.no_tc_spgemm:
            if args.tc_spgemm_bin:
                tc_spgemm_bin = Path(args.tc_spgemm_bin)
            else:
                try:
                    tc_spgemm_bin = compile_tc_spgemm(args.cuda_home, args.arch, bin_dir)
                except RuntimeError as e:
                    print(f"  bench_tc_spgemm failed — skipping: {e}")
        print()
    else:
        if not args.no_prisma:
            prisma_bin = Path(args.prisma_bin) if args.prisma_bin else bin_dir / "prisma_bench"
            if not prisma_bin.exists():
                print(f"prisma_bench not found in {bin_dir}; Prisma lanes will be skipped")
                prisma_bin = None
        if not args.no_taco_gpu:
            taco_bin = Path(args.taco_gpu_bin) if args.taco_gpu_bin else bin_dir / "bench_taco_gpu"
            if not taco_bin.exists():
                print(f"bench_taco_gpu not found in {bin_dir}; TACO GPU will be skipped")
                taco_bin = None
        if not args.no_tc_spgemm:
            tc_spgemm_bin = Path(args.tc_spgemm_bin) if args.tc_spgemm_bin else bin_dir / "bench_tc_spgemm"
            if not tc_spgemm_bin.exists():
                print(f"bench_tc_spgemm not found in {bin_dir}; TC_SpGEMM will be skipped")
                tc_spgemm_bin = None

    matrices = load_matrix_list(Path(args.csv))

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Prisma bin     : {prisma_bin      or '(disabled)'}")
    print(f"TileSpGEMM bin : {tilespgemm_bin  or '(disabled)'}")
    print(f"TACO GPU bin   : {taco_bin        or '(disabled)'}")
    print(f"TC_SpGEMM bin  : {tc_spgemm_bin   or '(disabled)'}")
    print(f"prisma_tc_block: {'included' if args.tc_block else 'skipped (ablation row)'}")
    print(f"Output CSV     : {csv_path}")
    print(f"Matrices       : {len(matrices)}")
    print(f"Runs/matrix    : {args.runs}  (run_id 0 = warmup)")
    print(f"Timeout (s)    : {args.timeout}")
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
            print(f"\n[{i}/{len(matrices)}] {name}")

            mtx = find_mtx(name, row.get("group", ""))
            if mtx is None:
                print("  MTX not found — skipping")
                continue

            benchmark_matrix(
                row=row,
                mtx=mtx,
                prisma_bin=prisma_bin,
                tilespgemm_bin=tilespgemm_bin,
                taco_bin=taco_bin,
                tc_spgemm_bin=tc_spgemm_bin,
                runs=args.runs,
                timeout=args.timeout,
                device=args.device,
                writer=writer,
                f_csv=f_csv,
                run_tc_block=args.tc_block,
            )

    print(f"\nDone. Results appended to {csv_path}")


if __name__ == "__main__":
    main()
