#!/usr/bin/env python3
"""
sweep.py — Block-sparse SpGEMM benchmark sweep.

Competitors per configuration (in execution order):
  prisma_tc_tile      gpu_dispatch_demo --tc-kernel tile   (generates A.mtx / B.mtx)
  prisma_tc_block     gpu_dispatch_demo --tc-kernel block  (reuses same TC matrices)
  prisma_cuda         gpu_dispatch_demo (no --tc-kernel)   (own internal matrices)
  tilespgemm            TileSpGEMM test binary               (reads TC A.mtx / B.mtx)
  cusparse_tilespgemm   cuSPARSE timing parsed from TileSpGEMM stdout
  tc_spgemm             bench_tc_spgemm binary               (reads TC A.mtx / B.mtx)

TileSpGEMM, cuSPARSE, and TC_SpGEMM always operate on the matrices written by the
TC-tile run.  tc_spgemm reports preproc+compute+postproc total; run_id=0 is the
warmup invocation (recorded), run_id=1..runs-1 are the timed runs.

CSV columns
  M, K, N, blocks_A, blocks_B,
  block_h_min, block_h_max, block_w_min, block_w_max,
  kernel, run_id, symbolic_ms, compute_ms, total_ms,
  n_pairs, n_groups, n_tc_descs, n_cuda_descs

Usage
  python sweep.py --mnk-list 1024 2048 --n-blocks-list 8 16 --runs 6
  python sweep.py --no-compile --tilespgemm-dir /path/to/TileSpGEMM/src
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_CORE_DIR = _SCRIPT_DIR.parent.parent / "core"
_BIN_CACHE = Path("/tmp/benchmark_bins")

# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

_NVCC_FLAGS = ["-O3", "--expt-relaxed-constexpr", "-std=c++20"]


def _needs_compile(out: Path, *srcs: Path) -> bool:
    if not out.exists():
        return True
    t = out.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in srcs)

_CORE_SRCS = [
    "block.cpp",
    "block_generator.cpp",
    "interval_tree.cpp",
    "matrix.cpp",
    "matrix_io.cpp",
    "pipeline.cpp",
    "segment_tree.cpp",
]


def compile_tc_spgemm(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "bench_tc_spgemm"
    if not _needs_compile(out, _SCRIPT_DIR / "bench_tc_spgemm.cu"):
        print("  bench_tc_spgemm … up-to-date")
        return out
    src = str(_SCRIPT_DIR / "bench_tc_spgemm.cu")
    cmd = [nvcc, "-O3", f"-arch={arch}", "-std=c++17", f"-I{_SCRIPT_DIR}", src, "-lcudart", "-o", str(out)]
    print("  Compiling bench_tc_spgemm … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(r.stderr[-2000:])
        raise RuntimeError("bench_tc_spgemm compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def compile_taco_gpu(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "bench_taco_gpu"
    if not _needs_compile(out, _SCRIPT_DIR / "bench_taco_gpu.cu"):
        print("  bench_taco_gpu … up-to-date")
        return out
    src = str(_SCRIPT_DIR / "bench_taco_gpu.cu")
    cmd = [nvcc, "-O3", f"-arch={arch}", f"-I{_SCRIPT_DIR}", src, "-lcudart", "-o", str(out)]
    print("  Compiling bench_taco_gpu … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(r.stderr[-2000:])
        raise RuntimeError("bench_taco_gpu compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def compile_gpu_dispatch(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "gpu_dispatch_demo"
    src_paths = [_CORE_DIR / s for s in _CORE_SRCS] + [_SCRIPT_DIR / "gpu_dispatch_demo.cu"]
    if not _needs_compile(out, *src_paths):
        print("  gpu_dispatch_demo … up-to-date")
        return out
    srcs = [str(p) for p in src_paths]
    cmd = [nvcc, *_NVCC_FLAGS, f"-arch={arch}", f"-I{_CORE_DIR}", f"-I{_SCRIPT_DIR}", *srcs, "-o", str(out)]
    print("  Compiling gpu_dispatch_demo … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)")
        print(r.stderr[-2000:])
        sys.exit("Compilation failed.")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# JSON parsing
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


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------

_TILE_TOT_RE = re.compile(r"CUDA TileSpGEMM run \d+ time is\s+([\d.]+)\s+ms")
_TILE_SYM_RE = re.compile(r"CUDA TileSpGEMM run \d+ symbolic is\s+([\d.]+)\s+ms")
_TILE_CMP_RE = re.compile(r"CUDA TileSpGEMM run \d+ compute is\s+([\d.]+)\s+ms")
_CSPARSE_TOT_RE = re.compile(r"CUDA cuSPARSE SpGEMM run \d+ time is\s+([\d.]+)\s+ms")
_CSPARSE_SYM_RE = re.compile(r"CUDA cuSPARSE SpGEMM run \d+ symbolic is\s+([\d.]+)\s+ms")
_CSPARSE_CMP_RE = re.compile(r"CUDA cuSPARSE SpGEMM run \d+ compute is\s+([\d.]+)\s+ms")


def run_gpu_dispatch(
    binary: Path,
    cfg: dict,
    tc_kernel: str,
    out_dir: Path,
    runs: int,
    seed: int,
    block_density: float = 1.0,
) -> dict:
    """
    Invoke gpu_dispatch_demo and return the parsed JSON dict.
    tc_kernel: "tile", "block", or "" for CUDA-only path.
    Raises RuntimeError on non-zero exit or missing JSON.
    """
    cli = [
        str(binary),
        "--M", str(cfg["M"]),
        "--K", str(cfg["K"]),
        "--N", str(cfg["N"]),
        "--blocks-A", str(cfg["n_blocks"]),
        "--blocks-B", str(cfg["n_blocks"]),
        "--block-h-min", str(cfg["block_h_min"]),
        "--block-h-max", str(cfg["block_h_max"]),
        "--block-w-min", str(cfg["block_w_min"]),
        "--block-w-max", str(cfg["block_w_max"]),
        "--seed", str(seed),
        "--runs", str(runs),
        "--block-density", str(block_density),
        "--out-dir", str(out_dir),
    ]
    if tc_kernel:
        cli.extend(["--tc-kernel", tc_kernel])

    r = subprocess.run(cli, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gpu_dispatch_demo exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError("gpu_dispatch_demo: no JSON block in stdout")
    return d


def run_taco_gpu(binary: Path, mtx_a: Path, mtx_b: Path, runs: int) -> list:
    """
    Run bench_taco_gpu once with --runs N.
    Returns list of (symbolic_ms, compute_ms, total_ms) triples, one per run.
    taco has no symbolic phase: symbolic_ms=0, compute_ms=total_ms=taco_ms.
    Raises RuntimeError on non-zero exit or parse failure.
    """
    cli = [str(binary), str(mtx_a), str(mtx_b), "--runs", str(runs)]
    r = subprocess.run(cli, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bench_taco_gpu exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    times = d.get("taco_ms")
    if not times:
        raise RuntimeError("bench_taco_gpu: could not parse taco_ms from JSON output")
    return [(0.0, float(t), float(t)) for t in times]


def run_tc_spgemm(binary: Path, mtx_a: Path, mtx_b: Path, runs: int) -> list:
    """
    Run bench_tc_spgemm once with --runs N.
    Returns list of (symbolic_ms, compute_ms, total_ms) triples.
    Raises RuntimeError on non-zero exit or parse failure.
    """
    cli = [str(binary), str(mtx_a), str(mtx_b), "--runs", str(runs)]
    r = subprocess.run(cli, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bench_tc_spgemm exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    syms  = d.get("tc_spgemm_symbolic_ms")
    cmps  = d.get("tc_spgemm_compute_ms")
    tots  = d.get("tc_spgemm_ms")
    if not syms or not cmps or not tots:
        raise RuntimeError("bench_tc_spgemm: could not parse phase timings from JSON output")
    return [(float(s), float(c), float(t)) for s, c, t in zip(syms, cmps, tots)]


def run_tilespgemm_and_cusparse(
    binary: Path, mtx_a: Path, mtx_b: Path, device: int, runs: int
) -> tuple[list, list, list, list, list, list]:
    """
    Run TileSpGEMM test binary once with --runs N.
    Returns (tile_sym, tile_cmp, tile_tot, csparse_sym, csparse_cmp, csparse_tot),
    each a list of runs+1 floats (index 0 = warmup, 1..N = timed).
    Raises RuntimeError on non-zero exit or parse failure.
    """
    cli = [str(binary), "-d", str(device), "--runs", str(runs), str(mtx_a), str(mtx_b)]
    r = subprocess.run(cli, capture_output=True, text=True)
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
# CSV
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "M", "K", "N", "blocks_A", "blocks_B",
    "block_h_min", "block_h_max", "block_w_min", "block_w_max", "block_density",
    "kernel", "run_id", "symbolic_ms", "compute_ms", "total_ms",
    "n_pairs", "n_groups", "n_tc_descs", "n_cuda_descs",
]

_NAN = float("nan")


def _fmt(v):
    return "nan" if v != v else f"{v:.4f}"


def _write_run_comment(f, args):
    """Write a # comment line recording all sweep parameters for this run."""
    mnk       = " ".join(f"{m},{k},{n}" for m, k, n in args.mnk_list)
    blocks    = " ".join(str(n) for n in args.n_blocks_list)
    ranges    = " ".join(f"{h0},{h1},{w0},{w1}"
                         for h0, h1, w0, w1 in args.block_ranges)
    densities = ",".join(str(d) for d in args.block_density)
    ts        = time.strftime("%Y-%m-%d %H:%M:%S")
    f.write(
        f"# {ts}"
        f"  --mnk-list {mnk}"
        f"  --n-blocks-list {blocks}"
        f"  --block-ranges {ranges}"
        f"  --block-density {densities}"
        f"  --runs {args.runs}"
        f"  --seed {args.seed}"
        f"  --device {args.device}"
        "\n"
    )


def _emit(writer, f_csv, base: dict, kernel: str, run_id: int,
          symbolic_ms, compute_ms, total_ms):
    writer.writerow({
        **base,
        "kernel":      kernel,
        "run_id":      run_id,
        "symbolic_ms": _fmt(symbolic_ms),
        "compute_ms":  _fmt(compute_ms),
        "total_ms":    _fmt(total_ms),
    })
    f_csv.flush()


# ---------------------------------------------------------------------------
# Per-config benchmark
# ---------------------------------------------------------------------------


def benchmark_config(
    cfg: dict,
    dispatch_bin: Path,
    tilespgemm_bin: Path,
    taco_gpu_bin,
    tc_spgemm_bin,
    runs: int,
    seed: int,
    device: int,
    work_dir: Path,
    writer,
    f_csv,
    block_density: float = 1.0,
):
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    n = cfg["n_blocks"]
    h_min, h_max = cfg["block_h_min"], cfg["block_h_max"]
    w_min, w_max = cfg["block_w_min"], cfg["block_w_max"]

    print(f"  M={M} K={K} N={N}  blocks={n}  h=[{h_min},{h_max}]  w=[{w_min},{w_max}]  density={block_density:.2f}")

    # Dedicated directory for the shared TC-snapped matrices
    mtx_dir = (
        work_dir
        / f"M{M}_K{K}_N{N}_b{n}_h{h_min}-{h_max}_w{w_min}-{w_max}_bd{block_density:.4f}"
    )
    mtx_dir.mkdir(parents=True, exist_ok=True)
    a_mtx = mtx_dir / "A.mtx"
    b_mtx = mtx_dir / "B.mtx"

    # ── Step 1: TC tile — generates the definitive A.mtx and B.mtx ──────────
    print("    [prisma_tc_tile]  ", end="", flush=True)
    try:
        d = run_gpu_dispatch(dispatch_bin, cfg, "tile", mtx_dir, runs, seed, block_density)
    except RuntimeError as e:
        print(f"SKIPPED  ({e})")
        return

    tc_ms  = d.get("tc_ms", [])
    plan_ms = d.get("plan_ms", [0.0] * len(tc_ms))
    if not tc_ms:
        print("0 timed runs (no intersecting pairs) — skipping config")
        return

    meta = {
        "M": M, "K": K, "N": N,
        "blocks_A": n, "blocks_B": n,
        "block_h_min": h_min, "block_h_max": h_max,
        "block_w_min": w_min, "block_w_max": w_max,
        "block_density": block_density,
        "n_pairs":     d.get("n_pairs", 0),
        "n_groups":    d.get("n_groups", 0),
        "n_tc_descs":  d.get("n_tc", 0),
        "n_cuda_descs": d.get("n_cuda", 0),
    }
    for run_id, (pm, tm) in enumerate(zip(plan_ms, tc_ms)):
        _emit(writer, f_csv, meta, "prisma_tc_tile", run_id, pm, tm, pm + tm)
    avg_tot = sum(p + t for p, t in zip(plan_ms, tc_ms)) / len(tc_ms)
    print(f"avg {avg_tot:.3f} ms  "
          f"(n_pairs={meta['n_pairs']} n_groups={meta['n_groups']})")

    # ── Step 2: TC block — same mtx_dir, same seed → identical TC matrices ───
    print("    [prisma_tc_block] ", end="", flush=True)
    try:
        d2 = run_gpu_dispatch(dispatch_bin, cfg, "block", mtx_dir, runs, seed, block_density)
        tc_ms2   = d2.get("tc_ms", [])
        plan_ms2 = d2.get("plan_ms", [0.0] * len(tc_ms2))
        for run_id, (pm, tm) in enumerate(zip(plan_ms2, tc_ms2)):
            _emit(writer, f_csv, meta, "prisma_tc_block", run_id, pm, tm, pm + tm)
        avg2 = sum(p + t for p, t in zip(plan_ms2, tc_ms2)) / len(tc_ms2) if tc_ms2 else 0.0
        print(f"avg {avg2:.3f} ms")
    except RuntimeError as e:
        print(f"FAILED  ({e})")

    # ── Step 3: CUDA dispatch — throwaway dir; kernel on non-snapped matrices ─
    print("    [prisma_cuda]     ", end="", flush=True)
    with tempfile.TemporaryDirectory(prefix="cuda_mtx_") as tmp:
        try:
            d3 = run_gpu_dispatch(dispatch_bin, cfg, "", Path(tmp), runs, seed, block_density)
            cuda_ms3  = d3.get("cuda_ms", [])
            plan_ms3  = d3.get("plan_ms", [0.0] * len(cuda_ms3))
            cuda_meta = {
                **meta,
                "n_pairs":     d3.get("n_pairs", 0),
                "n_groups":    d3.get("n_groups", 0),
                "n_tc_descs":  d3.get("n_tc", 0),
                "n_cuda_descs": d3.get("n_cuda", 0),
            }
            for run_id, (pm, cm) in enumerate(zip(plan_ms3, cuda_ms3)):
                _emit(writer, f_csv, cuda_meta, "prisma_cuda", run_id, pm, cm, pm + cm)
            avg3 = sum(p + c for p, c in zip(plan_ms3, cuda_ms3)) / len(cuda_ms3) if cuda_ms3 else 0.0
            print(f"avg {avg3:.3f} ms")
        except RuntimeError as e:
            print(f"FAILED  ({e})")

    # ── Step 4: TileSpGEMM + cuSPARSE — read TC matrices from mtx_dir ────────
    if not (a_mtx.exists() and b_mtx.exists()):
        print("    [tilespgemm]        SKIPPED (A.mtx / B.mtx not written by TC-tile run)")
        return

    tile_meta = {**meta, "n_tc_descs": 0, "n_cuda_descs": 0}

    print(f"    [tilespgemm+cusparse]          ", end="", flush=True)
    try:
        tile_sym, tile_cmp, tile_tot, cs_sym, cs_cmp, cs_tot = \
            run_tilespgemm_and_cusparse(tilespgemm_bin, a_mtx, b_mtx, device, runs)
        for run_id, (s, c, t) in enumerate(zip(tile_sym, tile_cmp, tile_tot)):
            _emit(writer, f_csv, tile_meta, "tilespgemm", run_id, s, c, t)
        for run_id, (s, c, t) in enumerate(zip(cs_sym, cs_cmp, cs_tot)):
            _emit(writer, f_csv, tile_meta, "cusparse_tilespgemm", run_id, s, c, t)
        timed = tile_tot[1:] or tile_tot
        print(f"avg {sum(timed)/len(timed):.3f} ms  ({len(tile_tot)} runs incl. warmup)")
    except RuntimeError as e:
        print(f"FAILED  ({e})")
        for run_id in range(runs + 1):
            _emit(writer, f_csv, tile_meta, "tilespgemm",           run_id, _NAN, _NAN, _NAN)
            _emit(writer, f_csv, tile_meta, "cusparse_tilespgemm",  run_id, _NAN, _NAN, _NAN)

    # ── Step 5: TACO GPU — reads same TC matrices from mtx_dir ───────────────
    if taco_gpu_bin:
        print(f"    [taco_gpu]                    ", end="", flush=True)
        try:
            taco_triples = run_taco_gpu(taco_gpu_bin, a_mtx, b_mtx, runs)
            for run_id, (s, c, t) in enumerate(taco_triples):
                _emit(writer, f_csv, tile_meta, "taco_gpu", run_id, s, c, t)
            timed = [t for _, _, t in taco_triples[1:]] or [t for _, _, t in taco_triples]
            print(f"avg {sum(timed)/len(timed):.3f} ms  ({len(taco_triples)} runs incl. warmup)")
        except RuntimeError as e:
            print(f"FAILED  ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, tile_meta, "taco_gpu", run_id, _NAN, _NAN, _NAN)

    # ── Step 6: TC_SpGEMM — reads same TC matrices from mtx_dir ─────────────
    # run_id=0 is the warmup; all runs+1 timings are recorded.
    if tc_spgemm_bin:
        print(f"    [tc_spgemm]                   ", end="", flush=True)
        try:
            tc_triples = run_tc_spgemm(tc_spgemm_bin, a_mtx, b_mtx, runs)
            for run_id, (s, c, t) in enumerate(tc_triples):
                _emit(writer, f_csv, tile_meta, "tc_spgemm", run_id, s, c, t)
            timed = [t for _, _, t in tc_triples[1:]] or [t for _, _, t in tc_triples]
            print(f"avg {sum(timed)/len(timed):.3f} ms  ({len(tc_triples)} runs incl. warmup)")
        except RuntimeError as e:
            print(f"FAILED  ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, tile_meta, "tc_spgemm", run_id, _NAN, _NAN, _NAN)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_range(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) == 2:
        return (parts[0], parts[1], parts[0], parts[1])
    if len(parts) == 4:
        return tuple(parts)
    raise argparse.ArgumentTypeError(
        f"block range must be 'h_min,h_max' or 'h_min,h_max,w_min,w_max', got: {s!r}"
    )


def _parse_mnk(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError(
        f"MNK must be 'N' (square) or 'M,K,N', got: {s!r}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Block-sparse SpGEMM benchmark sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g = p.add_argument_group("Matrix grid")
    g.add_argument(
        "--mnk-list", nargs="+", type=_parse_mnk,
        default=[(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
        metavar="M[,K,N]",
        help="matrix sizes — single int for square, or M,K,N  (default: 1024 2048 4096)",
    )
    g.add_argument(
        "--n-blocks-list", nargs="+", type=int, default=[8, 16, 32],
        metavar="N",
        help="blocks_A = blocks_B = N  (default: 8 16 32)",
    )
    g.add_argument(
        "--block-ranges", nargs="+", type=_parse_range,
        default=[(16, 32, 16, 32), (32, 64, 32, 64), (64, 128, 64, 128)],
        metavar="H_MIN,H_MAX,W_MIN,W_MAX",
        help="block dimension ranges  (default: 16,32,16,32  32,64,32,64  64,128,64,128)",
    )

    g.add_argument(
        "--block-density", type=lambda s: [float(x) for x in s.split(",")],
        default=[1.0], metavar="F[,F...]",
        help="comma-separated intra-block density values to sweep (default: 1.0)",
    )

    g = p.add_argument_group("Run control")
    g.add_argument("--runs", type=int, default=6,
                   help="executions per config; run_id 0..runs-1  (default: 6)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=int, default=0,
                   help="GPU device index passed to TileSpGEMM  (default: 0)")

    g = p.add_argument_group("Paths")
    g.add_argument("--out", default="results.csv",
                   help="output CSV, opened in append mode  (default: results.csv)")
    g.add_argument("--work-dir", default="",
                   help="directory for .mtx files  (default: auto tempdir)")
    g.add_argument(
        "--tilespgemm-dir",
        default="/home/kaio/artifacts/TileSpGEMM/src",
        metavar="PATH",
        help="directory containing TileSpGEMM 'test' binary  "
             "(default: /home/kaio/artifacts/TileSpGEMM/src)",
    )
    g.add_argument("--bin-dir", default="",
                   help=f"directory for compiled binaries (default: {_BIN_CACHE})")

    g.add_argument(
        "--taco-gpu-bin", default="", metavar="PATH",
        help="path to pre-built bench_taco_gpu binary (skips compilation)",
    )
    g.add_argument(
        "--tc-spgemm-bin", default="", metavar="PATH",
        help="path to pre-built bench_tc_spgemm binary (skips compilation)",
    )

    g = p.add_argument_group("Build")
    g.add_argument("--no-compile", action="store_true",
                   help="skip compilation; assume gpu_dispatch_demo already built")
    g.add_argument("--no-taco-gpu", action="store_true",
                   help="skip TACO GPU competitor entirely")
    g.add_argument("--no-tc-spgemm", action="store_true",
                   help="skip TC_SpGEMM competitor entirely")
    g.add_argument("--cuda-home", default="/usr/local/cuda")
    g.add_argument("--arch", default="sm_120")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _needs_header(csv_path: Path) -> bool:
    """Return True if the CSV file is missing or has a different schema than _CSV_FIELDS."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True
    expected = ",".join(_CSV_FIELDS)
    with open(csv_path, newline="") as f:
        for line in f:
            if not line.startswith("#"):
                return line.rstrip("\r\n") != expected
    return True


def main():
    args = parse_args()

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path(tempfile.mkdtemp(prefix="spgemm_sweep_"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    bin_dir = Path(args.bin_dir) if args.bin_dir else _BIN_CACHE
    bin_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    tilespgemm_bin = Path(args.tilespgemm_dir) / "test"
    if not tilespgemm_bin.exists():
        sys.exit(
            f"TileSpGEMM binary not found: {tilespgemm_bin}\n"
            f"Build it first with 'make' inside {args.tilespgemm_dir}"
        )

    # ── TACO GPU binary ────────────────────────────────────────────────────────
    taco_gpu_bin = None
    if not args.no_taco_gpu:
        if args.taco_gpu_bin:
            taco_gpu_bin = Path(args.taco_gpu_bin)
            if not taco_gpu_bin.exists():
                sys.exit(f"bench_taco_gpu binary not found: {taco_gpu_bin}")

    # ── TC_SpGEMM binary ───────────────────────────────────────────────────────
    tc_spgemm_bin = None
    if not args.no_tc_spgemm:
        if args.tc_spgemm_bin:
            tc_spgemm_bin = Path(args.tc_spgemm_bin)
            if not tc_spgemm_bin.exists():
                sys.exit(f"bench_tc_spgemm binary not found: {tc_spgemm_bin}")

    print(f"Work dir       : {work_dir}")
    print(f"CSV            : {csv_path}")
    print(f"TileSpGEMM bin : {tilespgemm_bin}")
    print()

    dispatch_bin = bin_dir / "gpu_dispatch_demo"
    if not args.no_compile:
        print("Compiling:")
        dispatch_bin = compile_gpu_dispatch(args.cuda_home, args.arch, bin_dir)
        if not args.no_taco_gpu and not args.taco_gpu_bin:
            try:
                taco_gpu_bin = compile_taco_gpu(args.cuda_home, args.arch, bin_dir)
            except RuntimeError:
                print("  bench_taco_gpu compilation failed — TACO GPU will be skipped")
                taco_gpu_bin = None
        if not args.no_tc_spgemm and not args.tc_spgemm_bin:
            try:
                tc_spgemm_bin = compile_tc_spgemm(args.cuda_home, args.arch, bin_dir)
            except RuntimeError:
                print("  bench_tc_spgemm compilation failed — TC_SpGEMM will be skipped")
                tc_spgemm_bin = None
        print()
    else:
        if not dispatch_bin.exists():
            sys.exit(
                f"Binary not found: {dispatch_bin}\n"
                f"Remove --no-compile or build first."
            )
        if not args.no_taco_gpu and not args.taco_gpu_bin:
            candidate = bin_dir / "bench_taco_gpu"
            taco_gpu_bin = candidate if candidate.exists() else None
            if taco_gpu_bin is None:
                print("bench_taco_gpu not found in bin-dir; TACO GPU will be skipped")
        if not args.no_tc_spgemm and not args.tc_spgemm_bin:
            candidate = bin_dir / "bench_tc_spgemm"
            tc_spgemm_bin = candidate if candidate.exists() else None
            if tc_spgemm_bin is None:
                print("bench_tc_spgemm not found in bin-dir; TC_SpGEMM will be skipped")

    configs = [
        {
            "M": M, "K": K, "N": N,
            "n_blocks": n_blocks,
            "block_h_min": h_min, "block_h_max": h_max,
            "block_w_min": w_min, "block_w_max": w_max,
            "block_density": bd,
        }
        for (M, K, N), n_blocks, (h_min, h_max, w_min, w_max), bd
        in product(args.mnk_list, args.n_blocks_list, args.block_ranges, args.block_density)
    ]

    print(f"TACO GPU bin   : {taco_gpu_bin or '(disabled)'}")
    print(f"TC_SpGEMM bin  : {tc_spgemm_bin or '(disabled)'}")
    print(f"Block densities: {args.block_density}")
    print(f"Configs        : {len(configs)}")
    print(f"Runs/config    : {args.runs}")
    taco_calls      = 1 if taco_gpu_bin else 0
    tc_spgemm_calls = 1 if tc_spgemm_bin else 0
    print(f"Est. calls     : ~{len(configs) * (3 + 1 + taco_calls + tc_spgemm_calls)} binary invocations")
    print()

    write_header = _needs_header(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        _write_run_comment(f_csv, args)
        writer = csv.DictWriter(f_csv, fieldnames=_CSV_FIELDS,
                                extrasaction="ignore", lineterminator="\n")
        if write_header:
            writer.writeheader()

        for i, cfg in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}]")
            benchmark_config(
                cfg=cfg,
                dispatch_bin=dispatch_bin,
                tilespgemm_bin=tilespgemm_bin,
                taco_gpu_bin=taco_gpu_bin,
                tc_spgemm_bin=tc_spgemm_bin,
                runs=args.runs,
                seed=args.seed,
                device=args.device,
                work_dir=work_dir,
                writer=writer,
                f_csv=f_csv,
                block_density=cfg["block_density"],
            )

    print(f"\nDone. Results appended to {csv_path}")


if __name__ == "__main__":
    main()
