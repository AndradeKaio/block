#!/usr/bin/env python3
"""
suite-sparse/benchmark.py — GPU SpGEMM benchmark on real SuiteSparse matrices.

Reads a CSV of SuiteSparse matrix names, downloads each as MatrixMarket (.mtx)
via ssgetpy, and benchmarks A×A through four GPU contenders:
  tilespgemm           TileSpGEMM test binary
  cusparse_tilespgemm  cuSPARSE timing parsed from TileSpGEMM stdout
  tc_spgemm            bench_tc_spgemm binary
  taco_gpu             bench_taco_gpu binary

The input CSV must have at least a 'name' column (matrix name in SuiteSparse).
Optional columns 'group', 'rows', 'cols', 'nnz' are carried through to output.

Usage:
  python benchmark.py matrices.csv
  python benchmark.py matrices.csv --out results.csv --runs 5
  python benchmark.py matrices.csv --no-compile --taco-gpu-bin /path/to/bin
"""

import argparse
import contextlib
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import ssgetpy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent  # /workspace/suite-sparse
_GPU_DIR = _SCRIPT_DIR.parent / "SpGEMM" / "GPU"  # /workspace/SpGEMM/GPU
_BIN_CACHE = Path("/tmp/benchmark_bins")

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
    "n_tc",
    "n_cuda",
]

_NAN = float("nan")


def _fmt(v: float) -> str:
    return "nan" if v != v else f"{v:.4f}"


def _needs_header(csv_path: Path) -> bool:
    """True if the CSV is absent/empty or its schema doesn't match _CSV_FIELDS."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True
    expected = ",".join(_CSV_FIELDS)
    with open(csv_path, newline="") as f:
        for line in f:
            if not line.startswith("#"):
                return line.rstrip("\r\n") != expected
    return True


def _emit(
    writer,
    f_csv,
    base: dict,
    kernel: str,
    run_id: int,
    symbolic_ms: float,
    compute_ms: float,
    total_ms: float,
) -> None:
    """Write one timing row to the CSV and flush."""
    writer.writerow(
        {
            **base,
            "kernel": kernel,
            "run_id": run_id,
            "symbolic_ms": _fmt(symbolic_ms),
            "compute_ms": _fmt(compute_ms),
            "total_ms": _fmt(total_ms),
        }
    )
    f_csv.flush()


# ---------------------------------------------------------------------------
# Matrix list loading
# ---------------------------------------------------------------------------


def load_matrix_list(csv_path: Path) -> list[dict]:
    """Read input CSV; return list of row dicts, each with at least 'name'."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        rows = list(reader)
    if not rows:
        sys.exit(f"No rows found in {csv_path}")
    if "name" not in rows[0]:
        sys.exit(f"Input CSV must have a 'name' column; got: {list(rows[0].keys())}")
    return rows


# ---------------------------------------------------------------------------
# Matrix download
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet_stderr():
    """Suppress stderr (suppresses ssgetpy's per-file tqdm download bars)."""
    with open(os.devnull, "w") as devnull:
        old, sys.stderr = sys.stderr, devnull
        try:
            yield
        finally:
            sys.stderr = old


_DATA_ROOT = Path("/home/kaio/datasets/suite-sparse")


def _readable(p: Path) -> bool:
    """True if p is a non-empty regular file we can actually open."""
    try:
        return p.is_file() and p.stat().st_size > 0 and os.access(p, os.R_OK)
    except OSError:
        return False


def download_matrix(name: str, group: str = "", timeout: int = 120) -> Path:
    """Return path to <name>.mtx from _DATA_ROOT.

    Raises FileNotFoundError if the matrix is not present in the local dataset.
    """
    mat_dir = _DATA_ROOT / group / name if group else _DATA_ROOT / name
    mtx_path = mat_dir / f"{name}.mtx"
    if _readable(mtx_path):
        return mtx_path
    # ssgetpy may store the matrix under a different group sub-directory;
    # do a quick search within _DATA_ROOT before giving up.
    candidates = [p for p in _DATA_ROOT.rglob(f"{name}.mtx") if _readable(p)]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Matrix '{name}' not found in dataset: {_DATA_ROOT}")


def ensure_real_general(mtx_path: Path, cache_dir: Path) -> Path:
    """Return a path to a real-general version of the matrix.

    If the file is already 'real general', returns mtx_path unchanged.
    Otherwise (pattern, symmetric, complex, …) reads with scipy and writes
    a normalised copy into cache_dir as <name>_real.mtx.
    """
    import scipy.io as sio
    import scipy.sparse as sp

    with open(mtx_path) as f:
        header = f.readline().lower()

    is_real = "real" in header or "integer" in header
    is_general = "general" in header

    if is_real and is_general:
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
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json_block(stdout: str) -> dict:
    """Extract and parse JSON between JSON_BEGIN / JSON_END sentinels."""
    if "JSON_BEGIN" not in stdout or "JSON_END" not in stdout:
        return {}
    s = stdout.index("JSON_BEGIN") + len("JSON_BEGIN")
    e = stdout.index("JSON_END")
    try:
        return json.loads(stdout[s:e].strip())
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Regex patterns for TileSpGEMM / cuSPARSE stdout
# ---------------------------------------------------------------------------

_TILE_TOT_RE = re.compile(r"CUDA TileSpGEMM run \d+ time is\s+([\d.]+)\s+ms")
_TILE_SYM_RE = re.compile(r"CUDA TileSpGEMM run \d+ symbolic is\s+([\d.]+)\s+ms")
_TILE_CMP_RE = re.compile(r"CUDA TileSpGEMM run \d+ compute is\s+([\d.]+)\s+ms")
_CSPARSE_TOT_RE = re.compile(r"CUDA cuSPARSE SpGEMM run \d+ time is\s+([\d.]+)\s+ms")
_CSPARSE_SYM_RE = re.compile(
    r"CUDA cuSPARSE SpGEMM run \d+ symbolic is\s+([\d.]+)\s+ms"
)
_CSPARSE_CMP_RE = re.compile(r"CUDA cuSPARSE SpGEMM run \d+ compute is\s+([\d.]+)\s+ms")


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------


def run_tilespgemm_and_cusparse(
    binary: Path,
    mtx: Path,
    device: int,
    runs: int,
    timeout: int = 300,
) -> tuple[
    list[float], list[float], list[float], list[float], list[float], list[float]
]:
    """Invoke TileSpGEMM test binary with A=B=mtx (A×A).

    Returns (tile_sym, tile_cmp, tile_tot, cs_sym, cs_cmp, cs_tot), each a
    list of runs+1 floats (index 0 = warmup, 1..runs = timed).
    Raises RuntimeError on non-zero exit or parse failure.
    """
    cli = [str(binary), "-d", str(device), "--runs", str(runs), str(mtx), str(mtx)]
    r = subprocess.run(cli, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"TileSpGEMM exited {r.returncode}:\n{r.stderr[-800:]}")
    tile_sym = [float(x) for x in _TILE_SYM_RE.findall(r.stdout)]
    tile_cmp = [float(x) for x in _TILE_CMP_RE.findall(r.stdout)]
    tile_tot = [float(x) for x in _TILE_TOT_RE.findall(r.stdout)]
    cs_sym = [float(x) for x in _CSPARSE_SYM_RE.findall(r.stdout)]
    cs_cmp = [float(x) for x in _CSPARSE_CMP_RE.findall(r.stdout)]
    cs_tot = [float(x) for x in _CSPARSE_TOT_RE.findall(r.stdout)]
    if not tile_tot:
        raise RuntimeError(
            "TileSpGEMM: could not parse any TileSpGEMM timings from stdout"
        )
    if not cs_tot:
        raise RuntimeError(
            "TileSpGEMM: could not parse any cuSPARSE timings from stdout"
        )
    return tile_sym, tile_cmp, tile_tot, cs_sym, cs_cmp, cs_tot


def run_tc_spgemm(
    binary: Path, mtx: Path, runs: int, timeout: int = 300
) -> list[tuple[float, float, float]]:
    """Invoke bench_tc_spgemm with A=B=mtx (A×A).

    Returns list of (symbolic_ms, compute_ms, total_ms) per run (index 0 = warmup).
    Raises RuntimeError on non-zero exit or parse failure.
    """
    cli = [str(binary), str(mtx), str(mtx), "--runs", str(runs)]
    r = subprocess.run(cli, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"bench_tc_spgemm exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    syms = d.get("tc_spgemm_symbolic_ms")
    cmps = d.get("tc_spgemm_compute_ms")
    tots = d.get("tc_spgemm_ms")
    if not syms or not cmps or not tots:
        raise RuntimeError(
            "bench_tc_spgemm: could not parse phase timings from JSON output"
        )
    return [(float(s), float(c), float(t)) for s, c, t in zip(syms, cmps, tots)]


def run_taco_gpu(
    binary: Path, mtx: Path, runs: int, timeout: int = 300
) -> list[tuple[float, float, float]]:
    """Invoke bench_taco_gpu with A=B=mtx (A×A).

    Returns list of (symbolic_ms, compute_ms, total_ms) per run.
    Raises RuntimeError on non-zero exit or parse failure.
    """
    cli = [str(binary), str(mtx), str(mtx), "--runs", str(runs)]
    r = subprocess.run(cli, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        stderr = r.stderr
        if "out of memory" in stderr or "GPUassert" in stderr:
            raise RuntimeError("OOM")
        raise RuntimeError(f"bench_taco_gpu exited {r.returncode}:\n{stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    sym_times = d.get("taco_symbolic_ms")
    comp_times = d.get("taco_compute_ms")
    if sym_times and comp_times:
        return [
            (float(s), float(c), float(s) + float(c))
            for s, c in zip(sym_times, comp_times)
        ]
    # Fallback for old binary without split timing
    times = d.get("taco_ms")
    if not times:
        raise RuntimeError("bench_taco_gpu: could not parse timing from JSON output")
    return [(0.0, float(t), float(t)) for t in times]


# ---------------------------------------------------------------------------
# PRISMA runner
# ---------------------------------------------------------------------------


def run_prisma(
    binary: Path,
    mtx: Path,
    tc_kernel: str,
    runs: int,
    timeout: int = 300,
) -> tuple[dict, str]:
    """Invoke prisma_bench with A=B=<name>.bsp (A×A squaring).

    The .bsp file is expected alongside the .mtx file with the same stem.
    Raises FileNotFoundError if the .bsp file does not exist.
    Raises RuntimeError on non-zero exit or missing JSON block.
    Returns (parsed_json_dict, non_json_stdout).
    """
    bsp = mtx.with_suffix(".bsp")
    if not bsp.exists():
        raise FileNotFoundError(f"BSP not found: {bsp}")
    cmd = [str(binary), str(bsp), str(bsp), "--runs", str(runs)]
    if tc_kernel:
        cmd += ["--tc-kernel", tc_kernel]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"prisma_bench exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError("prisma_bench: no JSON block in stdout")
    # Extract lines outside the JSON_BEGIN/JSON_END sentinels for display.
    extra_lines = []
    in_json = False
    for line in r.stdout.splitlines():
        if line.strip() == "JSON_BEGIN":
            in_json = True
            continue
        if line.strip() == "JSON_END":
            in_json = False
            continue
        if not in_json:
            extra_lines.append(line)
    return d, "\n".join(extra_lines)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


_CORE_SRCS = [
    "block.cpp",
    "block_generator.cpp",
    "interval_tree.cpp",
    "matrix.cpp",
    "matrix_io.cpp",
    "pipeline.cpp",
    "segment_tree.cpp",
]

_NVCC_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "-std=c++20",
    "-Xcompiler",
    "-fopenmp",
    "-lgomp",
]


def _needs_compile(out: Path, *srcs: Path) -> bool:
    if not out.exists():
        return True
    t = out.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in srcs)


def _hdf5_include() -> list[str]:
    """Return -I flags for the HDF5 headers, searching common install paths."""
    candidates = [
        Path("/usr/include/hdf5/serial"),
        Path("/usr/include/hdf5"),
        Path("/usr/local/include"),
    ]
    # also ask pkg-config if available
    try:
        r = subprocess.run(
            ["pkg-config", "--cflags-only-I", "hdf5"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return r.stdout.split()
    except FileNotFoundError:
        pass
    for p in candidates:
        if (p / "hdf5.h").exists():
            return [f"-I{p}"]
    return []


def _hdf5_link() -> list[str]:
    """Return -L / -l flags for linking HDF5."""
    try:
        r = subprocess.run(
            ["pkg-config", "--libs", "hdf5"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return r.stdout.split()
    except FileNotFoundError:
        pass
    # serial HDF5 on Debian/Ubuntu lives in a subdirectory
    serial = Path("/usr/lib/aarch64-linux-gnu/hdf5/serial")
    if not serial.exists():
        serial = Path("/usr/lib/x86_64-linux-gnu/hdf5/serial")
    if serial.exists():
        return [f"-L{serial}", "-lhdf5"]
    return ["-lhdf5"]


def compile_prisma_bench(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    """Compile GPU/prisma_bench.cu → bin_dir/prisma_bench."""
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "prisma_bench"
    core_dir = _GPU_DIR.parent.parent / "core"
    src_paths = [core_dir / s for s in _CORE_SRCS] + [_GPU_DIR / "prisma_bench.cu"]
    if not _needs_compile(out, *src_paths):
        print("  prisma_bench … up-to-date")
        return out
    srcs = [str(p) for p in src_paths]
    cmd = [
        nvcc,
        *_NVCC_FLAGS,
        f"-arch={arch}",
        "-DHAVE_HDF5",
        f"-I{core_dir}",
        f"-I{_GPU_DIR}",
        *_hdf5_include(),
        *srcs,
        *_hdf5_link(),
        "-o",
        str(out),
    ]
    print("  Compiling prisma_bench … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-2000:]}")
        raise RuntimeError("prisma_bench compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def compile_tc_spgemm(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    """Compile GPU/bench_tc_spgemm.cu → bin_dir/bench_tc_spgemm."""
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "bench_tc_spgemm"
    if not _needs_compile(out, _GPU_DIR / "bench_tc_spgemm.cu"):
        print("  bench_tc_spgemm … up-to-date")
        return out
    src = str(_GPU_DIR / "bench_tc_spgemm.cu")
    cmd = [
        nvcc,
        "-O3",
        f"-arch={arch}",
        "-std=c++17",
        f"-I{_GPU_DIR}",
        src,
        "-lcudart",
        "-o",
        str(out),
    ]
    print("  Compiling bench_tc_spgemm … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-2000:]}")
        raise RuntimeError("bench_tc_spgemm compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


def compile_taco_gpu(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    """Compile GPU/bench_taco_gpu.cu → bin_dir/bench_taco_gpu."""
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "bench_taco_gpu"
    if not _needs_compile(out, _GPU_DIR / "bench_taco_gpu.cu"):
        print("  bench_taco_gpu … up-to-date")
        return out
    src = str(_GPU_DIR / "bench_taco_gpu.cu")
    cmd = [
        nvcc,
        "-O3",
        f"-arch={arch}",
        f"-I{_GPU_DIR}",
        src,
        "-lcudart",
        "-o",
        str(out),
    ]
    print("  Compiling bench_taco_gpu … ", end="", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAILED ({time.time() - t0:.1f}s)\n{r.stderr[-2000:]}")
        raise RuntimeError("bench_taco_gpu compilation failed")
    print(f"ok ({time.time() - t0:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Per-matrix benchmark
# ---------------------------------------------------------------------------


def benchmark_matrix(
    matrix_row: dict,
    mtx: Path,
    tilespgemm_bin: Path,
    taco_gpu_bin,
    tc_spgemm_bin,
    prisma_bin,
    prisma_tc_kernel: str,
    runs: int,
    device: int,
    timeout: int,
    writer,
    f_csv,
    orig_mtx: Path = None,
    prisma_timing: bool = False,
) -> None:
    """Run all enabled contenders on a single matrix (A×A) and emit CSV rows."""
    base = {
        "matrix_name": matrix_row["name"],
        "group": matrix_row.get("group", ""),
        "rows": matrix_row.get("rows", ""),
        "cols": matrix_row.get("cols", ""),
        "nnz": matrix_row.get("nnz", ""),
    }

    # ── TileSpGEMM + cuSPARSE ────────────────────────────────────────────────
    print("  [tilespgemm+cusparse] ", end="", flush=True)
    try:
        tile_sym, tile_cmp, tile_tot, cs_sym, cs_cmp, cs_tot = (
            run_tilespgemm_and_cusparse(tilespgemm_bin, mtx, device, runs, timeout)
        )
        for run_id, (s, c, t) in enumerate(zip(tile_sym, tile_cmp, tile_tot)):
            _emit(writer, f_csv, base, "tilespgemm", run_id, s, c, t)
        for run_id, (s, c, t) in enumerate(zip(cs_sym, cs_cmp, cs_tot)):
            _emit(writer, f_csv, base, "cusparse_tilespgemm", run_id, s, c, t)
        timed = tile_tot[1:] or tile_tot
        print(
            f"avg {sum(timed) / len(timed):.3f} ms  ({len(tile_tot)} runs incl. warmup)"
        )
        if cs_tot:
            cs_timed = cs_tot[1:] or cs_tot
            print(
                f"  [cusparse]            "
                f"avg {sum(cs_timed) / len(cs_timed):.3f} ms  ({len(cs_tot)} runs incl. warmup)"
            )
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"FAILED ({e})")
        for run_id in range(runs + 1):
            _emit(writer, f_csv, base, "tilespgemm", run_id, _NAN, _NAN, _NAN)
            _emit(writer, f_csv, base, "cusparse_tilespgemm", run_id, _NAN, _NAN, _NAN)

    # ── TACO GPU ─────────────────────────────────────────────────────────────
    if taco_gpu_bin:
        print("  [taco_gpu]            ", end="", flush=True)
        try:
            triples = run_taco_gpu(taco_gpu_bin, mtx, runs, timeout)
            for run_id, (s, c, t) in enumerate(triples):
                _emit(writer, f_csv, base, "taco_gpu", run_id, s, c, t)
            timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
            print(
                f"avg {sum(timed) / len(timed):.3f} ms  ({len(triples)} runs incl. warmup)"
            )
        except subprocess.TimeoutExpired:
            print("FAILED (timeout)")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, "taco_gpu", run_id, _NAN, _NAN, _NAN)
        except RuntimeError as e:
            reason = str(e)
            print(f"FAILED ({reason})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, "taco_gpu", run_id, _NAN, _NAN, _NAN)

    # ── TC_SpGEMM ─────────────────────────────────────────────────────────────
    if tc_spgemm_bin:
        print("  [tc_spgemm]           ", end="", flush=True)
        try:
            triples = run_tc_spgemm(tc_spgemm_bin, mtx, runs, timeout)
            for run_id, (s, c, t) in enumerate(triples):
                _emit(writer, f_csv, base, "tc_spgemm", run_id, s, c, t)
            timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
            print(
                f"avg {sum(timed) / len(timed):.3f} ms  ({len(triples)} runs incl. warmup)"
            )
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"FAILED ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, "tc_spgemm", run_id, _NAN, _NAN, _NAN)

    # ── PRISMA ────────────────────────────────────────────────────────────────
    if prisma_bin:
        bsp_mtx = orig_mtx or mtx

        def _run_one_prisma(tc_kernel: str):
            label = f"prisma_tc_{tc_kernel}" if tc_kernel else "prisma_cuda"
            print(f"  [{label}]", " " * max(0, 22 - len(label)), end="", flush=True)
            try:
                d, extra_stdout = run_prisma(
                    prisma_bin, bsp_mtx, tc_kernel, runs, timeout
                )
                plan_ms = d.get("plan_ms", [])
                tc_ms = d.get("tc_ms", [])
                cuda_ms = d.get("cuda_ms", [])
                n_pairs = d.get("n_pairs", "")
                n_groups = d.get("n_groups", "")
                n_tc = d.get("n_tc", "")
                n_cuda = d.get("n_cuda", "")
                kernel = d.get("kernel", label)
                # One-time pipeline cost amortised over timed runs.
                pipe_total = d.get("pipe_total_ms", 0.0)
                n_timed = max(len(plan_ms) - 1, 1)
                amortised_sym = pipe_total / n_timed
                for run_id, (p, tc, cu) in enumerate(zip(plan_ms, tc_ms, cuda_ms)):
                    # run_id 0 is warmup: report only compute; timed runs carry
                    # the amortised pipeline cost as symbolic_ms.
                    sym = 0.0 if run_id == 0 else amortised_sym
                    compute = tc + cu
                    writer.writerow(
                        {
                            **base,
                            "kernel": kernel,
                            "run_id": run_id,
                            "symbolic_ms": _fmt(sym),
                            "compute_ms": _fmt(compute),
                            "total_ms": _fmt(sym + compute),
                            "n_pairs": n_pairs,
                            "n_groups": n_groups,
                            "n_tc": n_tc,
                            "n_cuda": n_cuda,
                        }
                    )
                f_csv.flush()
                timed = [
                    amortised_sym + tc + cu for tc, cu in zip(tc_ms[1:], cuda_ms[1:])
                ]
                if timed:
                    print(
                        f"avg {sum(timed) / len(timed):.3f} ms  ({len(plan_ms)} runs incl. warmup)"
                    )
                else:
                    print("ok")
                if prisma_timing and extra_stdout.strip():
                    for line in extra_stdout.splitlines():
                        print(f"    {line}")
            except FileNotFoundError:
                raise  # BSP missing is a hard error — propagate up
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"FAILED ({e})")
                for run_id in range(runs + 1):
                    _emit(writer, f_csv, base, label, run_id, _NAN, _NAN, _NAN)

        _run_one_prisma(prisma_tc_kernel)  # e.g. tc_tile
        _run_one_prisma("")  # cuda (no TC)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="GPU SpGEMM benchmark on real SuiteSparse matrices (A×A)",
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
        "--device", type=int, default=0, help="CUDA device index (default: 0)"
    )
    g.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-binary and per-download timeout in seconds (default: 300)",
    )

    g = p.add_argument_group("Paths")
    g.add_argument(
        "--out",
        default="gpu_results.csv",
        help="output CSV, append mode (default: gpu_results.csv)",
    )
    g.add_argument(
        "--work-dir",
        default="",
        help="directory for downloads and binaries (default: auto tempdir)",
    )
    g.add_argument(
        "--tilespgemm-dir",
        default="/home/kaio/artifacts/TileSpGEMM/src",
        metavar="PATH",
        help="directory containing the TileSpGEMM 'test' binary",
    )
    g.add_argument(
        "--taco-gpu-bin",
        default="",
        metavar="PATH",
        help="pre-built bench_taco_gpu binary (skips compilation)",
    )
    g.add_argument(
        "--tc-spgemm-bin",
        default="",
        metavar="PATH",
        help="pre-built bench_tc_spgemm binary (skips compilation)",
    )
    g.add_argument(
        "--prisma-bin",
        default="",
        metavar="PATH",
        help="pre-built prisma_bench binary (skips compilation)",
    )
    g.add_argument(
        "--prisma-tc-kernel",
        default="tile",
        choices=["tile", "block", ""],
        metavar='tile|block|""',
        help="TC kernel variant for PRISMA: tile, block, or empty for CUDA-only (default: tile)",
    )

    g = p.add_argument_group("Build")
    g.add_argument(
        "--no-compile",
        action="store_true",
        help=f"skip compilation; binaries must already exist in work-dir (default: {_BIN_CACHE})",
    )
    g.add_argument("--no-taco-gpu", action="store_true", help="skip TACO GPU contender")
    g.add_argument(
        "--no-tc-spgemm", action="store_true", help="skip TC_SpGEMM contender"
    )
    g.add_argument("--no-prisma", action="store_true", help="skip PRISMA contender")
    g.add_argument(
        "--prisma-timing",
        action="store_true",
        help="print per-phase symbolic timing breakdown for each PRISMA run",
    )
    g.add_argument("--cuda-home", default="/usr/local/cuda")
    g.add_argument("--arch", default="sm_120")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    work_dir = Path(args.work_dir) if args.work_dir else _BIN_CACHE
    work_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    tilespgemm_bin = Path(args.tilespgemm_dir) / "test"
    if not tilespgemm_bin.exists():
        sys.exit(
            f"TileSpGEMM binary not found: {tilespgemm_bin}\n"
            f"Build it first with 'make' inside {args.tilespgemm_dir}"
        )

    taco_gpu_bin: Path | None = None
    if not args.no_taco_gpu:
        if args.taco_gpu_bin:
            taco_gpu_bin = Path(args.taco_gpu_bin)
            if not taco_gpu_bin.exists():
                sys.exit(f"bench_taco_gpu not found: {taco_gpu_bin}")

    tc_spgemm_bin: Path | None = None
    if not args.no_tc_spgemm:
        if args.tc_spgemm_bin:
            tc_spgemm_bin = Path(args.tc_spgemm_bin)
            if not tc_spgemm_bin.exists():
                sys.exit(f"bench_tc_spgemm not found: {tc_spgemm_bin}")

    prisma_bin: Path | None = None
    if not args.no_prisma:
        if args.prisma_bin:
            prisma_bin = Path(args.prisma_bin)
            if not prisma_bin.exists():
                sys.exit(f"prisma_bench not found: {prisma_bin}")

    if not args.no_compile:
        print("Compiling:")
        if not args.no_taco_gpu and not args.taco_gpu_bin:
            try:
                taco_gpu_bin = compile_taco_gpu(args.cuda_home, args.arch, work_dir)
            except RuntimeError:
                print("  bench_taco_gpu compilation failed — TACO GPU will be skipped")
        if not args.no_tc_spgemm and not args.tc_spgemm_bin:
            try:
                tc_spgemm_bin = compile_tc_spgemm(args.cuda_home, args.arch, work_dir)
            except RuntimeError:
                print(
                    "  bench_tc_spgemm compilation failed — TC_SpGEMM will be skipped"
                )
        if not args.no_prisma and not args.prisma_bin:
            try:
                prisma_bin = compile_prisma_bench(args.cuda_home, args.arch, work_dir)
            except RuntimeError:
                print("  prisma_bench compilation failed — PRISMA will be skipped")
        print()
    else:
        if not args.no_taco_gpu and not args.taco_gpu_bin:
            candidate = work_dir / "bench_taco_gpu"
            taco_gpu_bin = candidate if candidate.exists() else None
            if not taco_gpu_bin:
                print(
                    f"bench_taco_gpu not found in {work_dir} — TACO GPU will be skipped"
                )
        if not args.no_tc_spgemm and not args.tc_spgemm_bin:
            candidate = work_dir / "bench_tc_spgemm"
            tc_spgemm_bin = candidate if candidate.exists() else None
            if not tc_spgemm_bin:
                print(
                    f"bench_tc_spgemm not found in {work_dir} — TC_SpGEMM will be skipped"
                )
        if not args.no_prisma and not args.prisma_bin:
            candidate = work_dir / "prisma_bench"
            prisma_bin = candidate if candidate.exists() else None
            if not prisma_bin:
                print(f"prisma_bench not found in {work_dir} — PRISMA will be skipped")

    matrices = load_matrix_list(Path(args.csv))

    print(f"CSV            : {csv_path}")
    print(f"Work dir       : {work_dir}")
    print(f"TileSpGEMM bin : {tilespgemm_bin}")
    print(f"TACO GPU bin   : {taco_gpu_bin or '(disabled)'}")
    print(f"TC_SpGEMM bin  : {tc_spgemm_bin or '(disabled)'}")
    print(f"PRISMA bin     : {prisma_bin or '(disabled)'}")
    print(f"Matrices       : {len(matrices)}")
    print(f"Runs/matrix    : {args.runs}")
    print(f"Timeout (s)    : {args.timeout}")
    print()

    write_header = _needs_header(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f_csv.write(
            f"# {ts}  input={args.csv}  runs={args.runs}  device={args.device}\n"
        )
        writer = csv.DictWriter(
            f_csv, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()

        for i, row in enumerate(matrices, 1):
            name = row["name"]
            print(f"\n[{i}/{len(matrices)}] {name}")

            try:
                orig_mtx = download_matrix(
                    name, group=row.get("group", ""), timeout=args.timeout
                )
                mtx = ensure_real_general(orig_mtx, Path("/tmp/mtx_cache"))
                print(f"  Matrix → {mtx}")
            except Exception as e:
                print(f"  DOWNLOAD FAILED: {e} — skipping")
                continue

            benchmark_matrix(
                matrix_row=row,
                mtx=mtx,
                orig_mtx=orig_mtx,
                tilespgemm_bin=tilespgemm_bin,
                taco_gpu_bin=taco_gpu_bin,
                tc_spgemm_bin=tc_spgemm_bin,
                prisma_bin=prisma_bin,
                prisma_tc_kernel=args.prisma_tc_kernel,
                runs=args.runs,
                device=args.device,
                timeout=args.timeout,
                writer=writer,
                f_csv=f_csv,
                prisma_timing=args.prisma_timing,
            )

    print(f"\nDone. Results appended to {csv_path}")


if __name__ == "__main__":
    main()
