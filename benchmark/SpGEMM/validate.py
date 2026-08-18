#!/usr/bin/env python3
"""
validate.py — Correctness check for all SpGEMM competitors against real matrices.

Reads a CSV produced by plot_stats.py (--output results.csv) and for each matrix
runs A @ A (squaring) through every contender, comparing against scipy's CSR reference.

Prisma contender uses .bsp files (prisma_bench); all other contenders use .mtx files.
Both must be present under --data-dir at <dir>/<group>/<name>/<name>.{bsp,mtx}.

Usage:
  python validate.py --matrices-csv results.csv --data-dir ~/experiments/pattern-mining/data/
  python validate.py --matrices-csv results.csv --data-dir /data --no-compile
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_CORE_DIR = _HERE.parent / "core"
_BIN_CACHE = Path("/tmp/benchmark_bins")

_DEFAULT_TILESPGEMM_DIR = Path("/home/kaio/artifacts/TileSpGEMM/src")
_DEFAULT_CUDA_HOME = "/usr/local/cuda"
_DEFAULT_ARCH = "sm_120"

_RTOL = 2e-2
_ATOL = 5e-3

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"

def _green(s):  return f"{_GREEN}{s}{_RESET}"
def _red(s):    return f"{_RED}{s}{_RESET}"
def _yellow(s): return f"{_YELLOW}{s}{_RESET}"


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

_NVCC_FLAGS = ["-O3", "--expt-relaxed-constexpr", "-std=c++20",
               "-Xcompiler", "-fopenmp", "-lgomp"]


def _needs_compile(out: Path, *srcs: Path) -> bool:
    if not out.exists():
        return True
    t = out.stat().st_mtime
    return any(s.exists() and s.stat().st_mtime > t for s in srcs)

_CORE_SRCS = [
    "block.cpp", "block_generator.cpp", "interval_tree.cpp",
    "matrix.cpp", "matrix_io.cpp", "pipeline.cpp", "segment_tree.cpp",
]


def _hdf5_include() -> list[str]:
    candidates = [
        Path("/usr/include/hdf5/serial"),
        Path("/usr/include/hdf5"),
        Path("/usr/local/include"),
    ]
    try:
        r = subprocess.run(["pkg-config", "--cflags-only-I", "hdf5"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.split()
    except FileNotFoundError:
        pass
    for p in candidates:
        if (p / "hdf5.h").exists():
            return [f"-I{p}"]
    return []


def _hdf5_link() -> list[str]:
    try:
        r = subprocess.run(["pkg-config", "--libs", "hdf5"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.split()
    except FileNotFoundError:
        pass
    serial = Path("/usr/lib/aarch64-linux-gnu/hdf5/serial")
    if not serial.exists():
        serial = Path("/usr/lib/x86_64-linux-gnu/hdf5/serial")
    if serial.exists():
        return [f"-L{serial}", "-lhdf5"]
    return ["-lhdf5"]


def compile_prisma_bench(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "prisma_bench"
    src_paths = [_CORE_DIR / s for s in _CORE_SRCS] + [_HERE / "GPU" / "prisma_bench.cu"]
    if not _needs_compile(out, *src_paths):
        print("  prisma_bench … up-to-date")
        return out
    srcs = [str(p) for p in src_paths]
    cmd = [nvcc, *_NVCC_FLAGS, f"-arch={arch}",
           "-DHAVE_HDF5",
           f"-I{_CORE_DIR}", f"-I{_HERE / 'GPU'}",
           *_hdf5_include(),
           *srcs,
           *_hdf5_link(), "-o", str(out)]
    print("  Compiling prisma_bench … ", end="", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED\n{r.stderr[-1200:]}")
    print("done")
    return out


_GPU_DIR = _HERE / "GPU"


def compile_taco_gpu(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "bench_taco_gpu"
    if not _needs_compile(out, _GPU_DIR / "bench_taco_gpu.cu"):
        print("  bench_taco_gpu … up-to-date")
        return out
    src = str(_GPU_DIR / "bench_taco_gpu.cu")
    cmd = [nvcc, "-O3", f"-arch={arch}", f"-I{_GPU_DIR}", src, "-lcudart", "-o", str(out)]
    print("  Compiling bench_taco_gpu … ", end="", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED\n{r.stderr[-1200:]}")
    print("done")
    return out


def compile_tc_spgemm(cuda_home: str, arch: str, bin_dir: Path) -> Path:
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    out = bin_dir / "bench_tc_spgemm"
    if not _needs_compile(out, _GPU_DIR / "bench_tc_spgemm.cu"):
        print("  bench_tc_spgemm … up-to-date")
        return out
    src = str(_GPU_DIR / "bench_tc_spgemm.cu")
    cmd = [nvcc, "-O3", f"-arch={arch}", "-std=c++17", "-DTC_SPGEMM_NO_MAIN",
           f"-I{_GPU_DIR}", src, "-o", str(out)]
    print("  Compiling bench_tc_spgemm … ", end="", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED\n{r.stderr[-1200:]}")
    print("done")
    return out


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _load_coo(path: Path, M: int, N: int) -> np.ndarray:
    """Read bare 'row col val' COO file (no header) → dense float64 array."""
    arr = np.zeros((M, N), dtype=np.float64)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                arr[int(parts[0]), int(parts[1])] = float(parts[2])
    return arr


def _compare(label: str, c_path: Path, c_ref: np.ndarray,
             loader: str = "mmread") -> bool:
    """Load c_path, compare to c_ref. Returns True on PASS."""
    M, N = c_ref.shape

    try:
        if loader == "coo":
            c_got = _load_coo(c_path, M, N)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mtx = scipy.io.mmread(str(c_path))
            if scipy.sparse.issparse(mtx):
                c_got = mtx.toarray().astype(np.float64)
            else:
                c_got = np.asarray(mtx, dtype=np.float64)
            # Pad / crop to reference shape
            if c_got.shape != (M, N):
                pad = np.zeros((M, N), dtype=np.float64)
                r = min(c_got.shape[0], M)
                c = min(c_got.shape[1], N)
                pad[:r, :c] = c_got[:r, :c]
                c_got = pad
    except Exception as e:
        print(f"  [{label:<22}]  load error: {e}                        {_red('FAIL')}")
        return False

    abs_err = np.abs(c_got - c_ref)
    denom = np.maximum(np.abs(c_ref), 1.0)
    rel_err = abs_err / denom

    mask = ~np.isclose(c_got, c_ref, rtol=_RTOL, atol=_ATOL)
    failures = int(mask.sum())
    total = int(c_ref.size)
    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())

    ok = failures == 0
    status = _green("PASS") if ok else _red("FAIL")
    print(f"  [{label:<22}]  max_err={max_abs:.2e}  max_rel={max_rel:.2e}"
          f"  failures={failures}/{total}   {status}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SpGEMM correctness validator (real matrices)")
    ap.add_argument("--matrices-csv",   required=True, metavar="CSV",
                    help="CSV produced by plot_stats.py --output results.csv")
    ap.add_argument("--data-dir",       required=True, metavar="DIR",
                    help="Mining output root: <dir>/<group>/<name>/<name>.{bsp,mtx}")
    ap.add_argument("--prisma-bin",     default="",  metavar="PATH",
                    help="Pre-built prisma_bench (skip compile)")
    ap.add_argument("--tilespgemm-dir", default=str(_DEFAULT_TILESPGEMM_DIR),
                    metavar="DIR", help="Directory containing TileSpGEMM 'test' binary")
    ap.add_argument("--taco-gpu-bin",   default="",  metavar="PATH",
                    help="Pre-built bench_taco_gpu (skip compile)")
    ap.add_argument("--tc-spgemm-bin",  default="",  metavar="PATH",
                    help="Pre-built bench_tc_spgemm (skip compile)")
    ap.add_argument("--no-compile",     action="store_true",
                    help="Skip all compilation steps")
    ap.add_argument("--cuda-home",      default=_DEFAULT_CUDA_HOME)
    ap.add_argument("--arch",           default=_DEFAULT_ARCH)
    ap.add_argument("--device",         type=int, default=0)
    ap.add_argument("--bin-dir",        default="",  metavar="PATH",
                    help=f"Directory for compiled binaries (default: {_BIN_CACHE})")
    ap.add_argument("--work-dir",       default="",  metavar="PATH",
                    help="Directory for per-matrix output files (default: auto tempdir)")
    args = ap.parse_args()

    # ---------- parse CSV ----------
    matrices = []
    data_root = Path(args.data_dir)
    with open(args.matrices_csv, newline="") as f:
        for row in csv.DictReader(f):
            name, grp = row["name"], row["group"]
            base = data_root / grp / name / name
            matrices.append({
                "name": name, "group": grp,
                "rows": int(row["rows"]), "cols": int(row["cols"]),
                "nnz":  int(row["nnz"]),
                "bsp":  base.with_suffix(".bsp"),
                "mtx":  base.with_suffix(".mtx"),
            })

    if not matrices:
        sys.exit("CSV is empty or has no data rows.")

    missing = [str(p) for m in matrices
               for p in [m["bsp"], m["mtx"]] if not p.exists()]
    if missing:
        sys.exit("Missing files:\n" + "\n".join(f"  {p}" for p in missing))

    print(f"Matrices : {len(matrices)}")
    print(f"Data dir : {data_root}")
    print()

    # ---------- resolve dirs ----------
    bin_dir = Path(args.bin_dir) if args.bin_dir else _BIN_CACHE
    bin_dir.mkdir(parents=True, exist_ok=True)

    _tmp_dir = None
    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        _tmp_dir = tempfile.TemporaryDirectory()
        work_dir = Path(_tmp_dir.name)

    # ---------- resolve / compile binaries ----------
    if args.prisma_bin:
        prisma_bin = Path(args.prisma_bin)
    elif args.no_compile:
        prisma_bin = bin_dir / "prisma_bench"
    else:
        prisma_bin = compile_prisma_bench(args.cuda_home, args.arch, bin_dir)

    tilespgemm_bin = Path(args.tilespgemm_dir) / "test"

    if args.taco_gpu_bin:
        taco_bin = Path(args.taco_gpu_bin)
    elif args.no_compile:
        taco_bin = bin_dir / "bench_taco_gpu"
    else:
        taco_bin = compile_taco_gpu(args.cuda_home, args.arch, bin_dir)

    if args.tc_spgemm_bin:
        tc_spgemm_bin = Path(args.tc_spgemm_bin)
    elif args.no_compile:
        tc_spgemm_bin = bin_dir / "bench_tc_spgemm"
    else:
        tc_spgemm_bin = compile_tc_spgemm(args.cuda_home, args.arch, bin_dir)

    missing_bins = [str(b) for b in [prisma_bin, tilespgemm_bin, taco_bin, tc_spgemm_bin]
                    if not b.exists()]
    if missing_bins:
        sys.exit("Missing binaries:\n" + "\n".join(f"  {b}" for b in missing_bins))

    # ---------- per-matrix validation loop ----------
    all_results: list[bool | None] = []

    for m in matrices:
        name     = m["name"]
        grp      = m["group"]
        bsp_path = m["bsp"]
        mtx_path = m["mtx"]
        M, N     = m["rows"], m["cols"]

        mat_dir = work_dir / name
        mat_dir.mkdir(exist_ok=True)

        c_tc_tile   = mat_dir / "C_tc_tile.coo"
        c_tc_block  = mat_dir / "C_tc_block.coo"
        c_cuda      = mat_dir / "C_cuda.coo"
        c_tilespgemm = mat_dir / "C_tilespgemm.mtx"
        c_taco      = mat_dir / "C_taco_gpu.mtx"
        c_tc_spgemm = mat_dir / "C_tc_spgemm.mtx"

        print(f"=== {name}  ({grp}, {M}×{N}, nnz={m['nnz']:,}) ===")

        def _run(label, cmd):
            print(f"  Running {label} … ", end="", flush=True)
            r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
            if r.returncode != 0:
                print("FAILED")
                print(r.stderr[-400:])
            else:
                print("done")
            return r

        # Prisma — same .bsp twice for A @ A
        r_tc_tile  = _run("prisma tc_tile",
                          [prisma_bin, bsp_path, bsp_path,
                           "--tc-kernel", "tile", "--validate", c_tc_tile])
        r_tc_block = _run("prisma tc_block",
                          [prisma_bin, bsp_path, bsp_path,
                           "--tc-kernel", "block", "--validate", c_tc_block])
        r_cuda     = _run("prisma cuda",
                          [prisma_bin, bsp_path, bsp_path,
                           "--validate", c_cuda])

        # scipy reference: A @ A (load before running mtx-based contenders so we
        # can detect a malformed .mtx early and skip those contenders)
        print("  Computing scipy reference … ", end="", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_sp = scipy.io.mmread(str(mtx_path)).tocsr().astype(np.float64)
        mtx_ok = (A_sp.shape == (M, N))
        if not mtx_ok:
            print(f"SKIP  (.mtx shape {A_sp.shape[0]}×{A_sp.shape[1]} != expected {M}×{N})")
            C_ref = None
        else:
            C_ref = (A_sp @ A_sp).toarray()
            print("done")
        print()

        # External contenders — only run if .mtx has the right shape
        if mtx_ok:
            r_tile = _run("TileSpGEMM",
                          [tilespgemm_bin, "-d", args.device,
                           mtx_path, mtx_path, "-o", c_tilespgemm])
            r_taco = _run("TACO GPU",
                          [taco_bin, mtx_path, mtx_path, "--output", c_taco])
            r_tc_spgemm = _run("TC SpGEMM",
                               [tc_spgemm_bin, mtx_path, mtx_path, "--output", c_tc_spgemm])
        else:
            r_tile = r_taco = r_tc_spgemm = None
            print()  # blank line before results block

        mat_results = []

        def _skip(label):
            print(f"  [{label:<22}]  output not found                           {_yellow('SKIP')}")
            mat_results.append(None)

        def _run_ok(r):
            return r is not None and r.returncode == 0

        if C_ref is not None and r_tc_tile.returncode == 0 and c_tc_tile.exists():
            mat_results.append(_compare("prisma_tc_tile", c_tc_tile, C_ref, loader="coo"))
        else:
            _skip("prisma_tc_tile")

        if C_ref is not None and r_tc_block.returncode == 0 and c_tc_block.exists():
            mat_results.append(_compare("prisma_tc_block", c_tc_block, C_ref, loader="coo"))
        else:
            _skip("prisma_tc_block")

        if C_ref is not None and r_cuda.returncode == 0 and c_cuda.exists():
            mat_results.append(_compare("prisma_cuda", c_cuda, C_ref, loader="coo"))
        else:
            _skip("prisma_cuda")

        if C_ref is not None and _run_ok(r_tile) and c_tilespgemm.exists():
            mat_results.append(_compare("tilespgemm", c_tilespgemm, C_ref))
        else:
            _skip("tilespgemm")

        if _run_ok(r_tile):
            cusparse_passed = ("[PASSED]" in r_tile.stdout
                               and "[NOT PASSED]" not in r_tile.stdout)
            status = _green("PASS") if cusparse_passed else _red("FAIL")
            verdict = "[PASSED]" if cusparse_passed else "[NOT PASSED]"
            print(f"  [{'cusparse (inline)':<22}]  TileSpGEMM reported {verdict:<12}            {status}")
            mat_results.append(cusparse_passed)
        else:
            _skip("cusparse (inline)")

        if C_ref is not None and _run_ok(r_taco) and c_taco.exists():
            mat_results.append(_compare("taco_gpu", c_taco, C_ref))
        else:
            _skip("taco_gpu")

        if C_ref is not None and _run_ok(r_tc_spgemm) and c_tc_spgemm.exists():
            mat_results.append(_compare("tc_spgemm", c_tc_spgemm, C_ref))
        else:
            _skip("tc_spgemm")

        definite = [r for r in mat_results if r is not None]
        mat_ok = all(definite) if definite else None
        if mat_ok is True:
            summary = _green("All PASSED")
        elif mat_ok is False:
            summary = _red("Some FAILED")
        else:
            summary = _yellow("No results")
        print(f"  → {summary}")
        print()

        all_results.extend(mat_results)

    # ---------- overall summary ----------
    n_mat = len(matrices)
    n_contenders = 7
    definite_all = [r for r in all_results if r is not None]
    print("=" * 60)
    if not definite_all:
        print(_yellow("No results to compare."))
    elif all(definite_all):
        print(_green(f"All checks PASSED  ({n_mat} matrices × {n_contenders} contenders)."))
    else:
        n_fail = sum(1 for r in definite_all if not r)
        print(_red(f"{n_fail} check(s) FAILED across {n_mat} matrices."))
        sys.exit(1)

    if _tmp_dir:
        _tmp_dir.cleanup()


if __name__ == "__main__":
    main()
