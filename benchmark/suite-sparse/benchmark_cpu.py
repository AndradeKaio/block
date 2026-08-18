#!/usr/bin/env python3
"""
suite-sparse/benchmark_cpu.py — CPU SpGEMM benchmark on real SuiteSparse matrices.

Reads a CSV of SuiteSparse matrix names, downloads each as MatrixMarket (.mtx)
via ssgetpy, and benchmarks A×A through CPU contenders:
  prisma_cpu     PRISMA block-sparse CPU executor
  taco_cpu       TACO-generated SpGEMM kernel (unoptimized)
  taco_cpu_opt   TACO-generated SpGEMM kernel (parallelized + reordered)

Usage:
  python benchmark_cpu.py matrices.csv
  python benchmark_cpu.py matrices.csv --out results_cpu.csv --runs 5
  python benchmark_cpu.py matrices.csv --no-compile --prisma-bin /path/to/bin
"""

import argparse
import contextlib
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_CPU_DIR    = _SCRIPT_DIR.parent / "SpGEMM" / "CPU"
_CORE_DIR   = _SCRIPT_DIR.parent / "core"

# ---------------------------------------------------------------------------
# CSV schema  (identical to GPU benchmark for interoperability)
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
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True
    expected = ",".join(_CSV_FIELDS)
    with open(csv_path, newline="") as f:
        for line in f:
            if not line.startswith("#"):
                return line.rstrip("\r\n") != expected
    return True


def _emit(writer, f_csv, base: dict, kernel: str, run_id: int,
          symbolic_ms: float, compute_ms: float, total_ms: float) -> None:
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
# Matrix list
# ---------------------------------------------------------------------------


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
# Matrix download (identical to GPU script)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet_stderr():
    with open(os.devnull, "w") as devnull:
        old, sys.stderr = sys.stderr, devnull
        try:
            yield
        finally:
            sys.stderr = old


_DATA_ROOT = Path("/home/kaio/datasets/suite-sparse")


def _readable(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0 and os.access(p, os.R_OK)
    except OSError:
        return False


def download_matrix(name: str, group: str = "", timeout: int = 120) -> Path:
    mat_dir  = _DATA_ROOT / group / name if group else _DATA_ROOT / name
    mtx_path = mat_dir / f"{name}.mtx"
    if _readable(mtx_path):
        return mtx_path
    candidates = [p for p in _DATA_ROOT.rglob(f"{name}.mtx") if _readable(p)]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Matrix '{name}' not found in dataset: {_DATA_ROOT}")


def ensure_real_general(mtx_path: Path, cache_dir: Path) -> Path:
    import scipy.io as sio
    import scipy.sparse as sp

    with open(mtx_path) as f:
        header = f.readline().lower()

    if "real" in header or "integer" in header:
        if "general" in header:
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
    if "JSON_BEGIN" not in stdout or "JSON_END" not in stdout:
        return {}
    s = stdout.index("JSON_BEGIN") + len("JSON_BEGIN")
    e = stdout.index("JSON_END")
    try:
        return json.loads(stdout[s:e].strip())
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Runner: TACO CPU
# ---------------------------------------------------------------------------

_TACO_ASM_RE  = re.compile(r"run_(\d+)_assemble_ns=(\d+)")
_TACO_COMP_RE = re.compile(r"run_(\d+)_compute_ns=(\d+)")


def run_taco_cpu(
    binary: Path,
    mtx: Path,
    runs: int,
    timeout: int = 300,
) -> list[tuple[float, float, float]]:
    """Returns [(symbolic_ms, compute_ms, total_ms)] for run_id 0..runs."""
    cmd = [str(binary), str(mtx), str(mtx), str(runs + 1)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"bench_taco_cpu exited {r.returncode}:\n{r.stderr[-800:]}")

    asm_ns:  dict[int, int] = {}
    comp_ns: dict[int, int] = {}
    for line in r.stdout.splitlines():
        m = _TACO_ASM_RE.match(line)
        if m:
            asm_ns[int(m.group(1))] = int(m.group(2))
        m = _TACO_COMP_RE.match(line)
        if m:
            comp_ns[int(m.group(1))] = int(m.group(2))

    if not asm_ns or not comp_ns:
        raise RuntimeError("bench_taco_cpu: could not parse timing from stdout")

    result = []
    for run_id in range(runs + 1):
        sym  = asm_ns.get(run_id, 0)  / 1e6
        comp = comp_ns.get(run_id, 0) / 1e6
        result.append((sym, comp, sym + comp))
    return result


# ---------------------------------------------------------------------------
# Runner: PRISMA CPU
# ---------------------------------------------------------------------------


def run_prisma_cpu(
    binary: Path,
    bsp: Path,
    runs: int,
    timeout: int = 300,
    specialized: bool = False,
) -> list[tuple[float, float, float]]:
    """Returns [(symbolic_ms, compute_ms, total_ms)] for run_id 0..runs."""
    if not bsp.exists():
        raise FileNotFoundError(f"BSP not found: {bsp}")
    cmd = [str(binary), str(bsp), str(bsp), "--runs", str(runs)]
    if specialized:
        cmd.append("--specialized-kernels")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"prisma_cpu_bench exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    compute_ms = d.get("compute_ms")
    if not compute_ms:
        raise RuntimeError("prisma_cpu_bench: could not parse compute_ms from JSON")
    pipe_total = float(d.get("pipe_total_ms", 0.0))
    n_timed = max(len(compute_ms) - 1, 1)
    amortised_sym = pipe_total / n_timed
    result = []
    for run_id, c in enumerate(compute_ms):
        sym = 0.0 if run_id == 0 else amortised_sym
        result.append((sym, float(c), sym + float(c)))
    return result


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


def _find_hdf5() -> tuple[str, str]:
    import platform, subprocess as _sp
    for candidate in [
        "/usr/include/hdf5/serial",
        "/usr/local/include",
        "/usr/include",
    ]:
        if Path(candidate + "/hdf5.h").exists() or Path(candidate).joinpath("hdf5.h").exists():
            inc = candidate
            break
    else:
        inc = "/usr/include/hdf5/serial"

    try:
        machine = platform.machine()
        arch_triplets = {
            "x86_64":  "x86_64-linux-gnu",
            "aarch64": "aarch64-linux-gnu",
            "arm":     "arm-linux-gnueabihf",
        }
        triplet = arch_triplets.get(machine, machine + "-linux-gnu")
    except Exception:
        triplet = "x86_64-linux-gnu"

    for candidate in [
        f"/usr/lib/{triplet}/hdf5/serial",
        f"/usr/lib/hdf5/serial",
        f"/usr/local/lib",
        f"/usr/lib/{triplet}",
    ]:
        if Path(candidate).joinpath("libhdf5.so").exists() or \
           Path(candidate).joinpath("libhdf5.a").exists():
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
        "gcc", "-O3", f'-DTACO_KERNEL_H="{kernel_h}"',
        f"-DNUM_THREADS={n_threads}",
        "-fopenmp",
        f"-I{_CPU_DIR}",
        src,
        "-lm", "-o", str(out),
    ]
    print(f"  {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcc failed:\n{r.stderr}")
    return out


def compile_prisma_cpu(out: Path, blas: bool) -> Path:
    """Compile prisma_cpu_bench.cpp with core sources."""
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    src  = str(_CPU_DIR / "prisma_cpu_bench.cpp")
    cmd  = [
        "g++", "-O3", "-std=c++20", "-fopenmp", "-march=native",
        "-DHAVE_HDF5",
        f"-I{_CORE_DIR}", f"-I{_CPU_DIR}", f"-I{_HDF5_INC}",
        *srcs, src,
        str(Path(_HDF5_LIB) / "libhdf5.so"),
        "-o", str(out),
    ]
    if blas:
        cmd[-1:-1] = ["-DHAVE_BLAS", "-lblas"]
    print(f"  {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"g++ failed:\n{r.stderr}")
    return out


# ---------------------------------------------------------------------------
# Per-matrix benchmark
# ---------------------------------------------------------------------------


def benchmark_matrix(
    matrix_row: dict,
    mtx: Path,
    orig_mtx: Path,
    taco_bin: Path | None,
    taco_opt_bin: Path | None,
    prisma_bin: Path | None,
    runs: int,
    timeout: int,
    threads: int,
    specialized: bool,
    writer,
    f_csv,
) -> None:
    name = matrix_row["name"]
    base = {
        "matrix_name": name,
        "group":       matrix_row.get("group", ""),
        "rows":        matrix_row.get("rows",  ""),
        "cols":        matrix_row.get("cols",  ""),
        "nnz":         matrix_row.get("nnz",   ""),
        "n_pairs":     "",
        "n_groups":    "",
        "n_tc":        "",
        "n_cuda":      "",
    }
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}

    def run_with_env(runner, *args, **kwargs):
        old = os.environ.copy()
        os.environ.update(env)
        try:
            return runner(*args, **kwargs)
        finally:
            os.environ.clear()
            os.environ.update(old)

    # ── TACO CPU ──────────────────────────────────────────────────────────────
    if taco_bin:
        print("  [taco_cpu]            ", end="", flush=True)
        try:
            triples = run_with_env(run_taco_cpu, taco_bin, mtx, runs, timeout)
            for run_id, (s, c, t) in enumerate(triples):
                _emit(writer, f_csv, base, "taco_cpu", run_id, s, c, t)
            timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
            print(f"avg {sum(timed)/len(timed):.3f} ms  ({len(triples)} runs incl. warmup)")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"FAILED ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, "taco_cpu", run_id, _NAN, _NAN, _NAN)

    # ── TACO CPU opt ──────────────────────────────────────────────────────────
    if taco_opt_bin:
        print("  [taco_cpu_opt]        ", end="", flush=True)
        try:
            triples = run_with_env(run_taco_cpu, taco_opt_bin, mtx, runs, timeout)
            for run_id, (s, c, t) in enumerate(triples):
                _emit(writer, f_csv, base, "taco_cpu_opt", run_id, s, c, t)
            timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
            print(f"avg {sum(timed)/len(timed):.3f} ms  ({len(triples)} runs incl. warmup)")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"FAILED ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, "taco_cpu_opt", run_id, _NAN, _NAN, _NAN)

    # ── PRISMA CPU ────────────────────────────────────────────────────────────
    if prisma_bin:
        print("  [prisma_cpu]          ", end="", flush=True)
        bsp = orig_mtx.with_suffix(".bsp")
        try:
            triples = run_with_env(run_prisma_cpu, prisma_bin, bsp, runs, timeout, specialized)
            n_pairs  = ""
            n_groups = ""
            for run_id, (s, c, t) in enumerate(triples):
                writer.writerow({
                    **base,
                    "kernel":      "prisma_cpu",
                    "run_id":      run_id,
                    "symbolic_ms": _fmt(s),
                    "compute_ms":  _fmt(c),
                    "total_ms":    _fmt(t),
                    "n_pairs":     n_pairs,
                    "n_groups":    n_groups,
                })
            f_csv.flush()
            timed = [t for _, _, t in triples[1:]] or [t for _, _, t in triples]
            print(f"avg {sum(timed)/len(timed):.3f} ms  ({len(triples)} runs incl. warmup)")
        except FileNotFoundError as e:
            print(f"SKIP ({e})")
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"FAILED ({e})")
            for run_id in range(runs + 1):
                _emit(writer, f_csv, base, "prisma_cpu", run_id, _NAN, _NAN, _NAN)

    # ── Finch (placeholder) ───────────────────────────────────────────────────
    for run_id in range(runs + 1):
        _emit(writer, f_csv, base, "finch", run_id, _NAN, _NAN, _NAN)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="CPU SpGEMM benchmark on real SuiteSparse matrices (A×A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", metavar="MATRICES.csv",
                   help="input CSV with at least a 'name' column")

    g = p.add_argument_group("Run control")
    g.add_argument("--runs", type=int, default=5,
                   help="timed repetitions per matrix; run_id 0 = warmup (default: 5)")
    g.add_argument("--timeout", type=int, default=300,
                   help="per-binary and per-download timeout in seconds (default: 300)")
    g.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                   help="OMP_NUM_THREADS for all runs (default: nproc)")

    g = p.add_argument_group("Paths")
    g.add_argument("--out", default="cpu_results.csv",
                   help="output CSV, append mode (default: cpu_results.csv)")
    g.add_argument("--work-dir", default="",
                   help="directory for downloads and binaries (default: auto tempdir)")
    g.add_argument("--taco-bin", default="", metavar="PATH",
                   help="pre-built bench_taco_cpu binary (skips compilation)")
    g.add_argument("--taco-opt-bin", default="", metavar="PATH",
                   help="pre-built bench_taco_cpu_opt binary (skips compilation)")
    g.add_argument("--prisma-bin", default="", metavar="PATH",
                   help="pre-built prisma_cpu_bench binary (skips compilation)")

    g = p.add_argument_group("Build")
    g.add_argument("--no-compile", action="store_true",
                   help="skip compilation; binaries must already exist in work-dir")
    g.add_argument("--no-taco",   action="store_true", help="skip both TACO contenders")
    g.add_argument("--no-prisma", action="store_true", help="skip PRISMA CPU contender")
    g.add_argument("--blas", action="store_true",
                   help="link BLAS when compiling prisma_cpu_bench")
    g.add_argument("--specialized-kernels", action="store_true", dest="specialized",
                   help="use compile-time specialized GEMM kernels for known shapes")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    work_dir = (
        Path(args.work_dir) if args.work_dir
        else Path(tempfile.mkdtemp(prefix="ss_cpu_bench_"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    taco_bin:     Path | None = None
    taco_opt_bin: Path | None = None
    prisma_bin:   Path | None = None

    if not args.no_compile:
        print("Compiling:")
        if not args.no_taco:
            if args.taco_bin:
                taco_bin = Path(args.taco_bin)
            else:
                try:
                    taco_bin = compile_taco_cpu(
                        "taco_kernel.h", work_dir / "bench_taco_cpu", args.threads)
                except RuntimeError as e:
                    print(f"  bench_taco_cpu failed — skipping: {e}")
            if args.taco_opt_bin:
                taco_opt_bin = Path(args.taco_opt_bin)
            else:
                try:
                    taco_opt_bin = compile_taco_cpu(
                        "taco_kernel_opt.h", work_dir / "bench_taco_cpu_opt", args.threads)
                except RuntimeError as e:
                    print(f"  bench_taco_cpu_opt failed — skipping: {e}")
        if not args.no_prisma:
            if args.prisma_bin:
                prisma_bin = Path(args.prisma_bin)
            else:
                try:
                    prisma_bin = compile_prisma_cpu(
                        work_dir / "prisma_cpu_bench", args.blas)
                except RuntimeError as e:
                    print(f"  prisma_cpu_bench failed — skipping: {e}")
        print()
    else:
        if not args.no_taco:
            taco_bin     = Path(args.taco_bin)     if args.taco_bin     else (work_dir / "bench_taco_cpu")
            taco_opt_bin = Path(args.taco_opt_bin) if args.taco_opt_bin else (work_dir / "bench_taco_cpu_opt")
            if not taco_bin.exists():     taco_bin     = None
            if not taco_opt_bin.exists(): taco_opt_bin = None
        if not args.no_prisma:
            prisma_bin = Path(args.prisma_bin) if args.prisma_bin else (work_dir / "prisma_cpu_bench")
            if not prisma_bin.exists(): prisma_bin = None

    matrices = load_matrix_list(Path(args.csv))

    print(f"CSV            : {csv_path}")
    print(f"Work dir       : {work_dir}")
    print(f"TACO CPU bin   : {taco_bin     or '(disabled)'}")
    print(f"TACO opt bin   : {taco_opt_bin or '(disabled)'}")
    print(f"PRISMA CPU bin : {prisma_bin   or '(disabled)'}")
    print(f"Matrices       : {len(matrices)}")
    print(f"Runs/matrix    : {args.runs}")
    print(f"Threads        : {args.threads}")
    print(f"Timeout (s)    : {args.timeout}")
    print()

    write_header = _needs_header(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f_csv.write(f"# {ts}  input={args.csv}  runs={args.runs}  threads={args.threads}\n")
        writer = csv.DictWriter(
            f_csv, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()

        for i, row in enumerate(matrices, 1):
            name = row["name"]
            print(f"\n[{i}/{len(matrices)}] {name}")

            try:
                orig_mtx = download_matrix(name, group=row.get("group", ""), timeout=args.timeout)
                mtx = ensure_real_general(orig_mtx, Path("/tmp/mtx_cache"))
                print(f"  Matrix → {mtx}")
            except Exception as e:
                print(f"  DOWNLOAD FAILED: {e} — skipping")
                continue

            benchmark_matrix(
                matrix_row=row,
                mtx=mtx,
                orig_mtx=orig_mtx,
                taco_bin=taco_bin,
                taco_opt_bin=taco_opt_bin,
                prisma_bin=prisma_bin,
                runs=args.runs,
                timeout=args.timeout,
                threads=args.threads,
                specialized=args.specialized,
                writer=writer,
                f_csv=f_csv,
            )

    print(f"\nDone. Results appended to {csv_path}")


if __name__ == "__main__":
    main()
