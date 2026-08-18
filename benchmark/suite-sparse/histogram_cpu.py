#!/usr/bin/env python3
"""
histogram_cpu.py — GEMM shape histogram over SuiteSparse matrices (A×A).

For each matrix in the input CSV, runs the PRISMA symbolic pipeline and records
the distribution of GEMM shapes (M, K, N) that arise during block-sparse SpGEMM.

Output: block_histogram.csv with one row per (matrix, shape).

Usage:
  python histogram_cpu.py matrices.csv
  python histogram_cpu.py matrices.csv --out block_histogram.csv --timeout 120
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


_SCRIPT_DIR = Path(__file__).parent
_CPU_DIR    = _SCRIPT_DIR.parent / "SpGEMM" / "CPU"
_CORE_DIR   = _SCRIPT_DIR.parent / "core"

_CSV_FIELDS = [
    "matrix_name",
    "group",
    "rows",
    "cols",
    "nnz",
    "total_calls",
    "total_flops",
    "M",
    "K",
    "N",
    "count",
    "flops_per_call",
    "pct_calls",
]

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
        "/usr/lib/hdf5/serial",
        "/usr/local/lib",
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


def compile_histogram_bin(out: Path) -> Path:
    srcs = [str(_CORE_DIR / s) for s in _CORE_SRCS]
    src  = str(_CPU_DIR / "histogram_bin.cpp")
    cmd  = [
        "g++", "-O2", "-std=c++20",
        "-DHAVE_HDF5",
        f"-I{_CORE_DIR}", f"-I{_CPU_DIR}", f"-I{_HDF5_INC}",
        *srcs, src,
        str(Path(_HDF5_LIB) / "libhdf5.so"),
        "-o", str(out),
    ]
    print(f"  {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"g++ failed:\n{r.stderr}")
    return out


def _parse_json_block(stdout: str) -> dict:
    if "JSON_BEGIN" not in stdout or "JSON_END" not in stdout:
        return {}
    s = stdout.index("JSON_BEGIN") + len("JSON_BEGIN")
    e = stdout.index("JSON_END")
    try:
        return json.loads(stdout[s:e].strip())
    except json.JSONDecodeError:
        return {}


def run_histogram(binary: Path, bsp: Path, timeout: int) -> dict:
    if not bsp.exists():
        raise FileNotFoundError(f"BSP not found: {bsp}")
    r = subprocess.run(
        [str(binary), str(bsp)],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"histogram_bin exited {r.returncode}:\n{r.stderr[-800:]}")
    d = _parse_json_block(r.stdout)
    if not d:
        raise RuntimeError("histogram_bin: could not parse JSON output")
    return d


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


def load_matrix_list(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        rows = list(reader)
    if not rows:
        sys.exit(f"No rows found in {csv_path}")
    if "name" not in rows[0]:
        sys.exit(f"Input CSV must have a 'name' column; got: {list(rows[0].keys())}")
    return rows


def _needs_header(csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True
    expected = ",".join(_CSV_FIELDS)
    with open(csv_path, newline="") as f:
        for line in f:
            if not line.startswith("#"):
                return line.rstrip("\r\n") != expected
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="GEMM shape histogram over SuiteSparse A×A matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", metavar="MATRICES.csv",
                   help="input CSV with at least a 'name' column")
    p.add_argument("--out", default="block_histogram.csv",
                   help="output CSV (default: block_histogram.csv)")
    p.add_argument("--bin", default="", dest="histogram_bin",
                   help="pre-built histogram_bin binary (skips compilation)")
    p.add_argument("--no-compile", action="store_true",
                   help="skip compilation; binary must already exist")
    p.add_argument("--timeout", type=int, default=120,
                   help="per-matrix timeout in seconds (default: 120)")
    p.add_argument("--work-dir", default="",
                   help="directory for downloads and binaries (default: auto tempdir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    work_dir = (
        Path(args.work_dir) if args.work_dir
        else Path(tempfile.mkdtemp(prefix="ss_hist_"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    hist_bin: Path | None = None

    if args.histogram_bin:
        hist_bin = Path(args.histogram_bin)
    elif args.no_compile:
        hist_bin = work_dir / "histogram_bin"
        if not hist_bin.exists():
            sys.exit(f"--no-compile set but binary not found: {hist_bin}")
    else:
        print("Compiling:")
        try:
            hist_bin = compile_histogram_bin(work_dir / "histogram_bin")
        except RuntimeError as e:
            sys.exit(f"Compilation failed: {e}")
        print()

    matrices = load_matrix_list(Path(args.csv))

    print(f"CSV          : {csv_path}")
    print(f"Binary       : {hist_bin}")
    print(f"Matrices     : {len(matrices)}")
    print(f"Timeout (s)  : {args.timeout}")
    print()

    write_header = _needs_header(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f_csv.write(f"# {ts}  input={args.csv}\n")
        writer = csv.DictWriter(
            f_csv, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n"
        )
        if write_header:
            writer.writeheader()

        for i, row in enumerate(matrices, 1):
            name = row["name"]
            print(f"[{i}/{len(matrices)}] {name}", end="  ", flush=True)

            try:
                orig_mtx = download_matrix(name, group=row.get("group", ""), timeout=args.timeout)
                ensure_real_general(orig_mtx, Path("/tmp/mtx_cache"))
            except Exception as e:
                print(f"DOWNLOAD FAILED: {e}")
                continue

            bsp = orig_mtx.with_suffix(".bsp")
            try:
                d = run_histogram(hist_bin, bsp, args.timeout)
            except FileNotFoundError as e:
                print(f"SKIP ({e})")
                continue
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"FAILED ({e})")
                continue

            total_calls = d.get("total_calls", 0)
            total_flops = d.get("total_flops", 0)
            shapes = d.get("shapes", [])

            base = {
                "matrix_name": name,
                "group":       row.get("group", ""),
                "rows":        row.get("rows",  ""),
                "cols":        row.get("cols",  ""),
                "nnz":         row.get("nnz",   ""),
                "total_calls": total_calls,
                "total_flops": total_flops,
            }

            if not shapes:
                print("no contributions")
                continue

            for s in shapes:
                writer.writerow({
                    **base,
                    "M":             s["M"],
                    "K":             s["K"],
                    "N":             s["N"],
                    "count":         s["count"],
                    "flops_per_call": s["flops_per_call"],
                    "pct_calls":     f"{s['pct']:.4f}",
                })
            f_csv.flush()
            print(f"{len(shapes)} shapes  total_calls={total_calls}  total_flops={total_flops}")

    print(f"\nDone. Results appended to {csv_path}")


if __name__ == "__main__":
    main()
