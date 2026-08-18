#!/usr/bin/env python3
"""
suite-sparse/perf_spmm.py — perf stat comparison across all SpMM contenders.

Runs every contender under `perf stat` on all matrices in a CSV list, collects
hardware counters (LLC misses, cycles, instructions) alongside timing, and
outputs perf_results.csv.  Mirrors benchmark_spmm.py in CLI and matrix lookup.

Usage:
  python perf_spmm.py MATRICES.csv
  python perf_spmm.py MATRICES.csv --runs 3 --out perf_results.csv
  python perf_spmm.py MATRICES.csv --no-taco --no-block-spmm-blas
  python perf_spmm.py MATRICES.csv --kernels taco,taco_opt0,prisma_specialized
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (mirror benchmark_spmm.py)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_SPMM_DIR   = _SCRIPT_DIR.parent / "SpMM"

_DATA_ROOT = Path("/home/kaio/datasets/suite-sparse")

# ---------------------------------------------------------------------------
# Contender table
# (label, binary_stem, input_ext, extra_flags)
# ---------------------------------------------------------------------------

CONTENDERS = [
    ("taco",               "bench_taco_spmm_taco",     ".mtx", []),
    ("taco_opt0",          "bench_taco_spmm_taco_opt0", ".mtx", []),
    ("taco_opt1",          "bench_taco_spmm_taco_opt1", ".mtx", []),
    ("prisma_cpu",         "prisma_cpu_spmm_bench",    ".bsp", []),
    ("prisma_specialized", "prisma_cpu_spmm_bench",    ".bsp", ["--specialized-kernels"]),
    ("prisma_static",      "prisma_cpu_spmm_bench",    ".bsp", ["--specialized-kernels", "--static"]),
    ("prisma_tiled",       "prisma_cpu_spmm_bench",    ".bsp", ["--specialized-kernels", "--tile-n", "512"]),
    ("prisma_auto",        "prisma_cpu_spmm_bench",    ".bsp", ["--specialized-kernels", "--auto"]),
]

# ---------------------------------------------------------------------------
# Output CSV schema
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "matrix_name", "group", "rows", "cols", "nnz",
    "kernel", "ms_med",
    "l3_misses", "l3_accesses",
    "l1_dcache_load_misses", "cache_misses",
    "dram_gb_per_run", "bw_gbs", "ipc",
    "cycles", "instructions", "fp_ops",
]

# ---------------------------------------------------------------------------
# Helpers shared with benchmark_spmm.py
# ---------------------------------------------------------------------------

_ASM_RE  = re.compile(r"run_(\d+)_assemble_ns=(\d+)")
_COMP_RE = re.compile(r"run_(\d+)_compute_ns=(\d+)")

# perf stat output line: leading whitespace, numeric value (with commas),
# whitespace, event name.  Example:
#   1,234,567,890      cycles
_PERF_RE = re.compile(r"^\s*([\d,]+)\s+([\w\-\.]+)", re.MULTILINE)


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


def _parse_json_block(stdout: str) -> dict:
    if "JSON_BEGIN" not in stdout or "JSON_END" not in stdout:
        return {}
    s = stdout.index("JSON_BEGIN") + len("JSON_BEGIN")
    e = stdout.index("JSON_END")
    try:
        return json.loads(stdout[s:e].strip())
    except json.JSONDecodeError:
        return {}


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
# perf stat invocation and parsing
# ---------------------------------------------------------------------------

# AMD Zen events: l3_misses is the correct L3 miss counter (replaces generic
# LLC-load-misses which returns 0 on Zen).  fp_ret_sse_avx_ops.all counts all
# SSE/AVX FP ops retired; Intel equivalent is fp_arith_inst_retired.*.
_PERF_EVENTS = (
    "cycles,instructions,"
    "l3_misses,l3_cache_accesses,"
    "L1-dcache-load-misses,cache-misses,"
    "fp_ret_sse_avx_ops.all"
)


def run_with_perf(binary: Path, extra_args: list[str],
                  timeout: int) -> tuple[str, str, int]:
    """Run binary under perf stat; return (stdout, stderr, returncode)."""
    cmd = ["perf", "stat", "-e", _PERF_EVENTS, "--", str(binary)] + extra_args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr, r.returncode


def parse_perf_stderr(stderr: str) -> dict[str, int]:
    """Extract hardware counter values from perf stat stderr."""
    counters: dict[str, int] = {}
    for m in _PERF_RE.finditer(stderr):
        raw  = m.group(1).replace(",", "")
        name = m.group(2)
        try:
            counters[name] = int(raw)
        except ValueError:
            pass
    return counters


# ---------------------------------------------------------------------------
# Timing extraction from binary stdout
# ---------------------------------------------------------------------------

def extract_ms_taco(stdout: str, runs: int) -> list[float]:
    """Parse run_N_compute_ns= lines; return ms values for run 0..runs."""
    comp_ns: dict[int, int] = {}
    for line in stdout.splitlines():
        m = _COMP_RE.match(line)
        if m:
            comp_ns[int(m.group(1))] = int(m.group(2))
    return [comp_ns.get(i, 0) / 1e6 for i in range(runs + 1)]


def extract_ms_json(stdout: str) -> list[float]:
    """Parse JSON_BEGIN..JSON_END block; return compute_ms list."""
    d = _parse_json_block(stdout)
    return d.get("compute_ms", [])


# ---------------------------------------------------------------------------
# Per-contender run
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "nan"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def run_contender(label: str, binary: Path, input_file: Path,
                  input_ext: str, extra_flags: list[str],
                  runs: int, timeout: int) -> dict | None:
    """
    Run one contender under perf stat.  Returns a result dict or None on failure.
    """
    args = [str(input_file), "--runs", str(runs)] + extra_flags

    try:
        stdout, stderr, rc = run_with_perf(binary, args, timeout)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT")
        return None

    if rc != 0:
        print(f"FAILED (exit {rc})")
        return None

    # Extract timing
    if input_ext == ".mtx":
        all_ms = extract_ms_taco(stdout, runs)
    else:
        all_ms = extract_ms_json(stdout)

    if not all_ms:
        print(f"FAILED (no timing output)")
        return None

    timed  = all_ms[1:] if len(all_ms) > 1 else all_ms
    ms_med = statistics.median(timed)

    # Extract perf counters
    ctr = parse_perf_stderr(stderr)
    cycles       = ctr.get("cycles", 0)
    instructions = ctr.get("instructions", 0)
    l3_miss      = ctr.get("l3_misses", 0)
    l3_acc       = ctr.get("l3_cache_accesses", 0)
    l1_ld_miss   = ctr.get("L1-dcache-load-misses", 0)
    cache_miss   = ctr.get("cache-misses", 0)
    fp_ops       = ctr.get("fp_ret_sse_avx_ops.all", 0)

    n_runs = len(timed)
    # L3 miss → DRAM: each miss fetches a 64-byte cache line
    dram_gb  = l3_miss * 64 / 1e9 / n_runs if n_runs else 0.0
    bw_gbs   = dram_gb / (ms_med / 1000) if ms_med > 0 else 0.0
    ipc      = instructions / cycles if cycles > 0 else 0.0

    return {
        "ms_med":                ms_med,
        "l3_misses":             l3_miss,
        "l3_accesses":           l3_acc,
        "l1_dcache_load_misses": l1_ld_miss,
        "cache_misses":          cache_miss,
        "dram_gb_per_run":       dram_gb,
        "bw_gbs":                bw_gbs,
        "ipc":                   ipc,
        "cycles":                cycles,
        "instructions":          instructions,
        "fp_ops":                fp_ops,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SpMM perf stat comparison (all contenders) on SuiteSparse matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("csv", metavar="MATRICES.csv",
                   help="input CSV with at least a 'name' column (same format as benchmark_spmm.py)")

    g = p.add_argument_group("Run control")
    g.add_argument("--runs", type=int, default=5,
                   help="timed repetitions per matrix (run 0 = warmup, default: 5)")
    g.add_argument("--timeout", type=int, default=600,
                   help="per-contender timeout in seconds (default: 600)")

    g = p.add_argument_group("Paths")
    g.add_argument("--out", default="perf_results.csv",
                   help="output CSV, append mode (default: perf_results.csv)")
    g.add_argument("--bin-dir", default="", dest="bin_dir",
                   help="directory containing compiled binaries (default: ../SpMM/)")

    g = p.add_argument_group("Kernel filter")
    g.add_argument("--no-taco", action="store_true",
                   help="skip taco, taco_opt0, taco_opt1")
    g.add_argument("--no-prisma", action="store_true",
                   help="skip all Prisma kernels")
    g.add_argument("--kernels", default="",
                   help="comma-separated list of kernel labels to run (overrides --no-* flags)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    matrices = load_matrix_list(Path(args.csv))
    bin_dir  = Path(args.bin_dir) if args.bin_dir else _SPMM_DIR

    # Build active contender list
    if args.kernels:
        keep = set(args.kernels.split(","))
        active = [(lbl, stem, ext, flags)
                  for lbl, stem, ext, flags in CONTENDERS if lbl in keep]
        if not active:
            sys.exit(f"No matching kernels for --kernels={args.kernels!r}; "
                     f"available: {[c[0] for c in CONTENDERS]}")
    else:
        def _skip(lbl: str) -> bool:
            if args.no_taco and lbl.startswith("taco"):
                return True
            if args.no_prisma and lbl.startswith("prisma"):
                return True
            return False
        active = [(lbl, stem, ext, flags)
                  for lbl, stem, ext, flags in CONTENDERS if not _skip(lbl)]

    # Verify binaries exist
    missing = []
    seen_stems: set[str] = set()
    for lbl, stem, _, _ in active:
        if stem not in seen_stems:
            b = bin_dir / stem
            if not b.exists():
                missing.append(f"  {stem}  (needed by {lbl})")
            seen_stems.add(stem)
    if missing:
        sys.exit("Missing binaries — build them first:\n" + "\n".join(missing))

    csv_path = Path(args.out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output  : {csv_path}")
    print(f"Matrices: {len(matrices)}")
    print(f"Kernels : {[c[0] for c in active]}")
    print(f"Runs    : {args.runs}  (run 0 = warmup)")
    print()

    write_header = _needs_header(csv_path)
    with open(csv_path, "a", newline="") as f_csv:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f_csv.write(f"# {ts}  input={args.csv}  runs={args.runs}\n")
        writer = csv.DictWriter(f_csv, fieldnames=_CSV_FIELDS,
                                extrasaction="ignore", lineterminator="\n")
        if write_header:
            writer.writeheader()

        for i, row in enumerate(matrices, 1):
            name  = row["name"]
            group = row.get("group", "")
            print(f"[{i}/{len(matrices)}] {name}")

            mtx = find_mtx(name, group)
            if mtx is None:
                print(f"  MTX not found — skipping\n")
                continue

            bsp = mtx.with_suffix(".bsp")

            base = {
                "matrix_name": name,
                "group":       group,
                "rows":        row.get("rows", ""),
                "cols":        row.get("cols", ""),
                "nnz":         row.get("nnz",  ""),
            }

            results_for_matrix: list[tuple[str, dict | None]] = []

            for lbl, stem, ext, extra_flags in active:
                binary = bin_dir / stem
                if ext == ".bsp" and not bsp.exists():
                    print(f"  [{lbl:<22}]  BSP not found — skipping")
                    results_for_matrix.append((lbl, None))
                    continue

                input_file = mtx if ext == ".mtx" else bsp
                print(f"  [{lbl:<22}]  … ", end="", flush=True)

                result = run_contender(lbl, binary, input_file, ext,
                                       extra_flags, args.runs, args.timeout)
                results_for_matrix.append((lbl, result))

                if result is None:
                    writer.writerow({**base, "kernel": lbl,
                                     **{k: "nan" for k in _CSV_FIELDS
                                        if k not in base and k != "kernel"}})
                else:
                    print(
                        f"{result['ms_med']:8.2f} ms  "
                        f"L3miss={result['l3_misses']/1e6:7.1f}M  "
                        f"L3acc={result['l3_accesses']/1e6:7.1f}M  "
                        f"L1_ld={result['l1_dcache_load_misses']/1e6:7.1f}M  "
                        f"DRAM={result['dram_gb_per_run']:6.2f}GB  "
                        f"BW={result['bw_gbs']:6.1f}GB/s  "
                        f"IPC={result['ipc']:.2f}  "
                        f"FP={result['fp_ops']/1e9:.2f}G"
                    )
                    writer.writerow({
                        **base,
                        "kernel":                lbl,
                        "ms_med":                _fmt(result["ms_med"]),
                        "l3_misses":             result["l3_misses"],
                        "l3_accesses":           result["l3_accesses"],
                        "l1_dcache_load_misses": result["l1_dcache_load_misses"],
                        "cache_misses":          result["cache_misses"],
                        "dram_gb_per_run":       _fmt(result["dram_gb_per_run"]),
                        "bw_gbs":                _fmt(result["bw_gbs"]),
                        "ipc":                   _fmt(result["ipc"]),
                        "cycles":                result["cycles"],
                        "instructions":          result["instructions"],
                        "fp_ops":                result["fp_ops"],
                    })
                f_csv.flush()

            # Per-matrix summary table
            valid = [(lbl, r) for lbl, r in results_for_matrix if r is not None]
            if len(valid) > 1:
                print()
                print(f"  {'kernel':<24} {'ms_med':>8} {'DRAM_GB':>9} {'BW_GB/s':>9} {'IPC':>6} {'FP_Gops':>9}")
                for lbl, r in valid:
                    print(f"  {lbl:<24} {r['ms_med']:8.2f} "
                          f"{r['dram_gb_per_run']:9.2f} {r['bw_gbs']:9.1f} {r['ipc']:6.2f} "
                          f"{r['fp_ops']/1e9:9.2f}")
            print()


if __name__ == "__main__":
    main()
