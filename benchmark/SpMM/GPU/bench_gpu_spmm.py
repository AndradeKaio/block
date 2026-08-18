#!/usr/bin/env python3
"""
bench_gpu_spmm.py — Benchmark Prisma GPU SpMM vs cuSPARSE on a list of matrices.

Matrices are loaded from /home/kaio/datasets/suite-sparse/<group>/<name>/<name>.bsp

Usage:
  python bench_gpu_spmm.py --matrices matrices.csv \
      --prisma /tmp/prisma_gpu_spmm_bench \
      --cusparse /tmp/cusparse_spmm_bench

  # fp32 only, 10 runs, skip cuSPARSE
  python bench_gpu_spmm.py --matrices matrices.csv \
      --prisma /tmp/prisma_gpu_spmm_bench \
      --precision fp32 --runs 10 --no-cusparse

  # Force CUDA cores (no TC), both precisions
  python bench_gpu_spmm.py --matrices matrices.csv \
      --prisma /tmp/prisma_gpu_spmm_bench \
      --cusparse /tmp/cusparse_spmm_bench \
      --force-cuda-fallback --precision fp32,fp64
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_DATA_ROOT = Path("/home/kaio/datasets/suite-sparse")


# ── Matrix lookup ─────────────────────────────────────────────────────────────

def find_bsp(name: str, group: str) -> Path | None:
    candidates = [
        _DATA_ROOT / group / name / f"{name}.bsp",
        _DATA_ROOT / name / f"{name}.bsp",
    ]
    if not group:
        candidates = [_DATA_ROOT / name / f"{name}.bsp"]
    for p in candidates:
        if p.exists():
            return p
    # slow fallback: glob
    hits = list(_DATA_ROOT.rglob(f"{name}.bsp"))
    return hits[0] if hits else None


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        rows = list(reader)
    if not rows:
        sys.exit(f"No rows in {path}")
    if "name" not in rows[0]:
        sys.exit(f"CSV must have a 'name' column; got: {list(rows[0].keys())}")
    return rows


# ── JSON extraction ───────────────────────────────────────────────────────────

def parse_json_block(stdout: str) -> dict | None:
    m = re.search(r"JSON_BEGIN\s*(\{.*?\})\s*JSON_END", stdout, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


# ── Run one binary ────────────────────────────────────────────────────────────

def run_bench(binary: Path, bsp: Path, precision: str, runs: int, seed: int,
              extra_flags: list[str], timeout: int) -> dict | str:
    """Returns parsed JSON dict on success, or an error string on failure."""
    cmd = [
        str(binary), str(bsp),
        "--runs",      str(runs),
        "--seed",      str(seed),
        "--precision", precision,
        *extra_flags,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"TIMEOUT after {timeout}s"
    if r.returncode != 0:
        last = r.stderr.strip().splitlines()
        hint = last[-1] if last else "(no stderr)"
        return f"exit {r.returncode}: {hint}"
    d = parse_json_block(r.stdout)
    if d is None:
        return "no JSON output"
    return d


# ── Summarise timing arrays ───────────────────────────────────────────────────

def median_excl_warmup(arr: list[float]) -> float:
    """Median of runs[1:] (exclude warmup run 0)."""
    timed = arr[1:] if len(arr) > 1 else arr
    s = sorted(timed)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


# ── Pretty table ──────────────────────────────────────────────────────────────

def _col(s, w): return str(s)[:w].ljust(w)
def _rcol(s, w): return str(s)[:w].rjust(w)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Benchmark Prisma GPU SpMM vs cuSPARSE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--matrices",   required=True, help="CSV file with at least name,group columns")
    ap.add_argument("--prisma",     default="",    help="path to prisma_gpu_spmm_bench binary")
    ap.add_argument("--cusparse",   default="",    help="path to cusparse_spmm_bench binary")
    ap.add_argument("--precision",  default="fp64",
                    help="comma-separated precisions to run: fp32,fp64 (default: fp64)")
    ap.add_argument("--runs",       type=int, default=5,
                    help="timed repetitions per matrix (run 0 = warmup, default: 5)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--timeout",    type=int, default=120, help="per-run timeout in seconds")
    ap.add_argument("--no-cusparse",    action="store_true")
    ap.add_argument("--no-prisma",      action="store_true")
    ap.add_argument("--force-cuda-fallback", action="store_true",
                    help="pass --force-cuda-fallback to prisma (disables TC kernel)")
    ap.add_argument("--tc-classify", default="",
                    help="TC classification threshold, e.g. 4x4 (default: 16x16)")
    ap.add_argument("--out",        default="", help="write CSV results to this file")
    return ap.parse_args()


def main():
    args = parse_args()

    precisions = [p.strip() for p in args.precision.split(",") if p.strip()]
    matrices   = load_csv(Path(args.matrices))

    prisma_bin   = Path(args.prisma)   if args.prisma   else None
    cusparse_bin = Path(args.cusparse) if args.cusparse else None

    if not args.no_prisma and prisma_bin and not prisma_bin.exists():
        sys.exit(f"prisma binary not found: {prisma_bin}")
    if not args.no_cusparse and cusparse_bin and not cusparse_bin.exists():
        sys.exit(f"cusparse binary not found: {cusparse_bin}")

    prisma_extra = []
    if args.force_cuda_fallback:
        prisma_extra.append("--force-cuda-fallback")
    if args.tc_classify:
        prisma_extra += ["--tc-classify", args.tc_classify]

    # CSV output setup
    out_rows = []
    out_fields = [
        "matrix", "group", "precision", "kernel",
        "tc_tiles", "cuda_tiles",
        "tc_ms", "cuda_ms", "compute_ms", "total_ms", "symbolic_ms",
    ]

    # Print header
    print(f"{'Matrix':<22} {'Prec':<5} {'Kernel':<12} "
          f"{'TC ms':>8} {'CUDA ms':>8} {'Compute':>8} {'Total':>8} "
          f"{'TC tiles':>9} {'CUDA tiles':>10}")
    print("-" * 96)

    for row in matrices:
        name  = row["name"]
        group = row.get("group", "")
        bsp   = find_bsp(name, group)

        if bsp is None:
            print(f"{'  '+name:<22} BSP not found — skipping")
            continue

        for prec in precisions:
            # ── Prisma ────────────────────────────────────────────────────
            if not args.no_prisma and prisma_bin:
                t0 = time.time()
                d  = run_bench(prisma_bin, bsp, prec, args.runs, args.seed,
                               prisma_extra, args.timeout)
                elapsed = time.time() - t0

                if isinstance(d, str):
                    print(f"  {name:<20} {prec:<5} {'prisma':<12} ERROR: {d}")
                else:
                    tc_ms_arr   = d.get("tc_ms",   [0.0])
                    cu_ms_arr   = d.get("cuda_ms", [0.0])
                    sym_ms_arr  = d.get("symbolic_ms", [0.0])
                    tc_med  = median_excl_warmup(tc_ms_arr)
                    cu_med  = median_excl_warmup(cu_ms_arr)
                    sym_med = median_excl_warmup(sym_ms_arr)
                    comp    = tc_med + cu_med
                    total   = sym_med + comp
                    tc_t    = d.get("n_tc_tiles",   "?")
                    cu_t    = d.get("n_cuda_tiles",  "?")

                    if args.force_cuda_fallback:
                        label = "prisma-cuda"
                    elif args.tc_classify:
                        label = f"prisma-tc-{args.tc_classify}"
                    else:
                        label = "prisma-tc"
                    print(f"  {name:<20} {prec:<5} {label:<12} "
                          f"{tc_med:>8.3f} {cu_med:>8.3f} {comp:>8.3f} {total:>8.3f} "
                          f"{str(tc_t):>9} {str(cu_t):>10}")
                    out_rows.append({
                        "matrix": name, "group": group, "precision": prec,
                        "kernel": label,
                        "tc_tiles": tc_t, "cuda_tiles": cu_t,
                        "tc_ms": f"{tc_med:.4f}", "cuda_ms": f"{cu_med:.4f}",
                        "compute_ms": f"{comp:.4f}", "total_ms": f"{total:.4f}",
                        "symbolic_ms": f"{sym_med:.4f}",
                    })

            # ── cuSPARSE ──────────────────────────────────────────────────
            if not args.no_cusparse and cusparse_bin:
                t0 = time.time()
                d  = run_bench(cusparse_bin, bsp, prec, args.runs, args.seed,
                               [], args.timeout)
                elapsed = time.time() - t0

                if isinstance(d, str):
                    print(f"  {name:<20} {prec:<5} {'cusparse':<12} ERROR: {d}")
                else:
                    comp_arr = d.get("compute_ms", [0.0])
                    sym_arr  = d.get("symbolic_ms", [0.0])
                    comp     = median_excl_warmup(comp_arr)
                    sym      = median_excl_warmup(sym_arr)
                    total    = sym + comp

                    print(f"  {name:<20} {prec:<5} {'cusparse':<12} "
                          f"{'':>8} {'':>8} {comp:>8.3f} {total:>8.3f} "
                          f"{'':>9} {'':>10}")
                    out_rows.append({
                        "matrix": name, "group": group, "precision": prec,
                        "kernel": "cusparse",
                        "tc_tiles": "", "cuda_tiles": "",
                        "tc_ms": "", "cuda_ms": "",
                        "compute_ms": f"{comp:.4f}", "total_ms": f"{total:.4f}",
                        "symbolic_ms": f"{sym:.4f}",
                    })

        # blank line between matrices when running multiple precisions
        if len(precisions) > 1:
            print()

    # ── Optional CSV output ───────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_fields, lineterminator="\n")
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nResults written to {out_path}")

    # ── Speedup summary ───────────────────────────────────────────────────────
    # Group by (matrix, precision) and print prisma vs cusparse ratio.
    if not args.no_prisma and not args.no_cusparse and prisma_bin and cusparse_bin:
        print()
        print(f"{'Speedup (cuSPARSE / Prisma compute_ms)':}")
        print(f"  {'Matrix':<22} {'Prec':<5} {'Prisma ms':>10} {'cuSPARSE ms':>12} {'Speedup':>8}")
        print("  " + "-" * 62)
        # index rows by (matrix, precision, kernel)
        by_key: dict[tuple, dict] = {}
        for r in out_rows:
            by_key[(r["matrix"], r["precision"], r["kernel"])] = r
        seen = set()
        for row in matrices:
            name  = row["name"]
            group = row.get("group", "")
            for prec in precisions:
                key = (name, prec)
                if key in seen:
                    continue
                seen.add(key)
                if args.force_cuda_fallback:
                    prisma_label = "prisma-cuda"
                elif args.tc_classify:
                    prisma_label = f"prisma-tc-{args.tc_classify}"
                else:
                    prisma_label = "prisma-tc"
                prisma_key   = (name, prec, prisma_label)
                cusparse_key = (name, prec, "cusparse")
                pr = by_key.get(prisma_key)
                cu = by_key.get(cusparse_key)
                if pr and cu and pr["compute_ms"] and cu["compute_ms"]:
                    p_ms = float(pr["compute_ms"])
                    c_ms = float(cu["compute_ms"])
                    ratio = c_ms / p_ms if p_ms > 0 else float("nan")
                    flag  = " ✓" if ratio > 1 else " ✗"
                    print(f"  {name:<22} {prec:<5} {p_ms:>10.3f} {c_ms:>12.3f} "
                          f"{ratio:>7.2f}x{flag}")


if __name__ == "__main__":
    main()
