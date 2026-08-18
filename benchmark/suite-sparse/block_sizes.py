#!/usr/bin/env python3
"""
block_sizes.py — Show block size distribution for a CSV list of matrices.

For each matrix, loads its .bsp file and reports the unique (h×w) block
shapes: how many blocks, how many dense cells, and what fraction of total
cells each shape represents.

Usage:
  python block_sizes.py matrices.csv
  python block_sizes.py matrices.csv --data-root /home/kaio/datasets/suite-sparse
  python block_sizes.py matrices.csv --top 10          # limit shape rows per matrix
  python block_sizes.py matrices.csv --summary          # cross-matrix shape table only
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

_DEFAULT_ROOT = Path("/home/kaio/datasets/suite-sparse")


def find_bsp(name: str, group: str, root: Path) -> Path | None:
    candidates = [
        root / group / name / f"{name}.bsp",
        root / name / f"{name}.bsp",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(root.rglob(f"{name}.bsp"))
    return hits[0] if hits else None


def load_block_shapes(bsp: Path):
    with h5py.File(str(bsp), "r") as f:
        M  = int(f.attrs["matrix_rows"])
        N  = int(f.attrs["matrix_cols"])
        bh = f["block_h"][:]
        bw = f["block_w"][:]
    return M, N, bh.astype(np.int32), bw.astype(np.int32)


def shape_table(bh, bw, top: int | None):
    """Return list of (h, w, count, cells, frac) sorted by cells desc."""
    from collections import Counter
    counts = Counter(zip(bh.tolist(), bw.tolist()))
    total_cells = int(np.sum(bh.astype(np.int64) * bw.astype(np.int64)))
    rows = []
    for (h, w), cnt in sorted(counts.items(), key=lambda x: -x[1]*x[0]*x[1]):
        cells = cnt * h * w
        rows.append((h, w, cnt, cells, cells / total_cells if total_cells else 0.0))
    if top:
        rows = rows[:top]
    return rows, total_cells


def print_matrix_report(name, M, N, bh, bw, top):
    rows, total_cells = shape_table(bh, bw, top)
    n_blocks = len(bh)
    print(f"\n{'─'*62}")
    print(f"  {name}  ({M}×{N})  blocks={n_blocks:,}  total_cells={total_cells:,}")
    print(f"  {'Shape':<10} {'Blocks':>8} {'Cells':>12} {'% cells':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*12} {'─'*8}")
    for h, w, cnt, cells, frac in rows:
        print(f"  {h}×{w:<8} {cnt:>8,} {cells:>12,} {frac:>7.1%}")
    if top and len(shape_table(bh, bw, None)[0]) > top:
        remaining = len(shape_table(bh, bw, None)[0]) - top
        print(f"  ... {remaining} more shapes")


def main():
    ap = argparse.ArgumentParser(description="Block size distribution across matrices")
    ap.add_argument("csv",        help="CSV file with name,group columns")
    ap.add_argument("--data-root", default=str(_DEFAULT_ROOT))
    ap.add_argument("--top",      type=int, default=None,
                    help="show only top N shapes per matrix (by cell count)")
    ap.add_argument("--summary",  action="store_true",
                    help="print a cross-matrix summary table instead of per-matrix detail")
    ap.add_argument("--out",      default="", help="write results to this file")
    args = ap.parse_args()

    root   = Path(args.data_root)
    out_f  = open(args.out, "w") if args.out else None

    import builtins
    _real_print = builtins.print
    def print(*a, **kw):
        _real_print(*a, **kw)
        if out_f:
            kw.pop("file", None)
            _real_print(*a, **kw, file=out_f)

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        matrices = list(reader)

    if not matrices:
        sys.exit("No rows in CSV")

    # ── Per-matrix detail ────────────────────────────────────────────────────
    # Also collect cross-matrix shape aggregates
    global_shapes: dict[tuple, dict] = defaultdict(lambda: {"blocks": 0, "cells": 0, "matrices": set()})
    summary_rows = []

    for row in matrices:
        name  = row["name"]
        group = row.get("group", "")
        bsp   = find_bsp(name, group, root)
        if bsp is None:
            print(f"  {name}: BSP not found — skipping", file=sys.stderr)
            continue

        M, N, bh, bw = load_block_shapes(bsp)
        total_cells = int(np.sum(bh.astype(np.int64) * bw.astype(np.int64)))
        n_shapes = len(set(zip(bh.tolist(), bw.tolist())))
        dominant_h, dominant_w = int(bh[0]), int(bw[0])  # will recompute below

        # find dominant shape by cell count
        from collections import Counter
        counts = Counter(zip(bh.tolist(), bw.tolist()))
        (dom_h, dom_w), _ = max(counts.items(), key=lambda x: x[1]*x[0]*x[1])

        summary_rows.append({
            "name": name, "M": M, "N": N,
            "blocks": len(bh), "shapes": n_shapes,
            "dominant": f"{dom_h}×{dom_w}",
            "total_cells": total_cells,
        })

        for (h, w), cnt in counts.items():
            global_shapes[(h, w)]["blocks"]   += cnt
            global_shapes[(h, w)]["cells"]    += cnt * h * w
            global_shapes[(h, w)]["matrices"].add(name)

        if not args.summary:
            print_matrix_report(name, M, N, bh, bw, args.top)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  SUMMARY")
    print(f"  {'Matrix':<22} {'Rows':>7} {'Blocks':>8} {'Shapes':>7} {'Dominant':>10} {'Cells':>12}")
    print(f"  {'─'*22} {'─'*7} {'─'*8} {'─'*7} {'─'*10} {'─'*12}")
    for r in summary_rows:
        print(f"  {r['name']:<22} {r['M']:>7,} {r['blocks']:>8,} {r['shapes']:>7} "
              f"{r['dominant']:>10} {r['total_cells']:>12,}")

    # ── Cross-matrix shape frequency ─────────────────────────────────────────
    if len(summary_rows) > 1:
        print(f"\n  SHAPES ACROSS ALL MATRICES (by total cells)")
        print(f"  {'Shape':<10} {'Blocks':>10} {'Cells':>14} {'Matrices':>10}")
        print(f"  {'─'*10} {'─'*10} {'─'*14} {'─'*10}")
        sorted_shapes = sorted(global_shapes.items(), key=lambda x: -x[1]["cells"])
        limit = args.top or 20
        for (h, w), info in sorted_shapes[:limit]:
            print(f"  {h}×{w:<8} {info['blocks']:>10,} {info['cells']:>14,} "
                  f"{len(info['matrices']):>10}")


    if out_f:
        out_f.close()
        _real_print(f"Results written to {args.out}")

if __name__ == "__main__":
    main()
