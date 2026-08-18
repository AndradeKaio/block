#!/usr/bin/env python3
"""
smtx_to_mtx.py — Convert structural CSR (.smtx) files to MatrixMarket (.mtx).

.smtx format (3 lines, text):
  Line 1: rows, cols, nnz
  Line 2: rowptr  (rows+1 space-separated integers, 0-indexed)
  Line 3: colidx  (nnz space-separated integers, 0-indexed)

All nonzero values are written as 1.0 (structural / pattern matrix).

Usage:
  python smtx_to_mtx.py file.smtx [file.mtx]
  python smtx_to_mtx.py *.smtx            # batch: each file.smtx -> file.mtx
"""

import argparse
import pathlib
import sys


def convert(src: pathlib.Path, dst: pathlib.Path) -> None:
    lines = src.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{src}: expected 3 lines, got {len(lines)}")

    rows, cols, nnz = [int(x.strip()) for x in lines[0].split(",")]
    rowptr = list(map(int, lines[1].split()))
    colidx = list(map(int, lines[2].split()))

    if len(rowptr) != rows + 1:
        raise ValueError(f"{src}: rowptr has {len(rowptr)} values, expected {rows + 1}")
    if len(colidx) != nnz:
        raise ValueError(f"{src}: colidx has {len(colidx)} values, expected {nnz}")
    if rowptr[-1] != nnz:
        raise ValueError(f"{src}: rowptr[-1]={rowptr[-1]} != nnz={nnz}")

    with open(dst, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{rows} {cols} {nnz}\n")
        for r in range(rows):
            for idx in range(rowptr[r], rowptr[r + 1]):
                f.write(f"{r + 1} {colidx[idx] + 1} 1.0\n")

    print(f"{src} -> {dst}  ({rows}x{cols}, NNZ={nnz})")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert .smtx to MatrixMarket .mtx")
    p.add_argument("inputs", nargs="+", metavar="FILE.smtx")
    p.add_argument("-o", "--output", metavar="FILE.mtx",
                   help="output path (only valid when converting a single file)")
    args = p.parse_args()

    if args.output and len(args.inputs) > 1:
        p.error("--output can only be used with a single input file")

    errors = 0
    for inp in args.inputs:
        src = pathlib.Path(inp)
        dst = pathlib.Path(args.output) if args.output else src.with_suffix(".mtx")
        try:
            convert(src, dst)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            errors += 1

    sys.exit(errors)


if __name__ == "__main__":
    main()
